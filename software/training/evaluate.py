from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from software.models.tiny_cnn import TinyCNN
from software.preprocessing.image_loader import (
    build_dataset_bundle,
    create_dataloaders,
    print_class_counts,
)
from software.training.metrics import (
    accumulate_batch_metrics,
    confusion_matrix_counts,
    finalize_metrics,
    predict_classes,
)
from software.utils.checkpoint import (
    checkpoint_path_from_config,
    count_trainable_parameters,
    restore_trained_model,
)
from software.utils.config import load_training_config, resolve_path
from software.utils.reproducibility import select_device, set_seed


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()
        except Exception:  # noqa: BLE001
            pass


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: list[str] | None = None,
    *,
    warmup_batches: int = 2,
    measure_inference: bool = False,
) -> dict[str, Any]:
    """Evaluate a model without modifying its parameters."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = [0 for _ in range(num_classes)]
    class_totals = [0 for _ in range(num_classes)]
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    measured_batches = 0
    measured_images = 0
    measured_seconds = 0.0
    warmup_remaining = warmup_batches if measure_inference else 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        if measure_inference and warmup_remaining > 0:
            _ = model(images)
            synchronize_device(device)
            warmup_remaining -= 1

        if measure_inference:
            synchronize_device(device)
            start = time.perf_counter()
            scores = model(images)
            synchronize_device(device)
            elapsed = time.perf_counter() - start
            measured_seconds += elapsed
            measured_batches += 1
            measured_images += int(images.shape[0])
        else:
            scores = model(images)

        loss = loss_function(scores, labels)
        predictions = predict_classes(scores)
        confusion += confusion_matrix_counts(predictions, labels, num_classes).cpu()

        updated = accumulate_batch_metrics(
            scores,
            labels,
            float(loss.item()),
            num_classes=num_classes,
            total_loss=total_loss,
            total_correct=total_correct,
            total_samples=total_samples,
            class_correct=class_correct,
            class_totals=class_totals,
        )
        total_loss = updated["total_loss"]
        total_correct = updated["total_correct"]
        total_samples = updated["total_samples"]
        class_correct = updated["class_correct"]
        class_totals = updated["class_totals"]

    metrics = finalize_metrics(
        total_loss=total_loss,
        total_correct=total_correct,
        total_samples=total_samples,
        class_correct=class_correct,
        class_totals=class_totals,
        class_names=class_names,
        confusion_matrix=confusion.tolist(),
    )
    if measure_inference and measured_batches > 0:
        metrics["average_inference_time_per_batch_s"] = (
            measured_seconds / measured_batches
        )
        metrics["average_inference_time_per_image_s"] = (
            measured_seconds / measured_images
        )
        metrics["measured_batches"] = measured_batches
        metrics["measured_images"] = measured_images
    else:
        metrics["average_inference_time_per_batch_s"] = None
        metrics["average_inference_time_per_image_s"] = None

    if was_training:
        model.train()
    else:
        model.eval()
    return metrics


def print_evaluation_metrics(metrics: dict[str, Any], *, title: str = "Evaluation") -> None:
    print(title)
    print("-" * len(title))
    print(f"loss: {metrics['loss']:.6f}")
    print(
        f"accuracy: {metrics['accuracy'] * 100:.2f}% "
        f"({metrics['correct']}/{metrics['total']})"
    )
    print("per-class accuracy:")
    for class_name, details in metrics["per_class"].items():
        print(
            f"  {class_name}: {details['accuracy'] * 100:.2f}% "
            f"({details['correct']}/{details['total']})"
        )
    confusion = metrics.get("confusion_matrix")
    if confusion is not None:
        class_names = list(metrics["per_class"])
        print("confusion matrix (rows=actual, columns=predicted):")
        print(" " * 20 + " ".join(f"{name[:8]:>8}" for name in class_names))
        for class_name, row in zip(class_names, confusion):
            print(f"  {class_name[:16]:<16} " + " ".join(f"{value:8d}" for value in row))
    if metrics.get("average_inference_time_per_batch_s") is not None:
        print(
            "average inference time per batch: "
            f"{metrics['average_inference_time_per_batch_s'] * 1000:.3f} ms"
        )
        print(
            "average inference time per image: "
            f"{metrics['average_inference_time_per_image_s'] * 1000:.3f} ms"
        )


def save_evaluation_artifacts(
    metrics: dict[str, Any],
    *,
    checkpoint_path: Path,
    device: torch.device,
    config: dict[str, Any],
    parameter_count: int,
    class_names: list[str],
) -> tuple[Path, Path, Path]:
    results_dir = resolve_path("results/accuracy")
    results_dir.mkdir(parents=True, exist_ok=True)

    confusion = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    npy_path = results_dir / "float32_confusion_matrix.npy"
    csv_path = results_dir / "float32_confusion_matrix.csv"
    json_path = results_dir / "float32_test_results.json"

    np.save(npy_path, confusion)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *class_names])
        for class_name, row in zip(class_names, confusion.tolist()):
            writer.writerow([class_name, *row])

    model_cfg = config["model_config"]["model"]
    payload = {
        "checkpoint_path": str(checkpoint_path),
        "model_architecture": {
            "conv1_output_channels": int(model_cfg["conv1_output_channels"]),
            "conv2_output_channels": int(model_cfg["conv2_output_channels"]),
            "classifier_in_features": int(model_cfg["conv2_output_channels"]),
            "num_classes": int(model_cfg["num_classes"]),
            "batch_normalization": bool(model_cfg.get("batch_normalization", True)),
            "conv1": "3x3 conv, stride 1, padding 1, bias=False",
            "conv2": "3x3 conv, stride 1, padding 1, bias=False",
            "pool": "2x2 max pool stride 2",
            "global_average_pool": "AdaptiveAvgPool2d(1,1)",
        },
        "input_shape": [3, 32, 32],
        "class_order": class_names,
        "overall_accuracy": float(metrics["accuracy"]),
        "test_loss": float(metrics["loss"]),
        "correct": int(metrics["correct"]),
        "total": int(metrics["total"]),
        "per_class_accuracy": {
            name: float(details["accuracy"])
            for name, details in metrics["per_class"].items()
        },
        "per_class_correct": {
            name: int(details["correct"])
            for name, details in metrics["per_class"].items()
        },
        "per_class_totals": {
            name: int(details["total"])
            for name, details in metrics["per_class"].items()
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "average_inference_time_per_batch_s": metrics.get(
            "average_inference_time_per_batch_s"
        ),
        "average_inference_time_per_image_s": metrics.get(
            "average_inference_time_per_image_s"
        ),
        "parameter_count": parameter_count,
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path, npy_path, csv_path


def evaluate_int8_model(
    *,
    config_path: str = "software/config/training_config.yaml",
    int8_export_directory: str = "software/exported_model/int8",
    float32_results_path: str = "results/accuracy/float32_test_results.json",
) -> dict[str, Any]:
    """Evaluate the integer NumPy golden model on the official filtered test set."""
    from software.inference.integer_inference import IntegerTinyCNN
    from software.quantization.saturation import INT8_MAX, INT8_MIN

    config = load_training_config(config_path)
    set_seed(int(config["seed"]))
    device = select_device(str(config["runtime"]["device"]))

    bundle = build_dataset_bundle(config)
    _, _, test_loader = create_dataloaders(
        bundle,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=0,
        seed=int(config["seed"]),
        device=device,
    )
    class_names = bundle.class_names
    num_classes = len(class_names)

    float_model, _, resolved_checkpoint = restore_trained_model(
        checkpoint_path_from_config(config),
        config,
        map_location=device,
    )
    float_model = float_model.to(device)
    float_model.eval()

    int_model = IntegerTinyCNN.from_export_directory(resolve_path(int8_export_directory))

    total = 0
    float_correct = 0
    int_correct = 0
    agreement = 0
    changed = 0
    class_correct_float = [0 for _ in range(num_classes)]
    class_correct_int = [0 for _ in range(num_classes)]
    class_totals = [0 for _ in range(num_classes)]
    confusion_int = np.zeros((num_classes, num_classes), dtype=np.int64)
    saturation_total = 0
    max_acc = {"conv1": 0, "conv2": 0, "classifier": 0}
    timed_seconds = 0.0
    timed_images = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels_np = labels.numpy()
            float_scores = float_model(images).detach().cpu().numpy()
            float_preds = np.argmax(float_scores, axis=1).astype(np.int64)

            float_input = images.detach().cpu().numpy()
            start = time.perf_counter()
            int_scores, intermediates = int_model.forward_with_intermediates(float_input)
            timed_seconds += time.perf_counter() - start
            timed_images += int(float_input.shape[0])
            int_preds = np.argmax(int_scores, axis=1).astype(np.int64)

            for index, true_label in enumerate(labels_np.tolist()):
                true_label = int(true_label)
                float_pred = int(float_preds[index])
                int_pred = int(int_preds[index])
                total += 1
                class_totals[true_label] += 1
                if float_pred == true_label:
                    float_correct += 1
                    class_correct_float[true_label] += 1
                if int_pred == true_label:
                    int_correct += 1
                    class_correct_int[true_label] += 1
                if float_pred == int_pred:
                    agreement += 1
                else:
                    changed += 1
                confusion_int[true_label, int_pred] += 1

            for name in ("input", "conv1", "relu1", "pool1", "conv2", "relu2", "pool2"):
                arr = intermediates[name]
                saturation_total += int(
                    np.count_nonzero((arr <= INT8_MIN) | (arr >= INT8_MAX))
                )
            max_acc["conv1"] = max(
                max_acc["conv1"], int(np.max(np.abs(intermediates["conv1_accumulator"])))
            )
            max_acc["conv2"] = max(
                max_acc["conv2"], int(np.max(np.abs(intermediates["conv2_accumulator"])))
            )
            max_acc["classifier"] = max(
                max_acc["classifier"],
                int(np.max(np.abs(intermediates["classifier_accumulator"]))),
            )

    float_accuracy = float_correct / total if total else 0.0
    int_accuracy = int_correct / total if total else 0.0
    float_results = resolve_path(float32_results_path)
    prior_float = None
    if float_results.is_file():
        prior_float = float(
            json.loads(float_results.read_text(encoding="utf-8"))["overall_accuracy"]
        )

    per_class_float = {
        name: (class_correct_float[i] / class_totals[i] if class_totals[i] else 0.0)
        for i, name in enumerate(class_names)
    }
    per_class_int = {
        name: (class_correct_int[i] / class_totals[i] if class_totals[i] else 0.0)
        for i, name in enumerate(class_names)
    }
    per_class_delta = {
        name: per_class_float[name] - per_class_int[name] for name in class_names
    }

    flags: list[str] = []
    drop = float_accuracy - int_accuracy
    if drop > 0.05:
        flags.append("accuracy_drop_exceeds_5pp")
    if agreement / total < 0.9 if total else False:
        flags.append("prediction_agreement_below_90pct")
    for name in class_names:
        if per_class_delta[name] > 0.1:
            flags.append(f"class_drop_{name}")
    if max(max_acc.values()) > 1_000_000_000:
        flags.append("accumulators_near_int32_limit")
    if saturation_total > total * 100:
        flags.append("frequent_saturation")

    report = {
        "checkpoint_path": str(resolved_checkpoint),
        "int8_export_directory": str(resolve_path(int8_export_directory)),
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "overall_accuracy": int_accuracy,
        "float32_accuracy_this_run": float_accuracy,
        "float32_accuracy_prior": prior_float,
        "accuracy_difference": drop,
        "prediction_agreement_rate": agreement / total if total else 0.0,
        "samples_with_changed_prediction": changed,
        "correct": int_correct,
        "total": total,
        "per_class_float32_accuracy": per_class_float,
        "per_class_int8_accuracy": per_class_int,
        "per_class_accuracy_difference": per_class_delta,
        "confusion_matrix": confusion_int.tolist(),
        "saturation_counts_total": saturation_total,
        "maximum_observed_accumulators": max_acc,
        "average_numpy_integer_inference_time_s": (
            timed_seconds / timed_images if timed_images else None
        ),
        "timing_note": "NumPy software reference timing only; not FPGA performance",
        "flags": flags,
        "qat_recommendation": (
            "Consider quantization-aware training later"
            if ("accuracy_drop_exceeds_5pp" in flags or any(f.startswith("class_drop_") for f in flags))
            else "Post-training quantization appears acceptable for this checkpoint"
        ),
        "class_order": class_names,
    }

    out_dir = resolve_path("results/accuracy")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "int8_test_results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(out_dir / "int8_confusion_matrix.npy", confusion_int)

    print("INT8 test evaluation")
    print("--------------------")
    print(f"float32 accuracy: {float_accuracy * 100:.2f}%")
    print(f"INT8 accuracy:    {int_accuracy * 100:.2f}%")
    print(f"accuracy drop:    {drop * 100:.2f} pp")
    print(f"agreement rate:   {report['prediction_agreement_rate'] * 100:.2f}%")
    print(f"changed preds:    {changed}")
    print(f"avg int inference:{report['average_numpy_integer_inference_time_s'] * 1000:.3f} ms/image")
    print(f"flags: {flags}")
    print(f"saved {out_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a TinyCNN checkpoint on the filtered GTSRB test split"
    )
    parser.add_argument(
        "--checkpoint",
        default="software/checkpoints/tiny_cnn_best.pth",
        help="Path to the trained checkpoint",
    )
    parser.add_argument(
        "--config",
        default="software/config/training_config.yaml",
        help="Path to the training configuration YAML",
    )
    parser.add_argument(
        "--mode",
        choices=["float32", "int8"],
        default="float32",
        help="Evaluation backend",
    )
    parser.add_argument(
        "--int8-export-directory",
        default="software/exported_model/int8",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "int8":
        evaluate_int8_model(
            config_path=args.config,
            int8_export_directory=args.int8_export_directory,
        )
        return

    config = load_training_config(args.config)
    set_seed(int(config["seed"]))
    device = select_device(str(config["runtime"]["device"]))
    print(f"Using device: {device}")

    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.is_file():
        checkpoint_path = checkpoint_path_from_config(config)

    model, checkpoint, resolved_checkpoint = restore_trained_model(
        checkpoint_path,
        config,
        map_location=device,
    )
    model = model.to(device)

    bundle = build_dataset_bundle(config)
    print_class_counts(bundle)
    print(bundle.split_notes)

    _, _, test_loader = create_dataloaders(
        bundle,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        seed=int(config["seed"]),
        device=device,
    )

    class_names = bundle.class_names
    model_settings = config["model_config"]["model"]
    print("\nActual model architecture")
    print("-------------------------")
    print(
        f"Conv1: 3 -> {model_settings['conv1_output_channels']}, "
        f"kernel {model_settings['conv1_kernel_size']}, "
        f"stride {model_settings['conv1_stride']}, "
        f"padding {model_settings['conv1_padding']}, BatchNorm, ReLU, MaxPool2d(2)"
    )
    print(
        f"Conv2: {model_settings['conv1_output_channels']} -> "
        f"{model_settings['conv2_output_channels']}, "
        f"kernel {model_settings['conv2_kernel_size']}, "
        f"stride {model_settings['conv2_stride']}, "
        f"padding {model_settings['conv2_padding']}, BatchNorm, ReLU, MaxPool2d(2)"
    )
    print(
        f"Classifier: {model_settings['conv2_output_channels']} -> "
        f"{model_settings['num_classes']}"
    )
    print(f"checkpoint: {resolved_checkpoint}")
    print(f"class order: {class_names}")

    loss_function = nn.CrossEntropyLoss()
    metrics = evaluate_model(
        model,
        test_loader,
        loss_function,
        device,
        num_classes=len(class_names),
        class_names=class_names,
        measure_inference=True,
    )
    print_evaluation_metrics(metrics, title="Test evaluation")
    json_path, npy_path, csv_path = save_evaluation_artifacts(
        metrics,
        checkpoint_path=resolved_checkpoint,
        device=device,
        config=config,
        parameter_count=count_trainable_parameters(model),
        class_names=class_names,
    )
    print(f"\nSaved results JSON: {json_path}")
    print(f"Saved confusion matrix NPY: {npy_path}")
    print(f"Saved confusion matrix CSV: {csv_path}")


if __name__ == "__main__":
    main()
