from __future__ import annotations

import argparse
import copy
from typing import Any

import torch
from torch import nn

from software.models.tiny_cnn import TinyCNN
from software.preprocessing.image_loader import (
    build_dataset_bundle,
    create_dataloaders,
    print_class_counts,
    verify_training_batch,
)
from software.training.evaluate import evaluate_model, print_evaluation_metrics
from software.training.metrics import (
    accumulate_batch_metrics,
    finalize_metrics,
)
from software.utils.checkpoint import checkpoint_path_from_config, save_checkpoint
from software.utils.config import ConfigurationError, load_training_config
from software.utils.reproducibility import (
    reproducibility_warning,
    select_device,
    set_seed,
)


def build_optimizer(
    model: nn.Module,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    name = str(config["optimizer"]["name"]).lower()
    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ConfigurationError(f"Unsupported optimizer name '{name}'")


def compute_class_weights(
    class_counts: dict[str, int],
    class_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Return balanced inverse-frequency weights for the training classes."""
    counts = torch.tensor(
        [class_counts[name] for name in class_names],
        dtype=torch.float32,
        device=device,
    )
    if torch.any(counts <= 0):
        raise ValueError("Every class must have training samples")
    weights = counts.sum() / (len(class_names) * counts)
    return weights


def build_loss_function(
    config: dict[str, Any],
    class_counts: dict[str, int],
    class_names: list[str],
    device: torch.device,
) -> tuple[nn.CrossEntropyLoss, torch.Tensor | None]:
    weighting = str(config["loss"]["class_weighting"])
    if weighting == "none":
        return nn.CrossEntropyLoss(), None
    if weighting == "inverse_frequency":
        weights = compute_class_weights(class_counts, class_names, device)
        return nn.CrossEntropyLoss(weight=weights), weights
    raise ConfigurationError(f"Unsupported class weighting '{weighting}'")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    scheduler_config = config["scheduler"]
    name = str(scheduler_config["name"])
    if name == "none":
        return None
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config["factor"]),
            patience=int(scheduler_config["patience"]),
            min_lr=float(scheduler_config["min_learning_rate"]),
        )
    raise ConfigurationError(f"Unsupported scheduler name '{name}'")


def train_one_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = [0 for _ in range(num_classes)]
    class_totals = [0 for _ in range(num_classes)]

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        scores = model(images)
        loss = loss_function(scores, labels)
        loss.backward()
        optimizer.step()

        updated = accumulate_batch_metrics(
            scores.detach(),
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

    return finalize_metrics(
        total_loss=total_loss,
        total_correct=total_correct,
        total_samples=total_samples,
        class_correct=class_correct,
        class_totals=class_totals,
        class_names=class_names,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TinyCNN on the filtered five-class GTSRB subset"
    )
    parser.add_argument(
        "--config",
        default="software/config/training_config.yaml",
        help="Path to the training configuration YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    set_seed(int(config["seed"]))
    print(reproducibility_warning())

    device = select_device(str(config["runtime"]["device"]))
    print(f"Using device: {device}")

    bundle = build_dataset_bundle(config)
    print_class_counts(bundle)
    print(bundle.split_notes)

    train_loader, validation_loader, _ = create_dataloaders(
        bundle,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        seed=int(config["seed"]),
        device=device,
    )

    images, labels = next(iter(train_loader))
    verify_training_batch(
        images,
        labels,
        int(config["training"]["batch_size"]),
        normalized=True,
    )

    class_names = bundle.class_names
    num_classes = len(class_names)
    model_settings = config["model_config"]["model"]
    model = TinyCNN(
        num_classes=num_classes,
        conv1_channels=int(model_settings["conv1_output_channels"]),
        conv2_channels=int(model_settings["conv2_output_channels"]),
    ).to(device)
    loss_function, class_weights = build_loss_function(
        config,
        bundle.class_counts["train"],
        class_names,
        device,
    )
    if class_weights is not None:
        print("Class weights:")
        for class_name, weight in zip(class_names, class_weights.tolist()):
            print(f"  {class_name}: {weight:.4f}")
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    best_validation_accuracy = -1.0
    best_epoch = 0
    best_validation_metrics: dict[str, Any] | None = None
    checkpoint_path = checkpoint_path_from_config(config)

    epochs = int(config["training"]["epochs"])
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
            num_classes=num_classes,
            class_names=class_names,
        )
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            loss_function,
            device,
            num_classes=num_classes,
            class_names=class_names,
        )
        if scheduler is not None:
            scheduler.step(float(validation_metrics["loss"]))
        learning_rate = float(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.6f} "
            f"acc {train_metrics['accuracy'] * 100:.2f}% | "
            f"val loss {validation_metrics['loss']:.6f} "
            f"acc {validation_metrics['accuracy'] * 100:.2f}% | "
            f"lr {learning_rate:.6g}"
        )

        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = float(validation_metrics["accuracy"])
            best_epoch = epoch
            best_validation_metrics = validation_metrics
            payload = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "scheduler_state_dict": (
                    copy.deepcopy(scheduler.state_dict())
                    if scheduler is not None
                    else None
                ),
                "epoch": epoch,
                "validation_loss": float(validation_metrics["loss"]),
                "validation_accuracy": float(validation_metrics["accuracy"]),
                "class_names": class_names,
                "gtsrb_to_project_label_mapping": {
                    int(key): int(value) for key, value in bundle.label_mapping.items()
                },
                "model_input_shape": [3, 32, 32],
                "num_classes": num_classes,
                "class_weights": (
                    class_weights.detach().cpu().tolist()
                    if class_weights is not None
                    else None
                ),
                "training_config": config,
            }
            save_checkpoint(checkpoint_path, payload)
            print(
                f"  saved new best checkpoint to {checkpoint_path} "
                f"(val acc {best_validation_accuracy * 100:.2f}%)"
            )

    print(
        f"Training complete. Best epoch={best_epoch}, "
        f"best validation accuracy={best_validation_accuracy * 100:.2f}%"
    )
    if best_validation_metrics is not None:
        print_evaluation_metrics(
            best_validation_metrics,
            title="Best validation metrics",
        )


if __name__ == "__main__":
    main()
