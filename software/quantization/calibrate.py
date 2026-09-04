from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from software.preprocessing.image_loader import build_dataset_bundle, create_dataloaders
from software.quantization.fixed_point import (
    calculate_symmetric_scale,
    load_quantization_config,
)
from software.utils.checkpoint import checkpoint_path_from_config, restore_trained_model
from software.utils.config import load_training_config, resolve_path
from software.utils.reproducibility import select_device, set_seed


CALIBRATION_TENSORS = (
    "input",
    "bn1",  # folded Conv1 + BatchNorm equivalent
    "relu1",
    "pool1",
    "bn2",  # folded Conv2 + BatchNorm equivalent
    "relu2",
    "pool2",
    "global_average_pool",
    "flatten",
    "scores",
)


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "p01": float(np.percentile(arr, 1)) if arr.size else 0.0,
        "p99": float(np.percentile(arr, 99)) if arr.size else 0.0,
        "count": int(arr.size),
    }


def select_calibration_indices(
    dataset_size: int,
    sample_count: int,
    seed: int,
) -> list[int]:
    if dataset_size < 1:
        raise RuntimeError("Calibration dataset is empty")
    count = min(int(sample_count), int(dataset_size))
    rng = np.random.default_rng(seed)
    indices = rng.choice(dataset_size, size=count, replace=False)
    return sorted(int(index) for index in indices)


@torch.no_grad()
def collect_calibration_statistics(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model.eval()
    collectors: dict[str, list[np.ndarray]] = {name: [] for name in CALIBRATION_TENSORS}

    for images, _ in data_loader:
        images = images.to(device)
        _, intermediates = model.forward_with_intermediates(images)
        collectors["input"].append(intermediates["input"].detach().cpu().numpy())
        collectors["bn1"].append(intermediates["bn1"].detach().cpu().numpy())
        collectors["relu1"].append(intermediates["relu1"].detach().cpu().numpy())
        collectors["pool1"].append(intermediates["pool1"].detach().cpu().numpy())
        collectors["bn2"].append(intermediates["bn2"].detach().cpu().numpy())
        collectors["relu2"].append(intermediates["relu2"].detach().cpu().numpy())
        collectors["pool2"].append(intermediates["pool2"].detach().cpu().numpy())
        collectors["global_average_pool"].append(
            intermediates["global_average_pool"].detach().cpu().numpy()
        )
        collectors["flatten"].append(intermediates["flatten"].detach().cpu().numpy())
        collectors["scores"].append(intermediates["scores"].detach().cpu().numpy())

    return {
        name: _stats(np.concatenate([arr.reshape(-1) for arr in arrays], axis=0))
        for name, arrays in collectors.items()
    }


def build_activation_scales(
    statistics: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Symmetric per-tensor scales. Folded conv outputs use bn* stats."""
    mapping = {
        "input": "input",
        "conv1": "bn1",
        "relu1": "relu1",
        "pool1": "pool1",
        "conv2": "bn2",
        "relu2": "relu2",
        "pool2": "pool2",
        "global_average_pool": "global_average_pool",
        "flatten": "flatten",
        "classifier_input": "flatten",
        "classifier_output": "scores",
    }
    scales: dict[str, float] = {}
    for name, source in mapping.items():
        max_abs = statistics[source]["max_abs"]
        scales[name] = calculate_symmetric_scale(np.array([max_abs, -max_abs]))
    return scales


def calibrate(
    *,
    config_path: str = "software/config/training_config.yaml",
    quantization_config_path: str = "software/config/quantization_config.yaml",
    checkpoint: str | None = None,
    output_path: str = "software/exported_model/int8/metadata/calibration.json",
) -> dict[str, Any]:
    training_config = load_training_config(config_path)
    quant_config = load_quantization_config(quantization_config_path)
    set_seed(int(quant_config["quantization"]["calibration_seed"]))
    device = select_device(str(training_config["runtime"]["device"]))

    checkpoint_path = resolve_path(
        checkpoint or checkpoint_path_from_config(training_config)
    )
    model, ckpt, resolved = restore_trained_model(
        checkpoint_path,
        training_config,
        map_location=device,
    )
    model = model.to(device)

    bundle = build_dataset_bundle(training_config)
    # Use validation split only; never the official test split.
    val_size = len(bundle.validation_dataset)
    indices = select_calibration_indices(
        val_size,
        int(quant_config["quantization"]["calibration_samples"]),
        int(quant_config["quantization"]["calibration_seed"]),
    )
    subset = Subset(bundle.validation_dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=int(training_config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    statistics = collect_calibration_statistics(model, loader, device)
    scales = build_activation_scales(statistics)

    payload = {
        "calibration_date": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(resolved),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "validation_accuracy": float(ckpt["validation_accuracy"]),
        "seed": int(quant_config["quantization"]["calibration_seed"]),
        "requested_samples": int(quant_config["quantization"]["calibration_samples"]),
        "used_samples": len(indices),
        "dataset_split": "validation",
        "indices": indices,
        "tensor_statistics": statistics,
        "activation_scales": scales,
        "notes": {
            "conv1_activation_source": "bn1",
            "conv2_activation_source": "bn2",
            "reason": (
                "BatchNorm is folded into convolution for hardware. "
                "Activation scales for conv outputs are calibrated on BN outputs."
            ),
            "activation_scheme": "symmetric_int8_zero_point_0",
        },
    }

    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate INT8 activation scales from the validation split"
    )
    parser.add_argument("--config", default="software/config/training_config.yaml")
    parser.add_argument(
        "--quantization-config",
        default="software/config/quantization_config.yaml",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--output",
        default="software/exported_model/int8/metadata/calibration.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = calibrate(
        config_path=args.config,
        quantization_config_path=args.quantization_config,
        checkpoint=args.checkpoint,
        output_path=args.output,
    )
    print("Calibration complete")
    print("--------------------")
    print(f"samples: {payload['used_samples']}")
    print(f"output: {args.output}")
    for name, scale in payload["activation_scales"].items():
        print(f"  {name:24s} scale={scale:.8g}")


if __name__ == "__main__":
    main()
