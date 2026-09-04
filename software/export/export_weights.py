from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from software.utils.batchnorm_fold import extract_float32_parameters
from software.utils.checkpoint import count_trainable_parameters


FLOAT32_EXPORT_LAYOUT = {
    "weights": [
        "conv1_weights_float32.npy",
        "conv2_weights_float32.npy",
        "classifier_weights_float32.npy",
    ],
    "biases": [
        "conv1_bias_float32.npy",
        "conv2_bias_float32.npy",
        "classifier_bias_float32.npy",
    ],
}


def ensure_export_root(output_directory: str | Path, *, force: bool = False) -> Path:
    root = Path(output_directory)
    marker = root / "metadata" / "model_metadata.json"
    if marker.is_file() and not force:
        existing = json.loads(marker.read_text(encoding="utf-8"))
        raise FileExistsError(
            f"Export directory already exists with metadata at {marker}. "
            "Pass --force to overwrite an incompatible or previous export. "
            f"Existing architecture summary: {existing.get('architecture_summary')}"
        )
    (root / "weights").mkdir(parents=True, exist_ok=True)
    (root / "biases").mkdir(parents=True, exist_ok=True)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    return root


def export_weights(
    model: nn.Module,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Export BatchNorm-folded and raw convolution/classifier weights."""
    root = Path(output_directory)
    params = extract_float32_parameters(model)
    paths: dict[str, Path] = {}

    folded = params["folded"]
    mapping = {
        "conv1_weights_float32.npy": folded["conv1_weights_float32"],
        "conv2_weights_float32.npy": folded["conv2_weights_float32"],
        "classifier_weights_float32.npy": folded["classifier_weights_float32"],
    }
    for filename, array in mapping.items():
        path = root / "weights" / filename
        np.save(path, array)
        paths[filename] = path

    raw = params["raw"]
    raw_mapping = {
        "conv1_weight_float32.npy": raw["conv1_weight"],
        "conv2_weight_float32.npy": raw["conv2_weight"],
        "classifier_weight_float32.npy": raw["classifier_weight"],
        "bn1_weight_float32.npy": raw["bn1_weight"],
        "bn2_weight_float32.npy": raw["bn2_weight"],
    }
    for filename, array in raw_mapping.items():
        path = root / "raw" / filename
        np.save(path, array)
        paths[filename] = path
    return paths


def export_biases(
    model: nn.Module,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Export BatchNorm-folded convolution biases and classifier bias."""
    root = Path(output_directory)
    params = extract_float32_parameters(model)
    paths: dict[str, Path] = {}

    folded = params["folded"]
    mapping = {
        "conv1_bias_float32.npy": folded["conv1_bias_float32"],
        "conv2_bias_float32.npy": folded["conv2_bias_float32"],
        "classifier_bias_float32.npy": folded["classifier_bias_float32"],
    }
    for filename, array in mapping.items():
        path = root / "biases" / filename
        np.save(path, array)
        paths[filename] = path

    raw = params["raw"]
    raw_mapping = {
        "classifier_bias_float32.npy": raw["classifier_bias"],
        "bn1_bias_float32.npy": raw["bn1_bias"],
        "bn1_running_mean_float32.npy": raw["bn1_running_mean"],
        "bn1_running_var_float32.npy": raw["bn1_running_var"],
        "bn2_bias_float32.npy": raw["bn2_bias"],
        "bn2_running_mean_float32.npy": raw["bn2_running_mean"],
        "bn2_running_var_float32.npy": raw["bn2_running_var"],
    }
    for filename, array in raw_mapping.items():
        path = root / "raw" / filename
        np.save(path, array)
        paths[filename] = path
    return paths


def build_model_metadata(
    model: nn.Module,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    config: dict[str, Any],
    *,
    test_accuracy: float | None = None,
) -> dict[str, Any]:
    params = extract_float32_parameters(model)
    model_cfg = config["model_config"]["model"]
    normalization = config["data"]["normalization"]
    class_names = list(checkpoint["class_names"])

    with torch.no_grad():
        sample = torch.zeros(1, 3, 32, 32)
        _, intermediates = model.forward_with_intermediates(sample)

    intermediate_shapes = {
        name: list(tensor.shape) for name, tensor in intermediates.items()
    }
    parameter_shapes = {
        name: list(parameter.shape)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    return {
        "model_name": "TinyCNN",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_accuracy": float(checkpoint["validation_accuracy"]),
        "validation_loss": float(checkpoint["validation_loss"]),
        "test_accuracy": test_accuracy,
        "export_date": datetime.now(timezone.utc).isoformat(),
        "input_shape": [3, 32, 32],
        "output_shape": [5],
        "class_order": class_names,
        "parameter_count": count_trainable_parameters(model),
        "architecture_summary": {
            "conv1": "3 -> 16, kernel 3, stride 1, padding 1, bias=False",
            "bn1": "BatchNorm2d(16)",
            "pool1": "MaxPool2d kernel 2 stride 2",
            "conv2": "16 -> 32, kernel 3, stride 1, padding 1, bias=False",
            "bn2": "BatchNorm2d(32)",
            "pool2": "MaxPool2d kernel 2 stride 2",
            "global_average_pool": "AdaptiveAvgPool2d(1, 1)",
            "classifier": "Linear(32 -> 5)",
        },
        "layer_order": [
            "input",
            "conv1",
            "bn1",
            "relu1",
            "pool1",
            "conv2",
            "bn2",
            "relu2",
            "pool2",
            "global_average_pool",
            "flatten",
            "scores",
        ],
        "intermediate_shapes": intermediate_shapes,
        "parameter_shapes": parameter_shapes,
        "layers": {
            "conv1": {
                "in_channels": 3,
                "out_channels": int(model_cfg["conv1_output_channels"]),
                "kernel_size": int(model_cfg["conv1_kernel_size"]),
                "stride": int(model_cfg["conv1_stride"]),
                "padding": int(model_cfg["conv1_padding"]),
                "bias": False,
                "weight_ordering": "[out_channels, in_channels, kernel_height, kernel_width]",
                "raw_weight_shape": list(params["raw"]["conv1_weight"].shape),
                "folded_weight_shape": list(params["folded"]["conv1_weights_float32"].shape),
                "folded_bias_shape": list(params["folded"]["conv1_bias_float32"].shape),
            },
            "conv2": {
                "in_channels": int(model_cfg["conv1_output_channels"]),
                "out_channels": int(model_cfg["conv2_output_channels"]),
                "kernel_size": int(model_cfg["conv2_kernel_size"]),
                "stride": int(model_cfg["conv2_stride"]),
                "padding": int(model_cfg["conv2_padding"]),
                "bias": False,
                "weight_ordering": "[out_channels, in_channels, kernel_height, kernel_width]",
                "raw_weight_shape": list(params["raw"]["conv2_weight"].shape),
                "folded_weight_shape": list(params["folded"]["conv2_weights_float32"].shape),
                "folded_bias_shape": list(params["folded"]["conv2_bias_float32"].shape),
            },
            "pool1": {
                "kernel_size": int(model_cfg["pool1_kernel_size"]),
                "stride": int(model_cfg["pool1_stride"]),
            },
            "pool2": {
                "kernel_size": int(model_cfg["pool2_kernel_size"]),
                "stride": int(model_cfg["pool2_stride"]),
            },
            "global_average_pool": {"output_size": [1, 1]},
            "classifier": {
                "in_features": int(model_cfg["conv2_output_channels"]),
                "out_features": int(model_cfg["num_classes"]),
                "weight_ordering": "[out_features, in_features]",
                "weight_shape": list(params["raw"]["classifier_weight"].shape),
                "bias_shape": list(params["raw"]["classifier_bias"].shape),
            },
        },
        "batchnorm": {
            "enabled": True,
            "bn1": {
                "num_features": int(model_cfg["conv1_output_channels"]),
                "eps": float(params["raw"]["bn1_eps"]),
            },
            "bn2": {
                "num_features": int(model_cfg["conv2_output_channels"]),
                "eps": float(params["raw"]["bn2_eps"]),
            },
            "folding_note": (
                "weights/ and biases/ contain BatchNorm-folded convolution "
                "parameters for future FPGA/INT8 use. raw/ contains the exact "
                "PyTorch parameters used by the independent NumPy model."
            ),
        },
        "tensor_dtypes": "float32",
        "preprocessing": {
            "rgb_conversion": True,
            "resize": [32, 32],
            "interpolation": "bilinear via torchvision.transforms.Resize",
            "to_tensor_scale": "[0, 255] -> [0.0, 1.0]",
            "channel_order": "NCHW",
            "normalization_mean": list(normalization["mean"]),
            "normalization_std": list(normalization["std"]),
            "note": (
                "Exported input.npy tensors are post-normalization network inputs."
            ),
        },
        "gtsrb_to_project_label_mapping": {
            str(key): int(value)
            for key, value in checkpoint["gtsrb_to_project_label_mapping"].items()
        },
    }


def export_model_metadata(
    model: nn.Module,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    config: dict[str, Any],
    output_directory: str | Path,
    *,
    test_accuracy: float | None = None,
) -> Path:
    metadata = build_model_metadata(
        model,
        checkpoint,
        checkpoint_path,
        config,
        test_accuracy=test_accuracy,
    )
    path = Path(output_directory) / "metadata" / "model_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path
