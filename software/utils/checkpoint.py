from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from software.models.tiny_cnn import TinyCNN
from software.utils.config import ConfigurationError, resolve_path


REQUIRED_CHECKPOINT_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "validation_loss",
    "validation_accuracy",
    "class_names",
    "gtsrb_to_project_label_mapping",
    "model_input_shape",
    "num_classes",
    "training_config",
)

EXPECTED_CLASS_ORDER = [
    "stop",
    "yield",
    "no_entry",
    "speed_limit_30",
    "keep_right",
]


def checkpoint_path_from_config(config: dict[str, Any]) -> Path:
    directory = resolve_path(config["checkpoint"]["directory"])
    return directory / str(config["checkpoint"]["filename"])


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save a checkpoint dictionary and create parent directories as needed."""
    checkpoint_path = resolve_path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    expected_class_names: list[str] | None = None,
    expected_num_classes: int | None = None,
) -> dict[str, Any]:
    """Load a checkpoint and validate class metadata when provided."""
    checkpoint_path = resolve_path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise ConfigurationError(
            f"Failed to load checkpoint {checkpoint_path}: {exc}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ConfigurationError(
            f"Incompatible checkpoint format in {checkpoint_path}: expected a dictionary"
        )

    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in checkpoint]
    if missing:
        raise ConfigurationError(
            "Incompatible checkpoint: missing key(s) "
            f"{', '.join(missing)} in {checkpoint_path}"
        )

    if "model_state_dict" not in checkpoint or not isinstance(
        checkpoint["model_state_dict"], dict
    ):
        raise ConfigurationError(
            f"Checkpoint is missing a valid model_state_dict: {checkpoint_path}"
        )

    class_names = list(checkpoint["class_names"])
    num_classes = int(checkpoint["num_classes"])

    if expected_num_classes is not None and num_classes != expected_num_classes:
        raise ConfigurationError(
            "Incompatible checkpoint: "
            f"num_classes={num_classes}, expected {expected_num_classes}"
        )

    if expected_class_names is not None and class_names != list(expected_class_names):
        raise ConfigurationError(
            "Incompatible checkpoint class names/order: "
            f"checkpoint={class_names}, expected={list(expected_class_names)}"
        )

    return checkpoint


def instantiate_model_from_config(config: dict[str, Any]) -> TinyCNN:
    model_settings = config["model_config"]["model"]
    return TinyCNN(
        num_classes=int(model_settings["num_classes"]),
        conv1_channels=int(model_settings["conv1_output_channels"]),
        conv2_channels=int(model_settings["conv2_output_channels"]),
    )


def validate_state_dict_shapes(
    model: nn.Module,
    state_dict: dict[str, Any],
) -> None:
    """Detect missing keys, unexpected keys, and tensor-shape mismatches."""
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(model_state))
    if missing:
        raise ConfigurationError(
            "Incompatible checkpoint architecture: missing key(s) "
            f"{', '.join(missing)}"
        )
    if unexpected:
        raise ConfigurationError(
            "Incompatible checkpoint architecture: unexpected key(s) "
            f"{', '.join(unexpected)}"
        )

    shape_errors: list[str] = []
    for key, expected in model_state.items():
        actual = state_dict[key]
        if tuple(actual.shape) != tuple(expected.shape):
            shape_errors.append(
                f"{key}: checkpoint {tuple(actual.shape)} vs model {tuple(expected.shape)}"
            )
    if shape_errors:
        raise ConfigurationError(
            "Incompatible checkpoint tensor shapes:\n  " + "\n  ".join(shape_errors)
        )


def load_model_weights(model: nn.Module, checkpoint: dict[str, Any]) -> nn.Module:
    state_dict = checkpoint["model_state_dict"]
    validate_state_dict_shapes(model, state_dict)
    model.load_state_dict(state_dict)
    return model


def restore_trained_model(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    *,
    map_location: str | torch.device | None = "cpu",
    strict_class_order: bool = True,
) -> tuple[TinyCNN, dict[str, Any], Path]:
    """Load and validate a checkpoint into a fresh TinyCNN instance."""
    resolved = resolve_path(checkpoint_path)
    expected_classes = list(config["model_config"]["classes"])
    if strict_class_order and expected_classes != EXPECTED_CLASS_ORDER:
        raise ConfigurationError(
            "Configured class order does not match the frozen project order: "
            f"config={expected_classes}, expected={EXPECTED_CLASS_ORDER}"
        )

    checkpoint = load_checkpoint(
        resolved,
        map_location=map_location,
        expected_class_names=expected_classes,
        expected_num_classes=int(config["model_config"]["model"]["num_classes"]),
    )

    checkpoint_classes = list(checkpoint["class_names"])
    if checkpoint_classes != EXPECTED_CLASS_ORDER:
        raise ConfigurationError(
            "Checkpoint class order is inconsistent with the frozen project order: "
            f"checkpoint={checkpoint_classes}, expected={EXPECTED_CLASS_ORDER}"
        )

    input_shape = list(checkpoint["model_input_shape"])
    if input_shape != [3, 32, 32]:
        raise ConfigurationError(
            f"Unexpected checkpoint model_input_shape={input_shape}, expected [3, 32, 32]"
        )

    checkpoint_model_config = checkpoint["training_config"].get("model_config")
    if checkpoint_model_config != config["model_config"]:
        raise ConfigurationError(
            "Checkpoint model architecture is incompatible with the current "
            "model_config.yaml. Retrain or update the configuration."
        )

    checkpoint_normalization = (
        checkpoint["training_config"].get("data", {}).get("normalization")
    )
    current_normalization = config["data"].get("normalization")
    if checkpoint_normalization != current_normalization:
        raise ConfigurationError(
            "Checkpoint preprocessing is incompatible with the current "
            "normalization configuration."
        )

    model = instantiate_model_from_config(config)
    load_model_weights(model, checkpoint)
    model.eval()
    return model, checkpoint, resolved


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
