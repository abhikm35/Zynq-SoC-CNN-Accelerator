from __future__ import annotations

from pathlib import Path

import pytest
import torch

from software.models.tiny_cnn import TinyCNN
from software.utils.checkpoint import (
    load_checkpoint,
    restore_trained_model,
    save_checkpoint,
    validate_state_dict_shapes,
)
from software.utils.config import ConfigurationError, load_training_config


def _minimal_checkpoint(model: TinyCNN) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "epoch": 1,
        "validation_loss": 1.0,
        "validation_accuracy": 0.5,
        "class_names": [
            "stop",
            "yield",
            "no_entry",
            "speed_limit_30",
            "keep_right",
        ],
        "gtsrb_to_project_label_mapping": {14: 0, 13: 1, 17: 2, 1: 3, 38: 4},
        "model_input_shape": [3, 32, 32],
        "num_classes": 5,
        "training_config": load_training_config(),
    }


def test_valid_checkpoint_loads_and_is_deterministic(tmp_path: Path) -> None:
    config = load_training_config()
    model = TinyCNN(num_classes=5, conv1_channels=16, conv2_channels=32)
    checkpoint = _minimal_checkpoint(model)
    path = tmp_path / "valid.pth"
    save_checkpoint(path, checkpoint)

    restored, loaded, resolved = restore_trained_model(path, config, map_location="cpu")
    assert resolved == path.resolve()
    assert loaded["num_classes"] == 5

    torch.manual_seed(0)
    inputs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        first = restored(inputs)
        second = restored(inputs)
    assert torch.allclose(first, second)


def test_missing_checkpoint_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint("software/checkpoints/does_not_exist.pth", map_location="cpu")


def test_missing_model_state_dict_raises(tmp_path: Path) -> None:
    model = TinyCNN(num_classes=5)
    checkpoint = _minimal_checkpoint(model)
    del checkpoint["model_state_dict"]
    path = tmp_path / "missing_state.pth"
    torch.save(checkpoint, path)
    with pytest.raises(ConfigurationError, match="model_state_dict"):
        load_checkpoint(path, map_location="cpu")


def test_wrong_classifier_size_detected(tmp_path: Path) -> None:
    config = load_training_config()
    model = TinyCNN(num_classes=5, conv1_channels=16, conv2_channels=32)
    checkpoint = _minimal_checkpoint(model)
    bad = TinyCNN(num_classes=4, conv1_channels=16, conv2_channels=32)
    checkpoint["model_state_dict"] = bad.state_dict()
    checkpoint["num_classes"] = 5
    path = tmp_path / "bad_classifier.pth"
    save_checkpoint(path, checkpoint)
    with pytest.raises(ConfigurationError, match="tensor shapes|unexpected|missing"):
        restore_trained_model(path, config, map_location="cpu")


def test_wrong_convolution_channels_detected() -> None:
    model = TinyCNN(num_classes=5, conv1_channels=16, conv2_channels=32)
    wrong = TinyCNN(num_classes=5, conv1_channels=8, conv2_channels=16)
    with pytest.raises(ConfigurationError, match="tensor shapes|missing|unexpected"):
        validate_state_dict_shapes(model, wrong.state_dict())


def test_incorrect_class_order_detected(tmp_path: Path) -> None:
    config = load_training_config()
    model = TinyCNN(num_classes=5, conv1_channels=16, conv2_channels=32)
    checkpoint = _minimal_checkpoint(model)
    checkpoint["class_names"] = [
        "yield",
        "stop",
        "no_entry",
        "speed_limit_30",
        "keep_right",
    ]
    path = tmp_path / "bad_classes.pth"
    save_checkpoint(path, checkpoint)
    with pytest.raises(ConfigurationError, match="class"):
        restore_trained_model(path, config, map_location="cpu")
