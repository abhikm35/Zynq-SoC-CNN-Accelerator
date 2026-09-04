from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from software.export.create_reference_set import ReferenceSample
from software.export.export_activations import export_sample_activations
from software.export.export_biases import export_biases
from software.export.export_model_metadata import export_model_metadata
from software.export.export_weights import ensure_export_root, export_weights
from software.models.tiny_cnn import TinyCNN
from software.preprocessing.image_loader import RemappedImageDataset, build_evaluation_transform
from software.utils.checkpoint import save_checkpoint
from software.utils.config import load_training_config


def _toy_checkpoint(model: TinyCNN) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "epoch": 7,
        "validation_loss": 0.25,
        "validation_accuracy": 0.91,
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


def test_float32_export_creates_matching_files(tmp_path: Path) -> None:
    config = load_training_config()
    model = TinyCNN(num_classes=5, conv1_channels=16, conv2_channels=32)
    model.eval()
    checkpoint = _toy_checkpoint(model)
    checkpoint_path = tmp_path / "ckpt.pth"
    save_checkpoint(checkpoint_path, checkpoint)

    export_root = tmp_path / "float32"
    ensure_export_root(export_root, force=True)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    weight_paths = export_weights(model, export_root)
    bias_paths = export_biases(model, export_root)
    metadata_path = export_model_metadata(
        model,
        checkpoint,
        checkpoint_path,
        config,
        export_root,
        test_accuracy=0.9098,
    )

    assert (export_root / "weights" / "conv1_weights_float32.npy").is_file()
    assert (export_root / "biases" / "conv1_bias_float32.npy").is_file()
    assert metadata_path.is_file()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["layers"]["conv1"]["out_channels"] == 16
    assert metadata["layers"]["conv2"]["out_channels"] == 32
    assert metadata["class_order"] == checkpoint["class_names"]

    folded_conv1 = np.load(export_root / "weights" / "conv1_weights_float32.npy")
    assert folded_conv1.shape == (16, 3, 3, 3)
    raw_conv1 = np.load(export_root / "raw" / "conv1_weight_float32.npy")
    np.testing.assert_array_equal(raw_conv1, model.conv1.weight.detach().cpu().numpy())

    for name, parameter in model.named_parameters():
        assert torch.equal(before[name], parameter)

    with pytest.raises(FileExistsError, match="--force"):
        ensure_export_root(export_root, force=False)

    ensure_export_root(export_root, force=True)
    assert weight_paths and bias_paths


def test_activation_export_for_mocked_sample(tmp_path: Path) -> None:
    model = TinyCNN(num_classes=5, conv1_channels=16, conv2_channels=32)
    model.eval()
    transform = build_evaluation_transform(
        32,
        32,
        normalization_mean=[0.4, 0.35, 0.37],
        normalization_std=[0.29, 0.27, 0.28],
    )
    samples = [
        (Image.new("RGB", (40, 40), color=(255, 0, 0)), 0),
    ]
    dataset = RemappedImageDataset(samples, transform=transform, class_names=["stop"])
    reference = [
        ReferenceSample(
            sample_id="sample_000",
            dataset_index=0,
            true_class_index=0,
            true_class_name="stop",
            predicted_class_index=0,
            predicted_class_name="stop",
            correct=True,
            scores=[1.0, 0.0, 0.0, 0.0, 0.0],
            confidence=1.0,
            source_identifier="mock",
            selection_reason="unit_test",
        )
    ]
    output_dir = tmp_path / "activations"
    written = export_sample_activations(
        model,
        dataset,
        reference,
        output_dir,
        torch.device("cpu"),
    )
    sample_dir = written[0]
    assert (sample_dir / "input.npy").is_file()
    assert (sample_dir / "scores.npy").is_file()
    assert (sample_dir / "bn1.npy").is_file()
    assert (sample_dir / "expected_class.txt").is_file()
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["sample_id"] == "sample_000"
    assert metadata["tensor_shapes"]["scores"] == [1, 5]
