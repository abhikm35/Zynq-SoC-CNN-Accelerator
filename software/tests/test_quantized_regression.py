from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from software.inference.integer_inference import IntegerTinyCNN
from software.utils.config import project_root


def test_quantized_regression_against_exported_vectors():
    root = project_root() / "software" / "exported_model" / "int8"
    manifest = root / "metadata" / "quantization_manifest.json"
    if not manifest.is_file():
        pytest.skip("INT8 export not present yet")

    model = IntegerTinyCNN.from_export_directory(root)
    float_root = project_root() / "software" / "exported_model" / "test_vectors" / "float32"
    int_root = root / "test_vectors"

    # Snapshot weights to ensure inference does not mutate parameters.
    weight_snapshot = {
        name: array.copy() for name, array in model.weights.items()
    }
    bias_snapshot = {name: array.copy() for name, array in model.biases.items()}

    sample_dirs = sorted(
        path for path in int_root.iterdir() if path.is_dir() and path.name.startswith("sample_")
    )
    assert sample_dirs, "Expected exported INT8 test vectors"

    for sample_dir in sample_dirs:
        float_input = np.load(float_root / sample_dir.name / "input.npy")
        scores, intermediates = model.forward_with_intermediates(float_input)
        for name, tensor in intermediates.items():
            expected = np.load(sample_dir / f"{name}.npy")
            np.testing.assert_array_equal(tensor, expected)
        meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        assert int(np.argmax(scores, axis=1)[0]) == meta["int8_predicted_class"]

        # Determinism
        scores2, intermediates2 = model.forward_with_intermediates(float_input)
        np.testing.assert_array_equal(scores, scores2)
        for name in intermediates:
            np.testing.assert_array_equal(intermediates[name], intermediates2[name])

    for name, array in model.weights.items():
        np.testing.assert_array_equal(array, weight_snapshot[name])
    for name, array in model.biases.items():
        np.testing.assert_array_equal(array, bias_snapshot[name])


def test_integer_inference_does_not_import_torch_nn_forward(monkeypatch):
    """Guardrail: integer path must not call torch.nn Module forward."""
    root = project_root() / "software" / "exported_model" / "int8"
    if not (root / "metadata" / "quantization_manifest.json").is_file():
        pytest.skip("INT8 export not present yet")

    import torch.nn as nn

    def boom(*args, **kwargs):
        raise AssertionError("torch.nn Module forward should not be used")

    monkeypatch.setattr(nn.Module, "__call__", boom)
    model = IntegerTinyCNN.from_export_directory(root)
    float_input = np.load(
        project_root()
        / "software"
        / "exported_model"
        / "test_vectors"
        / "float32"
        / "sample_000"
        / "input.npy"
    )
    scores, _ = model.forward_with_intermediates(float_input)
    assert scores.dtype == np.int32
