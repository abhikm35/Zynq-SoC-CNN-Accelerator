from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from software.inference.integer_inference import (
    IntegerTinyCNN,
    integer_conv2d_nchw,
    integer_global_average_pool_nchw,
    integer_linear,
    integer_max_pool2d_nchw,
    integer_relu,
)
from software.quantization.fixed_point import quantize_multiplier, requantize_int32
from software.utils.config import project_root


def test_integer_conv_single_channel_exact():
    # input 1x1x2x2, weight 1x1x2x2 all ones, bias 3, no pad stride 1
    inputs = np.array([[[[1, 2], [3, 4]]]], dtype=np.int8)
    weights = np.ones((1, 1, 2, 2), dtype=np.int8)
    bias = np.array([3], dtype=np.int32)
    out = integer_conv2d_nchw(inputs, weights, bias, stride=1, padding=0)
    assert out.shape == (1, 1, 1, 1)
    assert int(out[0, 0, 0, 0]) == 1 + 2 + 3 + 4 + 3


def test_integer_conv_padding_zero_point():
    inputs = np.array([[[[5]]]], dtype=np.int8)
    weights = np.ones((1, 1, 3, 3), dtype=np.int8)
    bias = np.array([0], dtype=np.int32)
    # pad with zp=1 => nine cells of centered (value-1); center cell = 4, edges = 0
    out = integer_conv2d_nchw(
        inputs,
        weights,
        bias,
        stride=1,
        padding=1,
        input_zero_point=1,
        weight_zero_point=0,
    )
    assert out.shape == (1, 1, 1, 1)
    assert int(out[0, 0, 0, 0]) == 4


def test_integer_conv_multi_channel():
    inputs = np.zeros((1, 2, 2, 2), dtype=np.int8)
    inputs[0, 0] = 1
    inputs[0, 1] = 2
    weights = np.zeros((2, 2, 1, 1), dtype=np.int8)
    weights[0, 0, 0, 0] = 3
    weights[0, 1, 0, 0] = 4
    weights[1, 0, 0, 0] = 5
    weights[1, 1, 0, 0] = 6
    bias = np.array([10, 20], dtype=np.int32)
    out = integer_conv2d_nchw(inputs, weights, bias, stride=1, padding=0)
    # channel0: 1*3 + 2*4 + 10 = 21
    # channel1: 1*5 + 2*6 + 20 = 37
    assert int(out[0, 0, 0, 0]) == 21
    assert int(out[0, 1, 0, 0]) == 37


def test_requantize_relu_saturation():
    acc = np.array([[[[1000], [-500]]]], dtype=np.int32)
    mult, shift = quantize_multiplier(0.1)
    out = requantize_int32(
        acc,
        np.array([mult], dtype=np.int32),
        np.array([shift], dtype=np.int32),
        qmin=-128,
        qmax=127,
    )
    relu = integer_relu(out, zero_point=0)
    assert int(relu.min()) >= 0


def test_integer_max_pool():
    inputs = np.array(
        [[[[-1, 3], [2, 0]], [[4, 1], [0, 5]]]],
        dtype=np.int8,
    )
    out = integer_max_pool2d_nchw(inputs, kernel_size=2, stride=2)
    assert out.shape == (1, 2, 1, 1)
    assert int(out[0, 0, 0, 0]) == 3
    assert int(out[0, 1, 0, 0]) == 5


def test_integer_max_pool_batch():
    inputs = np.arange(2 * 1 * 2 * 2, dtype=np.int8).reshape(2, 1, 2, 2)
    out = integer_max_pool2d_nchw(inputs, kernel_size=2, stride=2)
    assert out.shape == (2, 1, 1, 1)
    assert int(out[0, 0, 0, 0]) == 3
    assert int(out[1, 0, 0, 0]) == 7


def test_integer_gap_exact():
    inputs = np.array([[[[1, 2], [3, 4]]]], dtype=np.int8)  # sum=10, /4 = 2.5 -> 3
    out = integer_global_average_pool_nchw(inputs, zero_point=0)
    assert out.shape == (1, 1, 1, 1)
    assert int(out[0, 0, 0, 0]) == 3


def test_integer_gap_negative_and_zero_point():
    inputs = np.array([[[[0, 2], [2, 4]]]], dtype=np.int8)  # zp=2 => centered -2,0,0,2 sum=0
    out = integer_global_average_pool_nchw(inputs, zero_point=2)
    assert int(out[0, 0, 0, 0]) == 2


def test_integer_linear_exact():
    features = np.array([[1, -2, 3]], dtype=np.int8)
    weights = np.array([[2, 0, -1], [1, 1, 1]], dtype=np.int8)
    bias = np.array([5, -1], dtype=np.int32)
    out = integer_linear(features, weights, bias)
    # class0: 5 + 1*2 + (-2)*0 + 3*(-1) = 4
    # class1: -1 + 1 + (-2) + 3 = 1
    assert out.tolist() == [[4, 1]]
    assert int(np.argmax(out, axis=1)[0]) == 0


def test_complete_integer_model_if_exported():
    root = project_root() / "software" / "exported_model" / "int8"
    manifest = root / "metadata" / "quantization_manifest.json"
    if not manifest.is_file():
        pytest.skip("INT8 export not present yet")
    model = IntegerTinyCNN.from_export_directory(root)
    sample = root / "test_vectors" / "sample_000"
    if not sample.is_dir():
        pytest.skip("INT8 test vectors not present")
    float_input = np.load(
        project_root()
        / "software"
        / "exported_model"
        / "test_vectors"
        / "float32"
        / "sample_000"
        / "input.npy"
    )
    scores1, inter1 = model.forward_with_intermediates(float_input)
    scores2, inter2 = model.forward_with_intermediates(float_input)
    np.testing.assert_array_equal(scores1, scores2)
    for name in inter1:
        np.testing.assert_array_equal(inter1[name], inter2[name])
        assert inter1[name].dtype in (np.int8, np.int32)

    expected_scores = np.load(sample / "scores.npy")
    np.testing.assert_array_equal(scores1, expected_scores)
    meta = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    assert int(np.argmax(scores1, axis=1)[0]) == meta["int8_predicted_class"]
