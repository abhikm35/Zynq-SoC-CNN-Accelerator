from __future__ import annotations

import numpy as np
import pytest

from software.quantization.fixed_point import (
    QuantizationError,
    calculate_affine_scale_zero_point,
    calculate_symmetric_scale,
    dequantize,
    quantize_affine,
    quantize_bias_int32,
    quantize_multiplier,
    quantize_symmetric,
    requantize_int32,
    round_divide_int,
    round_to_nearest_away_from_zero,
)
from software.quantization.saturation import (
    saturate_int8,
    saturate_narrow_int8,
    saturate_uint8,
)


def test_symmetric_scale():
    scale = calculate_symmetric_scale(np.array([-2.54, 1.0, 2.54]), max_abs_int=127)
    assert scale == pytest.approx(2.54 / 127)


def test_affine_scale_zero_point():
    values = np.array([0.0, 1.0, 2.0])
    scale, zp = calculate_affine_scale_zero_point(values, qmin=0, qmax=255)
    assert scale == pytest.approx(2.0 / 255)
    dequant = dequantize(quantize_affine(values, scale, zp, qmin=0, qmax=255), scale, zp)
    assert dequant.min() == pytest.approx(0.0, abs=1e-2)
    assert dequant.max() == pytest.approx(2.0, abs=1e-2)


def test_round_to_nearest_away_from_zero_examples():
    values = np.array([2.4, 2.5, -2.4, -2.5, 0.5, -0.5])
    rounded = round_to_nearest_away_from_zero(values)
    np.testing.assert_array_equal(rounded, np.array([2, 3, -2, -3, 1, -1]))


def test_round_divide_int_ties():
    assert int(round_divide_int(np.int64(5), 2)) == 3
    assert int(round_divide_int(np.int64(-5), 2)) == -3
    assert int(round_divide_int(np.int64(64), 64)) == 1
    assert int(round_divide_int(np.int64(32), 64)) == 1  # 0.5 -> away from zero -> 1
    assert int(round_divide_int(np.int64(-32), 64)) == -1


def test_quantize_dequantize_symmetric():
    values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    scale = calculate_symmetric_scale(values, max_abs_int=127)
    q = quantize_symmetric(values, scale, qmin=-127, qmax=127)
    dq = dequantize(q, scale, 0)
    assert np.max(np.abs(dq - values)) < scale


def test_saturate_int8():
    np.testing.assert_array_equal(
        saturate_int8(np.array([165, -150, 72])),
        np.array([127, -128, 72], dtype=np.int8),
    )


def test_saturate_uint8():
    np.testing.assert_array_equal(
        saturate_uint8(np.array([-1, 0, 255, 300])),
        np.array([0, 0, 255, 255], dtype=np.uint8),
    )


def test_narrow_range_weights():
    values = np.array([-200, -128, 0, 128, 200], dtype=np.float64)
    q = quantize_symmetric(values, 1.0, qmin=-127, qmax=127)
    assert int(q.min()) >= -127
    assert int(q.max()) <= 127
    np.testing.assert_array_equal(saturate_narrow_int8(np.array([-128, 128])), np.array([-127, 127]))


def test_per_channel_scales():
    weights = np.array(
        [
            [[[1.0, -1.0]]],
            [[[2.0, -2.0]]],
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0 / 127, 2.0 / 127], dtype=np.float32)
    q = quantize_symmetric(weights, scales, qmin=-127, qmax=127)
    assert q.shape == weights.shape
    assert int(q[0].max()) == 127
    assert int(q[1].max()) == 127


def test_invalid_scales():
    with pytest.raises(QuantizationError):
        quantize_symmetric(np.array([1.0]), 0.0)
    with pytest.raises(QuantizationError):
        quantize_symmetric(np.array([1.0]), -1.0)
    with pytest.raises(QuantizationError):
        quantize_symmetric(np.array([1.0]), np.nan)
    with pytest.raises(QuantizationError):
        quantize_symmetric(np.array([1.0]), np.inf)


def test_bias_quantization():
    bias = np.array([0.5, -1.0], dtype=np.float32)
    weight_scales = np.array([0.1, 0.2], dtype=np.float32)
    q = quantize_bias_int32(bias, 0.05, weight_scales)
    expected = round_to_nearest_away_from_zero(
        np.array([0.5 / (0.05 * 0.1), -1.0 / (0.05 * 0.2)])
    ).astype(np.int32)
    np.testing.assert_array_equal(q, expected)


def test_quantize_multiplier_and_requantize():
    mult, shift = quantize_multiplier(0.25, multiplier_bits=31)
    assert mult > 0
    assert shift >= 0
    approx = mult / (1 << shift)
    assert approx == pytest.approx(0.25, rel=1e-8)

    acc = np.array([[[[100]]]], dtype=np.int32)  # NCHW one channel
    out = requantize_int32(
        acc,
        np.array([mult], dtype=np.int32),
        np.array([shift], dtype=np.int32),
        output_zero_point=0,
        qmin=-128,
        qmax=127,
    )
    assert out.shape == (1, 1, 1, 1)
    assert int(out[0, 0, 0, 0]) == 25


def test_saturation_boundaries():
    q = quantize_symmetric(np.array([10.0]), 0.01, qmin=-128, qmax=127)
    assert int(q[0]) == 127
    q2 = quantize_symmetric(np.array([-10.0]), 0.01, qmin=-128, qmax=127)
    assert int(q2[0]) == -128
