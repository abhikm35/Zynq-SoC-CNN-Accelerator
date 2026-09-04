from __future__ import annotations

from typing import Any

import numpy as np


class QuantizationError(ValueError):
    """Raised when a quantization configuration or value is invalid."""


def validate_scale(scale: float | np.ndarray, *, name: str = "scale") -> np.ndarray:
    values = np.asarray(scale, dtype=np.float64)
    if values.size == 0:
        raise QuantizationError(f"{name} must not be empty")
    if not np.isfinite(values).all():
        raise QuantizationError(f"{name} contains NaN or infinity")
    if np.any(values <= 0.0):
        raise QuantizationError(f"{name} must be strictly positive")
    return values


def round_to_nearest_away_from_zero(values: np.ndarray) -> np.ndarray:
    """Round half-away-from-zero. Do not use NumPy banker's rounding."""
    arr = np.asarray(values, dtype=np.float64)
    rounded = np.where(
        arr >= 0.0,
        np.floor(arr + 0.5),
        np.ceil(arr - 0.5),
    )
    return rounded


def round_divide(numerator: np.ndarray, denominator: float | np.ndarray) -> np.ndarray:
    """Rounded division using ties-away-from-zero (float path for calibration)."""
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    if np.any(den == 0.0):
        raise QuantizationError("Division by zero in round_divide")
    return round_to_nearest_away_from_zero(num / den)


def round_divide_int(numerator: np.ndarray, denominator: int) -> np.ndarray:
    """Integer rounded division with ties away from zero.

    For positive ``n``: ``(n + d // 2) // d``.
    For negative ``n``: ``-(((-n) + d // 2) // d)``.
    Power-of-two denominators use ``rounding_right_shift``.
    """
    d = int(denominator)
    if d <= 0:
        raise QuantizationError("denominator must be positive")
    num = np.asarray(numerator, dtype=np.int64)
    if d & (d - 1) == 0:
        shift = int(d.bit_length() - 1)
        return rounding_right_shift(num, shift)

    half = d // 2

    def _one(value: int) -> int:
        if value >= 0:
            return (value + half) // d
        return -(((-value) + half) // d)

    if num.ndim == 0:
        return np.int64(_one(int(num)))
    return np.fromiter((_one(int(v)) for v in num.reshape(-1)), dtype=np.int64).reshape(
        num.shape
    )


def calculate_symmetric_scale(
    values: np.ndarray,
    *,
    max_abs_int: int = 127,
    eps: float = 1e-12,
) -> float:
    """scale = max(abs(values)) / max_abs_int."""
    if max_abs_int <= 0:
        raise QuantizationError("max_abs_int must be positive")
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise QuantizationError("Cannot calibrate non-finite values")
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    scale = max(max_abs, eps) / float(max_abs_int)
    validate_scale(scale, name="symmetric_scale")
    return float(scale)


def calculate_affine_scale_zero_point(
    values: np.ndarray,
    *,
    qmin: int,
    qmax: int,
    eps: float = 1e-12,
) -> tuple[float, int]:
    """Affine quantization: real = scale * (q - zero_point)."""
    if qmax <= qmin:
        raise QuantizationError("qmax must be greater than qmin")
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise QuantizationError("Cannot calibrate non-finite values")
    vmin = float(arr.min()) if arr.size else 0.0
    vmax = float(arr.max()) if arr.size else 0.0
    if vmax < vmin:
        raise QuantizationError("Invalid calibration range")
    scale = max(vmax - vmin, eps) / float(qmax - qmin)
    validate_scale(scale, name="affine_scale")
    # zero_point chosen so that 0 maps near a representable integer.
    zero_point = int(round_to_nearest_away_from_zero(np.array([qmin - vmin / scale]))[0])
    zero_point = int(np.clip(zero_point, qmin, qmax))
    return float(scale), zero_point


def quantize_symmetric(
    values: np.ndarray,
    scale: float | np.ndarray,
    *,
    qmin: int = -127,
    qmax: int = 127,
) -> np.ndarray:
    scales = validate_scale(scale, name="weight_or_activation_scale")
    arr = np.asarray(values, dtype=np.float64)
    if scales.ndim == 0:
        quantized = round_to_nearest_away_from_zero(arr / float(scales))
    else:
        # Broadcast per-channel scales over leading axis.
        reshape = (scales.shape[0],) + (1,) * (arr.ndim - 1)
        quantized = round_to_nearest_away_from_zero(arr / scales.reshape(reshape))
    clipped = np.clip(quantized, qmin, qmax)
    return clipped.astype(np.int8)


def quantize_affine(
    values: np.ndarray,
    scale: float,
    zero_point: int,
    *,
    qmin: int,
    qmax: int,
) -> np.ndarray:
    validate_scale(scale, name="affine_scale")
    arr = np.asarray(values, dtype=np.float64)
    quantized = round_to_nearest_away_from_zero(arr / float(scale) + float(zero_point))
    clipped = np.clip(quantized, qmin, qmax)
    if qmin >= 0:
        return clipped.astype(np.uint8)
    return clipped.astype(np.int8)


def dequantize(
    quantized: np.ndarray,
    scale: float | np.ndarray,
    zero_point: int | np.ndarray = 0,
) -> np.ndarray:
    q = np.asarray(quantized, dtype=np.float64)
    scales = validate_scale(scale, name="dequant_scale")
    zp = np.asarray(zero_point, dtype=np.float64)
    if scales.ndim == 0:
        return ((q - zp) * float(scales)).astype(np.float32)
    reshape = (scales.shape[0],) + (1,) * (q.ndim - 1)
    return ((q - zp) * scales.reshape(reshape)).astype(np.float32)


def quantize_bias_int32(
    bias: np.ndarray,
    input_scale: float,
    weight_scales: np.ndarray,
) -> np.ndarray:
    """bias_int32 = round(bias / (input_scale * weight_scale))."""
    validate_scale(input_scale, name="input_scale")
    scales = validate_scale(weight_scales, name="weight_scales")
    bias_scales = float(input_scale) * scales.astype(np.float64)
    bias_arr = np.asarray(bias, dtype=np.float64).reshape(-1)
    if bias_arr.shape != bias_scales.shape:
        raise QuantizationError(
            f"Bias shape {bias_arr.shape} does not match scale shape {bias_scales.shape}"
        )
    quantized = round_to_nearest_away_from_zero(bias_arr / bias_scales)
    # Keep as int64 temporarily to detect overflow, then cast.
    if np.any(quantized < np.iinfo(np.int32).min) or np.any(
        quantized > np.iinfo(np.int32).max
    ):
        raise QuantizationError("Bias quantization overflowed int32")
    return quantized.astype(np.int32)


def quantize_multiplier(real_multiplier: float, *, multiplier_bits: int = 31) -> tuple[int, int]:
    """Convert positive real multiplier to (int_multiplier, right_shift).

    Representation:
        real_multiplier ≈ int_multiplier / 2^right_shift

    int_multiplier is in [2^(bits-1), 2^bits - 1] after normalization, except for
    tiny multipliers that may underflow after frexp normalization.
    """
    if not np.isfinite(real_multiplier):
        raise QuantizationError("real_multiplier must be finite")
    if real_multiplier <= 0.0:
        raise QuantizationError("real_multiplier must be positive")
    if multiplier_bits < 8 or multiplier_bits > 31:
        raise QuantizationError("multiplier_bits must be in [8, 31]")

    significand, exponent = np.frexp(float(real_multiplier))
    # significand in [0.5, 1), real = significand * 2^exponent
    q = int(round_to_nearest_away_from_zero(np.array([significand * (1 << multiplier_bits)]))[0])
    if q == (1 << multiplier_bits):
        q //= 2
        exponent += 1
    right_shift = int(multiplier_bits - exponent)
    if right_shift < 0:
        # Multiplier too large; left-shift significand instead by reducing shift to 0
        # and saturating is unsafe. Raise so calibration can be inspected.
        raise QuantizationError(
            f"real_multiplier {real_multiplier} requires negative shift {right_shift}"
        )
    return int(q), right_shift


def rounding_right_shift(values: np.ndarray, shift: int | np.ndarray) -> np.ndarray:
    """Arithmetic right shift with round-to-nearest, ties away from zero."""
    x = np.asarray(values, dtype=np.int64)
    shifts = np.asarray(shift, dtype=np.int64)
    if np.any(shifts < 0):
        raise QuantizationError("right shift must be non-negative")

    def _shift_one(value: int, sh: int) -> int:
        if sh == 0:
            return int(value)
        half = 1 << (sh - 1)
        if value >= 0:
            return int((value + half) >> sh)
        return int(-(((-value) + half) >> sh))

    if shifts.ndim == 0:
        sh = int(shifts)
        if x.ndim == 0:
            return np.int64(_shift_one(int(x), sh))
        return np.fromiter((_shift_one(int(v), sh) for v in x.reshape(-1)), dtype=np.int64).reshape(
            x.shape
        )

    # Per-channel shifts broadcast over channel axis.
    if x.ndim < 1:
        raise QuantizationError("Cannot broadcast shifts onto scalar without matching shape")
    out = np.empty(x.shape, dtype=np.int64)
    if x.ndim == 1:
        for index, (value, sh) in enumerate(zip(x, np.broadcast_to(shifts, x.shape))):
            out[index] = _shift_one(int(value), int(sh))
        return out

    # NCHW: shifts shaped [C]
    channels = x.shape[1]
    channel_shifts = np.broadcast_to(shifts.reshape(-1), (channels,))
    for channel in range(channels):
        sh = int(channel_shifts[channel])
        plane = x[:, channel, ...]
        out[:, channel, ...] = np.fromiter(
            (_shift_one(int(v), sh) for v in plane.reshape(-1)),
            dtype=np.int64,
        ).reshape(plane.shape)
    return out


def requantize_int32(
    accumulator: np.ndarray,
    multipliers: np.ndarray,
    shifts: np.ndarray,
    *,
    output_zero_point: int = 0,
    qmin: int = -128,
    qmax: int = 127,
) -> np.ndarray:
    """requantized = saturate(round(acc * multiplier / 2^shift) + zp)."""
    acc = np.asarray(accumulator, dtype=np.int64)
    mult = np.asarray(multipliers, dtype=np.int64)
    sh = np.asarray(shifts, dtype=np.int64)

    if acc.ndim == 4:
        # NCHW convolution output
        if mult.shape[0] != acc.shape[1]:
            raise QuantizationError("Multiplier channel count mismatch")
        products = acc * mult.reshape(1, -1, 1, 1)
        shifted = rounding_right_shift(products, sh)
    elif acc.ndim == 2:
        # Dense scores [N, C]
        if mult.shape[0] != acc.shape[1]:
            raise QuantizationError("Multiplier class count mismatch")
        products = acc * mult.reshape(1, -1)
        shifted = rounding_right_shift(products, sh)
    elif acc.ndim == 1:
        products = acc * mult
        shifted = rounding_right_shift(products, sh)
    else:
        raise QuantizationError(f"Unsupported accumulator rank {acc.ndim}")

    with_zp = shifted + int(output_zero_point)
    clipped = np.clip(with_zp, qmin, qmax)
    if qmin >= 0:
        return clipped.astype(np.uint8)
    if qmin == -128 and qmax == 127:
        return clipped.astype(np.int8)
    return clipped.astype(np.int32)


def load_quantization_config(path: str = "software/config/quantization_config.yaml") -> dict[str, Any]:
    from software.utils.config import load_yaml, ConfigurationError

    config = load_yaml(path)
    required = [
        "quantization",
        "weights",
        "activations",
        "accumulators",
        "rounding",
        "saturation",
        "requantization",
        "layers",
        "export",
    ]
    for key in required:
        if key not in config or not isinstance(config[key], dict):
            raise ConfigurationError(f"Missing quantization config section '{key}'")

    if config["rounding"]["mode"] != "round_to_nearest_away_from_zero":
        raise ConfigurationError("Only round_to_nearest_away_from_zero is supported")
    if not config["weights"].get("symmetric", False):
        raise ConfigurationError("Only symmetric weight quantization is supported")
    if not config["activations"].get("symmetric", False):
        raise ConfigurationError("Only symmetric activation quantization is supported")
    if str(config["weights"]["dtype"]) != "int8":
        raise ConfigurationError("Weight dtype must be int8")
    if str(config["activations"]["dtype"]) != "int8":
        raise ConfigurationError("Activation dtype must be int8")
    if str(config["accumulators"]["dtype"]) != "int32":
        raise ConfigurationError("Accumulator dtype must be int32")
    return config
