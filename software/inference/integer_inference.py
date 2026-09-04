from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from software.quantization.fixed_point import (
    QuantizationError,
    quantize_symmetric,
    requantize_int32,
    rounding_right_shift,
)
from software.quantization.saturation import INT8_MAX, INT8_MIN, saturate_int8


def pad_int_nchw(
    inputs: np.ndarray,
    padding: int,
    pad_value: int,
) -> np.ndarray:
    if padding == 0:
        return np.asarray(inputs)
    return np.pad(
        np.asarray(inputs),
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant",
        constant_values=int(pad_value),
    )


def integer_conv2d_nchw(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias_int32: np.ndarray,
    *,
    stride: int = 1,
    padding: int = 0,
    input_zero_point: int = 0,
    weight_zero_point: int = 0,
) -> np.ndarray:
    """INT8 x INT8 convolution into INT32 accumulators.

    Uses NumPy int64/int32 arithmetic (no float). Results match an explicit
    nested-loop MAC with the same zero points, stride, and padding.
    """
    x = pad_int_nchw(inputs, padding, pad_value=input_zero_point).astype(np.int32, copy=False)
    w = np.asarray(weights).astype(np.int32, copy=False)
    b = np.asarray(bias_int32, dtype=np.int32).reshape(-1)
    out_channels, in_channels, kernel_h, kernel_w = w.shape
    batch, channels, height, width = x.shape
    if channels != in_channels:
        raise QuantizationError("Convolution channel mismatch")

    out_h = (height - kernel_h) // stride + 1
    out_w = (width - kernel_w) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise QuantizationError("Invalid convolution output size")

    input_centered = x - np.int32(input_zero_point)
    weight_centered = w - np.int32(weight_zero_point)

    # Gather patches: [N, C, kH, kW, outH, outW]
    patches = np.lib.stride_tricks.as_strided(
        input_centered,
        shape=(batch, in_channels, kernel_h, kernel_w, out_h, out_w),
        strides=(
            input_centered.strides[0],
            input_centered.strides[1],
            input_centered.strides[2],
            input_centered.strides[3],
            input_centered.strides[2] * stride,
            input_centered.strides[3] * stride,
        ),
        writeable=False,
    )
    # products: sum over in_channels, kH, kW
    # weight_centered: [O, C, kH, kW]
    products = np.einsum(
        "nchwpq,ochw->nopq",
        patches.astype(np.int64),
        weight_centered.astype(np.int64),
        optimize=True,
    )
    output = products + b.reshape(1, -1, 1, 1).astype(np.int64)
    if np.any(output < np.iinfo(np.int32).min) or np.any(output > np.iinfo(np.int32).max):
        raise QuantizationError("Convolution accumulator overflowed int32")
    return output.astype(np.int32)


def integer_relu(values: np.ndarray, zero_point: int = 0) -> np.ndarray:
    """ReLU in quantized domain: max(value, zero_point)."""
    arr = np.asarray(values)
    return np.maximum(arr, int(zero_point)).astype(arr.dtype, copy=False)


def integer_max_pool2d_nchw(
    inputs: np.ndarray,
    *,
    kernel_size: int = 2,
    stride: int = 2,
) -> np.ndarray:
    values = np.asarray(inputs)
    batch, channels, height, width = values.shape
    out_h = (height - kernel_size) // stride + 1
    out_w = (width - kernel_size) // stride + 1
    # Vectorized max over pooling windows.
    patches = np.lib.stride_tricks.as_strided(
        values,
        shape=(batch, channels, out_h, out_w, kernel_size, kernel_size),
        strides=(
            values.strides[0],
            values.strides[1],
            values.strides[2] * stride,
            values.strides[3] * stride,
            values.strides[2],
            values.strides[3],
        ),
        writeable=False,
    )
    return patches.reshape(batch, channels, out_h, out_w, -1).max(axis=-1)


def integer_global_average_pool_nchw(
    inputs: np.ndarray,
    *,
    zero_point: int = 0,
) -> np.ndarray:
    """Average H*W integer activations with ties-away-from-zero rounding.

    For zero_point == 0 this is round(sum / (H*W)) in integer arithmetic.
    For nonzero zero_points, averages centered values then adds zero_point back.
    Output stays in the same activation scale and is saturated to the input dtype.
    """
    from software.quantization.fixed_point import round_divide_int

    values = np.asarray(inputs)
    batch, channels, height, width = values.shape
    denom = height * width
    if denom <= 0:
        raise QuantizationError("Invalid global-average-pool spatial size")

    plane_sums = values.astype(np.int64).reshape(batch, channels, -1)
    if zero_point == 0:
        totals = plane_sums.sum(axis=-1)
        averaged = round_divide_int(totals, denom)
    else:
        centered = plane_sums - np.int64(zero_point)
        totals = centered.sum(axis=-1)
        averaged = round_divide_int(totals, denom) + np.int64(zero_point)

    qmin = int(np.iinfo(values.dtype).min)
    qmax = int(np.iinfo(values.dtype).max)
    clipped = np.clip(averaged, qmin, qmax).astype(values.dtype)
    return clipped.reshape(batch, channels, 1, 1)


def integer_linear(
    features: np.ndarray,
    weights: np.ndarray,
    bias_int32: np.ndarray,
    *,
    input_zero_point: int = 0,
    weight_zero_point: int = 0,
) -> np.ndarray:
    """INT8 features x INT8 weights -> INT32 scores."""
    x = np.asarray(features)
    w = np.asarray(weights)
    b = np.asarray(bias_int32, dtype=np.int32).reshape(-1)
    if x.ndim != 2:
        raise QuantizationError("Features must be [batch, features]")
    batch, in_features = x.shape
    out_features, weight_in = w.shape
    if in_features != weight_in:
        raise QuantizationError("Linear feature mismatch")

    input_centered = x.astype(np.int64) - np.int64(input_zero_point)
    weight_centered = w.astype(np.int64) - np.int64(weight_zero_point)
    products = input_centered @ weight_centered.T
    output = products + b.astype(np.int64).reshape(1, -1)
    if np.any(output < np.iinfo(np.int32).min) or np.any(output > np.iinfo(np.int32).max):
        raise QuantizationError("Classifier accumulator overflowed int32")
    return output.astype(np.int32)

class IntegerTinyCNN:
    """Integer-only TinyCNN golden model using folded convolution parameters."""

    def __init__(self, package: dict[str, Any]) -> None:
        self.package = package
        self.meta = package["metadata"]
        self.weights = package["weights"]
        self.biases = package["biases"]
        self.scales = package["scales"]
        self.requant = package["requantization"]

    @classmethod
    def from_export_directory(cls, export_root: str | Path) -> "IntegerTinyCNN":
        root = Path(export_root)
        manifest = json.loads(
            (root / "metadata" / "quantization_manifest.json").read_text(encoding="utf-8")
        )
        package = {
            "metadata": manifest,
            "weights": {
                "conv1": np.load(root / "weights" / "conv1_weights_int8.npy"),
                "conv2": np.load(root / "weights" / "conv2_weights_int8.npy"),
                "classifier": np.load(root / "weights" / "classifier_weights_int8.npy"),
            },
            "biases": {
                "conv1": np.load(root / "biases" / "conv1_bias_int32.npy"),
                "conv2": np.load(root / "biases" / "conv2_bias_int32.npy"),
                "classifier": np.load(root / "biases" / "classifier_bias_int32.npy"),
            },
            "scales": {
                "input": float(np.load(root / "scales" / "input_scale_float32.npy")),
                "conv1_weight": np.load(root / "scales" / "conv1_weight_scales_float32.npy"),
                "conv1_output": float(
                    np.load(root / "scales" / "conv1_output_scale_float32.npy")
                ),
                "relu1": float(np.load(root / "scales" / "relu1_scale_float32.npy")),
                "pool1": float(np.load(root / "scales" / "pool1_scale_float32.npy")),
                "conv2_weight": np.load(root / "scales" / "conv2_weight_scales_float32.npy"),
                "conv2_output": float(
                    np.load(root / "scales" / "conv2_output_scale_float32.npy")
                ),
                "relu2": float(np.load(root / "scales" / "relu2_scale_float32.npy")),
                "pool2": float(np.load(root / "scales" / "pool2_scale_float32.npy")),
                "gap": float(
                    np.load(root / "scales" / "global_average_pool_scale_float32.npy")
                ),
                "flatten": float(np.load(root / "scales" / "flatten_scale_float32.npy")),
                "classifier_weight": np.load(
                    root / "scales" / "classifier_weight_scales_float32.npy"
                ),
                "classifier_output": float(
                    np.load(root / "scales" / "classifier_output_scale_float32.npy")
                ),
            },
            "requantization": {
                "conv1_multipliers": np.load(
                    root / "requantization" / "conv1_multipliers_int32.npy"
                ),
                "conv1_shifts": np.load(root / "requantization" / "conv1_shifts_int32.npy"),
                "conv2_multipliers": np.load(
                    root / "requantization" / "conv2_multipliers_int32.npy"
                ),
                "conv2_shifts": np.load(root / "requantization" / "conv2_shifts_int32.npy"),
                "gap_multiplier": np.load(root / "requantization" / "gap_multiplier_int32.npy"),
                "gap_shift": np.load(root / "requantization" / "gap_shift_int32.npy"),
                "classifier_multipliers": np.load(
                    root / "requantization" / "classifier_multipliers_int32.npy"
                ),
                "classifier_shifts": np.load(
                    root / "requantization" / "classifier_shifts_int32.npy"
                ),
            },
        }
        return cls(package)

    def quantize_input(self, float_input: np.ndarray) -> np.ndarray:
        return quantize_symmetric(
            float_input,
            self.scales["input"],
            qmin=INT8_MIN,
            qmax=INT8_MAX,
        )

    def forward_with_intermediates(
        self, float_or_quantized_input: np.ndarray, *, input_already_quantized: bool = False
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if input_already_quantized:
            input_q = np.asarray(float_or_quantized_input)
        else:
            input_q = self.quantize_input(float_or_quantized_input)

        layers = self.meta["layers"]
        conv1_acc = integer_conv2d_nchw(
            input_q,
            self.weights["conv1"],
            self.biases["conv1"],
            stride=int(layers["conv1"]["stride"]),
            padding=int(layers["conv1"]["padding"]),
            input_zero_point=0,
            weight_zero_point=0,
        )
        conv1 = requantize_int32(
            conv1_acc,
            self.requant["conv1_multipliers"],
            self.requant["conv1_shifts"],
            output_zero_point=0,
            qmin=INT8_MIN,
            qmax=INT8_MAX,
        )
        relu1 = integer_relu(conv1, zero_point=0).astype(np.int8)
        pool1 = integer_max_pool2d_nchw(
            relu1,
            kernel_size=int(layers["pool1"]["kernel_size"]),
            stride=int(layers["pool1"]["stride"]),
        )

        conv2_acc = integer_conv2d_nchw(
            pool1,
            self.weights["conv2"],
            self.biases["conv2"],
            stride=int(layers["conv2"]["stride"]),
            padding=int(layers["conv2"]["padding"]),
            input_zero_point=0,
            weight_zero_point=0,
        )
        conv2 = requantize_int32(
            conv2_acc,
            self.requant["conv2_multipliers"],
            self.requant["conv2_shifts"],
            output_zero_point=0,
            qmin=INT8_MIN,
            qmax=INT8_MAX,
        )
        relu2 = integer_relu(conv2, zero_point=0).astype(np.int8)
        pool2 = integer_max_pool2d_nchw(
            relu2,
            kernel_size=int(layers["pool2"]["kernel_size"]),
            stride=int(layers["pool2"]["stride"]),
        )
        gap = integer_global_average_pool_nchw(pool2, zero_point=0)
        # Requantize GAP from pool2 scale into calibrated flatten/GAP scale.
        gap_mult = np.asarray(self.requant["gap_multiplier"], dtype=np.int32).reshape(1)
        gap_shift = np.asarray(self.requant["gap_shift"], dtype=np.int32).reshape(1)
        gap_acc = gap.astype(np.int32)
        # Treat as NCHW with C=channels for per-tensor broadcast via fake channel mult.
        # Use dense-style path: reshape to [N, C] then back.
        flat_gap = gap_acc.reshape(gap_acc.shape[0], gap_acc.shape[1])
        # Shared scalar multiplier: duplicate across channels.
        mult = np.full((flat_gap.shape[1],), int(gap_mult.reshape(-1)[0]), dtype=np.int32)
        sh = np.full((flat_gap.shape[1],), int(gap_shift.reshape(-1)[0]), dtype=np.int32)
        gap_q = requantize_int32(
            flat_gap,
            mult,
            sh,
            output_zero_point=0,
            qmin=INT8_MIN,
            qmax=INT8_MAX,
        ).reshape(gap_acc.shape)
        flatten = gap_q.reshape(gap_q.shape[0], -1)

        classifier_acc = integer_linear(
            flatten,
            self.weights["classifier"],
            self.biases["classifier"],
            input_zero_point=0,
            weight_zero_point=0,
        )
        scores = requantize_int32(
            classifier_acc,
            self.requant["classifier_multipliers"],
            self.requant["classifier_shifts"],
            output_zero_point=0,
            qmin=np.iinfo(np.int32).min,
            qmax=np.iinfo(np.int32).max,
        ).astype(np.int32)

        intermediates = {
            "input": input_q.astype(np.int8, copy=False),
            "conv1_accumulator": conv1_acc,
            "conv1": conv1,
            "relu1": relu1,
            "pool1": pool1,
            "conv2_accumulator": conv2_acc,
            "conv2": conv2,
            "relu2": relu2,
            "pool2": pool2,
            "global_average_pool": gap_q,
            "flatten": flatten,
            "classifier_accumulator": classifier_acc,
            "scores": scores,
        }
        return scores, intermediates

    def forward(self, float_input: np.ndarray) -> np.ndarray:
        scores, _ = self.forward_with_intermediates(float_input)
        return scores

    def predict(self, float_input: np.ndarray) -> np.ndarray:
        scores = self.forward(float_input)
        return np.argmax(scores, axis=1)
