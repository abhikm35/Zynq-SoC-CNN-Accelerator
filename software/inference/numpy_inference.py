from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def pad_input(
    inputs: np.ndarray,
    padding: int | tuple[int, int] = 0,
) -> np.ndarray:
    """Zero-pad an NCHW tensor."""
    if isinstance(padding, int):
        pad_h = pad_w = int(padding)
    else:
        pad_h, pad_w = int(padding[0]), int(padding[1])
    if pad_h < 0 or pad_w < 0:
        raise ValueError("padding must be non-negative")
    if pad_h == 0 and pad_w == 0:
        return np.asarray(inputs, dtype=np.float32)
    return np.pad(
        np.asarray(inputs, dtype=np.float32),
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0.0,
    )


def conv2d_nchw(
    inputs: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
    *,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Explicit NCHW convolution without im2col or library convolution helpers."""
    x = pad_input(inputs, padding)
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 4:
        raise ValueError(f"weight must be 4D, got shape {weight.shape}")
    out_channels, in_channels, kernel_h, kernel_w = weight.shape
    batch, channels, height, width = x.shape
    if channels != in_channels:
        raise ValueError(
            f"Input channels {channels} do not match weight in_channels {in_channels}"
        )

    out_h = (height - kernel_h) // stride + 1
    out_w = (width - kernel_w) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Invalid convolution output size")

    if bias is None:
        bias_vec = np.zeros((out_channels,), dtype=np.float32)
    else:
        bias_vec = np.asarray(bias, dtype=np.float32).reshape(out_channels)

    output = np.zeros((batch, out_channels, out_h, out_w), dtype=np.float32)
    for batch_index in range(batch):
        for out_channel in range(out_channels):
            for out_row in range(out_h):
                for out_col in range(out_w):
                    # Accumulate in float64 to reduce float32 rounding drift versus cuDNN/PyTorch.
                    accumulator = float(bias_vec[out_channel])
                    row0 = out_row * stride
                    col0 = out_col * stride
                    for in_channel in range(in_channels):
                        for kernel_row in range(kernel_h):
                            for kernel_col in range(kernel_w):
                                accumulator += float(
                                    x[
                                        batch_index,
                                        in_channel,
                                        row0 + kernel_row,
                                        col0 + kernel_col,
                                    ]
                                ) * float(
                                    weight[
                                        out_channel,
                                        in_channel,
                                        kernel_row,
                                        kernel_col,
                                    ]
                                )
                    output[batch_index, out_channel, out_row, out_col] = np.float32(
                        accumulator
                    )
    return output


def relu(inputs: np.ndarray) -> np.ndarray:
    values = np.asarray(inputs, dtype=np.float32)
    return np.maximum(values, 0.0)


def max_pool2d_nchw(
    inputs: np.ndarray,
    *,
    kernel_size: int = 2,
    stride: int = 2,
) -> np.ndarray:
    values = np.asarray(inputs, dtype=np.float32)
    batch, channels, height, width = values.shape
    out_h = (height - kernel_size) // stride + 1
    out_w = (width - kernel_size) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Invalid max-pool output size")
    output = np.zeros((batch, channels, out_h, out_w), dtype=np.float32)
    for batch_index in range(batch):
        for channel in range(channels):
            for out_row in range(out_h):
                for out_col in range(out_w):
                    row0 = out_row * stride
                    col0 = out_col * stride
                    window = values[
                        batch_index,
                        channel,
                        row0 : row0 + kernel_size,
                        col0 : col0 + kernel_size,
                    ]
                    output[batch_index, channel, out_row, out_col] = float(window.max())
    return output


def global_average_pool_nchw(inputs: np.ndarray) -> np.ndarray:
    values = np.asarray(inputs, dtype=np.float32)
    return values.mean(axis=(2, 3), keepdims=True).astype(np.float32)


def linear(
    features: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Dense layer: scores = features @ weight.T + bias."""
    x = np.asarray(features, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    scores = x @ w.T
    if bias is not None:
        scores = scores + np.asarray(bias, dtype=np.float32).reshape(1, -1)
    return scores.astype(np.float32)


def batch_norm2d_nchw(
    inputs: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    *,
    eps: float = 1e-5,
) -> np.ndarray:
    """Inference-mode BatchNorm matching PyTorch BatchNorm2d."""
    values = np.asarray(inputs, dtype=np.float32)
    gamma = np.asarray(weight, dtype=np.float32).reshape(1, -1, 1, 1)
    beta = np.asarray(bias, dtype=np.float32).reshape(1, -1, 1, 1)
    mean = np.asarray(running_mean, dtype=np.float32).reshape(1, -1, 1, 1)
    var = np.asarray(running_var, dtype=np.float32).reshape(1, -1, 1, 1)
    return ((values - mean) / np.sqrt(var + float(eps)) * gamma + beta).astype(
        np.float32
    )


def predict_class(scores: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(scores, dtype=np.float32), axis=1)


class NumpyTinyCNN:
    """Independent NumPy floating-point TinyCNN with explicit BatchNorm."""

    def __init__(self, parameters: dict[str, Any], metadata: dict[str, Any]) -> None:
        self.parameters = parameters
        self.metadata = metadata
        conv = metadata["layers"]["conv1"]
        self.conv1_stride = int(conv["stride"])
        self.conv1_padding = int(conv["padding"])
        conv2 = metadata["layers"]["conv2"]
        self.conv2_stride = int(conv2["stride"])
        self.conv2_padding = int(conv2["padding"])
        pool1 = metadata["layers"]["pool1"]
        self.pool1_kernel = int(pool1["kernel_size"])
        self.pool1_stride = int(pool1["stride"])
        pool2 = metadata["layers"]["pool2"]
        self.pool2_kernel = int(pool2["kernel_size"])
        self.pool2_stride = int(pool2["stride"])

    @classmethod
    def from_export_directory(cls, export_root: str | Path) -> "NumpyTinyCNN":
        root = Path(export_root)
        metadata_path = root / "metadata" / "model_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        import json

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw_dir = root / "raw"
        parameters = {
            "conv1_weight": np.load(raw_dir / "conv1_weight_float32.npy"),
            "conv2_weight": np.load(raw_dir / "conv2_weight_float32.npy"),
            "classifier_weight": np.load(raw_dir / "classifier_weight_float32.npy"),
            "classifier_bias": np.load(raw_dir / "classifier_bias_float32.npy"),
            "bn1_weight": np.load(raw_dir / "bn1_weight_float32.npy"),
            "bn1_bias": np.load(raw_dir / "bn1_bias_float32.npy"),
            "bn1_running_mean": np.load(raw_dir / "bn1_running_mean_float32.npy"),
            "bn1_running_var": np.load(raw_dir / "bn1_running_var_float32.npy"),
            "bn1_eps": float(metadata["batchnorm"]["bn1"]["eps"]),
            "bn2_weight": np.load(raw_dir / "bn2_weight_float32.npy"),
            "bn2_bias": np.load(raw_dir / "bn2_bias_float32.npy"),
            "bn2_running_mean": np.load(raw_dir / "bn2_running_mean_float32.npy"),
            "bn2_running_var": np.load(raw_dir / "bn2_running_var_float32.npy"),
            "bn2_eps": float(metadata["batchnorm"]["bn2"]["eps"]),
        }
        return cls(parameters, metadata)

    def forward_with_intermediates(
        self, inputs: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x = np.asarray(inputs, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW input, got shape {x.shape}")

        conv1 = conv2d_nchw(
            x,
            self.parameters["conv1_weight"],
            bias=None,
            stride=self.conv1_stride,
            padding=self.conv1_padding,
        )
        bn1 = batch_norm2d_nchw(
            conv1,
            self.parameters["bn1_weight"],
            self.parameters["bn1_bias"],
            self.parameters["bn1_running_mean"],
            self.parameters["bn1_running_var"],
            eps=float(self.parameters["bn1_eps"]),
        )
        relu1 = relu(bn1)
        pool1 = max_pool2d_nchw(
            relu1,
            kernel_size=self.pool1_kernel,
            stride=self.pool1_stride,
        )

        conv2 = conv2d_nchw(
            pool1,
            self.parameters["conv2_weight"],
            bias=None,
            stride=self.conv2_stride,
            padding=self.conv2_padding,
        )
        bn2 = batch_norm2d_nchw(
            conv2,
            self.parameters["bn2_weight"],
            self.parameters["bn2_bias"],
            self.parameters["bn2_running_mean"],
            self.parameters["bn2_running_var"],
            eps=float(self.parameters["bn2_eps"]),
        )
        relu2 = relu(bn2)
        pool2 = max_pool2d_nchw(
            relu2,
            kernel_size=self.pool2_kernel,
            stride=self.pool2_stride,
        )
        global_average_pool = global_average_pool_nchw(pool2)
        flatten = global_average_pool.reshape(global_average_pool.shape[0], -1)
        scores = linear(
            flatten,
            self.parameters["classifier_weight"],
            self.parameters["classifier_bias"],
        )
        intermediates = {
            "input": x,
            "conv1": conv1,
            "bn1": bn1,
            "relu1": relu1,
            "pool1": pool1,
            "conv2": conv2,
            "bn2": bn2,
            "relu2": relu2,
            "pool2": pool2,
            "global_average_pool": global_average_pool,
            "flatten": flatten,
            "scores": scores,
        }
        return scores, intermediates

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        scores, _ = self.forward_with_intermediates(inputs)
        return scores
