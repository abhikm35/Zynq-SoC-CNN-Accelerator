"""Shared activation / weight memory layouts for Python and RTL.

Layouts match NumPy C-order (row-major) flattening of NCHW activations and
OIHW convolution weights used by the integer golden model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from software.export.export_fpga_hex import (
    flatten_row_major,
    parse_twos_complement_hex,
    twos_complement_hex,
    write_hex_file,
)
from software.quantization.fixed_point import QuantizationError


def activation_address(
    channel: int,
    row: int,
    column: int,
    *,
    height: int,
    width: int,
) -> int:
    """NCHW planar address: channel*H*W + row*W + column."""
    if channel < 0 or row < 0 or column < 0:
        raise QuantizationError("activation indices must be non-negative")
    if row >= height or column >= width:
        raise QuantizationError("activation indices out of range")
    return int(channel) * int(height) * int(width) + int(row) * int(width) + int(column)


def weight_address(
    output_channel: int,
    input_channel: int,
    kernel_row: int,
    kernel_column: int,
    *,
    input_channels: int,
    kernel_height: int,
    kernel_width: int,
) -> int:
    """OIHW address for convolution weights."""
    return (
        int(output_channel) * int(input_channels) * int(kernel_height) * int(kernel_width)
        + int(input_channel) * int(kernel_height) * int(kernel_width)
        + int(kernel_row) * int(kernel_width)
        + int(kernel_column)
    )


def flatten_activation_nchw(activation: np.ndarray) -> np.ndarray:
    """Flatten NCHW or CHW activation to C-order vector."""
    arr = np.asarray(activation)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise QuantizationError("Only batch size 1 supported for flatten helper")
        arr = arr[0]
    if arr.ndim != 3:
        raise QuantizationError(f"Expected CHW or NCHW activation, got {arr.shape}")
    return flatten_row_major(arr)


def unflatten_activation_chw(
    flat: np.ndarray,
    *,
    channels: int,
    height: int,
    width: int,
    dtype=np.int8,
) -> np.ndarray:
    values = np.asarray(flat).reshape(-1)
    expected = channels * height * width
    if values.size != expected:
        raise QuantizationError(
            f"Flat activation size {values.size} != {channels}*{height}*{width}"
        )
    return values.astype(dtype, copy=False).reshape(channels, height, width, order="C")


def flatten_weights_oihw(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights)
    if arr.ndim != 4:
        raise QuantizationError(f"Expected OIHW weights, got shape {arr.shape}")
    return flatten_row_major(arr)


def unflatten_weights_oihw(
    flat: np.ndarray,
    *,
    out_channels: int,
    in_channels: int,
    kernel_height: int,
    kernel_width: int,
    dtype=np.int8,
) -> np.ndarray:
    values = np.asarray(flat).reshape(-1)
    expected = out_channels * in_channels * kernel_height * kernel_width
    if values.size != expected:
        raise QuantizationError(
            f"Flat weight size {values.size} != expected {expected}"
        )
    return values.astype(dtype, copy=False).reshape(
        out_channels, in_channels, kernel_height, kernel_width, order="C"
    )


def write_mem_file(path, values: np.ndarray, *, bits: int):
    """Write `$readmemh`-compatible one-value-per-line HEX."""
    return write_hex_file(path, values, bits=bits)


def read_mem_file(path, *, bits: int, dtype):
    from software.export.export_fpga_hex import read_hex_file

    return read_hex_file(path, bits=bits, dtype=dtype)


def encode_signed_list(values: Sequence[int], *, bits: int) -> list[str]:
    return [twos_complement_hex(int(v), bits=bits) for v in values]


def decode_signed_list(lines: Sequence[str], *, bits: int) -> list[int]:
    return [parse_twos_complement_hex(line, bits=bits) for line in lines]
