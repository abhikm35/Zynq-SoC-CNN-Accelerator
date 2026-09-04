from __future__ import annotations

import numpy as np

from software.quantization.fixed_point import QuantizationError


INT8_MIN = -128
INT8_MAX = 127
UINT8_MIN = 0
UINT8_MAX = 255
NARROW_INT8_MIN = -127
NARROW_INT8_MAX = 127
INT32_MIN = -2147483648
INT32_MAX = 2147483647


def saturate(values: np.ndarray, qmin: int, qmax: int) -> np.ndarray:
    arr = np.asarray(values)
    if qmin > qmax:
        raise QuantizationError("qmin must be <= qmax")
    return np.clip(arr, qmin, qmax)


def saturate_int8(values: np.ndarray) -> np.ndarray:
    return saturate(values, INT8_MIN, INT8_MAX).astype(np.int8)


def saturate_uint8(values: np.ndarray) -> np.ndarray:
    return saturate(values, UINT8_MIN, UINT8_MAX).astype(np.uint8)


def saturate_narrow_int8(values: np.ndarray) -> np.ndarray:
    return saturate(values, NARROW_INT8_MIN, NARROW_INT8_MAX).astype(np.int8)


def saturate_int32(values: np.ndarray) -> np.ndarray:
    return saturate(values, INT32_MIN, INT32_MAX).astype(np.int32)


def count_saturations(values: np.ndarray, qmin: int, qmax: int) -> int:
    arr = np.asarray(values)
    return int(np.count_nonzero((arr < qmin) | (arr > qmax)))
