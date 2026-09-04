from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from software.quantization.fixed_point import QuantizationError


def twos_complement_hex(value: int, *, bits: int) -> str:
    """Encode a signed integer as fixed-width lowercase two's-complement hex."""
    if bits <= 0 or bits % 4 != 0:
        raise QuantizationError("bits must be a positive multiple of 4")
    max_unsigned = 1 << bits
    min_signed = -(1 << (bits - 1))
    max_signed = (1 << (bits - 1)) - 1
    value = int(value)
    if value < min_signed or value > max_signed:
        raise QuantizationError(
            f"Value {value} does not fit in signed {bits}-bit range "
            f"[{min_signed}, {max_signed}]"
        )
    unsigned = value % max_unsigned
    width = bits // 4
    return f"{unsigned:0{width}x}"


def parse_twos_complement_hex(text: str, *, bits: int) -> int:
    unsigned = int(text.strip(), 16)
    sign_bit = 1 << (bits - 1)
    if unsigned & sign_bit:
        return unsigned - (1 << bits)
    return unsigned


def flatten_row_major(array: np.ndarray) -> np.ndarray:
    """C-style row-major flatten."""
    return np.asarray(array).reshape(-1, order="C")


def write_hex_file(
    path: str | Path,
    values: np.ndarray,
    *,
    bits: int,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    flat = flatten_row_major(values)
    lines = [twos_complement_hex(int(v), bits=bits) for v in flat]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def read_hex_file(path: str | Path, *, bits: int, dtype: np.dtype | type) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return np.asarray([], dtype=dtype)
    values = [parse_twos_complement_hex(line, bits=bits) for line in text.splitlines()]
    return np.asarray(values, dtype=dtype)


def export_array_hex(
    array: np.ndarray,
    path: str | Path,
    *,
    bits: int,
) -> dict[str, Any]:
    write_hex_file(path, array, bits=bits)
    reloaded = read_hex_file(path, bits=bits, dtype=array.dtype)
    original_flat = flatten_row_major(array)
    if reloaded.shape != original_flat.shape or not np.array_equal(reloaded, original_flat):
        raise QuantizationError(f"HEX round-trip failed for {path}")
    return {
        "path": str(path),
        "bits": bits,
        "count": int(original_flat.size),
        "shape": list(np.asarray(array).shape),
        "dtype": str(np.asarray(array).dtype),
        "flattening": "C-style row-major",
    }
