from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from software.export.export_fpga_hex import (
    export_array_hex,
    flatten_row_major,
    parse_twos_complement_hex,
    read_hex_file,
    twos_complement_hex,
    write_hex_file,
)
from software.utils.config import project_root


def test_signed_int8_encoding():
    assert twos_complement_hex(0, bits=8) == "00"
    assert twos_complement_hex(127, bits=8) == "7f"
    assert twos_complement_hex(-1, bits=8) == "ff"
    assert twos_complement_hex(-128, bits=8) == "80"
    assert parse_twos_complement_hex("80", bits=8) == -128
    assert parse_twos_complement_hex("7f", bits=8) == 127


def test_signed_int32_encoding():
    assert twos_complement_hex(-1, bits=32) == "ffffffff"
    assert twos_complement_hex(1, bits=32) == "00000001"
    assert parse_twos_complement_hex("ffffffff", bits=32) == -1


def test_fixed_width_and_one_value_per_line(tmp_path: Path):
    values = np.array([1, -2, 127, -128], dtype=np.int8)
    path = tmp_path / "vals.hex"
    write_hex_file(path, values, bits=8)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines == ["01", "fe", "7f", "80"]
    for line in lines:
        assert len(line) == 2


def test_flatten_order():
    arr = np.arange(2 * 3 * 2 * 2, dtype=np.int8).reshape(2, 3, 2, 2)
    flat = flatten_row_major(arr)
    np.testing.assert_array_equal(flat, arr.reshape(-1, order="C"))


def test_round_trip(tmp_path: Path):
    arr = np.array([[-5, 0], [12, 40]], dtype=np.int8)
    path = tmp_path / "a.hex"
    meta = export_array_hex(arr, path, bits=8)
    assert meta["count"] == 4
    reloaded = read_hex_file(path, bits=8, dtype=np.int8).reshape(arr.shape)
    np.testing.assert_array_equal(reloaded, arr)


def test_width_overflow_rejected():
    with pytest.raises(Exception):
        twos_complement_hex(128, bits=8)
    with pytest.raises(Exception):
        twos_complement_hex(-129, bits=8)


def test_exported_fpga_hex_round_trip_if_present():
    root = project_root() / "software" / "exported_model" / "int8"
    weights = root / "weights" / "conv1_weights_int8.npy"
    hex_path = root / "fpga" / "conv1_weights.hex"
    if not weights.is_file() or not hex_path.is_file():
        pytest.skip("FPGA HEX export not present")
    original = np.load(weights)
    reloaded = read_hex_file(hex_path, bits=8, dtype=np.int8)
    np.testing.assert_array_equal(reloaded, flatten_row_major(original))
    reconstructed = reloaded.reshape(original.shape)
    np.testing.assert_array_equal(reconstructed, original)

    bias = np.load(root / "biases" / "conv1_bias_int32.npy")
    bias_hex = root / "fpga" / "conv1_bias.hex"
    reloaded_b = read_hex_file(bias_hex, bits=32, dtype=np.int32)
    np.testing.assert_array_equal(reloaded_b, flatten_row_major(bias))
