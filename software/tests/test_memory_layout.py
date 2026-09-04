from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from software.hardware.memory_layout import (
    activation_address,
    flatten_activation_nchw,
    flatten_weights_oihw,
    unflatten_activation_chw,
    unflatten_weights_oihw,
    weight_address,
)
from software.utils.config import project_root


def test_activation_address_formula():
    assert activation_address(0, 0, 0, height=32, width=32) == 0
    assert activation_address(1, 0, 0, height=32, width=32) == 1024
    assert activation_address(2, 31, 31, height=32, width=32) == 2 * 1024 + 31 * 32 + 31


def test_weight_address_conv1():
    assert weight_address(0, 0, 0, 0, input_channels=3, kernel_height=3, kernel_width=3) == 0
    assert weight_address(0, 1, 0, 0, input_channels=3, kernel_height=3, kernel_width=3) == 9
    assert weight_address(1, 0, 0, 0, input_channels=3, kernel_height=3, kernel_width=3) == 27
    assert weight_address(1, 2, 2, 2, input_channels=3, kernel_height=3, kernel_width=3) == 27 + 18 + 6 + 2


def test_activation_round_trip():
    rng = np.random.default_rng(0)
    original = rng.integers(-128, 128, size=(3, 32, 32), dtype=np.int8)
    flat = flatten_activation_nchw(original)
    restored = unflatten_activation_chw(flat, channels=3, height=32, width=32)
    np.testing.assert_array_equal(restored, original)
    # Spot-check addressing
    for c, r, col in [(0, 0, 0), (1, 10, 11), (2, 31, 31)]:
        assert flat[activation_address(c, r, col, height=32, width=32)] == original[c, r, col]


def test_weight_round_trip_exported_conv1():
    path = project_root() / "software/exported_model/int8/weights/conv1_weights_int8.npy"
    if not path.is_file():
        pytest.skip("INT8 conv1 weights missing")
    original = np.load(path)
    flat = flatten_weights_oihw(original)
    restored = unflatten_weights_oihw(
        flat,
        out_channels=original.shape[0],
        in_channels=original.shape[1],
        kernel_height=original.shape[2],
        kernel_width=original.shape[3],
    )
    np.testing.assert_array_equal(restored, original)
    oc, ic, kr, kc = 3, 1, 2, 0
    addr = weight_address(
        oc,
        ic,
        kr,
        kc,
        input_channels=original.shape[1],
        kernel_height=3,
        kernel_width=3,
    )
    assert flat[addr] == original[oc, ic, kr, kc]
