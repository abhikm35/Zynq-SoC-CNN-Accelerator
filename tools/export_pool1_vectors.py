#!/usr/bin/env python3
"""Export expected MaxPool1 tensor for RTL.

Trained model: Conv1 is 16 x 32 x 32, so Pool1 is 16 x 16 x 16 = 4096.
Prompt sketches with 8 channels / 2048 outputs are obsolete.
MaxPool is 2x2 stride 2, signed INT8 max only (no requant / ReLU).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from software.hardware.memory_layout import write_mem_file
from software.inference.integer_inference import (
    IntegerTinyCNN,
    integer_max_pool2d_nchw,
)
from software.utils.config import resolve_path


NUM_CHANNELS = 16
POOL_H = 16
POOL_W = 16
EXPECTED_COUNT = NUM_CHANNELS * POOL_H * POOL_W  # 4096

SELECTED_WINDOWS = [
    ("ch0_pr0_pc0", 0, 0, 0),
    ("ch0_pr15_pc15", 0, 15, 15),
    ("ch3_pr7_pc9", 3, 7, 9),
    ("ch5_pr12_pc4", 5, 12, 4),
    ("ch7_pr15_pc15", 7, 15, 15),
    ("ch15_pr15_pc15", 15, 15, 15),
]


def pool1_address(channel: int, pool_row: int, pool_col: int) -> int:
    return channel * 256 + pool_row * 16 + pool_col


def conv1_address(channel: int, row: int, col: int) -> int:
    return channel * 1024 + row * 32 + col


def window_coords(pool_row: int, pool_col: int) -> list[tuple[int, int]]:
    br, bc = pool_row * 2, pool_col * 2
    return [
        (br, bc),
        (br, bc + 1),
        (br + 1, bc),
        (br + 1, bc + 1),
    ]


def export_pool1(
    *,
    int8_root: str | Path = "software/exported_model/int8",
    sample_input: str | Path | None = None,
    output_dir: str | Path = "vectors/pool1",
) -> dict[str, Any]:
    root = resolve_path(int8_root)
    if sample_input is None:
        sample_path = resolve_path(
            "software/exported_model/int8/test_vectors/sample_000/input.npy"
        )
    else:
        sample_path = resolve_path(sample_input)
    input_q = np.load(sample_path)
    if input_q.ndim == 3:
        input_q = input_q[None, ...]

    model = IntegerTinyCNN.from_export_directory(root)
    _, inter = model.forward_with_intermediates(input_q, input_already_quantized=True)
    relu1 = np.asarray(inter["relu1"], dtype=np.int8)
    pool1 = np.asarray(inter["pool1"], dtype=np.int8)
    # Cross-check against explicit max-pool helper
    pool_check = integer_max_pool2d_nchw(relu1, kernel_size=2, stride=2).astype(np.int8)
    if not np.array_equal(pool1, pool_check):
        raise AssertionError("pool1 mismatch vs integer_max_pool2d_nchw")
    if pool1.shape != (1, NUM_CHANNELS, POOL_H, POOL_W):
        raise RuntimeError(f"Unexpected pool1 shape {pool1.shape}")

    tensor = pool1[0]
    flat = tensor.reshape(-1, order="C")
    if flat.size != EXPECTED_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_COUNT}, got {flat.size}")

    out = resolve_path(output_dir)
    ch_dir = out / "channel_summaries"
    traces_dir = out / "selected_window_traces"
    ch_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    write_mem_file(out / "pool1_expected.mem", flat, bits=8)
    # Also keep a copy reference of Conv1 tensor used as MaxPool input
    write_mem_file(
        out / "conv1_input_for_pool.mem",
        np.asarray(inter["relu1"][0], dtype=np.int8).reshape(-1, order="C"),
        bits=8,
    )

    zeros = int(np.sum(flat == 0))
    checksum = int(np.sum(flat.astype(np.int64)))
    channel_meta: list[dict[str, Any]] = []
    for ch in range(NUM_CHANNELS):
        ch_flat = tensor[ch].reshape(-1, order="C")
        info = {
            "channel": ch,
            "count": 256,
            "min": int(ch_flat.min()),
            "max": int(ch_flat.max()),
            "num_zeros": int(np.sum(ch_flat == 0)),
            "checksum_sum_int64": int(np.sum(ch_flat.astype(np.int64))),
            "address_base": ch * 256,
            "address_end": ch * 256 + 255,
        }
        channel_meta.append(info)
        (ch_dir / f"channel_{ch:02d}.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        )

    meta = {
        "architecture": "MaxPool1 2x2 s2 on Conv1 16x32x32 -> 16x16x16",
        "note": "Prompt sketches with 8 channels are obsolete; trained model uses 16.",
        "input_shape": [NUM_CHANNELS, 32, 32],
        "output_shape": [NUM_CHANNELS, POOL_H, POOL_W],
        "kernel": 2,
        "stride": 2,
        "operation": "signed_int8_max_of_four",
        "requantize": False,
        "relu": False,
        "flattening": "channel_first_then_row_major",
        "pool1_address": "channel * 256 + pool_row * 16 + pool_column",
        "conv1_address": "channel * 1024 + row * 32 + column",
        "output_count": EXPECTED_COUNT,
        "total_conv1_reads": EXPECTED_COUNT * 4,
        "min": int(flat.min()),
        "max": int(flat.max()),
        "num_zeros": zeros,
        "checksum_sum_int64": checksum,
        "channels": channel_meta,
        "source_input": str(sample_path),
    }
    (out / "pool1_expected.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "MaxPool1 expected tensor (signed INT8 max, no requant/ReLU)",
        f"shape: [{NUM_CHANNELS}, {POOL_H}, {POOL_W}]  entries: {EXPECTED_COUNT}",
        "address: ch*256 + prow*16 + pcol",
        f"min={meta['min']} max={meta['max']} zeros={zeros} checksum={checksum}",
        f"reads_per_output=4  total_reads={EXPECTED_COUNT * 4}",
        "",
        "Selected windows:",
    ]
    for name, ch, pr, pc in SELECTED_WINDOWS:
        addr = pool1_address(ch, pr, pc)
        lines.append(f"  {name}: ({ch},{pr},{pc}) addr={addr} value={int(tensor[ch, pr, pc])}")
    (out / "pool1_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    relu_chw = np.asarray(inter["relu1"][0], dtype=np.int8)
    for name, ch, pr, pc in SELECTED_WINDOWS:
        coords = window_coords(pr, pc)
        values = [int(relu_chw[ch, r, c]) for r, c in coords]
        addrs = [conv1_address(ch, r, c) for r, c in coords]
        maximum = max(values)  # Python int max is fine for signed values in range
        if maximum != int(tensor[ch, pr, pc]):
            raise AssertionError(f"Window max mismatch {name}")
        out_addr = pool1_address(ch, pr, pc)
        trace = {
            "name": name,
            "channel": ch,
            "pool_row": pr,
            "pool_column": pc,
            "window": [
                {
                    "window_index": i,
                    "input_row": coords[i][0],
                    "input_column": coords[i][1],
                    "conv1_address": addrs[i],
                    "value": values[i],
                }
                for i in range(4)
            ],
            "maximum": maximum,
            "pool1_address": out_addr,
            "pool1_value": int(tensor[ch, pr, pc]),
        }
        (traces_dir / f"{name}_metadata.json").write_text(
            json.dumps(trace, indent=2), encoding="utf-8"
        )
        text = [
            f"channel = {ch}",
            f"pool_row = {pr}",
            f"pool_column = {pc}",
            "",
        ]
        labels = ["A top-left", "B top-right", "C bottom-left", "D bottom-right"]
        for i, lab in enumerate(labels):
            text.extend(
                [
                    f"window {i} ({lab})",
                    f"  input_row = {coords[i][0]}",
                    f"  input_column = {coords[i][1]}",
                    f"  conv1_address = {addrs[i]}",
                    f"  value = {values[i]}",
                    "",
                ]
            )
        text.extend(
            [
                f"maximum = {maximum}",
                f"pool1_address = {out_addr}",
                f"pool1_value = {int(tensor[ch, pr, pc])}",
            ]
        )
        (traces_dir / f"{name}_trace.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
        write_mem_file(
            traces_dir / f"{name}_conv1_addrs.mem",
            np.asarray(addrs, dtype=np.int32),
            bits=32,
        )
        write_mem_file(
            traces_dir / f"{name}_values.mem",
            np.asarray(values, dtype=np.int8),
            bits=8,
        )
        write_mem_file(
            traces_dir / f"{name}_expected.mem",
            np.asarray([maximum], dtype=np.int8),
            bits=8,
        )
        write_mem_file(
            traces_dir / f"{name}_pool_addr.mem",
            np.asarray([out_addr], dtype=np.int32),
            bits=32,
        )

    print(f"Exported Pool1 vectors to {out}")
    print(
        f"  entries={EXPECTED_COUNT} channels={NUM_CHANNELS} "
        f"min={meta['min']} max={meta['max']} checksum={checksum}"
    )
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--int8-root", default="software/exported_model/int8")
    p.add_argument("--sample-input", default=None)
    p.add_argument("--output-dir", default="vectors/pool1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_pool1(
        int8_root=args.int8_root,
        sample_input=args.sample_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
