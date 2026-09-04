#!/usr/bin/env python3
"""Export expected MaxPool2 tensor for RTL.

Trained model: Conv2 is 32 x 16 x 16, so Pool2 is 32 x 8 x 8 = 2048.
Prompt sketches with 16 channels / 1024 outputs are obsolete.
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


NUM_CHANNELS = 32
IN_H = 16
IN_W = 16
POOL_H = 8
POOL_W = 8
EXPECTED_COUNT = NUM_CHANNELS * POOL_H * POOL_W  # 2048

SELECTED_WINDOWS = [
    ("ch0_pr0_pc0", 0, 0, 0),
    ("ch0_pr7_pc7", 0, 7, 7),
    ("ch3_pr4_pc5", 3, 4, 5),
    ("ch8_pr2_pc6", 8, 2, 6),
    ("ch15_pr7_pc7", 15, 7, 7),
    ("ch31_pr7_pc7", 31, 7, 7),
]


def pool2_address(channel: int, pool_row: int, pool_col: int) -> int:
    return channel * 64 + pool_row * 8 + pool_col


def conv2_address(channel: int, row: int, col: int) -> int:
    return channel * 256 + row * 16 + col


def window_coords(pool_row: int, pool_col: int) -> list[tuple[int, int]]:
    br, bc = pool_row * 2, pool_col * 2
    return [
        (br, bc),
        (br, bc + 1),
        (br + 1, bc),
        (br + 1, bc + 1),
    ]


def export_pool2(
    *,
    int8_root: str | Path = "software/exported_model/int8",
    sample_input: str | Path | None = None,
    output_dir: str | Path = "vectors/pool2",
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
    relu2 = np.asarray(inter["relu2"], dtype=np.int8)
    pool2 = np.asarray(inter["pool2"], dtype=np.int8)
    pool_check = integer_max_pool2d_nchw(relu2, kernel_size=2, stride=2).astype(np.int8)
    if not np.array_equal(pool2, pool_check):
        raise AssertionError("pool2 mismatch vs integer_max_pool2d_nchw")
    if pool2.shape != (1, NUM_CHANNELS, POOL_H, POOL_W):
        raise RuntimeError(f"Unexpected pool2 shape {pool2.shape}")
    if relu2.shape != (1, NUM_CHANNELS, IN_H, IN_W):
        raise RuntimeError(f"Unexpected relu2 shape {relu2.shape}")

    tensor = pool2[0]
    flat = tensor.reshape(-1, order="C")
    if flat.size != EXPECTED_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_COUNT}, got {flat.size}")

    out = resolve_path(output_dir)
    ch_dir = out / "channel_summaries"
    traces_dir = out / "selected_window_traces"
    ch_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    write_mem_file(out / "pool2_expected.mem", flat, bits=8)
    write_mem_file(
        out / "conv2_input.mem",
        np.asarray(relu2[0], dtype=np.int8).reshape(-1, order="C"),
        bits=8,
    )

    zeros = int(np.sum(flat == 0))
    checksum = int(np.sum(flat.astype(np.int64)))
    channel_meta: list[dict[str, Any]] = []
    for ch in range(NUM_CHANNELS):
        ch_flat = tensor[ch].reshape(-1, order="C")
        info = {
            "channel": ch,
            "count": 64,
            "min": int(ch_flat.min()),
            "max": int(ch_flat.max()),
            "num_zeros": int(np.sum(ch_flat == 0)),
            "checksum_sum_int64": int(np.sum(ch_flat.astype(np.int64))),
            "address_base": ch * 64,
            "address_end": ch * 64 + 63,
        }
        channel_meta.append(info)
        (ch_dir / f"channel_{ch:02d}.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        )

    meta = {
        "architecture": "MaxPool2 2x2 s2 on Conv2 32x16x16 -> 32x8x8",
        "note": "Prompt sketches with 16 channels are obsolete; trained model uses 32.",
        "input_shape": [NUM_CHANNELS, IN_H, IN_W],
        "output_shape": [NUM_CHANNELS, POOL_H, POOL_W],
        "kernel": 2,
        "stride": 2,
        "operation": "signed_int8_max_of_four",
        "requantize": False,
        "relu": False,
        "flattening": "channel_first_then_row_major",
        "pool2_address": "channel * 64 + pool_row * 8 + pool_column",
        "conv2_address": "channel * 256 + row * 16 + column",
        "output_count": EXPECTED_COUNT,
        "total_conv2_reads": EXPECTED_COUNT * 4,
        "min": int(flat.min()),
        "max": int(flat.max()),
        "num_zeros": zeros,
        "checksum_sum_int64": checksum,
        "channels": channel_meta,
        "source_input": str(sample_path),
    }
    (out / "pool2_expected.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "MaxPool2 expected tensor (signed INT8 max, no requant/ReLU)",
        f"shape: [{NUM_CHANNELS}, {POOL_H}, {POOL_W}]  entries: {EXPECTED_COUNT}",
        "address: ch*64 + prow*8 + pcol",
        f"min={meta['min']} max={meta['max']} zeros={zeros} checksum={checksum}",
        f"reads_per_output=4  total_reads={EXPECTED_COUNT * 4}",
        "",
        "Selected windows:",
    ]
    for name, ch, pr, pc in SELECTED_WINDOWS:
        addr = pool2_address(ch, pr, pc)
        lines.append(f"  {name}: ({ch},{pr},{pc}) addr={addr} value={int(tensor[ch, pr, pc])}")
    (out / "pool2_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    relu_chw = np.asarray(relu2[0], dtype=np.int8)
    for name, ch, pr, pc in SELECTED_WINDOWS:
        coords = window_coords(pr, pc)
        values = [int(relu_chw[ch, r, c]) for r, c in coords]
        addrs = [conv2_address(ch, r, c) for r, c in coords]
        maximum = max(values)
        if maximum != int(tensor[ch, pr, pc]):
            raise AssertionError(f"Window max mismatch {name}")
        out_addr = pool2_address(ch, pr, pc)
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
                    "conv2_address": addrs[i],
                    "value": values[i],
                }
                for i in range(4)
            ],
            "maximum": maximum,
            "pool2_address": out_addr,
            "pool2_value": int(tensor[ch, pr, pc]),
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
                    f"  conv2_address = {addrs[i]}",
                    f"  value = {values[i]}",
                    "",
                ]
            )
        text.extend(
            [
                f"maximum = {maximum}",
                f"pool2_address = {out_addr}",
                f"pool2_value = {int(tensor[ch, pr, pc])}",
            ]
        )
        (traces_dir / f"{name}_trace.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
        write_mem_file(
            traces_dir / f"{name}_conv2_addrs.mem",
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

    print(f"Exported Pool2 vectors to {out}")
    print(
        f"  entries={EXPECTED_COUNT} channels={NUM_CHANNELS} "
        f"min={meta['min']} max={meta['max']} checksum={checksum}"
    )
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--int8-root", default="software/exported_model/int8")
    p.add_argument("--sample-input", default=None)
    p.add_argument("--output-dir", default="vectors/pool2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_pool2(
        int8_root=args.int8_root,
        sample_input=args.sample_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
