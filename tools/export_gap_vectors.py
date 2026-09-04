#!/usr/bin/env python3
"""Export expected Global Average Pooling vectors for RTL.

Trained model: Pool2 is 32 x 8 x 8 = 2048, GAP is 32 INT8 values.

Python path (integer_inference.py):
  1) sum 64 INT8 activations per channel (zero_point == 0)
  2) averaged = round_divide_int(sum, 64)  # ties away from zero
  3) saturate to INT8 (still pool2 scale)
  4) requantize_int32(avg, gap_multiplier, gap_shift) -> INT8 flatten scale

Exported gap_expected.mem matches intermediate ``global_average_pool`` (post-requant).
Prompt sketches with 16 channels / 1024 Pool2 values are obsolete.
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
    integer_global_average_pool_nchw,
)
from software.quantization.fixed_point import requantize_int32, round_divide_int
from software.utils.config import resolve_path


NUM_CHANNELS = 32
POOL_H = 8
POOL_W = 8
ELEMS_PER_CH = POOL_H * POOL_W  # 64
POOL2_COUNT = NUM_CHANNELS * ELEMS_PER_CH  # 2048
GAP_COUNT = NUM_CHANNELS  # 32

SELECTED_CHANNELS = [0, 1, 5, 10, 15, 31]


def pool2_address(channel: int, element_index: int) -> int:
    return channel * ELEMS_PER_CH + element_index


def gap_address(channel: int) -> int:
    return channel


def export_gap(
    *,
    int8_root: str | Path = "software/exported_model/int8",
    sample_input: str | Path | None = None,
    output_dir: str | Path = "vectors/gap",
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
    pool2 = np.asarray(inter["pool2"], dtype=np.int8)
    gap_q = np.asarray(inter["global_average_pool"], dtype=np.int8)
    flatten = np.asarray(inter["flatten"], dtype=np.int8)

    if pool2.shape != (1, NUM_CHANNELS, POOL_H, POOL_W):
        raise RuntimeError(f"Unexpected pool2 shape {pool2.shape}")
    if gap_q.shape != (1, NUM_CHANNELS, 1, 1):
        raise RuntimeError(f"Unexpected gap shape {gap_q.shape}")
    if not np.array_equal(flatten.reshape(-1), gap_q.reshape(-1)):
        raise AssertionError("flatten must equal GAP post-requant")

    gap_mult = int(np.asarray(model.requant["gap_multiplier"]).reshape(-1)[0])
    gap_shift = int(np.asarray(model.requant["gap_shift"]).reshape(-1)[0])

    raw_avg = integer_global_average_pool_nchw(pool2, zero_point=0).astype(np.int8)
    # Re-check requant path
    flat_avg = raw_avg.reshape(1, NUM_CHANNELS).astype(np.int32)
    mult = np.full((NUM_CHANNELS,), gap_mult, dtype=np.int32)
    sh = np.full((NUM_CHANNELS,), gap_shift, dtype=np.int32)
    gap_check = requantize_int32(
        flat_avg, mult, sh, output_zero_point=0, qmin=-128, qmax=127
    ).astype(np.int8)
    if not np.array_equal(gap_check, gap_q.reshape(1, NUM_CHANNELS)):
        raise AssertionError("GAP requant mismatch vs golden intermediates")

    tensor = pool2[0]
    pool2_flat = tensor.reshape(-1, order="C")
    gap_flat = gap_q.reshape(-1, order="C")
    raw_flat = raw_avg.reshape(-1, order="C")

    out = resolve_path(output_dir)
    traces_dir = out / "channel_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    write_mem_file(out / "pool2_input.mem", pool2_flat, bits=8)
    write_mem_file(out / "gap_expected.mem", gap_flat, bits=8)
    write_mem_file(out / "gap_raw_avg_expected.mem", raw_flat, bits=8)
    write_mem_file(
        out / "gap_multiplier.mem",
        np.asarray([gap_mult], dtype=np.int32),
        bits=32,
    )
    write_mem_file(
        out / "gap_shift.mem",
        np.asarray([gap_shift], dtype=np.int32),
        bits=32,
    )

    # Arithmetic unit-test cases for divide-by-64 (ties away from zero).
    arith_sums = [
        0,
        1,
        31,
        32,
        33,
        63,
        64,
        65,
        -1,
        -31,
        -32,
        -33,
        -63,
        -64,
        -65,
        8128,
        -8192,
    ]
    arith_rows: list[dict[str, Any]] = []
    arith_mem_lines: list[str] = []
    for s in arith_sums:
        avg = int(round_divide_int(np.int64(s), 64))
        sat = int(np.clip(avg, -128, 127))
        # Post-requant of the saturated average (matches full GAP stage).
        gap_out = int(
            requantize_int32(
                np.asarray([sat], dtype=np.int32),
                np.asarray([gap_mult], dtype=np.int32),
                np.asarray([gap_shift], dtype=np.int32),
                output_zero_point=0,
                qmin=-128,
                qmax=127,
            )[0]
        )
        half = 32
        if s >= 0:
            adj = s + half
            shifted = adj >> 6
        else:
            adj = -((-s) + half)
            # magnitude path: -(((-s)+half)>>6)
            shifted = -(((-s) + half) >> 6)
        arith_rows.append(
            {
                "sum": s,
                "rounding_adjustment": half if s >= 0 else -half,
                "adjusted_magnitude_path": int(adj) if s >= 0 else int(-((-s) + half)),
                "averaged_before_sat": avg,
                "averaged_saturated": sat,
                "gap_requantized": gap_out,
            }
        )
        # mem line: sum (8 hex) then sat avg (2 hex) then gap_q (2 hex)
        arith_mem_lines.append(
            f"{(s & 0xFFFFFFFF):08x} {(sat & 0xFF):02x} {(gap_out & 0xFF):02x}"
        )

    (out / "gap_average_cases.json").write_text(
        json.dumps({"denominator": 64, "cases": arith_rows}, indent=2) + "\n"
    )
    (out / "gap_average_cases.mem").write_text("\n".join(arith_mem_lines) + "\n")

    channel_meta: list[dict[str, Any]] = []
    json_traces: dict[str, Any] = {}
    for ch in range(NUM_CHANNELS):
        ch_vals = tensor[ch].reshape(-1, order="C").astype(np.int64)
        running = 0
        steps: list[dict[str, Any]] = []
        lines = [
            f"channel: {ch}",
            f"addresses: {ch * 64} .. {ch * 64 + 63}",
            "",
        ]
        for elem in range(ELEMS_PER_CH):
            addr = pool2_address(ch, elem)
            val = int(ch_vals[elem])
            running += val
            steps.append(
                {
                    "element_index": elem,
                    "pool2_address": addr,
                    "value": val,
                    "running_sum": int(running),
                }
            )
            lines.append(
                f"elem={elem:2d} addr={addr:4d} value={val:4d} running_sum={running}"
            )

        final_sum = int(running)
        averaged = int(round_divide_int(np.int64(final_sum), 64))
        sat = int(np.clip(averaged, -128, 127))
        half = 32
        if final_sum >= 0:
            adj = final_sum + half
            after_div = adj >> 6
        else:
            after_div = -(((-final_sum) + half) >> 6)
            adj = -((-final_sum) + half)

        gap_out = int(gap_flat[ch])
        meta = {
            "channel": ch,
            "final_sum": final_sum,
            "denominator": 64,
            "rounding_adjustment": 32 if final_sum >= 0 else -32,
            "adjusted_sum_debug": int(adj),
            "value_after_division": int(after_div),
            "averaged_before_saturation": averaged,
            "averaged_after_saturation": sat,
            "gap_multiplier": gap_mult,
            "gap_shift": gap_shift,
            "gap_output": gap_out,
            "gap_address": gap_address(ch),
            "raw_avg_from_model": int(raw_flat[ch]),
        }
        channel_meta.append(meta)
        lines.extend(
            [
                "",
                f"final_sum: {final_sum}",
                f"division: round_divide_int(sum, 64) ties-away-from-zero",
                f"rounding_half: 32",
                f"value_after_division: {after_div}",
                f"before_saturation: {averaged}",
                f"after_saturation (pool2 scale): {sat}",
                f"requant multiplier: {gap_mult}",
                f"requant shift: {gap_shift}",
                f"final GAP output (flatten scale INT8): {gap_out}",
                f"gap_address: {ch}",
            ]
        )
        (traces_dir / f"channel_{ch:02d}.txt").write_text("\n".join(lines) + "\n")
        json_traces[f"channel_{ch:02d}"] = {"meta": meta, "steps": steps}

    (traces_dir / "all_channels.json").write_text(
        json.dumps(json_traces, indent=2) + "\n"
    )

    # Selected channel compact JSON for TB
    selected = {str(ch): json_traces[f"channel_{ch:02d}"] for ch in SELECTED_CHANNELS}
    (out / "selected_channels.json").write_text(json.dumps(selected, indent=2) + "\n")

    summary = {
        "operation": "global_average_pool",
        "note": (
            "Trained model uses 32 Pool2 channels (32x8x8). "
            "GAP output is post-requant INT8 matching Python global_average_pool."
        ),
        "input_shape": [NUM_CHANNELS, POOL_H, POOL_W],
        "input_datatype": "int8",
        "input_scale": "pool2 / conv2 scale",
        "input_zero_point": 0,
        "accumulator_datatype": "int32",
        "accumulator_width_bits": 32,
        "math_min_accumulator_bits": 14,
        "division": "round_divide_int(sum, 64)",
        "rounding": "ties away from zero (rounding_right_shift by 6)",
        "average_output_datatype": "int8",
        "average_saturation": "[-128, 127]",
        "requantize": True,
        "gap_multiplier": gap_mult,
        "gap_shift": gap_shift,
        "output_datatype": "int8",
        "output_zero_point": 0,
        "output_count": GAP_COUNT,
        "total_pool2_reads": POOL2_COUNT,
        "pool2_address": "channel * 64 + element_index",
        "gap_address": "channel",
        "flattening": "channel -> row -> column",
        "checksum_sum_int64": int(np.sum(gap_flat.astype(np.int64))),
        "raw_avg_checksum_sum_int64": int(np.sum(raw_flat.astype(np.int64))),
        "min": int(gap_flat.min()),
        "max": int(gap_flat.max()),
        "num_zeros": int(np.sum(gap_flat == 0)),
        "channels": channel_meta,
        "selected_channels": SELECTED_CHANNELS,
        "source_input": str(sample_path),
    }
    (out / "gap_expected.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "GAP vector export summary",
        f"Pool2 input: {NUM_CHANNELS} x {POOL_H} x {POOL_W} = {POOL2_COUNT}",
        f"GAP outputs: {GAP_COUNT} (post-requant INT8)",
        f"gap_multiplier={gap_mult} gap_shift={gap_shift}",
        f"checksum={summary['checksum_sum_int64']} min={summary['min']} max={summary['max']}",
        "",
        "channel  sum  raw_avg  gap_q",
    ]
    for m in channel_meta:
        lines.append(
            f"{m['channel']:7d} {m['final_sum']:4d} {m['averaged_after_saturation']:7d} "
            f"{m['gap_output']:5d}"
        )
    (out / "gap_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote GAP vectors under {out}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--int8-root", default="software/exported_model/int8")
    p.add_argument("--sample-input", default=None)
    p.add_argument("--output-dir", default="vectors/gap")
    args = p.parse_args()
    export_gap(
        int8_root=args.int8_root,
        sample_input=args.sample_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
