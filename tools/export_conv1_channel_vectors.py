#!/usr/bin/env python3
"""Export expected Conv1 output channel 0 (32x32) for RTL channel controller.

Trained model: Conv1 3->16. This phase only exports output channel 0
(1024 INT8 post-ReLU activations). Prompt sketches with 8 output channels
are obsolete for architecture width; only channel 0 is computed in RTL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from software.hardware.memory_layout import write_mem_file
from software.inference.integer_inference import IntegerTinyCNN
from software.quantization.saturation import INT8_MAX, INT8_MIN
from software.utils.config import resolve_path
from tools.export_conv1_memory_vectors import (
    enrich_trace_with_addresses,
    write_memory_trace,
)
from tools.export_rtl_vectors import _load_conv1_package, compute_conv1_output_trace


OUTPUT_CHANNEL = 0
SELECTED_PIXELS = [
    ("r0_c0", 0, 0),
    ("r0_c31", 0, 31),
    ("r10_c10", 10, 10),
    ("r15_c15", 15, 15),
    ("r31_c0", 31, 0),
    ("r31_c31", 31, 31),
]


def export_conv1_channel0(
    *,
    int8_root: str | Path = "software/exported_model/int8",
    sample_input: str | Path | None = None,
    output_dir: str | Path = "vectors/conv1_channel",
) -> dict[str, Any]:
    root = resolve_path(int8_root)
    pkg = _load_conv1_package(root)

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
    # Post-ReLU Conv1 activations (INT8), channel 0, shape 32x32
    channel = np.asarray(inter["relu1"][0, OUTPUT_CHANNEL], dtype=np.int8)
    if channel.shape != (32, 32):
        raise RuntimeError(f"Unexpected channel shape {channel.shape}")

    flat = channel.reshape(-1, order="C")  # row-major: addr = row*32 + col
    if flat.size != 1024:
        raise RuntimeError(f"Expected 1024 values, got {flat.size}")

    out = resolve_path(output_dir)
    traces_dir = out / "selected_pixel_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    write_mem_file(out / "conv1_channel0_expected.mem", flat, bits=8)

    zeros = int(np.sum(flat == 0))
    sat_hi = int(np.sum(flat == INT8_MAX))
    sat_lo = int(np.sum(flat == INT8_MIN))
    checksum = int(np.sum(flat.astype(np.int64)))
    meta = {
        "output_channel": OUTPUT_CHANNEL,
        "shape": [1, 32, 32],
        "flattening": "row_major_within_channel",
        "address_equation": "output_row * 32 + output_column",
        "future_full_conv1_address": "output_channel * 1024 + output_row * 32 + output_column",
        "note": "Trained Conv1 has 16 output channels; this export is channel 0 only.",
        "output_count": 1024,
        "dtype": "int8",
        "min": int(flat.min()),
        "max": int(flat.max()),
        "num_zeros": zeros,
        "num_saturated_pos127": sat_hi,
        "num_saturated_neg128": sat_lo,
        "num_saturated": sat_hi + sat_lo,
        "checksum_sum_int64": checksum,
        "source_input": str(sample_path),
        "macs_per_output": 27,
        "input_channels_combined": 3,
    }
    (out / "conv1_channel0_expected.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    summary_lines = [
        "Conv1 output channel 0 expected feature map",
        f"shape: [1, 32, 32]  entries: 1024",
        f"address: row*32 + col  (row-major)",
        f"min={meta['min']} max={meta['max']}",
        f"zeros={zeros} saturated={meta['num_saturated']}",
        f"checksum_sum={checksum}",
        f"macs_per_output=27 (3 input channels x 3x3)",
        "",
        "Selected pixels:",
    ]
    for name, row, col in SELECTED_PIXELS:
        addr = row * 32 + col
        summary_lines.append(
            f"  {name}: ({row},{col}) addr={addr} value={int(channel[row, col])}"
        )
    (out / "conv1_channel0_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    for name, row, col in SELECTED_PIXELS:
        trace = compute_conv1_output_trace(
            input_q,
            pkg["weights"],
            pkg["bias"],
            pkg["multipliers"],
            pkg["shifts"],
            out_channel=OUTPUT_CHANNEL,
            out_row=row,
            out_col=col,
        )
        trace = enrich_trace_with_addresses(trace)
        if int(trace["result_after_relu"]) != int(channel[row, col]):
            raise AssertionError(
                f"Trace relu mismatch at ({row},{col}): "
                f"{trace['result_after_relu']} vs {int(channel[row, col])}"
            )
        write_memory_trace(traces_dir / f"{name}_trace.txt", trace)
        (traces_dir / f"{name}_metadata.json").write_text(
            json.dumps(trace, indent=2), encoding="utf-8"
        )
        write_mem_file(
            traces_dir / f"{name}_act_addr.mem",
            np.asarray(
                [-1 if a is None else int(a) for a in trace["activation_addresses"]],
                dtype=np.int32,
            ),
            bits=32,
        )
        write_mem_file(
            traces_dir / f"{name}_wgt_addr.mem",
            np.asarray(trace["weight_addresses"], dtype=np.int32),
            bits=32,
        )
        write_mem_file(
            traces_dir / f"{name}_padding.mem",
            np.asarray([1 if m["padding"] else 0 for m in trace["macs"]], dtype=np.int8),
            bits=8,
        )
        write_mem_file(
            traces_dir / f"{name}_products.mem",
            np.asarray(trace["products"], dtype=np.int16),
            bits=16,
        )
        write_mem_file(
            traces_dir / f"{name}_acc_trace.mem",
            np.asarray(trace["acc_trace"], dtype=np.int32),
            bits=32,
        )
        write_mem_file(
            traces_dir / f"{name}_expected_acc.mem",
            np.asarray([trace["final_accumulator"]], dtype=np.int32),
            bits=32,
        )
        write_mem_file(
            traces_dir / f"{name}_expected_requantized.mem",
            np.asarray([trace["result_after_saturation"]], dtype=np.int8),
            bits=8,
        )
        write_mem_file(
            traces_dir / f"{name}_expected_relu.mem",
            np.asarray([trace["result_after_relu"]], dtype=np.int8),
            bits=8,
        )

    print(f"Exported Conv1 channel0 to {out}")
    print(
        f"  entries=1024 min={meta['min']} max={meta['max']} "
        f"checksum={checksum} zeros={zeros}"
    )
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--int8-root", default="software/exported_model/int8")
    p.add_argument("--sample-input", default=None)
    p.add_argument("--output-dir", default="vectors/conv1_channel")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_conv1_channel0(
        int8_root=args.int8_root,
        sample_input=args.sample_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
