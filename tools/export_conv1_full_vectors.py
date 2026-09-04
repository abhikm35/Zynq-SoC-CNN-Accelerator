#!/usr/bin/env python3
"""Export the complete Conv1 output tensor for the full-layer RTL controller.

Trained / quantized model: Conv1 3->16, so the full tensor is 16 x 32 x 32 =
16384 INT8 post-ReLU activations. Prompt sketches that mention 8 output
channels (8192) are obsolete; this export matches the trained model.
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


# Authoritative trained width (not the obsolete 8-channel sketch).
NUM_OUT_CHANNELS = 16
EXPECTED_COUNT = NUM_OUT_CHANNELS * 32 * 32  # 16384

SELECTED_OUTPUTS = [
    ("ch0_r0_c0", 0, 0, 0),
    ("ch0_r10_c10", 0, 10, 10),
    ("ch1_r0_c0", 1, 0, 0),
    ("ch3_r15_c15", 3, 15, 15),
    ("ch5_r12_c17", 5, 12, 17),
    ("ch7_r31_c31", 7, 31, 31),
    ("ch15_r31_c31", 15, 31, 31),
]


def export_conv1_full(
    *,
    int8_root: str | Path = "software/exported_model/int8",
    sample_input: str | Path | None = None,
    output_dir: str | Path = "vectors/conv1_full",
) -> dict[str, Any]:
    root = resolve_path(int8_root)
    pkg = _load_conv1_package(root)
    weights = pkg["weights"]
    if weights.shape[0] != NUM_OUT_CHANNELS:
        raise RuntimeError(
            f"Expected {NUM_OUT_CHANNELS} Conv1 out channels, got {weights.shape}"
        )

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
    relu = np.asarray(inter["relu1"][0], dtype=np.int8)  # [16,32,32]
    if relu.shape != (NUM_OUT_CHANNELS, 32, 32):
        raise RuntimeError(f"Unexpected relu1 shape {relu.shape}")

    # Channel-first flatten: oc -> row -> col
    flat = relu.reshape(-1, order="C")
    if flat.size != EXPECTED_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_COUNT} values, got {flat.size}")

    out = resolve_path(output_dir)
    ch_dir = out / "channel_summaries"
    traces_dir = out / "selected_output_traces"
    ch_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    write_mem_file(out / "conv1_expected.mem", flat, bits=8)

    zeros = int(np.sum(flat == 0))
    sat_hi = int(np.sum(flat == INT8_MAX))
    sat_lo = int(np.sum(flat == INT8_MIN))
    checksum = int(np.sum(flat.astype(np.int64)))

    channel_meta: list[dict[str, Any]] = []
    for oc in range(NUM_OUT_CHANNELS):
        ch = relu[oc]
        ch_flat = ch.reshape(-1, order="C")
        ch_sum = int(np.sum(ch_flat.astype(np.int64)))
        ch_zeros = int(np.sum(ch_flat == 0))
        info = {
            "output_channel": oc,
            "count": 1024,
            "min": int(ch_flat.min()),
            "max": int(ch_flat.max()),
            "num_zeros": ch_zeros,
            "num_saturated": int(
                np.sum(ch_flat == INT8_MAX) + np.sum(ch_flat == INT8_MIN)
            ),
            "checksum_sum_int64": ch_sum,
            "address_base": oc * 1024,
            "address_end": oc * 1024 + 1023,
        }
        channel_meta.append(info)
        (ch_dir / f"channel_{oc:02d}.json").write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        )

    meta = {
        "architecture": "Conv1 3->16 (trained model)",
        "note": (
            "Prompt sketches with 8 output channels / 8192 outputs are obsolete. "
            "This export is the complete Conv1 tensor: 16 x 32 x 32 = 16384."
        ),
        "shape": [NUM_OUT_CHANNELS, 32, 32],
        "flattening": "channel_first_then_row_major",
        "address_equation": "output_channel * 1024 + output_row * 32 + output_column",
        "output_count": EXPECTED_COUNT,
        "macs_per_output": 27,
        "total_macs": EXPECTED_COUNT * 27,
        "input_channels_combined": 3,
        "quantization": "per_output_channel bias, multiplier, shift; zp=0",
        "dtype": "int8",
        "min": int(flat.min()),
        "max": int(flat.max()),
        "num_zeros": zeros,
        "num_saturated_pos127": sat_hi,
        "num_saturated_neg128": sat_lo,
        "num_saturated": sat_hi + sat_lo,
        "checksum_sum_int64": checksum,
        "channels": channel_meta,
        "source_input": str(sample_path),
    }
    (out / "conv1_expected.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "Complete Conv1 output tensor (post-ReLU INT8)",
        f"shape: [{NUM_OUT_CHANNELS}, 32, 32]  entries: {EXPECTED_COUNT}",
        "address: oc*1024 + row*32 + col  (channel-first)",
        f"min={meta['min']} max={meta['max']} zeros={zeros} saturated={meta['num_saturated']}",
        f"checksum_sum={checksum}",
        f"macs_per_output=27  total_macs={EXPECTED_COUNT * 27}",
        "",
        "Per-channel checksums:",
    ]
    for info in channel_meta:
        lines.append(
            f"  ch{info['output_channel']:02d}: sum={info['checksum_sum_int64']} "
            f"min={info['min']} max={info['max']} zeros={info['num_zeros']}"
        )
    lines.append("")
    lines.append("Selected outputs:")
    for name, oc, row, col in SELECTED_OUTPUTS:
        addr = oc * 1024 + row * 32 + col
        lines.append(
            f"  {name}: ({oc},{row},{col}) addr={addr} value={int(relu[oc, row, col])}"
        )
    (out / "conv1_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    biases = np.asarray(pkg["bias"], dtype=np.int32).reshape(-1)
    mults = np.asarray(pkg["multipliers"], dtype=np.int32).reshape(-1)
    shifts = np.asarray(pkg["shifts"], dtype=np.int32).reshape(-1)

    for name, oc, row, col in SELECTED_OUTPUTS:
        trace = compute_conv1_output_trace(
            input_q,
            pkg["weights"],
            pkg["bias"],
            pkg["multipliers"],
            pkg["shifts"],
            out_channel=oc,
            out_row=row,
            out_col=col,
        )
        trace = enrich_trace_with_addresses(trace)
        if int(trace["result_after_relu"]) != int(relu[oc, row, col]):
            raise AssertionError(f"Trace mismatch {name}")
        trace["output_ram_address"] = oc * 1024 + row * 32 + col
        trace["quant_parameter_address"] = oc
        trace["bias_address"] = oc
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
            traces_dir / f"{name}_inputs.mem",
            np.asarray(trace["inputs"], dtype=np.int8),
            bits=8,
        )
        write_mem_file(
            traces_dir / f"{name}_weights.mem",
            np.asarray(trace["weights"], dtype=np.int8),
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
        write_mem_file(
            traces_dir / f"{name}_bias.mem",
            np.asarray([int(biases[oc])], dtype=np.int32),
            bits=32,
        )
        write_mem_file(
            traces_dir / f"{name}_multiplier.mem",
            np.asarray([int(mults[oc])], dtype=np.int32),
            bits=32,
        )
        write_mem_file(
            traces_dir / f"{name}_shift.mem",
            np.asarray([int(shifts[oc])], dtype=np.int32),
            bits=32,
        )

    print(f"Exported complete Conv1 tensor to {out}")
    print(
        f"  entries={EXPECTED_COUNT} channels={NUM_OUT_CHANNELS} "
        f"min={meta['min']} max={meta['max']} checksum={checksum}"
    )
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--int8-root", default="software/exported_model/int8")
    p.add_argument("--sample-input", default=None)
    p.add_argument("--output-dir", default="vectors/conv1_full")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_conv1_full(
        int8_root=args.int8_root,
        sample_input=args.sample_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
