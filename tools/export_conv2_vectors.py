#!/usr/bin/env python3
"""Export complete Conv2 memories, expected tensor, and selected-output traces.

Trained model: Pool1 16x16x16 -> Conv2 16->32, k=3,s=1,p=1 -> 32x16x16.
Prompt sketches with 8->16 channels / 72 MACs are obsolete.
MACs per output: 16*3*3 = 144. Total outputs: 32*16*16 = 8192.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from software.hardware.memory_layout import (
    activation_address,
    flatten_activation_nchw,
    flatten_weights_oihw,
    weight_address,
    write_mem_file,
)
from software.inference.integer_inference import IntegerTinyCNN
from software.utils.config import resolve_path
from tools.export_rtl_vectors import compute_conv1_output_trace


NUM_IN = 16
NUM_OUT = 32
SPATIAL = 16
MACS_PER = NUM_IN * 9  # 144
EXPECTED = NUM_OUT * SPATIAL * SPATIAL  # 8192
WGT_COUNT = NUM_OUT * MACS_PER  # 4608

SELECTED_CASES = [
    ("ch0_r0_c0", 0, 0, 0),
    ("ch0_r7_c7", 0, 7, 7),
    ("ch1_r0_c0", 1, 0, 0),
    ("ch5_r7_c10", 5, 7, 10),
    ("ch9_r12_c4", 9, 12, 4),
    ("ch15_r15_c15", 15, 15, 15),
    ("ch31_r15_c15", 31, 15, 15),
]

ADDRESS_CASES = SELECTED_CASES + [
    ("ch0_r0_c15", 0, 0, 15),
    ("ch0_r15_c0", 0, 15, 0),
]


def _load_conv2_package(int8_root: Path) -> dict[str, Any]:
    return {
        "weights": np.load(int8_root / "weights" / "conv2_weights_int8.npy"),
        "bias": np.load(int8_root / "biases" / "conv2_bias_int32.npy"),
        "multipliers": np.load(int8_root / "requantization" / "conv2_multipliers_int32.npy"),
        "shifts": np.load(int8_root / "requantization" / "conv2_shifts_int32.npy"),
        "input_scale": float(np.load(int8_root / "scales" / "conv1_output_scale_float32.npy")),
        "weight_scales": np.load(int8_root / "scales" / "conv2_weight_scales_float32.npy"),
        "output_scale": float(np.load(int8_root / "scales" / "conv2_output_scale_float32.npy")),
    }


def enrich_trace(trace: dict[str, Any]) -> dict[str, Any]:
    oc = int(trace["out_channel"])
    act_addrs: list[int | None] = []
    wgt_addrs: list[int] = []
    for mac in trace["macs"]:
        ic = int(mac["input_channel"])
        kr = int(mac["kernel_row"])
        kc = int(mac["kernel_column"])
        waddr = weight_address(
            oc, ic, kr, kc,
            input_channels=NUM_IN, kernel_height=3, kernel_width=3,
        )
        mac["weight_address"] = waddr
        mac["bias_address"] = oc
        if mac["padding"]:
            mac["activation_address"] = None
            act_addrs.append(None)
        else:
            aaddr = activation_address(
                ic, int(mac["input_row"]), int(mac["input_column"]),
                height=SPATIAL, width=SPATIAL,
            )
            mac["activation_address"] = aaddr
            act_addrs.append(aaddr)
        wgt_addrs.append(waddr)
    trace["activation_addresses"] = act_addrs
    trace["weight_addresses"] = wgt_addrs
    trace["bias_address"] = oc
    trace["output_address"] = oc * 256 + int(trace["out_row"]) * 16 + int(trace["out_col"])
    return trace


def write_trace_txt(path: Path, trace: dict[str, Any]) -> None:
    lines = [
        f"output_channel = {trace['out_channel']}",
        f"output_row = {trace['out_row']}",
        f"output_column = {trace['out_col']}",
        f"bias_address = {trace['bias_address']}",
        f"bias_value = {trace['bias']}",
        f"multiplier = {trace['multiplier']}",
        f"shift = {trace['shift']}",
        f"output_address = {trace['output_address']}",
        "",
    ]
    for mac in trace["macs"]:
        aaddr = mac["activation_address"]
        lines.extend(
            [
                f"MAC {mac['mac_index']}",
                f"input_channel = {mac['input_channel']}",
                f"kernel_row = {mac['kernel_row']}",
                f"kernel_column = {mac['kernel_column']}",
                f"calculated_input_row = {mac['input_row']}",
                f"calculated_input_column = {mac['input_column']}",
                f"padding = {str(mac['padding']).lower()}",
                f"pool1_activation_address = {'invalid' if aaddr is None else aaddr}",
                f"pool1_activation_value = {mac['input']}",
                f"conv2_weight_address = {mac['weight_address']}",
                f"conv2_weight_value = {mac['weight']}",
                f"product = {mac['product']}",
                f"running_accumulator = {mac['accumulator']}",
                "",
            ]
        )
    lines.extend(
        [
            f"final_accumulator = {trace['final_accumulator']}",
            f"requantized = {trace['result_after_saturation']}",
            f"relu = {trace['result_after_relu']}",
            f"conv2_output_address = {trace['output_address']}",
            f"conv2_output_value = {trace['result_after_relu']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_conv2(
    *,
    int8_root: str | Path = "software/exported_model/int8",
    sample_input: str | Path | None = None,
    output_dir: str | Path = "vectors/conv2",
) -> dict[str, Any]:
    root = resolve_path(int8_root)
    pkg = _load_conv2_package(root)
    weights = pkg["weights"]
    if weights.shape != (NUM_OUT, NUM_IN, 3, 3):
        raise RuntimeError(f"Unexpected Conv2 weights {weights.shape}")

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
    pool1 = np.asarray(inter["pool1"], dtype=np.int8)
    golden_acc = inter["conv2_accumulator"]
    golden_conv = inter["conv2"]
    golden_relu = inter["relu2"]
    if pool1.shape != (1, NUM_IN, SPATIAL, SPATIAL):
        raise RuntimeError(f"Unexpected pool1 shape {pool1.shape}")
    if golden_relu.shape != (1, NUM_OUT, SPATIAL, SPATIAL):
        raise RuntimeError(f"Unexpected relu2 shape {golden_relu.shape}")

    out = resolve_path(output_dir)
    traces_dir = out / "selected_output_traces"
    ch_dir = out / "channel_summaries"
    traces_dir.mkdir(parents=True, exist_ok=True)
    ch_dir.mkdir(parents=True, exist_ok=True)

    write_mem_file(out / "pool1_input.mem", flatten_activation_nchw(pool1), bits=8)
    write_mem_file(out / "conv2_weights.mem", flatten_weights_oihw(weights), bits=8)
    write_mem_file(out / "conv2_biases.mem", np.asarray(pkg["bias"], dtype=np.int32), bits=32)
    write_mem_file(
        out / "conv2_multipliers.mem",
        np.asarray(pkg["multipliers"], dtype=np.int32),
        bits=32,
    )
    write_mem_file(
        out / "conv2_shifts.mem",
        np.asarray(pkg["shifts"], dtype=np.int32),
        bits=32,
    )

    tensor = np.asarray(golden_relu[0], dtype=np.int8)
    flat = tensor.reshape(-1, order="C")
    if flat.size != EXPECTED:
        raise RuntimeError(f"Expected {EXPECTED} outputs, got {flat.size}")
    write_mem_file(out / "conv2_expected.mem", flat, bits=8)

    zeros = int(np.sum(flat == 0))
    sats = int(np.sum((flat == 127) | (flat == -128)))
    checksum = int(np.sum(flat.astype(np.int64)))
    channel_meta: list[dict[str, Any]] = []
    for ch in range(NUM_OUT):
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

    quant = {
        "architecture": "Conv2 16->32, k=3, s=1, p=1 (trained model)",
        "note": "Prompt sketches with 8->16 are obsolete; use 16->32.",
        "input_shape": [NUM_IN, SPATIAL, SPATIAL],
        "output_shape": [NUM_OUT, SPATIAL, SPATIAL],
        "weight_shape": [NUM_OUT, NUM_IN, 3, 3],
        "bias_shape": [NUM_OUT],
        "macs_per_output": MACS_PER,
        "total_outputs": EXPECTED,
        "total_macs": EXPECTED * MACS_PER,
        "zero_points": {"input": 0, "weight": 0, "output": 0},
        "input_scale": pkg["input_scale"],
        "output_scale": pkg["output_scale"],
        "weight_scales": [float(x) for x in np.asarray(pkg["weight_scales"]).tolist()],
        "multipliers": [int(x) for x in np.asarray(pkg["multipliers"]).tolist()],
        "shifts": [int(x) for x in np.asarray(pkg["shifts"]).tolist()],
        "quantization": "per_output_channel multipliers/shifts",
        "rounding": "ties_away_from_zero_right_shift",
        "saturation": "[-128, 127]",
        "relu": "after_saturation",
        "pool1_address": "ic * 256 + row * 16 + col",
        "weight_address": "oc * 144 + ic * 9 + kr * 3 + kc",
        "bias_address": "oc",
        "output_address": "oc * 256 + row * 16 + col",
        "source_input": str(sample_path),
    }
    (out / "conv2_quant_params.json").write_text(json.dumps(quant, indent=2), encoding="utf-8")

    meta = {
        **quant,
        "flattening": "channel_first_then_row_major",
        "min": int(flat.min()),
        "max": int(flat.max()),
        "num_zeros": zeros,
        "num_saturated": sats,
        "checksum_sum_int64": checksum,
        "channels": channel_meta,
    }
    (out / "conv2_expected.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "Conv2 expected tensor (signed INT8 after requant + ReLU)",
        f"shape: [{NUM_OUT}, {SPATIAL}, {SPATIAL}]  entries: {EXPECTED}",
        f"macs_per_output={MACS_PER}  total_macs={EXPECTED * MACS_PER}",
        f"min={meta['min']} max={meta['max']} zeros={zeros} sats={sats} checksum={checksum}",
        "address: oc*256 + row*16 + col",
        "",
        "Selected outputs:",
    ]
    for name, oc, row, col in SELECTED_CASES:
        addr = oc * 256 + row * 16 + col
        lines.append(
            f"  {name}: ({oc},{row},{col}) addr={addr} value={int(tensor[oc, row, col])}"
        )
    (out / "conv2_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for prefix, oc, row, col in ADDRESS_CASES:
        trace = compute_conv1_output_trace(
            pool1,
            pkg["weights"],
            pkg["bias"],
            pkg["multipliers"],
            pkg["shifts"],
            out_channel=oc,
            out_row=row,
            out_col=col,
        )
        if trace["mac_count"] != MACS_PER:
            raise AssertionError(f"{prefix}: expected {MACS_PER} MACs")
        trace = enrich_trace(trace)
        selected_names = {c[0] for c in SELECTED_CASES}
        if prefix in selected_names:
            if trace["final_accumulator"] != int(golden_acc[0, oc, row, col]):
                raise AssertionError(f"{prefix} acc mismatch")
            if trace["result_after_saturation"] != int(golden_conv[0, oc, row, col]):
                raise AssertionError(f"{prefix} requant mismatch")
            if trace["result_after_relu"] != int(golden_relu[0, oc, row, col]):
                raise AssertionError(f"{prefix} relu mismatch")

        write_trace_txt(traces_dir / f"{prefix}_trace.txt", trace)
        (traces_dir / f"{prefix}_metadata.json").write_text(
            json.dumps(trace, indent=2), encoding="utf-8"
        )
        act_addr_arr = np.asarray(
            [-1 if a is None else int(a) for a in trace["activation_addresses"]],
            dtype=np.int32,
        )
        write_mem_file(traces_dir / f"{prefix}_act_addr.mem", act_addr_arr, bits=32)
        write_mem_file(
            traces_dir / f"{prefix}_wgt_addr.mem",
            np.asarray(trace["weight_addresses"], dtype=np.int32),
            bits=32,
        )
        write_mem_file(
            traces_dir / f"{prefix}_input_row.mem",
            np.asarray([m["input_row"] for m in trace["macs"]], dtype=np.int8),
            bits=8,
        )
        write_mem_file(
            traces_dir / f"{prefix}_input_col.mem",
            np.asarray([m["input_column"] for m in trace["macs"]], dtype=np.int8),
            bits=8,
        )
        write_mem_file(
            traces_dir / f"{prefix}_padding.mem",
            np.asarray([1 if m["padding"] else 0 for m in trace["macs"]], dtype=np.int8),
            bits=8,
        )
        write_mem_file(
            traces_dir / f"{prefix}_inputs.mem",
            np.asarray(trace["inputs"], dtype=np.int8),
            bits=8,
        )
        write_mem_file(
            traces_dir / f"{prefix}_weights.mem",
            np.asarray(trace["weights"], dtype=np.int8),
            bits=8,
        )
        write_mem_file(
            traces_dir / f"{prefix}_products.mem",
            np.asarray(trace["products"], dtype=np.int16),
            bits=16,
        )
        write_mem_file(
            traces_dir / f"{prefix}_acc_trace.mem",
            np.asarray(trace["acc_trace"], dtype=np.int32),
            bits=32,
        )
        write_mem_file(
            traces_dir / f"{prefix}_expected_acc.mem",
            np.asarray([trace["final_accumulator"]], dtype=np.int32),
            bits=32,
        )
        write_mem_file(
            traces_dir / f"{prefix}_expected_requantized.mem",
            np.asarray([trace["result_after_saturation"]], dtype=np.int8),
            bits=8,
        )
        write_mem_file(
            traces_dir / f"{prefix}_expected_relu.mem",
            np.asarray([trace["result_after_relu"]], dtype=np.int8),
            bits=8,
        )
        write_mem_file(
            traces_dir / f"{prefix}_bias.mem",
            np.asarray([trace["bias"]], dtype=np.int32),
            bits=32,
        )
        write_mem_file(
            traces_dir / f"{prefix}_multiplier.mem",
            np.asarray([trace["multiplier"]], dtype=np.int32),
            bits=32,
        )
        write_mem_file(
            traces_dir / f"{prefix}_shift.mem",
            np.asarray([trace["shift"]], dtype=np.int32),
            bits=32,
        )
        write_mem_file(
            traces_dir / f"{prefix}_out_addr.mem",
            np.asarray([trace["output_address"]], dtype=np.int32),
            bits=32,
        )

    print(f"Exported Conv2 vectors to {out}")
    print(
        f"  entries={EXPECTED} channels={NUM_OUT} macs/out={MACS_PER} "
        f"min={meta['min']} max={meta['max']} checksum={checksum}"
    )
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--int8-root", default="software/exported_model/int8")
    p.add_argument("--sample-input", default=None)
    p.add_argument("--output-dir", default="vectors/conv2")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_conv2(
        int8_root=args.int8_root,
        sample_input=args.sample_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
