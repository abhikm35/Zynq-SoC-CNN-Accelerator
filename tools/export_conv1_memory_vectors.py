#!/usr/bin/env python3
"""Export full Conv1 memories and memory-driven single-output traces.

Uses the trained architecture: Conv1 3->16 (not the outdated 8-channel sketch).
Address equations match software.hardware.memory_layout.
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
from tools.export_rtl_vectors import _load_conv1_package, compute_conv1_output_trace


SELECTED_CASES = [
    ("center", 0, 10, 10),
    ("top_left", 0, 0, 0),
    ("bottom_right", 0, 31, 31),
    ("channel5", 5, 12, 17),
]

# Extra cases for address-generator coverage
ADDRESS_CASES = SELECTED_CASES + [
    ("top_right", 0, 0, 31),
    ("bottom_left", 0, 31, 0),
]


def enrich_trace_with_addresses(
    trace: dict[str, Any],
    *,
    height: int = 32,
    width: int = 32,
    in_channels: int = 3,
    kernel_h: int = 3,
    kernel_w: int = 3,
) -> dict[str, Any]:
    oc = int(trace["out_channel"])
    act_addrs: list[int | None] = []
    wgt_addrs: list[int] = []
    for mac in trace["macs"]:
        ic = int(mac["input_channel"])
        kr = int(mac["kernel_row"])
        kc = int(mac["kernel_column"])
        waddr = weight_address(
            oc,
            ic,
            kr,
            kc,
            input_channels=in_channels,
            kernel_height=kernel_h,
            kernel_width=kernel_w,
        )
        mac["weight_address"] = waddr
        mac["bias_address"] = oc
        if mac["padding"]:
            mac["activation_address"] = None
            act_addrs.append(None)
        else:
            aaddr = activation_address(
                ic,
                int(mac["input_row"]),
                int(mac["input_column"]),
                height=height,
                width=width,
            )
            mac["activation_address"] = aaddr
            act_addrs.append(aaddr)
        wgt_addrs.append(waddr)
    trace["activation_addresses"] = act_addrs
    trace["weight_addresses"] = wgt_addrs
    trace["bias_address"] = oc
    return trace


def write_memory_trace(path: Path, trace: dict[str, Any]) -> None:
    lines = [
        f"output_channel = {trace['out_channel']}",
        f"output_row = {trace['out_row']}",
        f"output_column = {trace['out_col']}",
        f"bias_address = {trace['bias_address']}",
        f"bias_value = {trace['bias']}",
        f"multiplier = {trace['multiplier']}",
        f"shift = {trace['shift']}",
        "",
    ]
    for mac in trace["macs"]:
        aaddr = mac["activation_address"]
        lines.extend(
            [
                f"MAC {mac['mac_index']}",
                f"output_channel = {trace['out_channel']}",
                f"output_row = {trace['out_row']}",
                f"output_column = {trace['out_col']}",
                f"input_channel = {mac['input_channel']}",
                f"kernel_row = {mac['kernel_row']}",
                f"kernel_column = {mac['kernel_column']}",
                f"calculated_input_row = {mac['input_row']}",
                f"calculated_input_column = {mac['input_column']}",
                f"padding = {str(mac['padding']).lower()}",
                f"activation_address = {'invalid' if aaddr is None else aaddr}",
                f"activation_value = {mac['input']}",
                f"weight_address = {mac['weight_address']}",
                f"weight_value = {mac['weight']}",
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
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_conv1_memories(
    *,
    int8_root: str | Path = "software/exported_model/int8",
    sample_input: str | Path | None = None,
    output_dir: str | Path = "vectors/conv1_memory",
) -> dict[str, Any]:
    root = resolve_path(int8_root)
    pkg = _load_conv1_package(root)
    weights = pkg["weights"]
    out_ch, in_ch, kh, kw = weights.shape
    if (out_ch, in_ch, kh, kw) != (16, 3, 3, 3):
        raise RuntimeError(
            f"Unexpected Conv1 weight shape {weights.shape}; expected (16,3,3,3)"
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

    out = resolve_path(output_dir)
    traces_dir = out / "selected_output_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    # Full memories
    write_mem_file(out / "input_image.mem", flatten_activation_nchw(input_q), bits=8)
    write_mem_file(out / "conv1_weights.mem", flatten_weights_oihw(weights), bits=8)
    write_mem_file(out / "conv1_biases.mem", np.asarray(pkg["bias"], dtype=np.int32), bits=32)
    write_mem_file(
        out / "conv1_multipliers.mem",
        np.asarray(pkg["multipliers"], dtype=np.int32),
        bits=32,
    )
    write_mem_file(
        out / "conv1_shifts.mem",
        np.asarray(pkg["shifts"], dtype=np.int32),
        bits=32,
    )

    quant = {
        "architecture": "Conv1 3->16, k=3, s=1, p=1 (trained model)",
        "note": "Prompt sketches with 8 output channels are obsolete; use 16.",
        "input_shape": [3, 32, 32],
        "weight_shape": [16, 3, 3, 3],
        "bias_shape": [16],
        "zero_points": {"input": 0, "weight": 0, "output": 0},
        "input_scale": pkg["input_scale"],
        "output_scale": pkg["output_scale"],
        "weight_scales": [float(x) for x in np.asarray(pkg["weight_scales"]).tolist()],
        "multipliers": [int(x) for x in np.asarray(pkg["multipliers"]).tolist()],
        "shifts": [int(x) for x in np.asarray(pkg["shifts"]).tolist()],
        "activation_address": "ic*1024 + row*32 + col",
        "weight_address": "oc*27 + ic*9 + kr*3 + kc",
        "bias_address": "oc",
        "rounding": "ties_away_from_zero_right_shift",
        "source_input": str(sample_path),
    }
    (out / "conv1_quant_params.json").write_text(json.dumps(quant, indent=2), encoding="utf-8")

    model = IntegerTinyCNN.from_export_directory(root)
    _, inter = model.forward_with_intermediates(input_q, input_already_quantized=True)
    golden_acc = inter["conv1_accumulator"]
    golden_conv = inter["conv1"]
    golden_relu = inter["relu1"]

    expected_outputs: dict[str, Any] = {"cases": {}}
    address_tables: dict[str, Any] = {"cases": {}}

    for prefix, oc, row, col in ADDRESS_CASES:
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
        selected_names = {c[0] for c in SELECTED_CASES}
        if prefix in selected_names:
            g_acc = int(golden_acc[0, oc, row, col])
            g_rq = int(golden_conv[0, oc, row, col])
            g_relu = int(golden_relu[0, oc, row, col])
            if trace["final_accumulator"] != g_acc:
                raise AssertionError(f"{prefix} acc mismatch")
            if trace["result_after_saturation"] != g_rq:
                raise AssertionError(f"{prefix} requant mismatch")
            if trace["result_after_relu"] != g_relu:
                raise AssertionError(f"{prefix} relu mismatch")

        write_memory_trace(traces_dir / f"{prefix}_trace.txt", trace)
        (traces_dir / f"{prefix}_metadata.json").write_text(
            json.dumps(trace, indent=2),
            encoding="utf-8",
        )

        # Compact mem dumps for TB (27 entries); use -1 for invalid act addr
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
        if prefix in selected_names:
            expected_outputs["cases"][prefix] = {
                "name": prefix,
                "output_channel": oc,
                "output_row": row,
                "output_column": col,
                "bias_address": oc,
                "bias": trace["bias"],
                "multiplier": trace["multiplier"],
                "shift": trace["shift"],
                "final_accumulator": trace["final_accumulator"],
                "requantized": trace["result_after_saturation"],
                "relu": trace["result_after_relu"],
                "mac_count": 27,
            }
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

        case_meta = {
            "name": prefix,
            "output_channel": oc,
            "output_row": row,
            "output_column": col,
            "bias_address": oc,
            "bias": trace["bias"],
            "multiplier": trace["multiplier"],
            "shift": trace["shift"],
            "final_accumulator": trace["final_accumulator"],
            "requantized": trace["result_after_saturation"],
            "relu": trace["result_after_relu"],
            "mac_count": 27,
        }
        address_tables["cases"][prefix] = {
            **case_meta,
            "macs": [
                {
                    "mac_index": m["mac_index"],
                    "input_channel": m["input_channel"],
                    "kernel_row": m["kernel_row"],
                    "kernel_column": m["kernel_column"],
                    "input_row": m["input_row"],
                    "input_column": m["input_column"],
                    "padding": m["padding"],
                    "activation_address": m["activation_address"],
                    "weight_address": m["weight_address"],
                    "bias_address": m["bias_address"],
                }
                for m in trace["macs"]
            ],
        }

    (out / "expected_selected_outputs.json").write_text(
        json.dumps(expected_outputs, indent=2), encoding="utf-8"
    )
    (out / "address_tables.json").write_text(
        json.dumps(address_tables, indent=2), encoding="utf-8"
    )

    print(f"Exported Conv1 memories to {out}")
    print(f"  input entries: 3072  weight entries: {out_ch * 27}  biases: {out_ch}")
    for name, info in expected_outputs["cases"].items():
        print(
            f"  {name}: oc={info['output_channel']} "
            f"({info['output_row']},{info['output_column']}) "
            f"acc={info['final_accumulator']} rq={info['requantized']} relu={info['relu']}"
        )
    return {"output_dir": str(out), "cases": expected_outputs["cases"]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--int8-root", default="software/exported_model/int8")
    p.add_argument("--sample-input", default=None)
    p.add_argument("--output-dir", default="vectors/conv1_memory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_conv1_memories(
        int8_root=args.int8_root,
        sample_input=args.sample_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
