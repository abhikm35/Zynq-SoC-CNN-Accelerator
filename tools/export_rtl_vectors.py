#!/usr/bin/env python3
"""Export bit-exact Conv1 single-output vectors for RTL simulation.

Uses the same integer equations as software.inference.integer_inference and
software.quantization.fixed_point.requantize_int32 / rounding_right_shift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from software.export.export_fpga_hex import twos_complement_hex, write_hex_file
from software.hardware.memory_layout import write_mem_file
from software.quantization.fixed_point import requantize_int32, rounding_right_shift
from software.quantization.saturation import INT8_MAX, INT8_MIN
from software.utils.config import project_root, resolve_path


def _load_conv1_package(int8_root: Path) -> dict[str, Any]:
    return {
        "weights": np.load(int8_root / "weights" / "conv1_weights_int8.npy"),
        "bias": np.load(int8_root / "biases" / "conv1_bias_int32.npy"),
        "multipliers": np.load(int8_root / "requantization" / "conv1_multipliers_int32.npy"),
        "shifts": np.load(int8_root / "requantization" / "conv1_shifts_int32.npy"),
        "input_scale": float(np.load(int8_root / "scales" / "input_scale_float32.npy")),
        "weight_scales": np.load(int8_root / "scales" / "conv1_weight_scales_float32.npy"),
        "output_scale": float(np.load(int8_root / "scales" / "conv1_output_scale_float32.npy")),
        "manifest": json.loads(
            (int8_root / "metadata" / "quantization_manifest.json").read_text(encoding="utf-8")
        ),
    }


def compute_conv1_output_trace(
    input_nchw: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    multipliers: np.ndarray,
    shifts: np.ndarray,
    *,
    out_channel: int,
    out_row: int,
    out_col: int,
    stride: int = 1,
    padding: int = 1,
    input_zero_point: int = 0,
    weight_zero_point: int = 0,
    output_zero_point: int = 0,
) -> dict[str, Any]:
    """Explicit 27-MAC Conv1 pixel matching the golden integer model."""
    x = np.asarray(input_nchw)
    if x.ndim == 3:
        x = x[None, ...]
    w = np.asarray(weights)
    _, in_channels, kernel_h, kernel_w = w.shape
    _, _, height, width = x.shape

    bias_val = int(np.asarray(bias, dtype=np.int32).reshape(-1)[out_channel])
    mult = int(np.asarray(multipliers, dtype=np.int32).reshape(-1)[out_channel])
    shift = int(np.asarray(shifts, dtype=np.int32).reshape(-1)[out_channel])

    macs: list[dict[str, Any]] = []
    inputs_used: list[int] = []
    weights_used: list[int] = []
    products: list[int] = []
    acc_trace: list[int] = []

    acc = np.int64(bias_val)
    mac_index = 0
    for ic in range(in_channels):
        for kr in range(kernel_h):
            for kc in range(kernel_w):
                in_r = out_row * stride + kr - padding
                in_c = out_col * stride + kc - padding
                is_pad = (
                    in_r < 0 or in_c < 0 or in_r >= height or in_c >= width
                )
                if is_pad:
                    xin = int(input_zero_point)
                else:
                    xin = int(x[0, ic, in_r, in_c])
                win = int(w[out_channel, ic, kr, kc])
                x_c = np.int32(xin) - np.int32(input_zero_point)
                w_c = np.int32(win) - np.int32(weight_zero_point)
                product = int(np.int64(x_c) * np.int64(w_c))
                acc = np.int64(acc) + np.int64(product)
                if acc < np.iinfo(np.int32).min or acc > np.iinfo(np.int32).max:
                    raise OverflowError("accumulator exceeded int32")
                acc_i32 = int(np.int32(acc))

                macs.append(
                    {
                        "mac_index": mac_index,
                        "input_channel": ic,
                        "kernel_row": kr,
                        "kernel_column": kc,
                        "input_row": in_r,
                        "input_column": in_c,
                        "padding": bool(is_pad),
                        "input": xin,
                        "weight": win,
                        "product": product,
                        "accumulator": acc_i32,
                    }
                )
                inputs_used.append(xin)
                weights_used.append(win)
                products.append(product)
                acc_trace.append(acc_i32)
                mac_index += 1

    final_acc = int(np.int32(acc))
    # Match requantize_int32 for a scalar via 1-D path.
    wide = np.int64(final_acc) * np.int64(mult)
    shifted = int(rounding_right_shift(np.int64(wide), shift))
    with_zp = shifted + int(output_zero_point)
    before_sat = with_zp
    after_sat = int(np.clip(with_zp, INT8_MIN, INT8_MAX))
    after_relu = max(after_sat, 0)

    # Cross-check vectorized helper
    acc_tensor = np.array([[[[final_acc]]]], dtype=np.int32)
    # Fake single-channel tensor with this channel's mult/shift placed at index 0
    rq = requantize_int32(
        acc_tensor,
        np.array([mult], dtype=np.int32),
        np.array([shift], dtype=np.int32),
        output_zero_point=output_zero_point,
        qmin=INT8_MIN,
        qmax=INT8_MAX,
    )
    if int(rq.flat[0]) != after_sat:
        raise AssertionError(
            f"requant mismatch: helper={int(rq.flat[0])} trace={after_sat}"
        )

    return {
        "out_channel": out_channel,
        "out_row": out_row,
        "out_col": out_col,
        "bias": bias_val,
        "multiplier": mult,
        "shift": shift,
        "output_zero_point": output_zero_point,
        "input_zero_point": input_zero_point,
        "weight_zero_point": weight_zero_point,
        "mac_count": mac_index,
        "macs": macs,
        "inputs": inputs_used,
        "weights": weights_used,
        "products": products,
        "acc_trace": acc_trace,
        "final_accumulator": final_acc,
        "wide_product": int(wide),
        "result_after_rounding_shift": shifted,
        "result_before_saturation": before_sat,
        "result_after_saturation": after_sat,
        "result_after_relu": after_relu,
        "rounding_offset": (1 << (shift - 1)) if shift > 0 else 0,
    }


def _write_case(out_dir: Path, prefix: str, trace: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_mem_file(out_dir / f"{prefix}_inputs.mem", np.asarray(trace["inputs"], dtype=np.int8), bits=8)
    write_mem_file(out_dir / f"{prefix}_weights.mem", np.asarray(trace["weights"], dtype=np.int8), bits=8)
    write_mem_file(out_dir / f"{prefix}_products.mem", np.asarray(trace["products"], dtype=np.int16), bits=16)
    write_mem_file(out_dir / f"{prefix}_acc_trace.mem", np.asarray(trace["acc_trace"], dtype=np.int32), bits=32)
    write_mem_file(out_dir / f"{prefix}_bias.mem", np.asarray([trace["bias"]], dtype=np.int32), bits=32)
    write_mem_file(
        out_dir / f"{prefix}_expected_acc.mem",
        np.asarray([trace["final_accumulator"]], dtype=np.int32),
        bits=32,
    )
    write_mem_file(
        out_dir / f"{prefix}_expected_requantized.mem",
        np.asarray([trace["result_after_saturation"]], dtype=np.int8),
        bits=8,
    )
    write_mem_file(
        out_dir / f"{prefix}_expected_relu.mem",
        np.asarray([trace["result_after_relu"]], dtype=np.int8),
        bits=8,
    )
    write_mem_file(
        out_dir / f"{prefix}_multiplier.mem",
        np.asarray([trace["multiplier"]], dtype=np.int32),
        bits=32,
    )
    write_mem_file(
        out_dir / f"{prefix}_shift.mem",
        np.asarray([trace["shift"]], dtype=np.int32),
        bits=32,
    )

    meta = {
        k: v
        for k, v in trace.items()
        if k not in {"macs"}
    }
    meta["macs"] = trace["macs"]
    (out_dir / f"{prefix}_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    lines = [
        f"Initial bias: {trace['bias']}",
        f"out_channel={trace['out_channel']} out_row={trace['out_row']} out_col={trace['out_col']}",
        f"multiplier={trace['multiplier']} shift={trace['shift']} output_zp={trace['output_zero_point']}",
        "",
    ]
    for mac in trace["macs"]:
        lines.extend(
            [
                f"MAC {mac['mac_index']}",
                f"input_channel = {mac['input_channel']}",
                f"kernel_row = {mac['kernel_row']}",
                f"kernel_column = {mac['kernel_column']}",
                f"input_row = {mac['input_row']}",
                f"input_column = {mac['input_column']}",
                f"padding = {str(mac['padding']).lower()}",
                f"input = {mac['input']}",
                f"weight = {mac['weight']}",
                f"product = {mac['product']}",
                f"accumulator = {mac['accumulator']}",
                "",
            ]
        )
    lines.extend(
        [
            f"Final INT32 accumulator: {trace['final_accumulator']}",
            f"wide_product (acc*mult): {trace['wide_product']}",
            f"rounding_offset: {trace['rounding_offset']}",
            f"after rounding/shift: {trace['result_after_rounding_shift']}",
            f"before saturation: {trace['result_before_saturation']}",
            f"after saturation (conv1): {trace['result_after_saturation']}",
            f"after ReLU (relu1): {trace['result_after_relu']}",
        ]
    )
    (out_dir / f"{prefix}_trace.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_vectors(
    *,
    int8_root: str | Path = "software/exported_model/int8",
    sample_input: str | Path | None = None,
    output_dir: str | Path = "vectors/conv1_single_output",
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

    # Verify against full-layer golden accumulator for selected pixels.
    from software.inference.integer_inference import IntegerTinyCNN

    model = IntegerTinyCNN.from_export_directory(root)
    # sample_000 input is already quantized int8
    _, inter = model.forward_with_intermediates(input_q, input_already_quantized=True)
    golden_acc = inter["conv1_accumulator"]
    golden_conv = inter["conv1"]
    golden_relu = inter["relu1"]

    cases = [
        ("center", 0, 10, 10),
        ("corner", 0, 0, 0),
        ("channel3", 3, 16, 8),
    ]
    out_dir = resolve_path(output_dir)
    exported = {}
    for prefix, oc, row, col in cases:
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
        g_acc = int(golden_acc[0, oc, row, col])
        g_rq = int(golden_conv[0, oc, row, col])
        g_relu = int(golden_relu[0, oc, row, col])
        if trace["final_accumulator"] != g_acc:
            raise AssertionError(
                f"{prefix}: acc mismatch trace={trace['final_accumulator']} golden={g_acc}"
            )
        if trace["result_after_saturation"] != g_rq:
            raise AssertionError(
                f"{prefix}: requant mismatch trace={trace['result_after_saturation']} golden={g_rq}"
            )
        if trace["result_after_relu"] != g_relu:
            raise AssertionError(
                f"{prefix}: relu mismatch trace={trace['result_after_relu']} golden={g_relu}"
            )
        if trace["mac_count"] != 27:
            raise AssertionError(f"{prefix}: expected 27 MACs, got {trace['mac_count']}")
        _write_case(out_dir, prefix, trace)
        exported[prefix] = {
            "out_channel": oc,
            "out_row": row,
            "out_col": col,
            "final_accumulator": trace["final_accumulator"],
            "requantized": trace["result_after_saturation"],
            "relu": trace["result_after_relu"],
        }

    summary = {
        "source_input": str(sample_path),
        "int8_root": str(root),
        "architecture_note": "Conv1 is 3->16 (trained model), not 3->8",
        "macs_per_output": 27,
        "cases": exported,
        "quantization": {
            "symmetric": True,
            "zero_points": 0,
            "weight_scales": "per_output_channel",
            "activation_scales": "per_tensor",
            "requantization": "round(acc * multiplier / 2^shift) ties away from zero",
        },
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Exported Conv1 single-output vectors to {out_dir}")
    for name, info in exported.items():
        print(
            f"  {name}: oc={info['out_channel']} ({info['out_row']},{info['out_col']}) "
            f"acc={info['final_accumulator']} rq={info['requantized']} relu={info['relu']}"
        )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--int8-root", default="software/exported_model/int8")
    p.add_argument("--sample-input", default=None)
    p.add_argument("--output-dir", default="vectors/conv1_single_output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_vectors(
        int8_root=args.int8_root,
        sample_input=args.sample_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
