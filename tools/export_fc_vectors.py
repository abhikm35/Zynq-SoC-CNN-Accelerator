#!/usr/bin/env python3
"""Export fully connected (classifier) vectors for RTL.

Trained model: Linear 32 -> 5 (GAP flatten length 32).
Prompt sketches with 16 inputs / 80 weights are obsolete.

Python path:
  acc[c] = bias[c] + Σ_i gap[i] * weight[c][i]   # ZP=0, INT32
  logit[c] = requantize_int32(acc[c], mult[c], shift[c],
                              zp=0, qmin=INT32_MIN, qmax=INT32_MAX)
No ReLU on logits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from software.hardware.memory_layout import write_mem_file
from software.inference.integer_inference import IntegerTinyCNN, integer_linear
from software.quantization.fixed_point import requantize_int32
from software.utils.config import resolve_path


NUM_FEATURES = 32
NUM_CLASSES = 5
NUM_WEIGHTS = NUM_CLASSES * NUM_FEATURES  # 160
SELECTED_CLASSES = [0, 2, 4]


def weight_address(class_index: int, input_index: int) -> int:
    return class_index * NUM_FEATURES + input_index


def export_fc(
    *,
    int8_root: str | Path = "software/exported_model/int8",
    sample_input: str | Path | None = None,
    output_dir: str | Path = "vectors/fc",
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

    gap = np.asarray(inter["flatten"], dtype=np.int8).reshape(-1)
    weights = np.asarray(model.weights["classifier"], dtype=np.int8)
    biases = np.asarray(model.biases["classifier"], dtype=np.int32).reshape(-1)
    mults = np.asarray(model.requant["classifier_multipliers"], dtype=np.int32).reshape(-1)
    shifts = np.asarray(model.requant["classifier_shifts"], dtype=np.int32).reshape(-1)
    scores = np.asarray(inter["scores"], dtype=np.int32).reshape(-1)
    acc_gold = np.asarray(inter["classifier_accumulator"], dtype=np.int32).reshape(-1)

    if gap.size != NUM_FEATURES:
        raise RuntimeError(f"Unexpected GAP/flatten length {gap.size}")
    if weights.shape != (NUM_CLASSES, NUM_FEATURES):
        raise RuntimeError(f"Unexpected weight shape {weights.shape}")
    if biases.shape != (NUM_CLASSES,):
        raise RuntimeError(f"Unexpected bias shape {biases.shape}")

    acc_check = integer_linear(
        gap.reshape(1, -1),
        weights,
        biases,
        input_zero_point=0,
        weight_zero_point=0,
    ).reshape(-1)
    if not np.array_equal(acc_check, acc_gold):
        raise AssertionError("classifier accumulator mismatch")
    scores_check = requantize_int32(
        acc_check.reshape(1, -1),
        mults,
        shifts,
        output_zero_point=0,
        qmin=np.iinfo(np.int32).min,
        qmax=np.iinfo(np.int32).max,
    ).reshape(-1).astype(np.int32)
    if not np.array_equal(scores_check, scores):
        raise AssertionError("scores mismatch vs requantize_int32")

    out = resolve_path(output_dir)
    traces_dir = out / "class_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    write_mem_file(out / "gap_input.mem", gap, bits=8)
    write_mem_file(out / "fc_weights.mem", weights.reshape(-1, order="C"), bits=8)
    write_mem_file(out / "fc_biases.mem", biases, bits=32)
    write_mem_file(out / "fc_multipliers.mem", mults, bits=32)
    write_mem_file(out / "fc_shifts.mem", shifts, bits=32)
    write_mem_file(out / "fc_logits_expected.mem", scores, bits=32)
    write_mem_file(out / "fc_accumulators_expected.mem", acc_gold, bits=32)

    quant = {
        "input_features": NUM_FEATURES,
        "num_classes": NUM_CLASSES,
        "gap_datatype": "int8",
        "gap_zero_point": 0,
        "weight_datatype": "int8",
        "weight_shape": [NUM_CLASSES, NUM_FEATURES],
        "weight_zero_point": 0,
        "bias_datatype": "int32",
        "bias_in_accumulator_units": True,
        "accumulator_datatype": "int32",
        "logit_datatype": "int32",
        "logit_zero_point": 0,
        "requantize": True,
        "multipliers": [int(x) for x in mults],
        "shifts": [int(x) for x in shifts],
        "saturation": [int(np.iinfo(np.int32).min), int(np.iinfo(np.int32).max)],
        "rounding": "ties_away_from_zero",
        "relu_on_logits": False,
        "weight_address": "class_index * 32 + input_index",
        "gap_address": "input_index",
        "bias_address": "class_index",
        "logit_address": "class_index",
    }
    (out / "fc_quant_params.json").write_text(json.dumps(quant, indent=2) + "\n")

    class_meta: list[dict[str, Any]] = []
    json_traces: dict[str, Any] = {}
    for c in range(NUM_CLASSES):
        running = int(biases[c])
        steps: list[dict[str, Any]] = []
        lines = [
            f"class_index: {c}",
            f"bias_address: {c}",
            f"bias_value: {int(biases[c])}",
            f"multiplier: {int(mults[c])}",
            f"shift: {int(shifts[c])}",
            "",
        ]
        for i in range(NUM_FEATURES):
            g = int(gap[i])
            w = int(weights[c, i])
            prod = g * w
            running += prod
            wa = weight_address(c, i)
            step = {
                "input_index": i,
                "gap_address": i,
                "gap_value": g,
                "gap_centered": g,
                "weight_address": wa,
                "weight_value": w,
                "weight_centered": w,
                "product": prod,
                "running_accumulator": running,
            }
            steps.append(step)
            lines.append(
                f"i={i:2d} gap_addr={i:2d} gap={g:4d} "
                f"wgt_addr={wa:3d} wgt={w:4d} prod={prod:7d} acc={running}"
            )

        final_acc = int(acc_gold[c])
        if running != final_acc:
            raise AssertionError(f"class {c} running sum {running} != gold {final_acc}")
        logit = int(scores[c])
        meta = {
            "class_index": c,
            "bias": int(biases[c]),
            "final_accumulator": final_acc,
            "multiplier": int(mults[c]),
            "shift": int(shifts[c]),
            "output_zero_point": 0,
            "logit": logit,
            "logit_hex": f"{(logit & 0xFFFFFFFF):08x}",
            "mac_count": NUM_FEATURES,
        }
        class_meta.append(meta)
        lines.extend(
            [
                "",
                f"final_accumulator: {final_acc}",
                f"requant multiplier: {int(mults[c])}",
                f"requant shift: {int(shifts[c])}",
                "rounding: ties away from zero",
                "saturation: full INT32 (no INT8 clip)",
                "ReLU: not applied",
                f"final_logit: {logit} (0x{(logit & 0xFFFFFFFF):08x})",
            ]
        )
        (traces_dir / f"class_{c}.txt").write_text("\n".join(lines) + "\n")
        json_traces[f"class_{c}"] = {"meta": meta, "steps": steps}

    (traces_dir / "all_classes.json").write_text(json.dumps(json_traces, indent=2) + "\n")
    selected = {str(c): json_traces[f"class_{c}"] for c in SELECTED_CLASSES}
    (out / "selected_classes.json").write_text(json.dumps(selected, indent=2) + "\n")

    summary = {
        "operation": "fully_connected_classifier",
        "note": "Trained Linear 32->5. Logits are INT32 after per-class requant.",
        "input_features": NUM_FEATURES,
        "num_classes": NUM_CLASSES,
        "num_weights": NUM_WEIGHTS,
        "macs_per_class": NUM_FEATURES,
        "total_macs": NUM_WEIGHTS,
        "gap_values": [int(x) for x in gap],
        "biases": [int(x) for x in biases],
        "accumulators": [int(x) for x in acc_gold],
        "logits": [int(x) for x in scores],
        "predicted_class_argmax": int(np.argmax(scores)),
        "classes": class_meta,
        "quant": quant,
        "source_input": str(sample_path),
    }
    (out / "fc_logits_expected.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "FC vector export summary",
        f"GAP inputs: {NUM_FEATURES}",
        f"Classes: {NUM_CLASSES}",
        f"Weights: {NUM_WEIGHTS}",
        f"Total MACs: {NUM_WEIGHTS}",
        f"argmax={summary['predicted_class_argmax']}",
        "",
        "class  bias    acc    logit",
    ]
    for m in class_meta:
        lines.append(
            f"{m['class_index']:5d} {m['bias']:6d} {m['final_accumulator']:7d} {m['logit']:7d}"
        )
    (out / "fc_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote FC vectors under {out}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--int8-root", default="software/exported_model/int8")
    p.add_argument("--sample-input", default=None)
    p.add_argument("--output-dir", default="vectors/fc")
    args = p.parse_args()
    export_fc(
        int8_root=args.int8_root,
        sample_input=args.sample_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
