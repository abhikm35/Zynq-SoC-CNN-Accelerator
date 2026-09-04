#!/usr/bin/env python3
"""Export signed argmax vectors for RTL.

Five signed INT32 FC logits -> predicted_class[2:0] and maximum_logit.
Tie-breaking matches numpy.argmax: lowest index among equal maxima
(update only on strictly greater).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from software.hardware.memory_layout import write_mem_file
from software.inference.integer_inference import IntegerTinyCNN
from software.utils.config import resolve_path


NUM_CLASSES = 5
CLASS_NAMES = ["stop", "yield", "no_entry", "speed_limit_30", "keep_right"]
INT32_MIN = int(np.iinfo(np.int32).min)
INT32_MAX = int(np.iinfo(np.int32).max)
RNG_SEED = 20260804
NUM_RANDOM = 512


def python_argmax(logits: np.ndarray) -> tuple[int, int]:
    """Lowest-index argmax (numpy.argmax / IntegerTinyCNN.predict)."""
    values = np.asarray(logits, dtype=np.int32).reshape(-1)
    if values.size != NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} logits, got {values.size}")
    idx = int(np.argmax(values))
    return idx, int(values[idx])


def comparison_trace(logits: list[int]) -> list[dict[str, Any]]:
    maximum = int(logits[0])
    pred = 0
    steps = [
        {
            "index": 0,
            "logit": int(logits[0]),
            "compared": False,
            "updated": True,
            "maximum_logit": maximum,
            "predicted_class": pred,
        }
    ]
    for i in range(1, NUM_CLASSES):
        cur = int(logits[i])
        updated = cur > maximum
        if updated:
            maximum = cur
            pred = i
        steps.append(
            {
                "index": i,
                "logit": cur,
                "compared": True,
                "updated": updated,
                "maximum_logit": maximum,
                "predicted_class": pred,
            }
        )
    return steps


def make_case(name: str, logits: list[int]) -> dict[str, Any]:
    pred, maximum = python_argmax(logits)
    return {
        "name": name,
        "logits": [int(x) for x in logits],
        "expected_maximum_logit": maximum,
        "expected_predicted_class": pred,
        "expected_class_name": CLASS_NAMES[pred],
        "comparison_decisions": comparison_trace(logits),
        "tie_breaking": "lowest_index_strict_greater",
    }


DIRECTED_CASES: list[tuple[str, list[int]]] = [
    ("largest_at_0", [500, 20, 10, 0, -1]),
    ("largest_at_1", [-10, 200, 30, 40, 50]),
    ("largest_at_2", [-10, -20, 300, -40, -50]),
    ("largest_at_3", [-100, -200, -300, -1, -400]),
    ("largest_at_4", [-100, -200, -300, -400, 0]),
    ("all_negative_max_at_1", [-10, -5, -20, -8, -30]),
    ("two_equal_maxima", [100, 500, 20, 500, -10]),
    ("all_equal", [7, 7, 7, 7, 7]),
    ("mixed_pos_neg", [-50, 10, -3, 8, -100]),
    ("exactly_one_positive", [-5, -9, -2, 1, -40]),
    ("three_equal_maxima", [1, 1, 5, 5, 5]),
    ("max_first_and_last_tie", [9, 1, 2, 3, 9]),
    ("int32_min_present", [INT32_MIN, -1, -2, -3, -4]),
    ("int32_max_present", [0, 1, INT32_MAX, 2, 3]),
    ("int32_minmax", [INT32_MIN, 0, INT32_MAX, -100, 100]),
    ("alternating", [1000, -1000, 999, -999, 998]),
]


def export_argmax(
    *,
    int8_root: str | Path = "software/exported_model/int8",
    sample_input: str | Path | None = None,
    output_dir: str | Path = "vectors/argmax",
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
    scores, _ = model.forward_with_intermediates(input_q, input_already_quantized=True)
    real_logits = np.asarray(scores, dtype=np.int32).reshape(-1)
    if real_logits.size != NUM_CLASSES:
        raise RuntimeError(f"unexpected score shape {real_logits.shape}")
    pred_real, max_real = python_argmax(real_logits)
    pred_via_scores = int(np.argmax(scores, axis=1)[0])
    if pred_via_scores != pred_real:
        raise AssertionError("argmax mismatch vs scores")

    out = resolve_path(output_dir)
    cases_dir = out / "test_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    write_mem_file(out / "logits_input.mem", real_logits, bits=32)

    real_case = make_case("real_fc_sample_000", [int(x) for x in real_logits])
    directed = [make_case(name, vals) for name, vals in DIRECTED_CASES]

    rng = np.random.default_rng(RNG_SEED)
    random_cases: list[dict[str, Any]] = []
    for n in range(NUM_RANDOM):
        # Mix ranges: full INT32, small, and forced-duplicate maxima
        mode = n % 5
        if mode == 0:
            vals = rng.integers(INT32_MIN, INT32_MAX, size=5, dtype=np.int64).astype(np.int32)
        elif mode == 1:
            vals = rng.integers(-1000, 1001, size=5, dtype=np.int32)
        elif mode == 2:
            base = int(rng.integers(-500, 501))
            vals = np.full(5, base, dtype=np.int32)
            vals[int(rng.integers(0, 5))] = base  # all equal
        elif mode == 3:
            vals = rng.integers(-200, 201, size=5, dtype=np.int32)
            hi = int(np.max(vals))
            vals[0] = hi
            vals[4] = hi  # duplicate max at ends
        else:
            vals = rng.integers(INT32_MIN // 2, INT32_MAX // 2, size=5, dtype=np.int64)
            vals = vals.astype(np.int32)
            hi = int(np.max(vals))
            idxs = rng.choice(5, size=2, replace=False)
            vals[idxs[0]] = hi
            vals[idxs[1]] = hi
        random_cases.append(make_case(f"random_{n:04d}", [int(x) for x in vals]))

    # Write directed + random case files (compact mem + json)
    all_cases = directed + random_cases
    directed_mem_lines: list[str] = []
    for case in directed:
        for v in case["logits"]:
            directed_mem_lines.append(f"{(v & 0xFFFFFFFF):08x}")
        directed_mem_lines.append(f"{(case['expected_maximum_logit'] & 0xFFFFFFFF):08x}")
        directed_mem_lines.append(f"{case['expected_predicted_class']:01x}")
    (cases_dir / "directed_cases.mem").write_text("\n".join(directed_mem_lines) + "\n")
    (cases_dir / "directed_cases.json").write_text(json.dumps(directed, indent=2) + "\n")

    random_mem_lines: list[str] = []
    for case in random_cases:
        for v in case["logits"]:
            random_mem_lines.append(f"{(v & 0xFFFFFFFF):08x}")
        random_mem_lines.append(f"{(case['expected_maximum_logit'] & 0xFFFFFFFF):08x}")
        random_mem_lines.append(f"{case['expected_predicted_class']:01x}")
    (cases_dir / "random_cases.mem").write_text("\n".join(random_mem_lines) + "\n")
    (cases_dir / "random_cases_meta.json").write_text(
        json.dumps(
            {
                "count": NUM_RANDOM,
                "seed": RNG_SEED,
                "format": "per case: 5 logit hex lines, max logit hex, class hex",
            },
            indent=2,
        )
        + "\n"
    )

    for case in directed:
        (cases_dir / f"{case['name']}.json").write_text(json.dumps(case, indent=2) + "\n")

    summary = {
        "operation": "signed_argmax5",
        "num_logits": NUM_CLASSES,
        "logit_datatype": "int32",
        "logit_width": 32,
        "logit_signed": True,
        "predicted_class_width": 3,
        "tie_breaking": "lowest_index_strict_greater",
        "python_reference": "numpy.argmax / IntegerTinyCNN.predict",
        "softmax": False,
        "requantize": False,
        "relu": False,
        "class_names": CLASS_NAMES,
        "real_case": real_case,
        "directed_count": len(directed),
        "random_count": NUM_RANDOM,
        "source_input": str(sample_path),
        "note": "Logits are FC post-requant INT32 scores.",
    }
    (out / "argmax_expected.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "Argmax vector export summary",
        f"Real logits: {real_case['logits']}",
        f"Expected max logit: {max_real}",
        f"Expected class: {pred_real} ({CLASS_NAMES[pred_real]})",
        "Tie-breaking: lowest index (strict >)",
        f"Directed cases: {len(directed)}",
        f"Random cases: {NUM_RANDOM}",
    ]
    (out / "argmax_summary.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote argmax vectors under {out}")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--int8-root", default="software/exported_model/int8")
    p.add_argument("--sample-input", default=None)
    p.add_argument("--output-dir", default="vectors/argmax")
    args = p.parse_args()
    export_argmax(
        int8_root=args.int8_root,
        sample_input=args.sample_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
