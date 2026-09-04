#!/usr/bin/env python3
"""Export end-to-end CNN inference vectors for RTL integration.

Trained model sizes (prompt sketches with 8/16 channels are obsolete):
  input  3x32x32  -> Conv1/ReLU1 16x32x32 -> Pool1 16x16x16
                  -> Conv2/ReLU2 32x16x16 -> Pool2 32x8x8
                  -> GAP/flatten 32 -> FC 5 INT32 logits -> argmax class
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


CLASS_NAMES = ["stop", "yield", "no_entry", "speed_limit_30", "keep_right"]
# One sample whose integer model predicts each class (class may differ from true label).
MULTI_SAMPLES = [
    ("sample_000", 0),
    ("sample_001", 1),
    ("sample_008", 2),
    ("sample_002", 3),
    ("sample_005", 4),
]


def _stats(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr).reshape(-1)
    return {
        "shape": list(np.asarray(arr).shape),
        "size": int(a.size),
        "dtype": str(a.dtype),
        "checksum": int(np.sum(a.astype(np.int64))),
        "min": int(a.min()),
        "max": int(a.max()),
        "zero_count": int(np.sum(a == 0)),
    }


def python_argmax(logits: np.ndarray) -> tuple[int, int]:
    values = np.asarray(logits, dtype=np.int32).reshape(-1)
    idx = int(np.argmax(values))
    return idx, int(values[idx])


def export_one_image(
    model: IntegerTinyCNN,
    sample_path: Path,
    out_dir: Path,
    *,
    write_full_tensors: bool,
) -> dict[str, Any]:
    input_q = np.load(sample_path / "input.npy")
    if input_q.ndim == 3:
        input_q = input_q[None, ...]

    _, inter = model.forward_with_intermediates(input_q, input_already_quantized=True)

    conv1 = np.asarray(inter["relu1"], dtype=np.int8).reshape(-1)
    pool1 = np.asarray(inter["pool1"], dtype=np.int8).reshape(-1)
    conv2 = np.asarray(inter["relu2"], dtype=np.int8).reshape(-1)
    pool2 = np.asarray(inter["pool2"], dtype=np.int8).reshape(-1)
    gap = np.asarray(inter["flatten"], dtype=np.int8).reshape(-1)
    logits = np.asarray(inter["scores"], dtype=np.int32).reshape(-1)
    inp = np.asarray(input_q, dtype=np.int8).reshape(-1)

    pred, max_logit = python_argmax(logits)
    true_class = int((sample_path / "expected_class.txt").read_text().strip())

    expected_sizes = {
        "input": 3072,
        "conv1": 16384,
        "pool1": 4096,
        "conv2": 8192,
        "pool2": 2048,
        "gap": 32,
        "logits": 5,
    }
    actual = {
        "input": inp.size,
        "conv1": conv1.size,
        "pool1": pool1.size,
        "conv2": conv2.size,
        "pool2": pool2.size,
        "gap": gap.size,
        "logits": logits.size,
    }
    for k, n in expected_sizes.items():
        if actual[k] != n:
            raise RuntimeError(f"{sample_path.name}: {k} size {actual[k]} != {n}")

    out_dir.mkdir(parents=True, exist_ok=True)

    write_mem_file(out_dir / "input_image.mem", inp, bits=8)
    write_mem_file(out_dir / "gap_expected.mem", gap, bits=8)
    write_mem_file(out_dir / "fc_logits_expected.mem", logits, bits=32)

    if write_full_tensors:
        write_mem_file(out_dir / "conv1_expected.mem", conv1, bits=8)
        write_mem_file(out_dir / "pool1_expected.mem", pool1, bits=8)
        write_mem_file(out_dir / "conv2_expected.mem", conv2, bits=8)
        write_mem_file(out_dir / "pool2_expected.mem", pool2, bits=8)

    meta = {
        "sample_id": sample_path.name,
        "true_class": true_class,
        "predicted_class": pred,
        "predicted_class_name": CLASS_NAMES[pred],
        "maximum_logit": max_logit,
        "logits": [int(x) for x in logits],
        "gap": [int(x) for x in gap],
        "tie_breaking": "lowest_index_strict_greater",
        "logit_datatype": "int32",
        "logit_width": 32,
        "tensor_stats": {
            "input": _stats(inp),
            "conv1_relu": _stats(conv1),
            "pool1": _stats(pool1),
            "conv2_relu": _stats(conv2),
            "pool2": _stats(pool2),
            "gap": _stats(gap),
            "logits": _stats(logits),
        },
        "shapes": {
            "input": [3, 32, 32],
            "conv1": [16, 32, 32],
            "pool1": [16, 16, 16],
            "conv2": [32, 16, 16],
            "pool2": [32, 8, 8],
            "gap": [32],
            "logits": [5],
        },
        "note": (
            "Trained channel counts are 16/32 (not the obsolete 8/16 prompt sketches). "
            "Conv1/Conv2 expected tensors are post-ReLU."
        ),
    }
    (out_dir / "inference_expected.json").write_text(json.dumps(meta, indent=2) + "\n")
    summary_lines = [
        f"sample: {sample_path.name}",
        f"true_class: {true_class} ({CLASS_NAMES[true_class]})",
        f"predicted_class: {pred} ({CLASS_NAMES[pred]})",
        f"maximum_logit: {max_logit}",
        f"logits: {[int(x) for x in logits]}",
        f"gap checksum: {meta['tensor_stats']['gap']['checksum']}",
        f"conv1 checksum: {meta['tensor_stats']['conv1_relu']['checksum']}",
        f"pool1 checksum: {meta['tensor_stats']['pool1']['checksum']}",
        f"conv2 checksum: {meta['tensor_stats']['conv2_relu']['checksum']}",
        f"pool2 checksum: {meta['tensor_stats']['pool2']['checksum']}",
        "",
    ]
    (out_dir / "inference_summary.txt").write_text("\n".join(summary_lines))
    return meta


def export_end_to_end(
    *,
    int8_root: str | Path = "software/exported_model/int8",
    sample_input: str | Path | None = None,
    output_dir: str | Path = "vectors/end_to_end",
) -> dict[str, Any]:
    root = resolve_path(int8_root)
    out = resolve_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if sample_input is None:
        primary = root / "test_vectors" / "sample_000"
    else:
        primary = resolve_path(sample_input)
        if primary.is_file():
            primary = primary.parent

    model = IntegerTinyCNN.from_export_directory(root)
    primary_meta = export_one_image(model, primary, out, write_full_tensors=True)

    dbg = out / "selected_debug_traces"
    dbg.mkdir(parents=True, exist_ok=True)
    (dbg / "README.txt").write_text(
        "Primary image intermediate checksums live in inference_expected.json.\n"
        "Full tensors: conv1/pool1/conv2/pool2/gap/fc_logits *.mem in parent.\n"
    )

    multi_root = out / "multi_image"
    multi_root.mkdir(parents=True, exist_ok=True)
    multi_meta: list[dict[str, Any]] = []
    for sample_id, expect_class in MULTI_SAMPLES:
        sample_path = root / "test_vectors" / sample_id
        img_dir = multi_root / sample_id
        meta = export_one_image(model, sample_path, img_dir, write_full_tensors=True)
        if meta["predicted_class"] != expect_class:
            raise AssertionError(
                f"{sample_id}: predicted {meta['predicted_class']} != {expect_class}"
            )
        multi_meta.append(
            {
                "sample_id": sample_id,
                "predicted_class": meta["predicted_class"],
                "maximum_logit": meta["maximum_logit"],
                "logits": meta["logits"],
                "checksums": {
                    k: meta["tensor_stats"][k]["checksum"]
                    for k in (
                        "input",
                        "conv1_relu",
                        "pool1",
                        "conv2_relu",
                        "pool2",
                        "gap",
                        "logits",
                    )
                },
            }
        )

    index = {
        "primary_sample": primary.name,
        "primary": primary_meta,
        "multi_image": multi_meta,
        "class_names": CLASS_NAMES,
        "trained_shapes": primary_meta["shapes"],
        "obsolete_prompt_shapes_note": (
            "Prompt text that lists 8/16 channels is obsolete; RTL uses 16/32."
        ),
    }
    (out / "end_to_end_index.json").write_text(json.dumps(index, indent=2) + "\n")
    (multi_root / "multi_image_index.json").write_text(
        json.dumps({"images": multi_meta, "class_names": CLASS_NAMES}, indent=2) + "\n"
    )
    print(f"Wrote end-to-end vectors under {out}")
    print(
        f"Primary {primary.name}: class={primary_meta['predicted_class']} "
        f"max_logit={primary_meta['maximum_logit']}"
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int8-root", default="software/exported_model/int8")
    parser.add_argument("--sample-input", default=None)
    parser.add_argument("--output-dir", default="vectors/end_to_end")
    args = parser.parse_args()
    export_end_to_end(
        int8_root=args.int8_root,
        sample_input=args.sample_input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
