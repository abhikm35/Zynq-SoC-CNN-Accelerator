from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from software.inference.numpy_inference import NumpyTinyCNN
from software.quantization.fixed_point import dequantize
from software.quantization.saturation import INT8_MAX, INT8_MIN
from software.utils.config import resolve_path


def compare_arrays(
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    if expected.shape != actual.shape:
        return {
            "match": False,
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            "max_abs_error": None,
            "mean_abs_error": None,
            "max_rel_error": None,
            "error": "shape_mismatch",
        }

    abs_error = np.abs(actual - expected)
    max_abs = float(abs_error.max()) if abs_error.size else 0.0
    mean_abs = float(abs_error.mean()) if abs_error.size else 0.0
    denom = np.maximum(np.abs(expected), atol)
    rel_error = abs_error / denom
    max_rel = float(rel_error.max()) if rel_error.size else 0.0
    flat_index = int(np.argmax(abs_error)) if abs_error.size else 0
    multi_index = np.unravel_index(flat_index, expected.shape) if expected.size else ()
    match = bool(np.allclose(actual, expected, rtol=rtol, atol=atol))
    return {
        "match": match,
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "max_rel_error": max_rel,
        "largest_error_index": [int(value) for value in multi_index],
        "expected_value_at_error": float(expected.flat[flat_index]) if expected.size else None,
        "actual_value_at_error": float(actual.flat[flat_index]) if actual.size else None,
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or x.shape != y.shape:
        return None
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return None
    return float(np.dot(x, y) / denom)


def compare_float32(
    *,
    export_directory: Path,
    activations_directory: Path,
    output_path: Path,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    model = NumpyTinyCNN.from_export_directory(export_directory)
    sample_dirs = sorted(
        path
        for path in activations_directory.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    )
    if not sample_dirs:
        raise FileNotFoundError(
            f"No exported activation sample directories found under {activations_directory}"
        )

    layer_names = list(model.metadata["layer_order"])
    report: dict[str, Any] = {
        "mode": "float32",
        "export_directory": str(export_directory),
        "activations_directory": str(activations_directory),
        "rtol": rtol,
        "atol": atol,
        "comparison_date": datetime.now(timezone.utc).isoformat(),
        "samples": [],
        "all_predictions_match": True,
        "all_layers_match": True,
    }

    for sample_dir in sample_dirs:
        expected_input = np.load(sample_dir / "input.npy")
        numpy_scores, numpy_intermediates = model.forward_with_intermediates(
            expected_input
        )
        sample_result: dict[str, Any] = {
            "sample_id": sample_dir.name,
            "layers": {},
            "predicted_class_match": False,
        }

        for layer_name in layer_names:
            expected_path = sample_dir / f"{layer_name}.npy"
            if not expected_path.is_file():
                raise FileNotFoundError(f"Missing exported activation: {expected_path}")
            expected = np.load(expected_path)
            actual = numpy_intermediates[layer_name]
            comparison = compare_arrays(
                expected,
                actual,
                rtol=rtol,
                atol=atol,
            )
            sample_result["layers"][layer_name] = comparison
            if not comparison["match"]:
                report["all_layers_match"] = False
                print(
                    f"MISMATCH sample={sample_dir.name} layer={layer_name} "
                    f"max_abs={comparison['max_abs_error']} "
                    f"index={comparison.get('largest_error_index')} "
                    f"expected={comparison.get('expected_value_at_error')} "
                    f"actual={comparison.get('actual_value_at_error')}"
                )
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Layer mismatch for {sample_dir.name}/{layer_name}",
                )

        expected_scores = np.load(sample_dir / "scores.npy")
        expected_class = int(np.argmax(expected_scores, axis=1)[0])
        actual_class = int(np.argmax(numpy_scores, axis=1)[0])
        sample_result["expected_class"] = expected_class
        sample_result["numpy_class"] = actual_class
        sample_result["predicted_class_match"] = expected_class == actual_class
        if not sample_result["predicted_class_match"]:
            report["all_predictions_match"] = False
            raise AssertionError(
                f"Prediction mismatch for {sample_dir.name}: "
                f"expected={expected_class}, numpy={actual_class}"
            )
        report["samples"].append(sample_result)
        print(
            f"OK {sample_dir.name}: all layers matched, predicted class={actual_class}"
        )

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote comparison report to {output_path}")
    return report


def compare_int8(
    *,
    float32_export_directory: Path,
    float32_activations_directory: Path,
    int8_export_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    from software.inference.integer_inference import IntegerTinyCNN

    float_model = NumpyTinyCNN.from_export_directory(float32_export_directory)
    int_model = IntegerTinyCNN.from_export_directory(int8_export_directory)
    scales = int_model.scales

    scale_map = {
        "input": scales["input"],
        "conv1": scales["conv1_output"],
        "relu1": scales["relu1"],
        "pool1": scales["pool1"],
        "conv2": scales["conv2_output"],
        "relu2": scales["relu2"],
        "pool2": scales["pool2"],
        "global_average_pool": scales["gap"],
        "flatten": scales["flatten"],
        "scores": scales["classifier_output"],
    }

    # Compare against folded BN outputs for conv stages when available.
    float_compare_alias = {
        "conv1": "bn1",
        "conv2": "bn2",
    }

    sample_dirs = sorted(
        path
        for path in float32_activations_directory.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    )
    report: dict[str, Any] = {
        "mode": "int8",
        "float32_export_directory": str(float32_export_directory),
        "int8_export_directory": str(int8_export_directory),
        "comparison_date": datetime.now(timezone.utc).isoformat(),
        "samples": [],
        "prediction_agreement_count": 0,
        "prediction_total": 0,
        "deterministic_repeat_ok": True,
    }

    compare_layers = [
        "input",
        "conv1",
        "relu1",
        "pool1",
        "conv2",
        "relu2",
        "pool2",
        "global_average_pool",
        "flatten",
        "scores",
    ]

    for sample_dir in sample_dirs:
        float_input = np.load(sample_dir / "input.npy")
        float_scores, float_intermediates = float_model.forward_with_intermediates(
            float_input
        )
        int_scores, int_intermediates = int_model.forward_with_intermediates(float_input)
        int_scores_2, int_intermediates_2 = int_model.forward_with_intermediates(
            float_input
        )

        for name, tensor in int_intermediates.items():
            if not np.array_equal(tensor, int_intermediates_2[name]):
                report["deterministic_repeat_ok"] = False
                raise AssertionError(f"Non-deterministic integer output for {name}")

        pytorch_scores = np.load(sample_dir / "scores.npy")
        pytorch_class = int(np.argmax(pytorch_scores, axis=1)[0])
        float_class = int(np.argmax(float_scores, axis=1)[0])
        int_class = int(np.argmax(int_scores, axis=1)[0])

        sample_result: dict[str, Any] = {
            "sample_id": sample_dir.name,
            "pytorch_class": pytorch_class,
            "float32_numpy_class": float_class,
            "int8_class": int_class,
            "classes_agree": pytorch_class == float_class == int_class,
            "layers": {},
        }
        report["prediction_total"] += 1
        if sample_result["classes_agree"]:
            report["prediction_agreement_count"] += 1

        for layer_name in compare_layers:
            int_tensor = int_intermediates[layer_name]
            alias = float_compare_alias.get(layer_name, layer_name)
            if alias in float_intermediates:
                float_tensor = float_intermediates[alias]
            else:
                float_tensor = float_intermediates[layer_name]
            # Prefer exported pytorch tensor when present for the alias.
            exported_path = sample_dir / f"{alias}.npy"
            if exported_path.is_file():
                pytorch_tensor = np.load(exported_path)
            else:
                pytorch_tensor = float_tensor

            scale = scale_map[layer_name]
            dequant = dequantize(int_tensor, scale, zero_point=0)
            abs_err = np.abs(dequant.astype(np.float64) - pytorch_tensor.astype(np.float64))
            sat = int(np.count_nonzero((int_tensor <= INT8_MIN) | (int_tensor >= INT8_MAX)))
            sample_result["layers"][layer_name] = {
                "shape": list(int_tensor.shape),
                "float_min": float(np.min(pytorch_tensor)),
                "float_max": float(np.max(pytorch_tensor)),
                "integer_min": int(np.min(int_tensor)),
                "integer_max": int(np.max(int_tensor)),
                "saturation_count": sat,
                "max_abs_dequant_error": float(abs_err.max()) if abs_err.size else 0.0,
                "mean_abs_error": float(abs_err.mean()) if abs_err.size else 0.0,
                "cosine_similarity": cosine_similarity(dequant, pytorch_tensor),
            }

        report["samples"].append(sample_result)
        print(
            f"{sample_dir.name}: pytorch={pytorch_class} float={float_class} "
            f"int8={int_class} agree={sample_result['classes_agree']}"
        )

    report["prediction_agreement_rate"] = (
        report["prediction_agreement_count"] / report["prediction_total"]
        if report["prediction_total"]
        else 0.0
    )
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote INT8 comparison report to {output_path}")
    print(
        f"agreement={report['prediction_agreement_rate'] * 100:.1f}% "
        f"deterministic={report['deterministic_repeat_ok']}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare NumPy TinyCNN inference against PyTorch exports"
    )
    parser.add_argument(
        "--mode",
        choices=["float32", "int8"],
        default="float32",
    )
    parser.add_argument(
        "--export-directory",
        default="software/exported_model/float32",
        help="Directory containing exported float32 parameters",
    )
    parser.add_argument(
        "--activations-directory",
        default="software/exported_model/test_vectors/float32",
        help="Directory containing exported PyTorch intermediate activations",
    )
    parser.add_argument(
        "--int8-export-directory",
        default="software/exported_model/int8",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the comparison JSON report",
    )
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "int8":
        output_path = resolve_path(
            args.output or "results/comparisons/float32_vs_int8.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compare_int8(
            float32_export_directory=resolve_path(args.export_directory),
            float32_activations_directory=resolve_path(args.activations_directory),
            int8_export_directory=resolve_path(args.int8_export_directory),
            output_path=output_path,
        )
        return

    output_path = resolve_path(
        args.output or "results/comparisons/pytorch_vs_numpy_float32.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    compare_float32(
        export_directory=resolve_path(args.export_directory),
        activations_directory=resolve_path(args.activations_directory),
        output_path=output_path,
        rtol=args.rtol,
        atol=args.atol,
    )


if __name__ == "__main__":
    main()
