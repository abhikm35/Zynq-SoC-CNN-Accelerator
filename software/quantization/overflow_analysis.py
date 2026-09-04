from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from software.utils.config import resolve_path


def theoretical_conv_accumulator_bound(
    *,
    in_channels: int,
    kernel_h: int,
    kernel_w: int,
    max_centered_input: int,
    max_centered_weight: int,
    max_bias_abs: int,
) -> int:
    mac_terms = int(in_channels) * int(kernel_h) * int(kernel_w)
    return int(mac_terms * max_centered_input * max_centered_weight + max_bias_abs)


def theoretical_linear_accumulator_bound(
    *,
    in_features: int,
    max_centered_input: int,
    max_centered_weight: int,
    max_bias_abs: int,
) -> int:
    return int(in_features * max_centered_input * max_centered_weight + max_bias_abs)


def required_signed_bits(magnitude: int) -> int:
    if magnitude < 0:
        raise ValueError("magnitude must be non-negative")
    if magnitude == 0:
        return 1
    # Need bits such that magnitude <= 2^(bits-1) - 1
    bits = 1
    while (1 << (bits - 1)) - 1 < magnitude:
        bits += 1
        if bits > 64:
            break
    return bits


def analyze_overflow(
    *,
    int8_export_directory: str = "software/exported_model/int8",
    output_path: str = "results/comparisons/int8_overflow_analysis.json",
) -> dict[str, Any]:
    root = resolve_path(int8_export_directory)
    manifest = json.loads(
        (root / "metadata" / "quantization_manifest.json").read_text(encoding="utf-8")
    )
    layers = manifest["layers"]

    conv1_w = np.load(root / "weights" / "conv1_weights_int8.npy")
    conv2_w = np.load(root / "weights" / "conv2_weights_int8.npy")
    clf_w = np.load(root / "weights" / "classifier_weights_int8.npy")
    conv1_b = np.load(root / "biases" / "conv1_bias_int32.npy")
    conv2_b = np.load(root / "biases" / "conv2_bias_int32.npy")
    clf_b = np.load(root / "biases" / "classifier_bias_int32.npy")

    # Symmetric zp=0: centered magnitude equals absolute quantized value.
    max_in = 128  # int8 activation after centering with zp=0 can be 128 in magnitude for -128
    max_w = 127

    theoretical = {
        "conv1": theoretical_conv_accumulator_bound(
            in_channels=int(layers["conv1"]["in_channels"]),
            kernel_h=int(layers["conv1"]["kernel_size"]),
            kernel_w=int(layers["conv1"]["kernel_size"]),
            max_centered_input=max_in,
            max_centered_weight=max_w,
            max_bias_abs=int(np.max(np.abs(conv1_b))),
        ),
        "conv2": theoretical_conv_accumulator_bound(
            in_channels=int(layers["conv2"]["in_channels"]),
            kernel_h=int(layers["conv2"]["kernel_size"]),
            kernel_w=int(layers["conv2"]["kernel_size"]),
            max_centered_input=max_in,
            max_centered_weight=max_w,
            max_bias_abs=int(np.max(np.abs(conv2_b))),
        ),
        "classifier": theoretical_linear_accumulator_bound(
            in_features=int(layers["classifier"]["in_features"]),
            max_centered_input=max_in,
            max_centered_weight=max_w,
            max_bias_abs=int(np.max(np.abs(clf_b))),
        ),
    }

    observed_calibration = dict(manifest.get("maximum_observed_accumulator_values", {}))
    observed_test = {}
    test_report = resolve_path("results/accuracy/int8_test_results.json")
    if test_report.is_file():
        payload = json.loads(test_report.read_text(encoding="utf-8"))
        observed_test = payload.get("maximum_observed_accumulators", {})

    int32_max = 2147483647
    report: dict[str, Any] = {
        "analysis_date": datetime.now(timezone.utc).isoformat(),
        "int32_range": [-2147483648, int32_max],
        "assumptions": {
            "max_centered_input": max_in,
            "max_centered_weight": max_w,
            "weight_arrays_checked": {
                "conv1_abs_max": int(np.max(np.abs(conv1_w))),
                "conv2_abs_max": int(np.max(np.abs(conv2_w))),
                "classifier_abs_max": int(np.max(np.abs(clf_w))),
            },
        },
        "layers": {},
        "any_theoretical_int32_overflow": False,
        "any_observed_int32_overflow": False,
    }

    for name, bound in theoretical.items():
        bits = required_signed_bits(bound)
        obs_cal = int(observed_calibration.get(name, 0) or 0)
        obs_test = int(observed_test.get(name, 0) or 0)
        layer_report = {
            "theoretical_maximum_magnitude": bound,
            "observed_calibration_or_reference_maximum": obs_cal,
            "observed_test_maximum": obs_test,
            "int32_limit": int32_max,
            "required_signed_bits_theoretical": bits,
            "theoretical_exceeds_int32": bound > int32_max,
            "observed_exceeds_int32": max(obs_cal, obs_test) > int32_max,
        }
        report["layers"][name] = layer_report
        report["any_theoretical_int32_overflow"] |= layer_report["theoretical_exceeds_int32"]
        report["any_observed_int32_overflow"] |= layer_report["observed_exceeds_int32"]

    out = resolve_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze INT8 accumulator overflow bounds")
    parser.add_argument(
        "--int8-export-directory",
        default="software/exported_model/int8",
    )
    parser.add_argument(
        "--output",
        default="results/comparisons/int8_overflow_analysis.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_overflow(
        int8_export_directory=args.int8_export_directory,
        output_path=args.output,
    )
    print("Overflow analysis")
    print("-----------------")
    for name, layer in report["layers"].items():
        print(
            f"{name}: theoretical={layer['theoretical_maximum_magnitude']} "
            f"bits={layer['required_signed_bits_theoretical']} "
            f"obs_ref={layer['observed_calibration_or_reference_maximum']} "
            f"obs_test={layer['observed_test_maximum']} "
            f"overflow_theory={layer['theoretical_exceeds_int32']}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
