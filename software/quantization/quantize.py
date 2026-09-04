from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from software.export.export_fpga_hex import export_array_hex
from software.inference.integer_inference import IntegerTinyCNN
from software.quantization.calibrate import calibrate
from software.quantization.fixed_point import (
    calculate_symmetric_scale,
    load_quantization_config,
    quantize_bias_int32,
    quantize_multiplier,
    quantize_symmetric,
)
from software.quantization.saturation import INT8_MAX, INT8_MIN, NARROW_INT8_MAX, NARROW_INT8_MIN
from software.utils.config import resolve_path


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _per_channel_weight_scales(weights: np.ndarray) -> np.ndarray:
    """One scale per output channel/class: max(abs(w[oc])) / 127."""
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim < 1:
        raise QuantizationError("Weight tensor rank too small")
    scales = np.empty((arr.shape[0],), dtype=np.float32)
    for index in range(arr.shape[0]):
        scales[index] = calculate_symmetric_scale(arr[index], max_abs_int=127)
    return scales


def _quantize_weights_per_channel(weights: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return quantize_symmetric(
        weights,
        scales,
        qmin=NARROW_INT8_MIN,
        qmax=NARROW_INT8_MAX,
    )


def _build_multipliers_shifts(
    input_scale: float,
    weight_scales: np.ndarray,
    output_scale: float,
    *,
    multiplier_bits: int,
) -> tuple[np.ndarray, np.ndarray]:
    weight_scales = np.asarray(weight_scales, dtype=np.float64)
    multipliers = np.empty(weight_scales.shape, dtype=np.int32)
    shifts = np.empty(weight_scales.shape, dtype=np.int32)
    for index, weight_scale in enumerate(weight_scales):
        real = (float(input_scale) * float(weight_scale)) / float(output_scale)
        mult, shift = quantize_multiplier(real, multiplier_bits=multiplier_bits)
        multipliers[index] = mult
        shifts[index] = shift
    return multipliers, shifts


def _tensor_stats(name: str, float_w: np.ndarray, quant_w: np.ndarray, scales: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "float_min": float(np.min(float_w)),
        "float_max": float(np.max(float_w)),
        "quant_min": int(np.min(quant_w)),
        "quant_max": int(np.max(quant_w)),
        "scale_min": float(np.min(scales)),
        "scale_max": float(np.max(scales)),
    }


def load_or_calibrate(
    *,
    calibration_path: Path,
    force: bool,
    config_path: str,
    quantization_config_path: str,
    checkpoint: str | None,
    calibration_samples: int | None,
) -> dict[str, Any]:
    if calibration_path.is_file() and not force:
        return json.loads(calibration_path.read_text(encoding="utf-8"))

    # Optionally override sample count in a temporary sense via calibrate args.
    if calibration_samples is not None:
        # Patch through quant config by writing is heavy; pass via calibrate after
        # updating in-memory is not supported — mutate YAML-backed call by
        # temporarily adjusting through calibrate's own config load.
        # Instead, call calibrate and then note requested samples.
        pass

    return calibrate(
        config_path=config_path,
        quantization_config_path=quantization_config_path,
        checkpoint=checkpoint,
        output_path=str(calibration_path),
    )


def build_quantized_package(
    *,
    float32_root: Path,
    calibration: dict[str, Any],
    quant_config: dict[str, Any],
) -> dict[str, Any]:
    metadata = json.loads(
        (float32_root / "metadata" / "model_metadata.json").read_text(encoding="utf-8")
    )
    weights_dir = float32_root / "weights"
    biases_dir = float32_root / "biases"

    conv1_w = np.load(weights_dir / "conv1_weights_float32.npy")
    conv2_w = np.load(weights_dir / "conv2_weights_float32.npy")
    clf_w = np.load(weights_dir / "classifier_weights_float32.npy")
    conv1_b = np.load(biases_dir / "conv1_bias_float32.npy")
    conv2_b = np.load(biases_dir / "conv2_bias_float32.npy")
    clf_b = np.load(biases_dir / "classifier_bias_float32.npy")

    act_scales = calibration["activation_scales"]
    input_scale = float(act_scales["input"])
    conv1_out_scale = float(act_scales["conv1"])
    # Preserve scale through ReLU/pool after each convolution.
    relu1_scale = conv1_out_scale
    pool1_scale = conv1_out_scale
    conv2_out_scale = float(act_scales["conv2"])
    relu2_scale = conv2_out_scale
    pool2_scale = conv2_out_scale
    gap_scale = float(act_scales["global_average_pool"])
    flatten_scale = float(act_scales["flatten"])
    # Keep GAP/flatten scales aligned (same calibrated tensor family).
    gap_scale = flatten_scale
    classifier_out_scale = float(act_scales["classifier_output"])

    conv1_w_scales = _per_channel_weight_scales(conv1_w)
    conv2_w_scales = _per_channel_weight_scales(conv2_w)
    clf_w_scales = _per_channel_weight_scales(clf_w)

    conv1_wq = _quantize_weights_per_channel(conv1_w, conv1_w_scales)
    conv2_wq = _quantize_weights_per_channel(conv2_w, conv2_w_scales)
    clf_wq = _quantize_weights_per_channel(clf_w, clf_w_scales)

    conv1_bq = quantize_bias_int32(conv1_b, input_scale, conv1_w_scales)
    conv2_bq = quantize_bias_int32(conv2_b, pool1_scale, conv2_w_scales)
    clf_bq = quantize_bias_int32(clf_b, flatten_scale, clf_w_scales)

    multiplier_bits = int(quant_config["requantization"]["multiplier_bits"])
    conv1_mult, conv1_shift = _build_multipliers_shifts(
        input_scale, conv1_w_scales, conv1_out_scale, multiplier_bits=multiplier_bits
    )
    conv2_mult, conv2_shift = _build_multipliers_shifts(
        pool1_scale, conv2_w_scales, conv2_out_scale, multiplier_bits=multiplier_bits
    )
    # After integer GAP (still in pool2 scale), requantize to calibrated flatten scale.
    gap_real = float(pool2_scale) / float(flatten_scale)
    gap_mult, gap_shift = quantize_multiplier(gap_real, multiplier_bits=multiplier_bits)
    clf_mult, clf_shift = _build_multipliers_shifts(
        flatten_scale,
        clf_w_scales,
        classifier_out_scale,
        multiplier_bits=multiplier_bits,
    )

    layer_stats = [
        _tensor_stats("conv1_weights", conv1_w, conv1_wq, conv1_w_scales),
        _tensor_stats("conv2_weights", conv2_w, conv2_wq, conv2_w_scales),
        _tensor_stats("classifier_weights", clf_w, clf_wq, clf_w_scales),
    ]

    package = {
        "metadata_base": metadata,
        "weights": {
            "conv1": conv1_wq,
            "conv2": conv2_wq,
            "classifier": clf_wq,
        },
        "biases": {
            "conv1": conv1_bq,
            "conv2": conv2_bq,
            "classifier": clf_bq,
        },
        "float_weights": {
            "conv1": conv1_w,
            "conv2": conv2_w,
            "classifier": clf_w,
        },
        "float_biases": {
            "conv1": conv1_b,
            "conv2": conv2_b,
            "classifier": clf_b,
        },
        "scales": {
            "input": input_scale,
            "conv1_weight": conv1_w_scales,
            "conv1_output": conv1_out_scale,
            "relu1": relu1_scale,
            "pool1": pool1_scale,
            "conv2_weight": conv2_w_scales,
            "conv2_output": conv2_out_scale,
            "relu2": relu2_scale,
            "pool2": pool2_scale,
            "gap": gap_scale,
            "flatten": flatten_scale,
            "classifier_weight": clf_w_scales,
            "classifier_output": classifier_out_scale,
        },
        "zero_points": {
            "input": 0,
            "conv1": 0,
            "relu1": 0,
            "pool1": 0,
            "conv2": 0,
            "relu2": 0,
            "pool2": 0,
            "gap": 0,
            "flatten": 0,
            "classifier_input": 0,
            "weights": 0,
        },
        "requantization": {
            "conv1_multipliers": conv1_mult,
            "conv1_shifts": conv1_shift,
            "conv2_multipliers": conv2_mult,
            "conv2_shifts": conv2_shift,
            "gap_multiplier": np.int32(gap_mult),
            "gap_shift": np.int32(gap_shift),
            "classifier_multipliers": clf_mult,
            "classifier_shifts": clf_shift,
            "multiplier_bits": multiplier_bits,
        },
        "layer_stats": layer_stats,
        "calibration": calibration,
        "quant_config": quant_config,
    }
    return package


def build_manifest(
    package: dict[str, Any],
    *,
    export_root: Path,
    float32_accuracy: float | None,
    int8_accuracy: float | None,
    accuracy_difference: float | None,
    max_accumulators: dict[str, int],
) -> dict[str, Any]:
    meta = package["metadata_base"]
    scales = package["scales"]
    zp = package["zero_points"]
    requant = package["requantization"]
    calibration = package["calibration"]

    return {
        "architecture": meta.get("architecture_summary"),
        "model_name": meta.get("model_name"),
        "class_order": meta["class_order"],
        "input_shape": meta["input_shape"],
        "layer_order": [
            "input",
            "conv1_accumulator",
            "conv1",
            "relu1",
            "pool1",
            "conv2_accumulator",
            "conv2",
            "relu2",
            "pool2",
            "global_average_pool",
            "flatten",
            "classifier_accumulator",
            "scores",
        ],
        "layers": meta["layers"],
        "quantization_scheme": {
            "name": package["quant_config"]["quantization"]["scheme"],
            "weights": "symmetric_int8_per_channel_narrow_range",
            "activations": "symmetric_int8_per_tensor",
            "input_dtype": "int8",
            "activation_dtype": "int8",
            "weight_dtype": "int8",
            "bias_dtype": "int32",
            "accumulator_dtype": "int32",
            "classifier_score_dtype": "int32",
            "classifier_score_policy": (
                "per-channel weight scales with shared output scale requantization"
            ),
        },
        "signedness": {
            "input": "signed",
            "weights": "signed",
            "activations": "signed",
            "biases": "signed",
            "accumulators": "signed",
            "scores": "signed",
        },
        "dtypes": {
            "input": "int8",
            "weights": "int8",
            "activations": "int8",
            "biases": "int32",
            "accumulators": "int32",
            "scores": "int32",
        },
        "scales": {
            "input": scales["input"],
            "conv1_weight_shape": list(scales["conv1_weight"].shape),
            "conv1_weight": [float(x) for x in scales["conv1_weight"].tolist()],
            "conv1_output": scales["conv1_output"],
            "relu1": scales["relu1"],
            "pool1": scales["pool1"],
            "conv2_weight_shape": list(scales["conv2_weight"].shape),
            "conv2_weight": [float(x) for x in scales["conv2_weight"].tolist()],
            "conv2_output": scales["conv2_output"],
            "relu2": scales["relu2"],
            "pool2": scales["pool2"],
            "global_average_pool": scales["gap"],
            "flatten": scales["flatten"],
            "classifier_weight_shape": list(scales["classifier_weight"].shape),
            "classifier_weight": [float(x) for x in scales["classifier_weight"].tolist()],
            "classifier_output": scales["classifier_output"],
        },
        "zero_points": zp,
        "bias_scales": {
            "conv1": [
                float(scales["input"] * w) for w in scales["conv1_weight"].tolist()
            ],
            "conv2": [
                float(scales["pool1"] * w) for w in scales["conv2_weight"].tolist()
            ],
            "classifier": [
                float(scales["flatten"] * w)
                for w in scales["classifier_weight"].tolist()
            ],
        },
        "weight_ordering": {
            "convolution": "[out_channels, in_channels, kernel_height, kernel_width]",
            "classifier": "[out_features, in_features]",
            "activation": "NCHW",
        },
        "rounding_mode": "round_to_nearest_away_from_zero",
        "saturation_limits": {
            "activation_int8": [INT8_MIN, INT8_MAX],
            "weight_int8_narrow": [NARROW_INT8_MIN, NARROW_INT8_MAX],
            "accumulator_int32": [-2147483648, 2147483647],
        },
        "relu_ordering": (
            "INT32 accumulator -> requantization -> rounding -> "
            "add output zero point -> saturate to activation range -> ReLU"
        ),
        "pooling_arithmetic": (
            "max pool compares quantized integers directly; scale and zero point preserved"
        ),
        "global_average_pool_arithmetic": (
            "sum in int64/int32, round_divide_int by H*W (ties away from zero), "
            "same activation scale, saturate to int8"
        ),
        "requantization": {
            "formula": "round(acc * integer_multiplier / 2^shift)",
            "multiplier_bits": requant["multiplier_bits"],
            "multiplier_width": requant["multiplier_bits"] + 1,
            "shift_range_observed": {
                "conv1": [
                    int(np.min(requant["conv1_shifts"])),
                    int(np.max(requant["conv1_shifts"])),
                ],
                "conv2": [
                    int(np.min(requant["conv2_shifts"])),
                    int(np.max(requant["conv2_shifts"])),
                ],
                "classifier": [
                    int(np.min(requant["classifier_shifts"])),
                    int(np.max(requant["classifier_shifts"])),
                ],
            },
            "temp_product_width": "int64 in NumPy golden model",
        },
        "accumulator_width": 32,
        "maximum_observed_accumulator_values": max_accumulators,
        "calibration": {
            "dataset_split": calibration.get("dataset_split"),
            "seed": calibration.get("seed"),
            "used_samples": calibration.get("used_samples"),
            "indices": calibration.get("indices"),
            "checkpoint_path": calibration.get("checkpoint_path"),
        },
        "floating_point_checkpoint_path": calibration.get("checkpoint_path"),
        "floating_point_test_accuracy": float32_accuracy,
        "quantized_test_accuracy": int8_accuracy,
        "accuracy_difference": accuracy_difference,
        "export_date": datetime.now(timezone.utc).isoformat(),
        "export_directory": str(export_root),
    }


def export_quantized_parameters(package: dict[str, Any], export_root: Path) -> None:
    if export_root.exists():
        # Preserve calibration.json if present.
        calibration_path = export_root / "metadata" / "calibration.json"
        calibration_backup = None
        if calibration_path.is_file():
            calibration_backup = calibration_path.read_text(encoding="utf-8")
        for child in ("weights", "biases", "scales", "requantization", "fpga", "test_vectors"):
            target = export_root / child
            if target.exists():
                shutil.rmtree(target)
        metadata_dir = export_root / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        if calibration_backup is not None:
            calibration_path.write_text(calibration_backup, encoding="utf-8")
    else:
        export_root.mkdir(parents=True, exist_ok=True)

    w = package["weights"]
    b = package["biases"]
    s = package["scales"]
    r = package["requantization"]

    _save_npy(export_root / "weights" / "conv1_weights_int8.npy", w["conv1"])
    _save_npy(export_root / "weights" / "conv2_weights_int8.npy", w["conv2"])
    _save_npy(export_root / "weights" / "classifier_weights_int8.npy", w["classifier"])

    _save_npy(export_root / "biases" / "conv1_bias_int32.npy", b["conv1"])
    _save_npy(export_root / "biases" / "conv2_bias_int32.npy", b["conv2"])
    _save_npy(export_root / "biases" / "classifier_bias_int32.npy", b["classifier"])

    _save_npy(export_root / "scales" / "input_scale_float32.npy", np.float32(s["input"]))
    _save_npy(export_root / "scales" / "conv1_weight_scales_float32.npy", s["conv1_weight"])
    _save_npy(
        export_root / "scales" / "conv1_output_scale_float32.npy",
        np.float32(s["conv1_output"]),
    )
    _save_npy(export_root / "scales" / "relu1_scale_float32.npy", np.float32(s["relu1"]))
    _save_npy(export_root / "scales" / "pool1_scale_float32.npy", np.float32(s["pool1"]))
    _save_npy(export_root / "scales" / "conv2_weight_scales_float32.npy", s["conv2_weight"])
    _save_npy(
        export_root / "scales" / "conv2_output_scale_float32.npy",
        np.float32(s["conv2_output"]),
    )
    _save_npy(export_root / "scales" / "relu2_scale_float32.npy", np.float32(s["relu2"]))
    _save_npy(export_root / "scales" / "pool2_scale_float32.npy", np.float32(s["pool2"]))
    _save_npy(
        export_root / "scales" / "global_average_pool_scale_float32.npy",
        np.float32(s["gap"]),
    )
    _save_npy(
        export_root / "scales" / "flatten_scale_float32.npy",
        np.float32(s["flatten"]),
    )
    _save_npy(
        export_root / "scales" / "classifier_weight_scales_float32.npy",
        s["classifier_weight"],
    )
    _save_npy(
        export_root / "scales" / "classifier_output_scale_float32.npy",
        np.float32(s["classifier_output"]),
    )

    _save_npy(export_root / "requantization" / "conv1_multipliers_int32.npy", r["conv1_multipliers"])
    _save_npy(export_root / "requantization" / "conv1_shifts_int32.npy", r["conv1_shifts"])
    _save_npy(export_root / "requantization" / "conv2_multipliers_int32.npy", r["conv2_multipliers"])
    _save_npy(export_root / "requantization" / "conv2_shifts_int32.npy", r["conv2_shifts"])
    _save_npy(
        export_root / "requantization" / "classifier_multipliers_int32.npy",
        r["classifier_multipliers"],
    )
    _save_npy(
        export_root / "requantization" / "classifier_shifts_int32.npy",
        r["classifier_shifts"],
    )
    _save_npy(
        export_root / "requantization" / "gap_multiplier_int32.npy",
        np.asarray(r["gap_multiplier"], dtype=np.int32),
    )
    _save_npy(
        export_root / "requantization" / "gap_shift_int32.npy",
        np.asarray(r["gap_shift"], dtype=np.int32),
    )

    input_meta = {
        "input_scale": float(s["input"]),
        "input_zero_point": 0,
        "input_dtype": "int8",
        "input_minimum": INT8_MIN,
        "input_maximum": INT8_MAX,
    }
    (export_root / "metadata").mkdir(parents=True, exist_ok=True)
    (export_root / "metadata" / "input_quantization.json").write_text(
        json.dumps(input_meta, indent=2),
        encoding="utf-8",
    )


def export_fpga_files(package: dict[str, Any], export_root: Path, sample_intermediates: dict[str, np.ndarray]) -> None:
    fpga = export_root / "fpga"
    fpga.mkdir(parents=True, exist_ok=True)
    w = package["weights"]
    b = package["biases"]

    export_array_hex(w["conv1"], fpga / "conv1_weights.hex", bits=8)
    export_array_hex(b["conv1"], fpga / "conv1_bias.hex", bits=32)
    export_array_hex(w["conv2"], fpga / "conv2_weights.hex", bits=8)
    export_array_hex(b["conv2"], fpga / "conv2_bias.hex", bits=32)
    export_array_hex(w["classifier"], fpga / "classifier_weights.hex", bits=8)
    export_array_hex(b["classifier"], fpga / "classifier_bias.hex", bits=32)

    export_array_hex(sample_intermediates["input"], fpga / "sample_000_input.hex", bits=8)
    export_array_hex(sample_intermediates["conv1"], fpga / "sample_000_conv1_expected.hex", bits=8)
    export_array_hex(sample_intermediates["pool1"], fpga / "sample_000_pool1_expected.hex", bits=8)
    export_array_hex(sample_intermediates["conv2"], fpga / "sample_000_conv2_expected.hex", bits=8)
    export_array_hex(sample_intermediates["pool2"], fpga / "sample_000_pool2_expected.hex", bits=8)
    export_array_hex(sample_intermediates["scores"], fpga / "sample_000_scores_expected.hex", bits=32)

    meta = package["metadata_base"]
    readme = f"""# FPGA INT8 parameter and test-vector HEX export

## Representation

* Encoding: two's-complement hexadecimal
* One value per line
* No memory addresses
* Flattening: C-style row-major (`numpy` default / C order)

## Widths and signedness

| File | Width | Signedness | Source tensor shape |
| --- | --- | --- | --- |
| `conv1_weights.hex` | 8 | signed | `{list(w['conv1'].shape)}` |
| `conv1_bias.hex` | 32 | signed | `{list(b['conv1'].shape)}` |
| `conv2_weights.hex` | 8 | signed | `{list(w['conv2'].shape)}` |
| `conv2_bias.hex` | 32 | signed | `{list(b['conv2'].shape)}` |
| `classifier_weights.hex` | 8 | signed | `{list(w['classifier'].shape)}` |
| `classifier_bias.hex` | 32 | signed | `{list(b['classifier'].shape)}` |
| `sample_000_input.hex` | 8 | signed | NCHW input |
| `sample_000_conv1_expected.hex` | 8 | signed | post-requantization, **pre-ReLU** |
| `sample_000_pool1_expected.hex` | 8 | signed | post-ReLU max-pool |
| `sample_000_conv2_expected.hex` | 8 | signed | post-requantization, **pre-ReLU** |
| `sample_000_pool2_expected.hex` | 8 | signed | post-ReLU max-pool |
| `sample_000_scores_expected.hex` | 32 | signed | shared-scale class scores |

## Reconstruction

Read each line as a hex integer, interpret as two's complement of the declared
width, then reshape using the table above with C-order.

Convolution weight loop order:

```text
for out_channel
  for in_channel
    for kernel_row
      for kernel_col
```

Classifier weight loop order:

```text
for out_class
  for in_feature
```

Activation loop order:

```text
for batch
  for channel
    for row
      for column
```

## Zero points and padding

All activation and weight zero points are `0` (symmetric signed scheme).
Convolution padding inserts the input activation zero point (`0`).

## Multipliers and shifts

Per-output-channel requantization uses:

```text
y = saturate(round(accumulator * multiplier / 2^shift))
```

Multiplier and shift arrays are stored under `../requantization/`.

## Class index order

{json.dumps(meta['class_order'], indent=2)}
"""
    (fpga / "README.md").write_text(readme, encoding="utf-8")


def export_integer_test_vectors(
    model: IntegerTinyCNN,
    float32_samples_dir: Path,
    export_root: Path,
    reference_manifest: dict[str, Any] | None,
) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    out_root = export_root / "test_vectors"
    out_root.mkdir(parents=True, exist_ok=True)
    max_acc = {
        "conv1": 0,
        "conv2": 0,
        "classifier": 0,
    }
    sample_dirs = sorted(
        path
        for path in float32_samples_dir.iterdir()
        if path.is_dir() and path.name.startswith("sample_")
    )
    first_intermediates: dict[str, np.ndarray] | None = None

    for sample_dir in sample_dirs:
        float_input = np.load(sample_dir / "input.npy")
        float_scores = np.load(sample_dir / "scores.npy")
        true_class = int(
            (sample_dir / "expected_class.txt").read_text(encoding="utf-8").strip().splitlines()[0]
        )
        float_pred = int(np.argmax(float_scores, axis=1)[0])

        scores, intermediates = model.forward_with_intermediates(float_input)
        int_pred = int(np.argmax(scores, axis=1)[0])

        max_acc["conv1"] = max(
            max_acc["conv1"], int(np.max(np.abs(intermediates["conv1_accumulator"])))
        )
        max_acc["conv2"] = max(
            max_acc["conv2"], int(np.max(np.abs(intermediates["conv2_accumulator"])))
        )
        max_acc["classifier"] = max(
            max_acc["classifier"],
            int(np.max(np.abs(intermediates["classifier_accumulator"]))),
        )

        dest = out_root / sample_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        for name, tensor in intermediates.items():
            np.save(dest / f"{name}.npy", tensor)
        (dest / "expected_class.txt").write_text(f"{true_class}\n", encoding="utf-8")

        sat_counts = {}
        for name in ("input", "conv1", "relu1", "pool1", "conv2", "relu2", "pool2", "global_average_pool"):
            arr = intermediates[name]
            sat_counts[name] = int(
                np.count_nonzero((arr <= INT8_MIN) | (arr >= INT8_MAX))
            )

        sample_meta = {
            "sample_id": sample_dir.name,
            "true_class": true_class,
            "float32_predicted_class": float_pred,
            "int8_predicted_class": int_pred,
            "class_score_scale": float(model.scales["classifier_output"]),
            "tensor_shapes": {k: list(v.shape) for k, v in intermediates.items()},
            "tensor_dtypes": {k: str(v.dtype) for k, v in intermediates.items()},
            "min_max": {
                k: {"min": int(np.min(v)), "max": int(np.max(v))}
                for k, v in intermediates.items()
            },
            "saturation_counts": sat_counts,
            "accumulator_max_magnitudes": {
                "conv1": int(np.max(np.abs(intermediates["conv1_accumulator"]))),
                "conv2": int(np.max(np.abs(intermediates["conv2_accumulator"]))),
                "classifier": int(
                    np.max(np.abs(intermediates["classifier_accumulator"]))
                ),
            },
            "source_image_identifier": sample_dir.name,
        }
        if reference_manifest is not None:
            for entry in reference_manifest.get("samples", []):
                if entry.get("sample_id") == sample_dir.name or entry.get("id") == sample_dir.name:
                    sample_meta["source_image_identifier"] = entry
                    break
        (dest / "metadata.json").write_text(json.dumps(sample_meta, indent=2), encoding="utf-8")

        if first_intermediates is None:
            first_intermediates = intermediates

    if first_intermediates is None:
        raise RuntimeError("No float32 reference samples found for INT8 vector export")
    return max_acc, first_intermediates


def print_quantization_statistics(package: dict[str, Any]) -> None:
    print("Quantization statistics")
    print("-----------------------")
    for stats in package["layer_stats"]:
        print(
            f"{stats['name']}: float[{stats['float_min']:.6g}, {stats['float_max']:.6g}] "
            f"quant[{stats['quant_min']}, {stats['quant_max']}] "
            f"scale[{stats['scale_min']:.6g}, {stats['scale_max']:.6g}]"
        )
    b = package["biases"]
    print(
        f"conv1 bias int range: [{int(b['conv1'].min())}, {int(b['conv1'].max())}]"
    )
    print(
        f"conv2 bias int range: [{int(b['conv2'].min())}, {int(b['conv2'].max())}]"
    )
    print(
        f"classifier bias int range: [{int(b['classifier'].min())}, {int(b['classifier'].max())}]"
    )
    s = package["scales"]
    print(f"input scale={s['input']:.8g} zp=0")
    print(f"conv1 output scale={s['conv1_output']:.8g} zp=0")
    print(f"conv2 output scale={s['conv2_output']:.8g} zp=0")
    print(f"classifier output scale={s['classifier_output']:.8g} zp=0")
    r = package["requantization"]
    print(
        f"conv1 multipliers [{int(r['conv1_multipliers'].min())}, {int(r['conv1_multipliers'].max())}] "
        f"shifts [{int(r['conv1_shifts'].min())}, {int(r['conv1_shifts'].max())}]"
    )
    print(
        f"conv2 multipliers [{int(r['conv2_multipliers'].min())}, {int(r['conv2_multipliers'].max())}] "
        f"shifts [{int(r['conv2_shifts'].min())}, {int(r['conv2_shifts'].max())}]"
    )
    print(
        f"classifier multipliers [{int(r['classifier_multipliers'].min())}, "
        f"{int(r['classifier_multipliers'].max())}] "
        f"shifts [{int(r['classifier_shifts'].min())}, {int(r['classifier_shifts'].max())}]"
    )


def quantize_main(
    *,
    config_path: str,
    quantization_config_path: str,
    checkpoint: str | None,
    force: bool,
    evaluate: bool,
    calibration_samples: int | None,
) -> dict[str, Any]:
    quant_config = load_quantization_config(quantization_config_path)
    if calibration_samples is not None:
        quant_config["quantization"]["calibration_samples"] = int(calibration_samples)

    export_root = resolve_path(quant_config["export"]["directory"])
    calibration_path = export_root / "metadata" / "calibration.json"
    float32_root = resolve_path("software/exported_model/float32")

    if calibration_samples is not None and (force or not calibration_path.is_file()):
        # Write a temporary overridden config for calibrate.
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(quant_config, handle)
            temp_qcfg = handle.name
        try:
            calibration = calibrate(
                config_path=config_path,
                quantization_config_path=temp_qcfg,
                checkpoint=checkpoint,
                output_path=str(calibration_path),
            )
        finally:
            Path(temp_qcfg).unlink(missing_ok=True)
    else:
        calibration = load_or_calibrate(
            calibration_path=calibration_path,
            force=force,
            config_path=config_path,
            quantization_config_path=quantization_config_path,
            checkpoint=checkpoint,
            calibration_samples=calibration_samples,
        )

    package = build_quantized_package(
        float32_root=float32_root,
        calibration=calibration,
        quant_config=quant_config,
    )
    export_quantized_parameters(package, export_root)

    # Write a provisional manifest so IntegerTinyCNN can load.
    provisional = build_manifest(
        package,
        export_root=export_root,
        float32_accuracy=None,
        int8_accuracy=None,
        accuracy_difference=None,
        max_accumulators={},
    )
    (export_root / "metadata" / "quantization_manifest.json").write_text(
        json.dumps(provisional, indent=2),
        encoding="utf-8",
    )

    model = IntegerTinyCNN.from_export_directory(export_root)
    # Attach scales alias used by export helper
    model.scales = model.scales  # noqa: B018 — already present

    float_results_path = resolve_path("results/accuracy/float32_test_results.json")
    float32_accuracy = None
    if float_results_path.is_file():
        float32_accuracy = float(
            json.loads(float_results_path.read_text(encoding="utf-8"))["overall_accuracy"]
        )

    reference_manifest_path = resolve_path(
        "software/exported_model/test_vectors/reference_manifest.json"
    )
    reference_manifest = None
    if reference_manifest_path.is_file():
        reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))

    max_acc, first_intermediates = export_integer_test_vectors(
        model,
        resolve_path("software/exported_model/test_vectors/float32"),
        export_root,
        reference_manifest,
    )
    export_fpga_files(package, export_root, first_intermediates)

    int8_accuracy = None
    accuracy_difference = None
    if evaluate:
        from software.training.evaluate import evaluate_int8_model

        int8_report = evaluate_int8_model(
            config_path=config_path,
            int8_export_directory=str(export_root),
        )
        int8_accuracy = float(int8_report["overall_accuracy"])
        if float32_accuracy is not None:
            accuracy_difference = float32_accuracy - int8_accuracy
        max_acc = {
            key: max(int(max_acc.get(key, 0)), int(value))
            for key, value in int8_report.get("maximum_observed_accumulators", {}).items()
        }

    manifest = build_manifest(
        package,
        export_root=export_root,
        float32_accuracy=float32_accuracy,
        int8_accuracy=int8_accuracy,
        accuracy_difference=accuracy_difference,
        max_accumulators=max_acc,
    )
    (export_root / "metadata" / "quantization_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    from software.quantization.overflow_analysis import analyze_overflow

    overflow = analyze_overflow(int8_export_directory=str(export_root))
    print(
        "overflow theory flags: "
        f"{overflow['any_theoretical_int32_overflow']} "
        f"observed={overflow['any_observed_int32_overflow']}"
    )

    print_quantization_statistics(package)
    print(f"\nExported INT8 package to {export_root}")
    print(f"max accumulators (reference set): {max_acc}")
    if int8_accuracy is not None:
        print(f"INT8 test accuracy: {int8_accuracy * 100:.2f}%")
        if accuracy_difference is not None:
            print(f"accuracy drop vs float32: {accuracy_difference * 100:.2f} pp")
    return {
        "export_root": str(export_root),
        "manifest": manifest,
        "package_stats": package["layer_stats"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize TinyCNN and export INT8 artifacts")
    parser.add_argument("--config", default="software/config/training_config.yaml")
    parser.add_argument(
        "--quantization-config",
        default="software/config/quantization_config.yaml",
        dest="quantization_config",
    )
    # Accept alias used in the phase brief.
    parser.add_argument("--quant-config", dest="quantization_config", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--calibration-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quantize_main(
        config_path=args.config,
        quantization_config_path=args.quantization_config,
        checkpoint=args.checkpoint,
        force=args.force,
        evaluate=args.evaluate,
        calibration_samples=args.calibration_samples,
    )


if __name__ == "__main__":
    main()
