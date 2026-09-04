from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from software.export.create_reference_set import create_reference_set
from software.export.export_activations import export_sample_activations
from software.export.export_biases import export_biases
from software.export.export_model_metadata import export_model_metadata
from software.export.export_weights import ensure_export_root, export_weights
from software.preprocessing.image_loader import build_dataset_bundle
from software.utils.checkpoint import checkpoint_path_from_config, restore_trained_model
from software.utils.config import load_training_config, resolve_path
from software.utils.reproducibility import select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export float32 TinyCNN weights, biases, and reference activations"
    )
    parser.add_argument(
        "--checkpoint",
        default="software/checkpoints/tiny_cnn_best.pth",
        help="Path to the trained checkpoint",
    )
    parser.add_argument(
        "--config",
        default="software/config/training_config.yaml",
        help="Path to the training configuration YAML",
    )
    parser.add_argument(
        "--output-directory",
        default="software/exported_model/float32",
        help="Root directory for float32 parameter exports",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=4,
        help="Number of deterministic reference images per class",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing float32 export directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    set_seed(int(config["seed"]))
    device = select_device(str(config["runtime"]["device"]))

    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.is_file():
        checkpoint_path = checkpoint_path_from_config(config)

    model, checkpoint, resolved_checkpoint = restore_trained_model(
        checkpoint_path,
        config,
        map_location=device,
    )
    model = model.to(device)

    export_root = ensure_export_root(args.output_directory, force=args.force)
    weight_paths = export_weights(model, export_root)
    bias_paths = export_biases(model, export_root)
    metadata_path = export_model_metadata(
        model,
        checkpoint,
        resolved_checkpoint,
        config,
        export_root,
    )

    bundle = build_dataset_bundle(config)
    selected = create_reference_set(
        model,
        bundle.test_dataset,
        bundle.class_names,
        output_directory="software/exported_model/test_vectors",
        samples_per_class=int(args.samples_per_class),
        seed=int(config["seed"]),
        device=device,
        checkpoint_path=str(resolved_checkpoint),
    )
    activation_dirs = export_sample_activations(
        model,
        bundle.test_dataset,
        selected,
        Path("software/exported_model/test_vectors/float32"),
        device,
    )

    summary = {
        "checkpoint": str(resolved_checkpoint),
        "export_root": str(export_root),
        "weight_files": sorted(str(path) for path in weight_paths.values()),
        "bias_files": sorted(str(path) for path in bias_paths.values()),
        "metadata": str(metadata_path),
        "reference_samples": len(selected),
        "activation_directories": [str(path) for path in activation_dirs],
        "class_order": list(checkpoint["class_names"]),
        "architecture": {
            "conv1_channels": int(
                config["model_config"]["model"]["conv1_output_channels"]
            ),
            "conv2_channels": int(
                config["model_config"]["model"]["conv2_output_channels"]
            ),
            "classifier_in_features": int(
                config["model_config"]["model"]["conv2_output_channels"]
            ),
            "num_classes": int(config["model_config"]["model"]["num_classes"]),
            "batch_normalization": True,
        },
    }
    summary_path = export_root / "metadata" / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Float32 export complete")
    print("-----------------------")
    print(f"checkpoint: {resolved_checkpoint}")
    print(f"export root: {export_root}")
    print(f"metadata: {metadata_path}")
    print(f"reference samples: {len(selected)}")
    print(f"activation dirs: {len(activation_dirs)}")
    print(
        "architecture: "
        f"conv1={summary['architecture']['conv1_channels']}, "
        f"conv2={summary['architecture']['conv2_channels']}, "
        f"classifier={summary['architecture']['classifier_in_features']}->"
        f"{summary['architecture']['num_classes']}, BatchNorm=True"
    )


if __name__ == "__main__":
    main()
