from __future__ import annotations

import argparse

import torch

from software.preprocessing.image_loader import (
    build_dataset_bundle,
    create_dataloaders,
    print_class_counts,
    verify_training_batch,
)
from software.utils.config import load_training_config
from software.utils.reproducibility import select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the filtered five-class GTSRB dataset bundle"
    )
    parser.add_argument(
        "--config",
        default="software/config/training_config.yaml",
        help="Path to the training configuration YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    set_seed(int(config["seed"]))
    device = select_device(str(config["runtime"]["device"]))

    bundle = build_dataset_bundle(config)
    train_size = len(bundle.train_dataset)
    validation_size = len(bundle.validation_dataset)
    test_size = len(bundle.test_dataset)

    print("Dataset sizes")
    print("-------------")
    print(f"train: {train_size}")
    print(f"validation: {validation_size}")
    print(f"test: {test_size}")
    print()
    print_class_counts(bundle)
    print()
    print(bundle.split_notes)
    print()
    print("Label mapping (GTSRB ID -> project label)")
    print("-----------------------------------------")
    for gtsrb_id, project_label in sorted(bundle.label_mapping.items()):
        class_name = bundle.class_names[project_label]
        print(f"{gtsrb_id} -> {project_label} ({class_name})")

    sample_image, sample_label = bundle.train_dataset[0]
    label_value = int(sample_label.item())
    print()
    print("Single training sample")
    print("----------------------")
    print(f"image shape: {list(sample_image.shape)}")
    print(f"image dtype: {sample_image.dtype}")
    print(f"normalized value min: {float(sample_image.min()):.6f}")
    print(f"normalized value max: {float(sample_image.max()):.6f}")
    print(f"label: {label_value} ({bundle.class_names[label_value]})")

    train_loader, _, _ = create_dataloaders(
        bundle,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        seed=int(config["seed"]),
        device=device,
    )
    images, labels = next(iter(train_loader))
    verify_training_batch(
        images,
        labels,
        int(config["training"]["batch_size"]),
        normalized=True,
    )

    print()
    print("Training DataLoader batch")
    print("-------------------------")
    print(f"images shape: {list(images.shape)}")
    print(f"images dtype: {images.dtype}")
    print(f"labels shape: {list(labels.shape)}")
    print(f"labels dtype: {labels.dtype}")
    print(f"labels: {labels.tolist()}")


if __name__ == "__main__":
    main()
