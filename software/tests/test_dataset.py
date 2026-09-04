from __future__ import annotations

from typing import Sequence

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from software.preprocessing.image_loader import (
    DatasetError,
    RemappedImageDataset,
    build_evaluation_transform,
    build_training_transform,
    create_dataloaders,
    filter_and_remap_samples,
    stratified_train_validation_split,
    verify_training_batch,
)
from software.preprocessing import image_loader as image_loader_module


CLASS_NAMES = [
    "stop",
    "yield",
    "no_entry",
    "speed_limit_30",
    "keep_right",
]

GTSRB_TO_PROJECT = {
    14: 0,
    13: 1,
    17: 2,
    1: 3,
    38: 4,
}


def _make_image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (40, 40), color=color)


def _fake_samples() -> list[tuple[Image.Image, int]]:
    # Include unselected GTSRB labels that must be dropped.
    samples: list[tuple[Image.Image, int]] = []
    color_cycle = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]
    selected_ids = [14, 13, 17, 1, 38]
    for gtsrb_id, color in zip(selected_ids, color_cycle):
        for _ in range(4):
            samples.append((_make_image(color), gtsrb_id))
    samples.append((_make_image((10, 10, 10)), 0))  # unselected
    samples.append((_make_image((20, 20, 20)), 99))  # unselected
    return samples


def test_filter_and_remap_keeps_selected_labels_only() -> None:
    filtered = filter_and_remap_samples(_fake_samples(), GTSRB_TO_PROJECT)
    assert len(filtered) == 20
    remapped_labels = sorted({label for _, label in filtered})
    assert remapped_labels == [0, 1, 2, 3, 4]


def test_empty_selected_subset_raises() -> None:
    with pytest.raises(DatasetError, match="No selected samples"):
        filter_and_remap_samples(
            [(_make_image((1, 2, 3)), 0), (_make_image((4, 5, 6)), 2)],
            GTSRB_TO_PROJECT,
        )


def test_dataset_tensor_contract() -> None:
    filtered = filter_and_remap_samples(_fake_samples(), GTSRB_TO_PROJECT)
    dataset = RemappedImageDataset(
        samples=filtered,
        transform=build_training_transform(32, 32),
        class_names=CLASS_NAMES,
        gtsrb_to_project_label=GTSRB_TO_PROJECT,
    )

    image, label = dataset[0]
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    assert label.dtype == torch.int64
    assert 0 <= int(label.item()) <= 4
    assert set(dataset.labels) == {0, 1, 2, 3, 4}


def test_evaluation_normalization_is_deterministic() -> None:
    transform = build_evaluation_transform(
        32,
        32,
        normalization_mean=[0.4, 0.35, 0.37],
        normalization_std=[0.29, 0.27, 0.28],
    )
    image = _make_image((128, 64, 192))
    first = transform(image)
    second = transform(image)

    assert torch.equal(first, second)
    assert first.shape == (3, 32, 32)
    assert first.dtype == torch.float32
    assert torch.isfinite(first).all()


def test_training_augmentation_preserves_tensor_contract() -> None:
    transform = build_training_transform(
        32,
        32,
        normalization_mean=[0.4, 0.35, 0.37],
        normalization_std=[0.29, 0.27, 0.28],
        rotation_degrees=8,
        translate_fraction=0.08,
        scale_min=0.9,
        scale_max=1.1,
        brightness=0.15,
        contrast=0.15,
    )
    tensor = transform(_make_image((128, 64, 192)))

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_train_validation_indices_do_not_overlap_and_are_deterministic() -> None:
    filtered = filter_and_remap_samples(_fake_samples(), GTSRB_TO_PROJECT)
    train_a, val_a, _ = stratified_train_validation_split(
        filtered,
        num_classes=5,
        validation_fraction=0.25,
        seed=42,
    )
    train_b, val_b, _ = stratified_train_validation_split(
        filtered,
        num_classes=5,
        validation_fraction=0.25,
        seed=42,
    )
    train_c, val_c, _ = stratified_train_validation_split(
        filtered,
        num_classes=5,
        validation_fraction=0.25,
        seed=7,
    )

    assert train_a == train_b
    assert val_a == val_b
    assert set(train_a).isdisjoint(set(val_a))
    assert set(train_a) | set(val_a) == set(range(len(filtered)))
    assert (train_a, val_a) != (train_c, val_c)


def test_group_aware_track_split_keeps_tracks_together(tmp_path) -> None:
    samples: list[tuple[str, int]] = []
    # Two tracks per class, three frames each, with GTSRB-style filenames.
    for class_label in range(5):
        class_dir = tmp_path / f"{class_label:05d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for track in (0, 1):
            for frame in range(3):
                named = class_dir / f"{track:05d}_{frame:05d}.ppm"
                Image.new("RGB", (8, 8), color=(class_label * 40, 0, 0)).save(named)
                samples.append((str(named), class_label))

    train_indices, val_indices, used_group_aware = stratified_train_validation_split(
        samples,
        num_classes=5,
        validation_fraction=0.5,
        seed=0,
    )
    assert used_group_aware

    def tracks_for(indices: Sequence[int]) -> set[tuple[int, str]]:
        result: set[tuple[int, str]] = set()
        for index in indices:
            path, label = samples[index]
            track = image_loader_module.extract_track_id(path)
            assert track is not None
            result.add((label, track))
        return result

    assert tracks_for(train_indices).isdisjoint(tracks_for(val_indices))


def test_training_batch_shapes() -> None:
    filtered = filter_and_remap_samples(_fake_samples(), GTSRB_TO_PROJECT)
    dataset = RemappedImageDataset(
        samples=filtered,
        transform=build_evaluation_transform(32, 32),
        class_names=CLASS_NAMES,
        gtsrb_to_project_label=GTSRB_TO_PROJECT,
    )
    train_indices, val_indices, _ = stratified_train_validation_split(
        filtered,
        num_classes=5,
        validation_fraction=0.25,
        seed=42,
    )

    from software.preprocessing.image_loader import DatasetBundle

    bundle = DatasetBundle(
        train_dataset=torch.utils.data.Subset(dataset, train_indices),
        validation_dataset=torch.utils.data.Subset(dataset, val_indices),
        test_dataset=dataset,
        class_names=CLASS_NAMES,
        class_counts={
            "train": {},
            "validation": {},
            "test": {},
        },
        label_mapping=GTSRB_TO_PROJECT,
        used_group_aware_split=False,
        split_notes="test",
    )

    train_loader, validation_loader, test_loader = create_dataloaders(
        bundle,
        batch_size=4,
        num_workers=0,
        seed=42,
        pin_memory=False,
    )
    assert isinstance(validation_loader, DataLoader)
    images, labels = next(iter(train_loader))
    verify_training_batch(images, labels, batch_size=4)
    assert images.shape[0] == 4
    assert labels.shape == (4,)

    # Deterministic order when shuffle is disabled.
    first = next(iter(test_loader))[1].tolist()
    second = next(iter(test_loader))[1].tolist()
    assert first == second
