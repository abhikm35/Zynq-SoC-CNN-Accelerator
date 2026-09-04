from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB

from software.utils.config import resolve_path
from software.utils.reproducibility import seed_worker


class DatasetError(RuntimeError):
    """Raised when filtered dataset construction or splitting fails."""


Transform = Callable[[Image.Image], torch.Tensor]
SampleSource = str | Path | Image.Image


def build_training_transform(
    image_width: int = 32,
    image_height: int = 32,
    *,
    normalization_mean: Sequence[float] | None = None,
    normalization_std: Sequence[float] | None = None,
    rotation_degrees: float = 0.0,
    translate_fraction: float = 0.0,
    scale_min: float = 1.0,
    scale_max: float = 1.0,
    brightness: float = 0.0,
    contrast: float = 0.0,
) -> transforms.Compose:
    """Build training-only geometric/color augmentation and normalization."""
    operations: list[Callable[[Any], Any]] = [
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize((image_height, image_width)),
        transforms.RandomAffine(
            degrees=rotation_degrees,
            translate=(translate_fraction, translate_fraction),
            scale=(scale_min, scale_max),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
        transforms.ToTensor(),
    ]
    if normalization_mean is not None and normalization_std is not None:
        operations.append(
            transforms.Normalize(
                mean=list(normalization_mean),
                std=list(normalization_std),
            )
        )
    return transforms.Compose(operations)


def build_evaluation_transform(
    image_width: int = 32,
    image_height: int = 32,
    *,
    normalization_mean: Sequence[float] | None = None,
    normalization_std: Sequence[float] | None = None,
) -> transforms.Compose:
    """Build deterministic resize/tensor/normalization evaluation transforms."""
    operations: list[Callable[[Any], Any]] = [
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor(),
    ]
    if normalization_mean is not None and normalization_std is not None:
        operations.append(
            transforms.Normalize(
                mean=list(normalization_mean),
                std=list(normalization_std),
            )
        )
    return transforms.Compose(operations)


def extract_gtsrb_sample_metadata(dataset: Any) -> list[tuple[str, int]]:
    """Return GTSRB ``(path, original_label)`` pairs without opening images.

    Torchvision's ``GTSRB`` stores sample metadata on the private ``_samples``
    attribute. This helper isolates that compatibility dependency so the rest of
    the project never touches private torchvision attributes directly.
    """
    samples = getattr(dataset, "_samples", None)
    if samples is None:
        raise DatasetError(
            "Installed torchvision GTSRB dataset does not expose sample metadata "
            "via '_samples'. Update this compatibility helper for the installed "
            "torchvision version."
        )

    metadata: list[tuple[str, int]] = []
    for entry in samples:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise DatasetError(
                "Unexpected GTSRB sample metadata format; expected (path, label)"
            )
        path, label = entry
        metadata.append((str(path), int(label)))
    return metadata


def build_label_mapping(
    selected_classes: Sequence[dict[str, Any]],
) -> dict[int, int]:
    return {
        int(entry["gtsrb_id"]): int(entry["project_label"])
        for entry in selected_classes
    }


def class_names_from_config(selected_classes: Sequence[dict[str, Any]]) -> list[str]:
    ordered = sorted(selected_classes, key=lambda entry: int(entry["project_label"]))
    return [str(entry["name"]) for entry in ordered]


def filter_and_remap_samples(
    samples: Sequence[tuple[SampleSource, int]],
    gtsrb_to_project_label: dict[int, int],
) -> list[tuple[SampleSource, int]]:
    """Keep selected GTSRB IDs and remap them to project labels."""
    filtered: list[tuple[SampleSource, int]] = []
    for source, original_label in samples:
        project_label = gtsrb_to_project_label.get(int(original_label))
        if project_label is None:
            continue
        filtered.append((source, int(project_label)))

    if not filtered:
        selected = sorted(gtsrb_to_project_label)
        raise DatasetError(
            "No selected samples found for GTSRB class IDs "
            f"{selected}. Check the dataset split and label mapping."
        )
    return filtered


def extract_track_id(source: SampleSource) -> str | None:
    """Parse a GTSRB training track id from ``XXXXX_YYYYY.ppm`` filenames."""
    if isinstance(source, Image.Image):
        return None
    stem = Path(source).stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return parts[0]
    return None


class RemappedImageDataset(Dataset):
    """Dataset returning remapped traffic-sign tensors and labels."""

    def __init__(
        self,
        samples: Sequence[tuple[SampleSource, int]],
        transform: Transform | None = None,
        class_names: Sequence[str] | None = None,
        gtsrb_to_project_label: dict[int, int] | None = None,
    ) -> None:
        if not samples:
            raise DatasetError("Cannot construct a dataset from an empty sample list")
        self.samples = list(samples)
        self.transform = transform
        self.class_names = list(class_names) if class_names is not None else []
        self.gtsrb_to_project_label = dict(gtsrb_to_project_label or {})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source, label = self.samples[index]
        if isinstance(source, Image.Image):
            image = source.convert("RGB")
        else:
            image = Image.open(source).convert("RGB")

        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = transforms.ToTensor()(image)

        label_tensor = torch.tensor(int(label), dtype=torch.int64)
        return tensor, label_tensor

    @property
    def labels(self) -> list[int]:
        return [int(label) for _, label in self.samples]


def count_labels(labels: Sequence[int], num_classes: int) -> dict[int, int]:
    counts = {index: 0 for index in range(num_classes)}
    for label in labels:
        counts[int(label)] += 1
    return counts


def _split_class_indices_by_track(
    class_indices: list[int],
    samples: Sequence[tuple[SampleSource, int]],
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[tuple[list[int], list[int]], bool]:
    """Split one class with track groups when metadata supports it."""
    groups: dict[str, list[int]] = {}
    ungrouped: list[int] = []
    for index in class_indices:
        track_id = extract_track_id(samples[index][0])
        if track_id is None:
            ungrouped.append(index)
        else:
            groups.setdefault(track_id, []).append(index)

    # Fall back when track metadata is missing or too sparse for a group split.
    if ungrouped or len(groups) < 2:
        return (
            _split_class_indices_image_level(
                class_indices, validation_fraction, rng
            ),
            False,
        )

    track_ids = list(groups.keys())
    rng.shuffle(track_ids)

    total = len(class_indices)
    target_val = int(round(total * validation_fraction))
    target_val = min(max(target_val, 1), total - 1)

    best_k = 1
    best_diff = abs(sum(len(groups[track_id]) for track_id in track_ids[:1]) - target_val)
    for split_at in range(1, len(track_ids)):
        val_count = sum(len(groups[track_id]) for track_id in track_ids[:split_at])
        train_count = total - val_count
        if val_count < 1 or train_count < 1:
            continue
        diff = abs(val_count - target_val)
        if diff < best_diff:
            best_diff = diff
            best_k = split_at

    validation = [
        index
        for track_id in track_ids[:best_k]
        for index in groups[track_id]
    ]
    training = [
        index
        for track_id in track_ids[best_k:]
        for index in groups[track_id]
    ]

    if not training or not validation:
        return (
            _split_class_indices_image_level(
                class_indices, validation_fraction, rng
            ),
            False,
        )

    return (sorted(training), sorted(validation)), True


def _split_class_indices_image_level(
    class_indices: list[int],
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    if len(class_indices) < 2:
        raise DatasetError(
            "Each selected class needs at least two samples to create "
            "non-empty training and validation subsets"
        )

    shuffled = list(class_indices)
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * validation_fraction))
    val_count = min(max(val_count, 1), len(shuffled) - 1)
    validation = sorted(shuffled[:val_count])
    training = sorted(shuffled[val_count:])
    return training, validation


def stratified_train_validation_split(
    samples: Sequence[tuple[SampleSource, int]],
    *,
    num_classes: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], bool]:
    """Create a deterministic stratified split, preferring track groups.

    Returns:
        train_indices, validation_indices, used_group_aware_split
    """
    if not 0.0 < validation_fraction < 1.0:
        raise DatasetError("validation_fraction must be strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = {index: [] for index in range(num_classes)}
    for index, (_, label) in enumerate(samples):
        by_class[int(label)].append(index)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    group_aware = True

    for class_index in range(num_classes):
        class_indices = by_class[class_index]
        if len(class_indices) < 2:
            raise DatasetError(
                f"Class {class_index} has only {len(class_indices)} sample(s); "
                "need at least two samples for a stratified train/validation split"
            )

        (class_train, class_val), class_group_aware = _split_class_indices_by_track(
            class_indices, samples, validation_fraction, rng
        )
        if not class_group_aware:
            group_aware = False
        train_indices.extend(class_train)
        validation_indices.extend(class_val)

    train_set = set(train_indices)
    validation_set = set(validation_indices)
    if train_set & validation_set:
        raise DatasetError("Train and validation indices overlap")
    if train_set | validation_set != set(range(len(samples))):
        raise DatasetError("Train/validation split does not cover all samples")

    return sorted(train_indices), sorted(validation_indices), group_aware


@dataclass
class DatasetBundle:
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    class_names: list[str]
    class_counts: dict[str, dict[str, int]]
    label_mapping: dict[int, int]
    used_group_aware_split: bool
    split_notes: str


def _subset_label_counts(dataset: RemappedImageDataset, indices: Sequence[int], num_classes: int) -> dict[int, int]:
    labels = [dataset.labels[index] for index in indices]
    return count_labels(labels, num_classes)


def _format_counts(counts: dict[int, int], class_names: Sequence[str]) -> dict[str, int]:
    return {class_names[index]: int(counts[index]) for index in range(len(class_names))}


def load_filtered_gtsrb_samples(
    *,
    root: str | Path,
    split: str,
    download: bool,
    gtsrb_to_project_label: dict[int, int],
) -> list[tuple[str, int]]:
    try:
        dataset = GTSRB(root=str(root), split=split, download=download)
    except Exception as exc:  # noqa: BLE001 - surface download/path failures clearly
        raise DatasetError(
            f"Failed to load GTSRB split '{split}' from '{root}': {exc}"
        ) from exc

    metadata = extract_gtsrb_sample_metadata(dataset)
    return filter_and_remap_samples(metadata, gtsrb_to_project_label)


def build_dataset_bundle(config: dict[str, Any]) -> DatasetBundle:
    """Build filtered train/validation/test datasets from configuration."""
    data_cfg = config["data"]
    selected_classes = data_cfg["selected_classes"]
    class_names = class_names_from_config(selected_classes)
    label_mapping = build_label_mapping(selected_classes)
    num_classes = len(class_names)

    image_width = int(data_cfg["image_width"])
    image_height = int(data_cfg["image_height"])
    normalization = data_cfg["normalization"]
    augmentation = data_cfg["augmentation"]
    normalization_mean = normalization["mean"]
    normalization_std = normalization["std"]
    train_transform = build_training_transform(
        image_width,
        image_height,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        rotation_degrees=float(augmentation["rotation_degrees"]),
        translate_fraction=float(augmentation["translate_fraction"]),
        scale_min=float(augmentation["scale_min"]),
        scale_max=float(augmentation["scale_max"]),
        brightness=float(augmentation["brightness"]),
        contrast=float(augmentation["contrast"]),
    )
    eval_transform = build_evaluation_transform(
        image_width,
        image_height,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
    )

    root = resolve_path(data_cfg["root"])
    download = bool(data_cfg["download"])

    train_samples = load_filtered_gtsrb_samples(
        root=root,
        split="train",
        download=download,
        gtsrb_to_project_label=label_mapping,
    )
    test_samples = load_filtered_gtsrb_samples(
        root=root,
        split="test",
        download=download,
        gtsrb_to_project_label=label_mapping,
    )

    full_train_dataset = RemappedImageDataset(
        samples=train_samples,
        transform=train_transform,
        class_names=class_names,
        gtsrb_to_project_label=label_mapping,
    )
    # Validation uses the evaluation transform; rebuild sample view without augmentation.
    full_train_for_split = RemappedImageDataset(
        samples=train_samples,
        transform=eval_transform,
        class_names=class_names,
        gtsrb_to_project_label=label_mapping,
    )

    train_indices, validation_indices, used_group_aware_split = (
        stratified_train_validation_split(
            train_samples,
            num_classes=num_classes,
            validation_fraction=float(data_cfg["validation_fraction"]),
            seed=int(config["seed"]),
        )
    )

    if used_group_aware_split:
        split_notes = (
            "Train/validation split is stratified by class and group-aware by "
            "GTSRB track id when filename metadata is available."
        )
    else:
        split_notes = (
            "Train/validation split is deterministic and stratified by class at "
            "the image level. Group-aware track splitting was not applied for "
            "every class because track metadata was missing or too sparse."
        )

    # Training subset should use training transform; validation uses eval transform.
    train_dataset = Subset(
        RemappedImageDataset(
            samples=train_samples,
            transform=train_transform,
            class_names=class_names,
            gtsrb_to_project_label=label_mapping,
        ),
        train_indices,
    )
    validation_dataset = Subset(
        RemappedImageDataset(
            samples=train_samples,
            transform=eval_transform,
            class_names=class_names,
            gtsrb_to_project_label=label_mapping,
        ),
        validation_indices,
    )
    test_dataset = RemappedImageDataset(
        samples=test_samples,
        transform=eval_transform,
        class_names=class_names,
        gtsrb_to_project_label=label_mapping,
    )

    train_counts = _subset_label_counts(
        full_train_dataset, train_indices, num_classes
    )
    validation_counts = _subset_label_counts(
        full_train_for_split, validation_indices, num_classes
    )
    test_counts = count_labels(test_dataset.labels, num_classes)

    class_counts = {
        "train": _format_counts(train_counts, class_names),
        "validation": _format_counts(validation_counts, class_names),
        "test": _format_counts(test_counts, class_names),
    }

    return DatasetBundle(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        class_names=class_names,
        class_counts=class_counts,
        label_mapping=label_mapping,
        used_group_aware_split=used_group_aware_split,
        split_notes=split_notes,
    )


def print_class_counts(bundle: DatasetBundle) -> None:
    print("Per-class sample counts")
    print("-----------------------")
    for split_name in ("train", "validation", "test"):
        print(f"{split_name}:")
        for class_name, count in bundle.class_counts[split_name].items():
            print(f"  {class_name}: {count}")
        print(f"  total: {sum(bundle.class_counts[split_name].values())}")


def create_dataloaders(
    bundle: DatasetBundle,
    *,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool | None = None,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test dataloaders with the required shuffle policy."""
    if batch_size < 1:
        raise DatasetError("batch_size must be >= 1")
    if num_workers < 0:
        raise DatasetError("num_workers must be >= 0")

    if pin_memory is None:
        pin_memory = device is not None and device.type == "cuda"

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )
    validation_loader = DataLoader(
        bundle.validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        bundle.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    return train_loader, validation_loader, test_loader


def verify_training_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    *,
    normalized: bool = False,
) -> None:
    """Validate one training batch against the phase contract."""
    if images.dtype != torch.float32:
        raise DatasetError(f"Expected float32 images, got {images.dtype}")
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32):
        raise DatasetError(f"Expected image shape [N, 3, 32, 32], got {tuple(images.shape)}")
    if images.shape[0] > batch_size:
        raise DatasetError("Unexpected batch size larger than configured batch_size")
    if labels.dtype != torch.int64:
        raise DatasetError(f"Expected int64 labels, got {labels.dtype}")
    if labels.shape != (images.shape[0],):
        raise DatasetError(f"Expected label shape [{images.shape[0]}], got {tuple(labels.shape)}")
    if labels.numel() and (labels.min() < 0 or labels.max() > 4):
        raise DatasetError("Labels must be in the inclusive range [0, 4]")
    if not torch.isfinite(images).all():
        raise DatasetError("Images contain non-finite pixel values")
    if (
        not normalized
        and images.numel()
        and (images.min() < 0.0 or images.max() > 1.0)
    ):
        raise DatasetError("Pixel values must be in the inclusive range [0.0, 1.0]")
