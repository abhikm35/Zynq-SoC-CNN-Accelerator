from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from software.utils.config import resolve_path
from software.utils.reproducibility import set_seed


@dataclass
class ReferenceSample:
    sample_id: str
    dataset_index: int
    true_class_index: int
    true_class_name: str
    predicted_class_index: int
    predicted_class_name: str
    correct: bool
    scores: list[float]
    confidence: float
    source_identifier: str
    selection_reason: str


def _subset_base_and_index(dataset: Dataset, index: int) -> tuple[Any, int]:
    current: Any = dataset
    current_index = int(index)
    while hasattr(current, "dataset") and hasattr(current, "indices"):
        current_index = int(current.indices[current_index])
        current = current.dataset
    return current, current_index


def resolve_source_identifier(dataset: Dataset, index: int) -> str:
    base, base_index = _subset_base_and_index(dataset, index)
    if hasattr(base, "samples"):
        source, _ = base.samples[base_index]
        return str(source)
    return f"dataset_index_{index}"


def score_reference_candidates(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    class_names: Sequence[str],
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            image, label = dataset[index]
            label_value = int(label.item())
            scores = model(image.unsqueeze(0).to(device)).detach().cpu().squeeze(0)
            probabilities = torch.softmax(scores, dim=0)
            predicted = int(torch.argmax(scores).item())
            confidence = float(probabilities[predicted].item())
            records.append(
                {
                    "dataset_index": index,
                    "true_class_index": label_value,
                    "true_class_name": class_names[label_value],
                    "predicted_class_index": predicted,
                    "predicted_class_name": class_names[predicted],
                    "correct": predicted == label_value,
                    "scores": [float(value) for value in scores.tolist()],
                    "confidence": confidence,
                    "true_class_confidence": float(probabilities[label_value].item()),
                    "source_identifier": resolve_source_identifier(dataset, index),
                }
            )
    return records


def select_reference_samples(
    records: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    *,
    samples_per_class: int = 4,
    seed: int = 42,
) -> list[ReferenceSample]:
    """Deterministically select a fixed reference set.

    Selection policy per class:
    1. Highest-confidence correct prediction
    2. One misclassified sample when available, else lowest-confidence correct
    3. Lowest-confidence correct remaining sample
    4. Next remaining sample by (confidence, dataset_index) for visual variety
    """
    rng_order_key = seed
    selected: list[ReferenceSample] = []
    sample_counter = 0

    for class_index, class_name in enumerate(class_names):
        class_records = [
            record
            for record in records
            if int(record["true_class_index"]) == class_index
        ]
        if len(class_records) < samples_per_class:
            raise RuntimeError(
                f"Class {class_name} has only {len(class_records)} test samples; "
                f"need at least {samples_per_class}"
            )

        correct = [record for record in class_records if record["correct"]]
        incorrect = [record for record in class_records if not record["correct"]]
        correct_high = sorted(
            correct,
            key=lambda item: (-item["confidence"], item["dataset_index"]),
        )
        correct_low = sorted(
            correct,
            key=lambda item: (item["confidence"], item["dataset_index"]),
        )
        incorrect_sorted = sorted(
            incorrect,
            key=lambda item: (item["confidence"], item["dataset_index"]),
        )
        variety = sorted(
            class_records,
            key=lambda item: (
                item["dataset_index"] % max(samples_per_class, 1),
                item["dataset_index"],
                rng_order_key,
            ),
        )

        picks: list[dict[str, Any]] = []
        reasons: list[str] = []

        def add_pick(record: dict[str, Any], reason: str) -> None:
            if any(
                existing["dataset_index"] == record["dataset_index"] for existing in picks
            ):
                return
            if len(picks) >= samples_per_class:
                return
            picks.append(record)
            reasons.append(reason)

        if correct_high:
            add_pick(correct_high[0], "high_confidence_correct")
        if incorrect_sorted:
            add_pick(incorrect_sorted[0], "misclassified")
        elif correct_low:
            add_pick(correct_low[0], "low_confidence_correct_fallback")
        if correct_low:
            add_pick(correct_low[0], "low_confidence_correct")
        for record in variety:
            add_pick(record, "deterministic_variety")

        if len(picks) < samples_per_class:
            raise RuntimeError(
                f"Unable to select {samples_per_class} reference samples for {class_name}"
            )

        for record, reason in zip(picks, reasons):
            sample_id = f"sample_{sample_counter:03d}"
            selected.append(
                ReferenceSample(
                    sample_id=sample_id,
                    dataset_index=int(record["dataset_index"]),
                    true_class_index=int(record["true_class_index"]),
                    true_class_name=str(record["true_class_name"]),
                    predicted_class_index=int(record["predicted_class_index"]),
                    predicted_class_name=str(record["predicted_class_name"]),
                    correct=bool(record["correct"]),
                    scores=[float(value) for value in record["scores"]],
                    confidence=float(record["confidence"]),
                    source_identifier=str(record["source_identifier"]),
                    selection_reason=reason,
                )
            )
            sample_counter += 1

    return selected


def save_reference_set(
    model: nn.Module,
    dataset: Dataset,
    selected: Sequence[ReferenceSample],
    output_directory: str | Path,
    *,
    seed: int,
    checkpoint_path: str,
) -> Path:
    root = resolve_path(output_directory)
    image_dir = root / "source_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Save human-viewable RGB images before network normalization.
    from torchvision import transforms

    to_pil = transforms.ToPILImage()
    denorm_note = (
        "source_images are reconstructed from normalized tensors for inspection "
        "when original PIL access is unavailable; prefer identifiers in the manifest."
    )

    manifest_entries: list[dict[str, Any]] = []
    for sample in selected:
        image_tensor, _ = dataset[sample.dataset_index]
        # image_tensor is normalized; also keep original path-based RGB when possible.
        base, base_index = _subset_base_and_index(dataset, sample.dataset_index)
        filename = f"{sample.sample_id}_{sample.true_class_name}.png"
        destination = image_dir / filename
        if hasattr(base, "samples"):
            source, _ = base.samples[base_index]
            if not isinstance(source, Image.Image):
                Image.open(source).convert("RGB").resize((32, 32)).save(destination)
            else:
                source.convert("RGB").resize((32, 32)).save(destination)
        else:
            # Fallback: clip normalized tensor into a displayable image.
            display = image_tensor.detach().cpu().clone()
            display = display - display.min()
            if float(display.max()) > 0:
                display = display / display.max()
            to_pil(display).save(destination)

        manifest_entries.append(
            {
                "sample_id": sample.sample_id,
                "dataset_index": sample.dataset_index,
                "source_class": sample.true_class_name,
                "source_class_index": sample.true_class_index,
                "predicted_class": sample.predicted_class_name,
                "predicted_class_index": sample.predicted_class_index,
                "correct": sample.correct,
                "raw_class_scores": sample.scores,
                "confidence": sample.confidence,
                "source_image_filename": filename,
                "source_identifier": sample.source_identifier,
                "selection_reason": sample.selection_reason,
            }
        )

    manifest = {
        "seed": seed,
        "checkpoint_path": checkpoint_path,
        "samples_per_class": 4,
        "total_samples": len(selected),
        "denormalization_note": denorm_note,
        "samples": manifest_entries,
    }
    manifest_path = root / "reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def create_reference_set(
    model: nn.Module,
    test_dataset: Dataset,
    class_names: Sequence[str],
    *,
    output_directory: str | Path = "software/exported_model/test_vectors",
    samples_per_class: int = 4,
    seed: int = 42,
    device: torch.device | None = None,
    checkpoint_path: str = "software/checkpoints/tiny_cnn_best.pth",
) -> list[ReferenceSample]:
    set_seed(seed)
    device = device or torch.device("cpu")
    model = model.to(device)
    records = score_reference_candidates(model, test_dataset, device, class_names)
    selected = select_reference_samples(
        records,
        class_names,
        samples_per_class=samples_per_class,
        seed=seed,
    )
    save_reference_set(
        model,
        test_dataset,
        selected,
        output_directory,
        seed=seed,
        checkpoint_path=checkpoint_path,
    )
    return selected
