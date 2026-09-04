from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def predict_classes(scores: Tensor) -> Tensor:
    """Return predicted class indices from unnormalized class scores."""
    return scores.argmax(dim=1)


def count_correct(predictions: Tensor, labels: Tensor) -> int:
    """Count how many predictions match the ground-truth labels."""
    if predictions.shape != labels.shape:
        raise ValueError(
            f"Prediction/label shape mismatch: {tuple(predictions.shape)} vs "
            f"{tuple(labels.shape)}"
        )
    return int((predictions == labels).sum().item())


def overall_accuracy(correct: int, total: int) -> float:
    """Return overall accuracy, or 0.0 when no samples are present."""
    if total <= 0:
        return 0.0
    return float(correct) / float(total)


def per_class_correct_counts(
    predictions: Tensor,
    labels: Tensor,
    num_classes: int,
) -> list[int]:
    """Count correct predictions for each class."""
    counts = [0 for _ in range(num_classes)]
    for class_index in range(num_classes):
        mask = labels == class_index
        if mask.any():
            counts[class_index] = int((predictions[mask] == class_index).sum().item())
    return counts


def per_class_sample_counts(labels: Tensor, num_classes: int) -> list[int]:
    """Count ground-truth samples for each class."""
    counts = [0 for _ in range(num_classes)]
    for class_index in range(num_classes):
        counts[class_index] = int((labels == class_index).sum().item())
    return counts


def per_class_accuracy(
    correct_counts: list[int],
    sample_counts: list[int],
) -> list[float]:
    """Return per-class accuracies, using 0.0 for classes with zero samples."""
    if len(correct_counts) != len(sample_counts):
        raise ValueError("correct_counts and sample_counts must have equal length")
    accuracies: list[float] = []
    for correct, total in zip(correct_counts, sample_counts):
        accuracies.append(0.0 if total <= 0 else float(correct) / float(total))
    return accuracies


def weighted_average_loss(total_loss: float, total_samples: int) -> float:
    """Return a sample-weighted average loss."""
    if total_samples <= 0:
        return 0.0
    return float(total_loss) / float(total_samples)


def confusion_matrix_counts(
    predictions: Tensor,
    labels: Tensor,
    num_classes: int,
) -> Tensor:
    """Return a confusion matrix with actual rows and predicted columns."""
    if predictions.shape != labels.shape:
        raise ValueError(
            f"Prediction/label shape mismatch: {tuple(predictions.shape)} vs "
            f"{tuple(labels.shape)}"
        )
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    encoded = labels.to(torch.int64) * num_classes + predictions.to(torch.int64)
    return torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def accumulate_batch_metrics(
    scores: Tensor,
    labels: Tensor,
    loss_value: float,
    *,
    num_classes: int,
    total_loss: float,
    total_correct: int,
    total_samples: int,
    class_correct: list[int],
    class_totals: list[int],
) -> dict[str, Any]:
    """Update running evaluation totals for one batch."""
    predictions = predict_classes(scores)
    batch_size = int(labels.shape[0])
    batch_correct = count_correct(predictions, labels)

    total_loss += float(loss_value) * batch_size
    total_correct += batch_correct
    total_samples += batch_size

    batch_class_correct = per_class_correct_counts(predictions, labels, num_classes)
    batch_class_totals = per_class_sample_counts(labels, num_classes)
    for index in range(num_classes):
        class_correct[index] += batch_class_correct[index]
        class_totals[index] += batch_class_totals[index]

    return {
        "total_loss": total_loss,
        "total_correct": total_correct,
        "total_samples": total_samples,
        "class_correct": class_correct,
        "class_totals": class_totals,
    }


def finalize_metrics(
    *,
    total_loss: float,
    total_correct: int,
    total_samples: int,
    class_correct: list[int],
    class_totals: list[int],
    class_names: list[str] | None = None,
    confusion_matrix: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Convert running totals into a structured metrics dictionary."""
    accuracies = per_class_accuracy(class_correct, class_totals)
    per_class: dict[str, dict[str, float | int]] = {}
    for index, accuracy in enumerate(accuracies):
        name = class_names[index] if class_names is not None else str(index)
        per_class[name] = {
            "correct": int(class_correct[index]),
            "total": int(class_totals[index]),
            "accuracy": float(accuracy),
        }

    return {
        "loss": weighted_average_loss(total_loss, total_samples),
        "accuracy": overall_accuracy(total_correct, total_samples),
        "correct": int(total_correct),
        "total": int(total_samples),
        "per_class": per_class,
        "per_class_accuracy": accuracies,
        "per_class_correct": list(class_correct),
        "per_class_totals": list(class_totals),
        "confusion_matrix": confusion_matrix,
    }
