from __future__ import annotations

import torch

from software.training.metrics import (
    confusion_matrix_counts,
    count_correct,
    finalize_metrics,
    overall_accuracy,
    per_class_accuracy,
    per_class_correct_counts,
    per_class_sample_counts,
    predict_classes,
    weighted_average_loss,
)


def test_predict_classes_uses_argmax() -> None:
    scores = torch.tensor(
        [
            [0.1, 0.9, 0.0, -1.0, 0.2],
            [2.0, 0.5, 0.4, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.0, 3.0],
        ]
    )
    assert predict_classes(scores).tolist() == [1, 0, 4]


def test_correct_and_overall_accuracy() -> None:
    predictions = torch.tensor([0, 1, 2, 2])
    labels = torch.tensor([0, 1, 1, 2])
    correct = count_correct(predictions, labels)
    assert correct == 3
    assert overall_accuracy(correct, 4) == 0.75
    assert overall_accuracy(0, 0) == 0.0


def test_per_class_metrics_and_zero_sample_class() -> None:
    predictions = torch.tensor([0, 1, 1, 3])
    labels = torch.tensor([0, 1, 2, 3])
    correct_counts = per_class_correct_counts(predictions, labels, num_classes=5)
    sample_counts = per_class_sample_counts(labels, num_classes=5)
    accuracies = per_class_accuracy(correct_counts, sample_counts)

    assert correct_counts == [1, 1, 0, 1, 0]
    assert sample_counts == [1, 1, 1, 1, 0]
    assert accuracies == [1.0, 1.0, 0.0, 1.0, 0.0]


def test_weighted_average_loss() -> None:
    assert weighted_average_loss(10.0, 4) == 2.5
    assert weighted_average_loss(5.0, 0) == 0.0


def test_confusion_matrix_uses_actual_rows_and_predicted_columns() -> None:
    predictions = torch.tensor([0, 2, 1, 2, 0])
    labels = torch.tensor([0, 1, 1, 2, 2])
    matrix = confusion_matrix_counts(predictions, labels, num_classes=3)

    assert matrix.tolist() == [
        [1, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
    ]


def test_finalize_metrics_structure() -> None:
    metrics = finalize_metrics(
        total_loss=4.0,
        total_correct=3,
        total_samples=4,
        class_correct=[1, 1, 0, 1, 0],
        class_totals=[1, 1, 1, 1, 0],
        class_names=["a", "b", "c", "d", "e"],
        confusion_matrix=[
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ],
    )
    assert metrics["loss"] == 1.0
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"]["e"]["accuracy"] == 0.0
    assert metrics["per_class"]["a"]["correct"] == 1
    assert metrics["confusion_matrix"][2][1] == 1
