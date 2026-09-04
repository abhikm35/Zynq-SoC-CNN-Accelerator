from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from software.models.tiny_cnn import TinyCNN
from software.training.evaluate import evaluate_model
from software.training.train import compute_class_weights, train_one_epoch


def _parameter_snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in model.parameters()]


def test_inverse_frequency_weights_emphasize_smaller_classes() -> None:
    class_names = ["stop", "yield", "no_entry", "speed_limit_30", "keep_right"]
    weights = compute_class_weights(
        {
            "stop": 420,
            "yield": 1140,
            "no_entry": 600,
            "speed_limit_30": 1200,
            "keep_right": 1110,
        },
        class_names,
        torch.device("cpu"),
    )

    counts = torch.tensor([420, 1140, 600, 1200, 1110], dtype=torch.float32)
    assert weights.shape == (5,)
    assert torch.isclose((weights * counts).sum() / counts.sum(), torch.tensor(1.0))
    assert weights[0] > weights[3]


def test_training_smoke_updates_weights_and_eval_is_side_effect_free() -> None:
    torch.manual_seed(0)
    images = torch.rand(8, 3, 32, 32)
    labels = torch.randint(0, 5, (8,), dtype=torch.int64)
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = TinyCNN(num_classes=5)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = _parameter_snapshot(model)
    train_metrics = train_one_epoch(
        model,
        loader,
        loss_function,
        optimizer,
        device=torch.device("cpu"),
        num_classes=5,
        class_names=["a", "b", "c", "d", "e"],
    )
    after_train = _parameter_snapshot(model)

    assert train_metrics["total"] == 8
    assert torch.isfinite(torch.tensor(train_metrics["loss"]))
    assert any(
        not torch.equal(left, right) for left, right in zip(before, after_train)
    )

    sample = images[:2]
    with torch.no_grad():
        scores = model(sample)
    assert scores.shape == (2, 5)

    before_eval = _parameter_snapshot(model)
    eval_metrics = evaluate_model(
        model,
        loader,
        loss_function,
        device=torch.device("cpu"),
        num_classes=5,
        class_names=["a", "b", "c", "d", "e"],
    )
    after_eval = _parameter_snapshot(model)

    assert torch.isfinite(torch.tensor(eval_metrics["loss"]))
    assert all(
        torch.equal(left, right) for left, right in zip(before_eval, after_eval)
    )
    # Ensure evaluate restored training mode when called from train context.
    model.train()
    evaluate_model(
        model,
        loader,
        loss_function,
        device=torch.device("cpu"),
        num_classes=5,
    )
    assert model.training
