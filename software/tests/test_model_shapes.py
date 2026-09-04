from __future__ import annotations

import torch

from software.models.tiny_cnn import TinyCNN


def test_model_and_intermediate_shapes() -> None:
    model = TinyCNN(num_classes=5)
    model.eval()
    inputs = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        output = model(inputs)
        intermediate_output, intermediates = model.forward_with_intermediates(inputs)

    assert output.shape == (1, 5)
    assert intermediate_output.shape == (1, 5)

    expected_shapes = {
        "input": (1, 3, 32, 32),
        "conv1": (1, 16, 32, 32),
        "bn1": (1, 16, 32, 32),
        "relu1": (1, 16, 32, 32),
        "pool1": (1, 16, 16, 16),
        "conv2": (1, 32, 16, 16),
        "bn2": (1, 32, 16, 16),
        "relu2": (1, 32, 16, 16),
        "pool2": (1, 32, 8, 8),
        "global_average_pool": (1, 32, 1, 1),
        "flatten": (1, 32),
        "scores": (1, 5),
    }

    assert set(intermediates) == set(expected_shapes)
    for name, expected_shape in expected_shapes.items():
        assert intermediates[name].shape == expected_shape


def test_model_supports_larger_batches() -> None:
    model = TinyCNN(num_classes=5)
    model.eval()
    inputs = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        output = model(inputs)

    assert output.shape == (4, 5)


def test_model_parameter_count_and_batchnorm_layout() -> None:
    model = TinyCNN(num_classes=5)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    assert trainable_parameters == 5_301
    assert isinstance(model.bn1, torch.nn.BatchNorm2d)
    assert isinstance(model.bn2, torch.nn.BatchNorm2d)
    assert model.conv1.bias is None
    assert model.conv2.bias is None
