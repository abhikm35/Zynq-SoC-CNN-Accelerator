from __future__ import annotations

import numpy as np
import pytest
import torch

from software.inference.numpy_inference import (
    NumpyTinyCNN,
    batch_norm2d_nchw,
    conv2d_nchw,
    global_average_pool_nchw,
    linear,
    max_pool2d_nchw,
    pad_input,
    predict_class,
    relu,
)
from software.models.tiny_cnn import TinyCNN
from software.utils.batchnorm_fold import extract_float32_parameters


def test_padding_shapes_and_zeros() -> None:
    inputs = np.ones((1, 1, 2, 2), dtype=np.float32)
    assert pad_input(inputs, 0).shape == (1, 1, 2, 2)
    padded = pad_input(inputs, 1)
    assert padded.shape == (1, 1, 4, 4)
    assert padded[0, 0, 0, 0] == 0.0
    assert padded[0, 0, 1, 1] == 1.0


def test_convolution_hand_calculated_single_channel() -> None:
    inputs = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
        dtype=np.float32,
    )
    weight = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    bias = np.array([0.5], dtype=np.float32)
    output = conv2d_nchw(inputs, weight, bias, stride=1, padding=0)
    # Windows: [[1,2],[4,5]] -> 1+5=6; [[2,3],[5,6]] -> 2+6=8;
    # [[4,5],[7,8]] -> 4+8=12; [[5,6],[8,9]] -> 5+9=14; plus bias 0.5
    assert output.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(
        output,
        np.array([[[[6.5, 8.5], [12.5, 14.5]]]], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )


def test_convolution_multi_channel_with_padding() -> None:
    inputs = np.arange(1 * 2 * 3 * 3, dtype=np.float32).reshape(1, 2, 3, 3)
    weight = np.ones((2, 2, 3, 3), dtype=np.float32)
    bias = np.array([1.0, -1.0], dtype=np.float32)
    output = conv2d_nchw(inputs, weight, bias, stride=1, padding=1)
    assert output.shape == (1, 2, 3, 3)
    assert np.isfinite(output).all()


def test_relu_and_pooling() -> None:
    values = np.array([[[[-1.0, 2.0], [3.0, -4.0]]]], dtype=np.float32)
    activated = relu(values)
    np.testing.assert_array_equal(activated, np.array([[[[0.0, 2.0], [3.0, 0.0]]]]))
    pooled = max_pool2d_nchw(
        np.array([[[[1.0, 2.0], [3.0, 0.0]]]], dtype=np.float32),
        kernel_size=2,
        stride=2,
    )
    assert pooled.shape == (1, 1, 1, 1)
    assert pooled.item() == 3.0


def test_global_average_and_linear() -> None:
    values = np.array(
        [[[[1.0, 3.0], [5.0, 7.0]], [[2.0, 2.0], [2.0, 2.0]]]],
        dtype=np.float32,
    )
    pooled = global_average_pool_nchw(values)
    np.testing.assert_allclose(pooled, np.array([[[[4.0]], [[2.0]]]]), atol=1e-6)
    features = pooled.reshape(1, -1)
    weight = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    bias = np.array([0.5, -0.5, 0.0], dtype=np.float32)
    scores = linear(features, weight, bias)
    np.testing.assert_allclose(scores, np.array([[4.5, 1.5, 6.0]], dtype=np.float32))
    assert predict_class(scores).tolist() == [2]


def test_batchnorm_matches_pytorch() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(2, 3, 4, 4)
    bn = torch.nn.BatchNorm2d(3)
    with torch.no_grad():
        bn.weight.copy_(torch.tensor([1.1, 0.9, 1.0]))
        bn.bias.copy_(torch.tensor([0.1, -0.2, 0.0]))
        bn.running_mean.copy_(torch.tensor([0.2, -0.1, 0.05]))
        bn.running_var.copy_(torch.tensor([1.5, 0.8, 1.2]))
    bn.eval()
    with torch.no_grad():
        expected = bn(inputs).numpy()
    actual = batch_norm2d_nchw(
        inputs.numpy(),
        bn.weight.detach().numpy(),
        bn.bias.detach().numpy(),
        bn.running_mean.detach().numpy(),
        bn.running_var.detach().numpy(),
        eps=float(bn.eps),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_numpy_model_matches_pytorch_one_sample() -> None:
    torch.manual_seed(1)
    model = TinyCNN(num_classes=5, conv1_channels=16, conv2_channels=32)
    model.eval()
    params = extract_float32_parameters(model)
    metadata = {
        "layer_order": [
            "input",
            "conv1",
            "bn1",
            "relu1",
            "pool1",
            "conv2",
            "bn2",
            "relu2",
            "pool2",
            "global_average_pool",
            "flatten",
            "scores",
        ],
        "layers": {
            "conv1": {"stride": 1, "padding": 1},
            "conv2": {"stride": 1, "padding": 1},
            "pool1": {"kernel_size": 2, "stride": 2},
            "pool2": {"kernel_size": 2, "stride": 2},
        },
        "batchnorm": {
            "bn1": {"eps": params["raw"]["bn1_eps"]},
            "bn2": {"eps": params["raw"]["bn2_eps"]},
        },
    }
    numpy_model = NumpyTinyCNN(params["raw"], metadata)
    inputs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        _, torch_intermediates = model.forward_with_intermediates(inputs)
    _, numpy_intermediates = numpy_model.forward_with_intermediates(inputs.numpy())

    for name, tensor in torch_intermediates.items():
        np.testing.assert_allclose(
            numpy_intermediates[name],
            tensor.numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Mismatch at layer {name}",
        )
    assert predict_class(numpy_intermediates["scores"]).tolist() == [
        int(torch.argmax(torch_intermediates["scores"], dim=1).item())
    ]
