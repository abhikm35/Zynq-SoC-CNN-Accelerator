from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


def fold_conv_batchnorm(
    conv_weight: torch.Tensor | np.ndarray,
    bn_weight: torch.Tensor | np.ndarray,
    bn_bias: torch.Tensor | np.ndarray,
    bn_mean: torch.Tensor | np.ndarray,
    bn_var: torch.Tensor | np.ndarray,
    *,
    eps: float = 1e-5,
    conv_bias: torch.Tensor | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold BatchNorm into convolution weights and biases.

    Returns:
        folded_weight with shape [out_channels, in_channels, kh, kw]
        folded_bias with shape [out_channels]
    """
    weight = np.asarray(conv_weight, dtype=np.float32)
    gamma = np.asarray(bn_weight, dtype=np.float32)
    beta = np.asarray(bn_bias, dtype=np.float32)
    mean = np.asarray(bn_mean, dtype=np.float32)
    var = np.asarray(bn_var, dtype=np.float32)
    scale = gamma / np.sqrt(var + float(eps))
    folded_weight = weight * scale.reshape(-1, 1, 1, 1)
    if conv_bias is None:
        bias = np.zeros_like(mean)
    else:
        bias = np.asarray(conv_bias, dtype=np.float32)
    folded_bias = beta + scale * (bias - mean)
    return folded_weight.astype(np.float32), folded_bias.astype(np.float32)


def extract_float32_parameters(model: nn.Module) -> dict[str, Any]:
    """Extract raw and BatchNorm-folded float32 parameters from TinyCNN."""
    if not hasattr(model, "conv1") or not hasattr(model, "bn1"):
        raise ValueError("Model does not expose the expected TinyCNN BatchNorm layout")

    conv1_w = model.conv1.weight.detach().cpu().numpy().astype(np.float32)
    conv2_w = model.conv2.weight.detach().cpu().numpy().astype(np.float32)
    classifier_w = model.classifier.weight.detach().cpu().numpy().astype(np.float32)
    classifier_b = model.classifier.bias.detach().cpu().numpy().astype(np.float32)

    bn1 = model.bn1
    bn2 = model.bn2
    folded_conv1_w, folded_conv1_b = fold_conv_batchnorm(
        conv1_w,
        bn1.weight.detach().cpu().numpy(),
        bn1.bias.detach().cpu().numpy(),
        bn1.running_mean.detach().cpu().numpy(),
        bn1.running_var.detach().cpu().numpy(),
        eps=float(bn1.eps),
    )
    folded_conv2_w, folded_conv2_b = fold_conv_batchnorm(
        conv2_w,
        bn2.weight.detach().cpu().numpy(),
        bn2.bias.detach().cpu().numpy(),
        bn2.running_mean.detach().cpu().numpy(),
        bn2.running_var.detach().cpu().numpy(),
        eps=float(bn2.eps),
    )

    return {
        "raw": {
            "conv1_weight": conv1_w,
            "conv2_weight": conv2_w,
            "classifier_weight": classifier_w,
            "classifier_bias": classifier_b,
            "bn1_weight": bn1.weight.detach().cpu().numpy().astype(np.float32),
            "bn1_bias": bn1.bias.detach().cpu().numpy().astype(np.float32),
            "bn1_running_mean": bn1.running_mean.detach().cpu().numpy().astype(np.float32),
            "bn1_running_var": bn1.running_var.detach().cpu().numpy().astype(np.float32),
            "bn1_eps": float(bn1.eps),
            "bn2_weight": bn2.weight.detach().cpu().numpy().astype(np.float32),
            "bn2_bias": bn2.bias.detach().cpu().numpy().astype(np.float32),
            "bn2_running_mean": bn2.running_mean.detach().cpu().numpy().astype(np.float32),
            "bn2_running_var": bn2.running_var.detach().cpu().numpy().astype(np.float32),
            "bn2_eps": float(bn2.eps),
        },
        "folded": {
            "conv1_weights_float32": folded_conv1_w,
            "conv1_bias_float32": folded_conv1_b,
            "conv2_weights_float32": folded_conv2_w,
            "conv2_bias_float32": folded_conv2_b,
            "classifier_weights_float32": classifier_w,
            "classifier_bias_float32": classifier_b,
        },
    }
