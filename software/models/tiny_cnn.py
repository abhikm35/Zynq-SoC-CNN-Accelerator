from __future__ import annotations

import torch
from torch import Tensor, nn


class TinyCNN(nn.Module):
    """FPGA-oriented CNN with foldable BatchNorm for five traffic-sign classes."""

    def __init__(
        self,
        num_classes: int = 5,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=conv1_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=conv1_channels,
            out_channels=conv2_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(conv2_channels)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(
            in_features=conv2_channels,
            out_features=num_classes,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return class scores for a batch of input images."""
        scores, _ = self.forward_with_intermediates(inputs)
        return scores

    def forward_with_intermediates(
        self, inputs: Tensor
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return class scores and all named intermediate tensors."""
        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)
        pool1 = self.pool1(relu1)

        conv2 = self.conv2(pool1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu2(bn2)
        pool2 = self.pool2(relu2)

        global_average_pool = self.global_average_pool(pool2)
        flatten = torch.flatten(global_average_pool, start_dim=1)
        scores = self.classifier(flatten)

        intermediates = {
            "input": inputs,
            "conv1": conv1,
            "bn1": bn1,
            "relu1": relu1,
            "pool1": pool1,
            "conv2": conv2,
            "bn2": bn2,
            "relu2": relu2,
            "pool2": pool2,
            "global_average_pool": global_average_pool,
            "flatten": flatten,
            "scores": scores,
        }
        return scores, intermediates
