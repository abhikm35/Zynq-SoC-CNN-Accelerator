from __future__ import annotations

import torch

from software.models.tiny_cnn import TinyCNN


def main() -> None:
    """Run one sample through TinyCNN and print model details."""
    model = TinyCNN(num_classes=5)
    model.eval()
    inputs = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        scores, intermediates = model.forward_with_intermediates(inputs)

    print("Intermediate tensor shapes")
    print("--------------------------")
    for name, tensor in intermediates.items():
        print(f"{name:20s}: {list(tensor.shape)}")

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"\nTrainable parameters: {trainable_parameters:,}")
    print("\nFinal output tensor")
    print("-------------------")
    print(scores)


if __name__ == "__main__":
    main()
