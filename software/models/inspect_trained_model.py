from __future__ import annotations

import argparse

import torch

from software.utils.checkpoint import (
    checkpoint_path_from_config,
    count_trainable_parameters,
    restore_trained_model,
)
from software.utils.config import load_training_config, resolve_path
from software.utils.reproducibility import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the trained TinyCNN checkpoint and architecture"
    )
    parser.add_argument(
        "--checkpoint",
        default="software/checkpoints/tiny_cnn_best.pth",
        help="Path to the trained checkpoint",
    )
    parser.add_argument(
        "--config",
        default="software/config/training_config.yaml",
        help="Path to the training configuration YAML",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Print a small preview of each parameter tensor",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    set_seed(int(config["seed"]))

    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.is_file():
        checkpoint_path = checkpoint_path_from_config(config)

    model, checkpoint, resolved = restore_trained_model(
        checkpoint_path,
        config,
        map_location="cpu",
    )

    print("Checkpoint metadata")
    print("-------------------")
    print(f"path: {resolved}")
    print(f"epoch: {checkpoint['epoch']}")
    print(f"validation accuracy: {checkpoint['validation_accuracy'] * 100:.2f}%")
    print(f"validation loss: {checkpoint['validation_loss']:.6f}")
    print(f"class names: {checkpoint['class_names']}")
    print(f"num classes: {checkpoint['num_classes']}")
    print(f"input shape: {checkpoint['model_input_shape']}")
    print("checkpoint loaded cleanly")

    print("\nTrainable parameters")
    print("--------------------")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        print(f"{name:30s} {list(parameter.shape)}")
        if args.show_values:
            flat = parameter.detach().cpu().reshape(-1)
            preview = flat[:8].tolist()
            print(f"  preview[:8]={preview}")
    print(f"\nTotal trainable parameters: {count_trainable_parameters(model):,}")

    print("\nClass order")
    print("-----------")
    for index, name in enumerate(checkpoint["class_names"]):
        print(f"{index} -> {name}")

    torch.manual_seed(0)
    inputs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        scores, intermediates = model.forward_with_intermediates(inputs)

    print("\nIntermediate tensor shapes")
    print("--------------------------")
    for name, tensor in intermediates.items():
        print(f"{name:20s}: {list(tensor.shape)}")

    print("\nDeterministic sample scores")
    print("---------------------------")
    print(scores)


if __name__ == "__main__":
    main()
