from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for practical reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Best-effort deterministic settings. These must not crash unsupported builds.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers when ``num_workers`` is greater than zero."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(preference: str = "auto") -> torch.device:
    """Select CUDA, Apple MPS, or CPU based on preference and availability."""
    choice = preference.strip().lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if choice.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device(choice)

    if choice == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")

    if choice == "cpu":
        return torch.device("cpu")

    raise RuntimeError(f"Unsupported runtime.device value: {preference}")


def reproducibility_warning() -> str:
    return (
        "Warning: exact numerical reproducibility may still vary across device "
        "types, PyTorch builds, and library versions."
    )


def count_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
