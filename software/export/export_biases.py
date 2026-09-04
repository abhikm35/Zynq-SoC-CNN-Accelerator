from __future__ import annotations

from pathlib import Path

from torch import nn

from software.export.export_weights import export_biases as _export_biases


def export_biases(model: nn.Module, output_directory: str | Path) -> dict[str, Path]:
    """Compatibility wrapper around the shared float32 bias exporter."""
    return _export_biases(model, output_directory)
