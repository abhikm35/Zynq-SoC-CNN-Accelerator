from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from software.export.export_weights import export_model_metadata as _export_model_metadata


def export_model_metadata(
    model: nn.Module,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    config: dict[str, Any],
    output_directory: str | Path,
    *,
    test_accuracy: float | None = None,
) -> Path:
    """Compatibility wrapper around the shared metadata exporter."""
    return _export_model_metadata(
        model,
        checkpoint,
        checkpoint_path,
        config,
        output_directory,
        test_accuracy=test_accuracy,
    )
