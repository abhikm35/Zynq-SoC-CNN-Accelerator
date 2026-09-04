from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from software.export.create_reference_set import ReferenceSample


def export_sample_activations(
    model: nn.Module,
    dataset: Dataset,
    samples: Sequence[ReferenceSample],
    output_directory: str | Path,
    device: torch.device,
) -> list[Path]:
    """Export every PyTorch intermediate tensor for the reference samples."""
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    model.eval()
    written: list[Path] = []

    with torch.no_grad():
        for sample in samples:
            image, label = dataset[sample.dataset_index]
            assert int(label.item()) == sample.true_class_index
            batch = image.unsqueeze(0).to(device)
            scores, intermediates = model.forward_with_intermediates(batch)
            predicted = int(torch.argmax(scores, dim=1).item())
            probabilities = torch.softmax(scores, dim=1).squeeze(0)
            confidence = float(probabilities[predicted].item())

            sample_dir = root / sample.sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            tensor_shapes: dict[str, list[int]] = {}
            tensor_dtypes: dict[str, str] = {}
            for name, tensor in intermediates.items():
                array = tensor.detach().cpu().numpy().astype(np.float32)
                np.save(sample_dir / f"{name}.npy", array)
                tensor_shapes[name] = list(array.shape)
                tensor_dtypes[name] = str(array.dtype)

            (sample_dir / "expected_class.txt").write_text(
                f"{sample.true_class_index}\n{sample.true_class_name}\n",
                encoding="utf-8",
            )
            metadata: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "true_class": sample.true_class_name,
                "true_class_index": sample.true_class_index,
                "predicted_class": sample.predicted_class_name,
                "predicted_class_index": predicted,
                "correct": predicted == sample.true_class_index,
                "confidence": confidence,
                "tensor_shapes": tensor_shapes,
                "tensor_dtypes": tensor_dtypes,
                "source_dataset_index": sample.dataset_index,
                "source_identifier": sample.source_identifier,
                "source_image_filename": f"{sample.sample_id}_{sample.true_class_name}.png",
            }
            (sample_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            written.append(sample_dir)
    return written
