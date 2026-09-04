from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a project configuration file is missing or invalid."""


def project_root() -> Path:
    """Return the repository root containing the ``software`` package."""
    cwd = Path.cwd()
    if (cwd / "software" / "config").is_dir():
        return cwd.resolve()
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve a possibly relative path against the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = base if base is not None else project_root()
    return (root / candidate).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping with clear errors for common failures."""
    config_path = resolve_path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc

    if data is None:
        raise ConfigurationError(f"Configuration file is empty: {config_path}")
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration root must be a mapping in {config_path}"
        )
    return data


def _require_mapping(data: dict[str, Any], key: str, *, source: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Missing or invalid '{key}' section in {source}")
    return value


def _require_keys(section: dict[str, Any], keys: list[str], *, section_name: str) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ConfigurationError(
            f"Missing required field(s) in '{section_name}': {', '.join(missing)}"
        )


def load_model_config(
    path: str | Path = "software/config/model_config.yaml",
) -> dict[str, Any]:
    """Load and lightly validate the model configuration."""
    source = str(path)
    config = load_yaml(path)
    model = _require_mapping(config, "model", source=source)
    _require_keys(model, ["num_classes"], section_name="model")
    classes = config.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ConfigurationError(f"Missing or invalid 'classes' list in {source}")
    if int(model["num_classes"]) != len(classes):
        raise ConfigurationError(
            "model.num_classes must equal the number of configured class names"
        )
    return config


def load_training_config(
    path: str | Path = "software/config/training_config.yaml",
    *,
    model_config_path: str | Path = "software/config/model_config.yaml",
) -> dict[str, Any]:
    """Load and validate the floating-point training configuration."""
    source = str(path)
    config = load_yaml(path)
    model_config = load_model_config(model_config_path)

    _require_keys(
        config,
        [
            "seed",
            "data",
            "training",
            "optimizer",
            "loss",
            "scheduler",
            "checkpoint",
            "runtime",
        ],
        section_name="root",
    )

    data = _require_mapping(config, "data", source=source)
    training = _require_mapping(config, "training", source=source)
    optimizer = _require_mapping(config, "optimizer", source=source)
    loss = _require_mapping(config, "loss", source=source)
    scheduler = _require_mapping(config, "scheduler", source=source)
    checkpoint = _require_mapping(config, "checkpoint", source=source)
    runtime = _require_mapping(config, "runtime", source=source)

    _require_keys(
        data,
        [
            "root",
            "image_width",
            "image_height",
            "channels",
            "validation_fraction",
            "download",
            "num_workers",
            "normalization",
            "augmentation",
            "selected_classes",
        ],
        section_name="data",
    )
    _require_keys(
        training,
        ["batch_size", "epochs", "learning_rate", "weight_decay"],
        section_name="training",
    )
    _require_keys(optimizer, ["name"], section_name="optimizer")
    _require_keys(loss, ["class_weighting"], section_name="loss")
    _require_keys(
        scheduler,
        ["name", "factor", "patience", "min_learning_rate"],
        section_name="scheduler",
    )
    _require_keys(checkpoint, ["directory", "filename"], section_name="checkpoint")
    _require_keys(runtime, ["device"], section_name="runtime")

    validation_fraction = float(data["validation_fraction"])
    if not 0.0 < validation_fraction < 1.0:
        raise ConfigurationError(
            "data.validation_fraction must be strictly between 0 and 1"
        )

    batch_size = int(training["batch_size"])
    if batch_size < 1:
        raise ConfigurationError("training.batch_size must be >= 1")

    epochs = int(training["epochs"])
    if epochs < 1:
        raise ConfigurationError("training.epochs must be >= 1")

    if int(data["num_workers"]) < 0:
        raise ConfigurationError("data.num_workers must be >= 0")

    normalization = _require_mapping(data, "normalization", source=source)
    _require_keys(normalization, ["mean", "std"], section_name="data.normalization")
    for key in ("mean", "std"):
        values = normalization[key]
        if not isinstance(values, list) or len(values) != 3:
            raise ConfigurationError(
                f"data.normalization.{key} must contain exactly three values"
            )
        normalization[key] = [float(value) for value in values]
    if any(value <= 0.0 for value in normalization["std"]):
        raise ConfigurationError("data.normalization.std values must be positive")

    augmentation = _require_mapping(data, "augmentation", source=source)
    _require_keys(
        augmentation,
        [
            "rotation_degrees",
            "translate_fraction",
            "scale_min",
            "scale_max",
            "brightness",
            "contrast",
        ],
        section_name="data.augmentation",
    )
    if float(augmentation["rotation_degrees"]) < 0.0:
        raise ConfigurationError("augmentation.rotation_degrees must be >= 0")
    if not 0.0 <= float(augmentation["translate_fraction"]) < 1.0:
        raise ConfigurationError(
            "augmentation.translate_fraction must be in [0, 1)"
        )
    if not 0.0 < float(augmentation["scale_min"]) <= float(
        augmentation["scale_max"]
    ):
        raise ConfigurationError(
            "augmentation scales must satisfy 0 < scale_min <= scale_max"
        )
    if float(augmentation["brightness"]) < 0.0:
        raise ConfigurationError("augmentation.brightness must be >= 0")
    if float(augmentation["contrast"]) < 0.0:
        raise ConfigurationError("augmentation.contrast must be >= 0")

    optimizer_name = str(optimizer["name"]).strip().lower()
    if optimizer_name not in {"adam"}:
        raise ConfigurationError(
            f"Unsupported optimizer name '{optimizer['name']}'. Supported: adam"
        )
    optimizer["name"] = optimizer_name

    weighting = str(loss["class_weighting"]).strip().lower()
    if weighting not in {"none", "inverse_frequency"}:
        raise ConfigurationError(
            "loss.class_weighting must be 'none' or 'inverse_frequency'"
        )
    loss["class_weighting"] = weighting

    scheduler_name = str(scheduler["name"]).strip().lower()
    if scheduler_name not in {"none", "reduce_on_plateau"}:
        raise ConfigurationError(
            "scheduler.name must be 'none' or 'reduce_on_plateau'"
        )
    scheduler["name"] = scheduler_name
    if not 0.0 < float(scheduler["factor"]) < 1.0:
        raise ConfigurationError("scheduler.factor must be strictly between 0 and 1")
    if int(scheduler["patience"]) < 0:
        raise ConfigurationError("scheduler.patience must be >= 0")
    if float(scheduler["min_learning_rate"]) < 0.0:
        raise ConfigurationError("scheduler.min_learning_rate must be >= 0")

    selected_classes = data["selected_classes"]
    if not isinstance(selected_classes, list) or not selected_classes:
        raise ConfigurationError("data.selected_classes must be a non-empty list")

    expected_names = list(model_config["classes"])
    expected_labels = list(range(len(expected_names)))
    names: list[str] = []
    project_labels: list[int] = []
    gtsrb_ids: list[int] = []

    for index, entry in enumerate(selected_classes):
        if not isinstance(entry, dict):
            raise ConfigurationError(
                f"data.selected_classes[{index}] must be a mapping"
            )
        for key in ("name", "gtsrb_id", "project_label"):
            if key not in entry:
                raise ConfigurationError(
                    f"data.selected_classes[{index}] is missing '{key}'"
                )
        name = str(entry["name"])
        gtsrb_id = int(entry["gtsrb_id"])
        project_label = int(entry["project_label"])
        names.append(name)
        gtsrb_ids.append(gtsrb_id)
        project_labels.append(project_label)

    if names != expected_names:
        raise ConfigurationError(
            "Selected class names/order must match model_config.yaml classes: "
            f"expected {expected_names}, got {names}"
        )

    if sorted(project_labels) != expected_labels or project_labels != expected_labels:
        raise ConfigurationError(
            "project_label values must be exactly "
            f"{expected_labels} in class order, got {project_labels}"
        )

    if len(set(project_labels)) != len(project_labels):
        raise ConfigurationError("Duplicate project labels are not allowed")

    if len(set(gtsrb_ids)) != len(gtsrb_ids):
        raise ConfigurationError("Duplicate GTSRB class IDs are not allowed")

    if int(data["image_width"]) != int(model_config["input"]["width"]):
        raise ConfigurationError("data.image_width must match model_config input.width")
    if int(data["image_height"]) != int(model_config["input"]["height"]):
        raise ConfigurationError(
            "data.image_height must match model_config input.height"
        )
    if int(data["channels"]) != int(model_config["input"]["channels"]):
        raise ConfigurationError(
            "data.channels must match model_config input.channels"
        )

    config["model_config"] = model_config
    return config
