"""Architecture-owned models and shared construction/loading entry points."""

import json
from pathlib import Path

from ventris.models.rope import ModelConfig as RopeConfig
from ventris.models.rope import Ventris as RopeVentris
from ventris.models.vanilla import ModelConfig as VanillaConfig
from ventris.models.vanilla import Ventris as VanillaVentris

Model = VanillaVentris | RopeVentris
ModelConfig = VanillaConfig | RopeConfig


def create_model(architecture: str) -> Model:
    """Construct a fresh default model for the selected architecture."""
    if architecture == "vanilla":
        return VanillaVentris()
    if architecture == "rope":
        return RopeVentris()
    raise ValueError(f"unsupported architecture: {architecture!r}")


def load_model(path: Path, *, expected_architecture: str | None = None) -> Model:
    """Load a local model using its saved architecture/version identity."""
    if expected_architecture not in (None, "vanilla", "rope"):
        raise ValueError(f"unsupported expected architecture: {expected_architecture!r}")
    saved_config = json.loads((path / "config.json").read_text())
    model_type = saved_config.get("model_type")
    if model_type == "ventris-vanilla-v1":
        architecture, model_class = "vanilla", VanillaVentris
    elif model_type == "ventris-rope-v1":
        architecture, model_class = "rope", RopeVentris
    else:
        raise ValueError(f"missing or unsupported saved model_type: {model_type!r}")
    if expected_architecture is not None and expected_architecture != architecture:
        raise ValueError(
            f"requested architecture {expected_architecture!r} conflicts with saved architecture "
            f"{architecture!r}"
        )
    return model_class.from_pretrained(path, local_files_only=True)
