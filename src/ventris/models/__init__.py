"""Architecture-owned models and shared construction/loading entry points."""

import json
from pathlib import Path

from ventris.models.vanilla import Ventris


def create_model(architecture: str = "vanilla") -> Ventris:
    """Construct a fresh default model for the selected architecture."""
    if architecture == "vanilla":
        return Ventris()
    raise ValueError(f"unsupported architecture: {architecture!r}")


def load_model(path: Path, *, expected_architecture: str | None = None) -> Ventris:
    """Load a local model using its saved architecture/version identity."""
    if expected_architecture not in (None, "vanilla"):
        raise ValueError(f"unsupported expected architecture: {expected_architecture!r}")
    saved_config = json.loads((path / "config.json").read_text())
    model_type = saved_config.get("model_type")
    if model_type != "ventris-vanilla-v1":
        raise ValueError(f"missing or unsupported saved model_type: {model_type!r}")
    return Ventris.from_pretrained(path, local_files_only=True)
