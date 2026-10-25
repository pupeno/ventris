"""Local MLflow storage and checkpoint references shared by Ventris entry points."""

import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

import mlflow
from mlflow import MlflowClient

DEFAULT_MLFLOW_ROOT = "data/mlflow"


def resolve_checkpoint(
    reference: str | Path, root: str = DEFAULT_MLFLOW_ROOT
) -> tuple[Path, str | None]:
    """Resolve a local bundle, exact LoggedModel URI, or native registry alias."""
    value = str(reference)
    if not value.startswith("models:/"):
        return Path(reference).resolve(), None
    configure_tracking(root)
    identity = value.removeprefix("models:/")
    if "@" in identity:
        name, alias = identity.split("@", 1)
        version = MlflowClient().get_model_version_by_alias(name, alias)
        value = version.source
        assert value is not None
        identity = value.removeprefix("models:/")
    if not identity.startswith("m-") or "/" in identity:
        raise ValueError(
            "checkpoint must identify a LoggedModel or an alias registered from models:/<model-id>"
        )
    model = MlflowClient().get_logged_model(identity)
    if str(model.status) != "READY":
        raise ValueError(f"checkpoint {identity} is not READY")
    return local_artifact_path(model.artifact_location), model.model_id


def prune_checkpoint(model_id: str) -> None:
    """Remove an obsolete snapshot unless an evaluation, fork, or registry retains it."""
    client = MlflowClient()
    model = client.get_logged_model(model_id)
    if any(
        model.tags.get(key) == "true"
        for key in ("ventris.milestone", "ventris.evaluated", "ventris.forked")
    ):
        return
    for source in (f"models:/{model_id}", model.artifact_location):
        if client.search_model_versions(filter_string=f"source_path = '{source}'", max_results=1):
            return
    shutil.rmtree(local_artifact_path(model.artifact_location))
    client.delete_logged_model(model_id)


def configure_tracking(root: str = DEFAULT_MLFLOW_ROOT) -> str:
    """Select the local SQLite store; writing needs no tracking server."""
    directory = Path(root).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{directory / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("ventris")
    if experiment is None:
        return client.create_experiment(
            "ventris", artifact_location=(directory / "artifacts").as_uri()
        )
    return experiment.experiment_id


def local_artifact_path(uri: str) -> Path:
    """Convert MLflow's local artifact location to its existing directory."""
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise ValueError("Ventris checkpoint storage requires local file artifacts")
    return Path(unquote(parsed.path))
