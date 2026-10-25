"""Test-only evaluation to exercise MLflow checkpoint and result linkage."""

import hashlib
import json
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from mlflow import MlflowClient
from mlflow.entities import Dataset, DatasetInput, LoggedModelInput

from ventris.common import evaluation_mode
from ventris.data import load_tokenizer
from ventris.models import load_model
from ventris.tracking import configure_tracking, resolve_checkpoint

FIXTURE = "The sun warms the ground. Water flows downhill. A book rests on the table.\n"


def evaluate_fixture(checkpoint: str | Path, *, mlflow_root: str) -> str:
    """Reload exact tracked weights and record a separate, reproducible fixture evaluation."""
    experiment_id = configure_tracking(mlflow_root)
    path, model_id = resolve_checkpoint(checkpoint, mlflow_root)
    if model_id is None:
        raise ValueError("evaluation requires a tracked models:/ checkpoint or registry alias")
    client = MlflowClient()
    client.set_logged_model_tags(model_id, {"ventris.evaluated": "true"})
    with mlflow.start_run(experiment_id=experiment_id, run_name="fixture-loss") as run:
        mlflow.set_tags({"ventris.kind": "evaluation", "ventris.checkpoint": model_id})
        model = load_model(path)
        tokenizer = load_tokenizer(path)
        tokens = tokenizer.encode(FIXTURE, add_special_tokens=False)
        context_length = model.config.max_position_embeddings
        digest = hashlib.sha256(FIXTURE.encode()).hexdigest()
        dataset = Dataset(
            name="ventris-fixture-v1",
            digest=digest[:32],
            source_type="code",
            source=json.dumps({"fixture": FIXTURE}),
        )
        client.log_inputs(
            run.info.run_id, datasets=[DatasetInput(dataset)], models=[LoggedModelInput(model_id)]
        )
        mlflow.log_params(
            {
                "fixture_sha256": digest,
                "context_length": context_length,
                "reduction": "mean-over-target-tokens",
                "device": "cpu",
                "precision": "float32",
                "add_special_tokens": False,
            }
        )
        mlflow.log_text(FIXTURE, "fixture.txt")
        total_loss = 0.0
        total_tokens = 0
        with evaluation_mode(model):
            for start in range(0, len(tokens) - 1, context_length):
                window = torch.tensor([tokens[start : start + context_length + 1]])
                logits = model(window[:, :-1]).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    window[:, 1:].reshape(-1),
                    reduction="sum",
                )
                total_loss += loss.item()
                total_tokens += window.size(1) - 1
        mean_loss = total_loss / total_tokens
        client.log_metric(
            run.info.run_id,
            "fixture/loss",
            mean_loss,
            model_id=model_id,
            dataset_name=dataset.name,
            dataset_digest=dataset.digest,
            synchronous=True,
        )
        mlflow.log_metric("fixture/tokens", total_tokens)
        mlflow.log_dict(
            {"checkpoint": model_id, "loss": mean_loss, "tokens": total_tokens}, "results.json"
        )
    return run.info.run_id
