"""Evaluate one tracked model on the pinned zero-shot PIQA validation task."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import mlflow
import torch
import yaml
from huggingface_hub.utils import validate_repo_id  # pyright: ignore[reportPrivateImportUsage]
from lm_eval.api.task import ConfigurableTask  # pyright: ignore[reportMissingTypeStubs]
from lm_eval.evaluator import simple_evaluate  # pyright: ignore[reportMissingTypeStubs]
from lm_eval.models.huggingface import HFLM  # pyright: ignore[reportMissingTypeStubs]
from lm_eval.utils import handle_non_serializable  # pyright: ignore[reportMissingTypeStubs]
from mlflow import MlflowClient
from mlflow.data.huggingface_dataset_source import HuggingFaceDatasetSource
from mlflow.entities import Dataset, DatasetInput, LoggedModelInput
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from ventris.common import default_device
from ventris.data import load_tokenizer
from ventris.models import load_model
from ventris.tracking import DEFAULT_MLFLOW_ROOT, configure_tracking, resolve_checkpoint

PIQA_REVISION = "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1"
HARNESS_REVISION = "d6de81643928d653435c431bae19945d41d32520"
GPT2_REPOSITORY = "openai-community/gpt2"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"


def evaluate(
    checkpoint: str | Path | None = None,
    *,
    hf_model: str | None = None,
    hf_revision: str | None = None,
    mlflow_root: str = DEFAULT_MLFLOW_ROOT,
    device: str | None = None,
    precision: str = "float32",
    device_batch_size: int = 4,
) -> str:
    """Score the complete validation split and return its new MLflow run ID."""
    if (checkpoint is None) == (hf_model is None):
        raise ValueError("select exactly one tracked checkpoint or Hugging Face model")
    if hf_model is not None:
        validate_repo_id(hf_model)
        if hf_revision is None or re.fullmatch(r"[0-9a-f]{40}", hf_revision) is None:
            raise ValueError("Hugging Face evaluation requires a full 40-character commit revision")
    elif hf_revision is not None:
        raise ValueError("hf_revision requires hf_model")
    if device_batch_size < 1:
        raise ValueError("device_batch_size must be positive")
    experiment_id = configure_tracking(mlflow_root)
    target = torch.device(device) if device else default_device()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        precision
    ]
    client = MlflowClient()
    with mlflow.start_run(experiment_id=experiment_id, run_name="piqa-validation") as run:
        model, tokenizer, model_id, identity = _load_evaluation_model(
            checkpoint, hf_model, hf_revision, mlflow_root, experiment_id
        )
        model_name = identity.get(
            "huggingface_repository", model.config.model_type.removeprefix("ventris-")
        )
        revision = identity.get("huggingface_revision", model_id.removeprefix("m-"))
        mlflow.set_tags(
            {
                "ventris.kind": "evaluation",
                "ventris.model": model_id,
                "mlflow.runName": f"piqa-{model_name.replace('/', '-')}-{revision[:8]}",
            }
        )
        if checkpoint is not None:
            mlflow.set_tag("ventris.checkpoint", model_id)
        tokenizer_sha256 = _digest(
            {
                "backend": tokenizer.backend_tokenizer.to_str(),
                "special_tokens": tokenizer.special_tokens_map,
            }
        )
        model.to(device=target, dtype=dtype).eval()  # pyright: ignore[reportCallIssue]
        # The harness's untyped registry decorator obscures the HFLM constructor.
        wrapper = cast(Any, HFLM)(
            pretrained=model,
            tokenizer=tokenizer,
            backend="causal",
            batch_size=device_batch_size,
            max_length=model.config.max_position_embeddings,
            add_bos_token=False,
        )
        task = _piqa_task()
        documents = list(task.validation_docs())
        dataset_sha256 = _digest(documents)
        source = HuggingFaceDatasetSource(
            path="baber/piqa", revision=PIQA_REVISION, split="validation"
        )
        dataset = Dataset(
            name="piqa-validation",
            digest=dataset_sha256[:32],
            source_type="hugging_face",
            source=source.to_json(),
        )
        client.log_inputs(
            run.info.run_id, datasets=[DatasetInput(dataset)], models=[LoggedModelInput(model_id)]
        )
        configuration = {
            "model_id": model_id,
            "model_name": model_name,
            **identity,
            "tokenizer_sha256": tokenizer_sha256,
            "dataset_repository": "baber/piqa",
            "dataset_revision": PIQA_REVISION,
            "dataset_split": "validation",
            "dataset_sha256": dataset_sha256,
            "sample_count": len(documents),
            "num_fewshot": 0,
            "apply_chat_template": False,
            "context_length": wrapper.max_length,
            "add_bos_token": False,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "prefix_token_id": wrapper.prefix_token_id,
            "padding_side": "right",
            "overlong_inputs": "left-truncate-to-context",
            "device": str(model.device),
            "precision": str(model.dtype).removeprefix("torch."),
            "device_batch_size": device_batch_size,
            "harness_version": version("lm-eval"),
            "harness_revision": HARNESS_REVISION,
        }
        mlflow.log_params(configuration)
        mlflow.log_dict(configuration, "configuration.json")
        results = simple_evaluate(
            model=wrapper,
            tasks=[task],
            num_fewshot=0,
            log_samples=True,
            apply_chat_template=False,
            random_seed=0,
            numpy_random_seed=0,
            torch_random_seed=0,
            fewshot_random_seed=0,
        )
        assert results is not None and "results" in results
        samples = results.pop("samples")["piqa"]
        mlflow.log_text(json.dumps(samples, default=handle_non_serializable), "samples.json")
        mlflow.log_dict(dict(results), "results.json")
        mlflow.log_table(_question_table(samples), "piqa_questions.json")
        for name, value in results["results"]["piqa"].items():
            if name.endswith(",none") and isinstance(value, int | float):
                client.log_metric(
                    run.info.run_id,
                    f"piqa/{name.removesuffix(',none')}",
                    float(value),
                    model_id=model_id,
                    dataset_name=dataset.name,
                    dataset_digest=dataset.digest,
                    synchronous=True,
                )
    return run.info.run_id


def _load_evaluation_model(
    checkpoint: str | Path | None,
    hf_model: str | None,
    hf_revision: str | None,
    mlflow_root: str,
    experiment_id: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast, str, dict[str, str]]:
    client = MlflowClient()
    if checkpoint is not None:
        path, model_id = resolve_checkpoint(checkpoint, mlflow_root)
        if model_id is None:
            raise ValueError("evaluation requires a tracked models:/ checkpoint or registry alias")
        client.set_logged_model_tags(model_id, {"ventris.evaluated": "true"})
        return (
            load_model(path),
            load_tokenizer(path),
            model_id,
            {"model_reference": str(checkpoint), "tokenizer_identity": f"models:/{model_id}"},
        )

    assert hf_model is not None and hf_revision is not None
    model = AutoModelForCausalLM.from_pretrained(hf_model, revision=hf_revision)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, revision=hf_revision)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    matches = client.search_logged_models(
        [experiment_id],
        filter_string=(
            f"params.huggingface_repository = '{hf_model}' "
            f"AND params.huggingface_revision = '{hf_revision}' "
            "AND tags.`mlflow.model.isExternal` = 'true'"
        ),
        max_results=1,
    )
    logged = (
        matches[0]
        if matches
        else mlflow.create_external_model(
            name=hf_model.replace("/", "-"),
            experiment_id=experiment_id,
            model_type=model.config.model_type,
            params={"huggingface_repository": hf_model, "huggingface_revision": hf_revision},
        )
    )
    identity = f"{hf_model}@{hf_revision}"
    return (
        model,
        tokenizer,
        logged.model_id,
        {
            "model_reference": identity,
            "tokenizer_identity": identity,
            "huggingface_repository": hf_model,
            "huggingface_revision": hf_revision,
        },
    )


def _question_table(samples: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Present harness results as a native MLflow table for comparing answers."""
    rows = []
    for sample in samples:
        document = sample["doc"]
        choices = [document["sol1"], document["sol2"]]
        scores = [float(response[0]) for response in sample["filtered_resps"]]
        # PIQA uses answer character counts, with the first choice winning ties.
        normalized = [score / len(choice) for score, choice in zip(scores, choices, strict=True)]
        predicted = max(range(2), key=lambda index: scores[index])
        predicted_normalized = max(range(2), key=lambda index: normalized[index])
        rows.append(
            {
                "question_id": sample["doc_id"],
                "question": document["goal"],
                "choice_1": choices[0],
                "choice_2": choices[1],
                "correct_answer": choices[int(document["label"])],
                "prediction": choices[predicted],
                "normalized_prediction": choices[predicted_normalized],
                "correct": bool(sample["acc"]),
                "normalized_correct": bool(sample["acc_norm"]),
                "choice_1_score": scores[0],
                "choice_2_score": scores[1],
                "choice_1_normalized_score": normalized[0],
                "choice_2_normalized_score": normalized[1],
            }
        )
    return {column: [row[column] for row in rows] for column in rows[0]}


def _piqa_task() -> ConfigurableTask:
    configuration = yaml.safe_load(files("lm_eval.tasks").joinpath("piqa/piqa.yaml").read_text())
    configuration.update(dataset_kwargs={"revision": PIQA_REVISION}, num_fewshot=0)
    return ConfigurableTask(config=configuration)


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()
