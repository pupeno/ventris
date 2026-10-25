import json
import math
from pathlib import Path

import datasets
import mlflow
import pytest
import torch
from datasets import Dataset, DatasetDict
from mlflow import MlflowClient
from mlflow.data import get_source
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel

from tests.helpers import tiny_model
from tests.test_train import write_tiny_tokenizer
from ventris.tracking import configure_tracking, local_artifact_path


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
def test_evaluation_saves_harness_scores_and_exact_model_dataset_inputs(
    local_questions, tmp_path, architecture
):
    from ventris.evaluate import evaluate

    root = str(tmp_path / "mlflow")
    model_id, checkpoint = tracked_model(root, architecture, uniform=True)
    weights = (checkpoint / "model.safetensors").read_bytes()
    run_id = evaluate(f"models:/{model_id}", mlflow_root=root, device="cpu")

    client = MlflowClient()
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.info.run_name == f"piqa-{architecture}-v1-{model_id[2:10]}"
    assert run.data.params["model_name"] == f"{architecture}-v1"
    assert run.data.tags["ventris.kind"] == "evaluation"
    assert run.data.tags["ventris.checkpoint"] == model_id
    assert run.inputs is not None
    assert run.inputs.model_inputs[0].model_id == model_id
    dataset = run.inputs.dataset_inputs[0].dataset
    assert dataset.name == "piqa-validation"
    assert json.loads(dataset.source)["revision"] == "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1"
    source = get_source(dataset)
    assert source.to_dict()["path"] == "baber/piqa"
    assert source.to_dict()["split"] == "validation"
    assert run.data.metrics["piqa/acc"] == 1
    assert run.data.metrics["piqa/acc_norm"] == pytest.approx(1 / 3)
    assert run.data.metrics["piqa/acc_norm_stderr"] > 0
    assert run.data.params["sample_count"] == "3"
    assert run.data.params["num_fewshot"] == "0"

    assert run.info.artifact_uri is not None
    artifacts = local_artifact_path(run.info.artifact_uri)
    samples = json.loads((artifacts / "samples.json").read_text())
    assert [sample["doc"] for sample in samples] == local_questions
    assert [sample["target"] for sample in samples] == [0, 1, 1]
    table = client.load_table(run.info.experiment_id, "piqa_questions.json", run_ids=[run_id])
    assert table["question_id"].tolist() == [0, 1, 2]
    assert table["question"].tolist() == [question["goal"] for question in local_questions]
    assert table["correct_answer"].tolist() == ["token2", "token6", "token10"]
    assert table["prediction"].tolist() == ["token2", "token6", "token10"]
    assert table["normalized_prediction"].tolist() == ["token3 token4", "token4 token5", "token10"]
    assert table["correct"].tolist() == [True, True, True]
    assert table["normalized_correct"].tolist() == [False, False, True]
    assert table["correct"].to_numpy().mean() == run.data.metrics["piqa/acc"]
    assert table["normalized_correct"].to_numpy().mean() == run.data.metrics["piqa/acc_norm"]
    assert table["choice_1_score"].tolist() == pytest.approx(
        [-math.log(32), -2 * math.log(32), -3 * math.log(32)]
    )
    assert table["choice_2_normalized_score"].tolist() == pytest.approx(
        [-2 * math.log(32) / 13, -math.log(32) / 6, -math.log(32) / 7]
    )
    expected_lengths = [[1, 2], [2, 1], [3, 1]]
    for sample, lengths in zip(samples, expected_lengths, strict=True):
        scores = [response[0] for response in sample["filtered_resps"]]
        assert scores == pytest.approx([-length * math.log(32) for length in lengths])
    results = json.loads((artifacts / "results.json").read_text())
    assert results["results"]["piqa"]["acc,none"] == run.data.metrics["piqa/acc"]
    config = json.loads((artifacts / "configuration.json").read_text())
    assert config["model_id"] == model_id
    assert config["tokenizer_identity"] == f"models:/{model_id}"
    assert config["tokenizer_sha256"]
    assert config["dataset_sha256"].startswith(dataset.digest)
    assert config["dataset_split"] == "validation"
    assert config["context_length"] == 8
    assert config["add_bos_token"] is False
    assert config["apply_chat_template"] is False
    assert config["precision"] == "float32"
    assert config["device"] == "cpu"
    assert config["harness_version"]
    assert config["harness_revision"] == "d6de81643928d653435c431bae19945d41d32520"
    model = client.get_logged_model(model_id)
    assert model.tags["ventris.evaluated"] == "true"
    assert model.metrics is not None
    assert {metric.key for metric in model.metrics} >= {"piqa/acc", "piqa/acc_norm"}
    assert all(metric.dataset_digest == dataset.digest for metric in model.metrics)
    assert (checkpoint / "model.safetensors").read_bytes() == weights
    assert len(list(Path(root).rglob("model.safetensors"))) == 1


def test_external_evaluations_reuse_exact_identity_without_copying_weights(
    local_questions, tmp_path, monkeypatch
):
    from ventris.evaluate import evaluate

    source = tmp_path / "hub-cache"
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=8,
            n_embd=32,
            n_layer=1,
            n_head=4,
            bos_token_id=31,
            eos_token_id=31,
        )
    ).save_pretrained(source)
    write_tiny_tokenizer(source)
    model_loader = AutoModelForCausalLM.from_pretrained
    tokenizer_loader = AutoTokenizer.from_pretrained
    requested = []

    def load_model(repository, *, revision):
        requested.append(("model", repository, revision))
        return model_loader(source, local_files_only=True)

    def load_tokenizer(repository, *, revision):
        requested.append(("tokenizer", repository, revision))
        return tokenizer_loader(source, local_files_only=True)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    root = str(tmp_path / "mlflow")
    repository = "openai-community/gpt2"
    revision = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    first = evaluate(
        hf_model=repository,
        hf_revision=revision,
        mlflow_root=root,
        device="cpu",
        device_batch_size=1,
    )
    second = evaluate(
        hf_model=repository,
        hf_revision=revision,
        mlflow_root=root,
        device="cpu",
        device_batch_size=3,
    )
    assert first != second
    external_run = MlflowClient().get_run(first)
    assert external_run.info.run_name == "piqa-openai-community-gpt2-607a30d7"
    assert external_run.data.params["model_name"] == repository
    combined = MlflowClient().load_table(
        external_run.info.experiment_id,
        "piqa_questions.json",
        run_ids=[first, second],
        extra_columns=["run_id"],
    )
    assert len(combined) == 6
    assert set(combined["run_id"]) == {first, second}
    assert combined.groupby("question_id").size().tolist() == [2, 2, 2]
    assert requested == [
        (kind, repository, revision) for _ in range(2) for kind in ("model", "tokenizer")
    ]
    client = MlflowClient()
    runs = [client.get_run(run_id) for run_id in (first, second)]
    model_ids = []
    for run in runs:
        assert run.inputs is not None
        model_ids.append(run.inputs.model_inputs[0].model_id)
    assert model_ids[0] == model_ids[1]
    model = client.get_logged_model(model_ids[0])
    assert model.tags["mlflow.model.isExternal"] == "true"
    assert model.params["huggingface_repository"] == repository
    assert model.params["huggingface_revision"] == revision
    assert len(client.search_logged_models([model.experiment_id])) == 1
    assert not list(Path(root).rglob("*.safetensors"))
    assert (source / "model.safetensors").is_file()
    saved_scores = []
    for run in runs:
        assert run.info.status == "FINISHED"
        assert run.info.artifact_uri is not None
        artifacts = local_artifact_path(run.info.artifact_uri)
        config = json.loads((artifacts / "configuration.json").read_text())
        assert config["huggingface_repository"] == repository
        assert config["huggingface_revision"] == revision
        assert config["tokenizer_identity"] == f"{repository}@{revision}"
        samples = json.loads((artifacts / "samples.json").read_text())
        saved_scores.append(
            [response[0] for sample in samples for response in sample["filtered_resps"]]
        )
    assert saved_scores[0] == pytest.approx(saved_scores[1], rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
def test_mixed_length_scoring_agrees_and_alias_changes_preserve_previous_results(
    local_questions, tmp_path, architecture
):
    from ventris.evaluate import evaluate

    root = str(tmp_path / "mlflow")
    original_id, _ = tracked_model(root, architecture)
    client = MlflowClient()
    client.create_registered_model("selected")
    original_version = client.create_model_version("selected", f"models:/{original_id}")
    client.set_registered_model_alias("selected", "selected", original_version.version)
    alias = "models:/selected@selected"
    single_id = evaluate(alias, mlflow_root=root, device="cpu", device_batch_size=1)
    batched_id = evaluate(alias, mlflow_root=root, device="cpu", device_batch_size=3)
    assert single_id != batched_id
    single, batched = (client.get_run(identity) for identity in (single_id, batched_id))
    assert single.info.artifact_uri is not None
    assert batched.info.artifact_uri is not None
    single_samples = json.loads(
        (local_artifact_path(single.info.artifact_uri) / "samples.json").read_text()
    )
    batched_samples = json.loads(
        (local_artifact_path(batched.info.artifact_uri) / "samples.json").read_text()
    )
    for one, many in zip(single_samples, batched_samples, strict=True):
        assert one["doc_id"] == many["doc_id"]
        assert [response[0] for response in one["filtered_resps"]] == pytest.approx(
            [response[0] for response in many["filtered_resps"]], rel=1e-5, abs=1e-5
        )
        assert one["acc"] == many["acc"]
        assert one["acc_norm"] == many["acc_norm"]

    replacement_id, _ = tracked_model(root, architecture)
    replacement_version = client.create_model_version("selected", f"models:/{replacement_id}")
    client.set_registered_model_alias("selected", "selected", replacement_version.version)
    replacement_run = client.get_run(evaluate(alias, mlflow_root=root, device="cpu"))
    assert replacement_run.inputs is not None
    assert replacement_run.inputs.model_inputs[0].model_id == replacement_id
    for run_id in (single_id, batched_id):
        old = client.get_run(run_id)
        assert old.inputs is not None
        assert old.inputs.model_inputs[0].model_id == original_id
        assert old.data.tags["ventris.checkpoint"] == original_id


@pytest.mark.parametrize(
    "selection, message",
    [
        ({"checkpoint": "/untracked/weights"}, "tracked models:/"),
        ({"hf_model": "openai-community/gpt2", "hf_revision": "main"}, "full 40-character"),
        ({"checkpoint": "models:/m-example", "hf_model": "openai-community/gpt2"}, "exactly one"),
    ],
)
def test_evaluation_rejects_untracked_or_ambiguous_model_identity(tmp_path, selection, message):
    from ventris.evaluate import evaluate

    with pytest.raises(ValueError, match=message):
        evaluate(**selection, mlflow_root=str(tmp_path / "mlflow"), device="cpu")


@pytest.fixture
def local_questions(monkeypatch):
    questions = [
        {"goal": "token1", "sol1": "token2", "sol2": "token3 token4", "label": 0},
        {
            "goal": "token1 token2 token3",
            "sol1": "token4 token5",
            "sol2": "token6",
            "label": 1,
        },
        {
            "goal": "token1 token2",
            "sol1": "token7 token8 token9",
            "sol2": "token10",
            "label": 1,
        },
    ]
    documents = Dataset.from_list(questions)
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *args, **kwargs: DatasetDict({"train": documents, "validation": documents}),
    )
    return questions


def tracked_model(root, architecture, *, uniform=False):
    experiment_id = configure_tracking(root)
    model = tiny_model(architecture)
    if uniform:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
    logged = mlflow.initialize_logged_model(
        name=architecture, model_type=model.config.model_type, experiment_id=experiment_id
    )
    path = local_artifact_path(logged.artifact_location)
    model.save_pretrained(path)
    write_tiny_tokenizer(path)
    mlflow.finalize_logged_model(logged.model_id, "READY")
    return logged.model_id, path
