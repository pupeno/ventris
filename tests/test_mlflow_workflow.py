from dataclasses import replace
from pathlib import Path

import mlflow
import pytest
import torch
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts

import ventris.train as train_module
import ventris.training_run as training_run_module
from tests.helpers import tiny_model
from tests.test_train import short_run, short_training, tiny_data, write_tiny_tokenizer
from ventris.checkpoint import TRAINING_STATE_FILE, load_checkpoint


@pytest.fixture
def local_training(tmp_path, monkeypatch):
    monkeypatch.setattr(train_module, "DATA_DIR", tmp_path)
    write_tiny_tokenizer(tmp_path)
    monkeypatch.setattr(train_module, "_training_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(training_run_module, "create_model", tiny_model)
    monkeypatch.setattr(train_module, "load_prepared_data", lambda: tiny_data(20))
    monkeypatch.setattr(train_module, "_generate_prompt_samples", lambda *args: [])
    return replace(short_run(), mlflow_root=str(tmp_path / "mlflow"))


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
def test_training_stores_complete_identifiable_checkpoints_once(local_training, architecture):
    training = short_training(3)
    checkpoint = train_module.train(training, local_training, architecture=architecture)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("ventris")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.params["model_type"] == f"ventris-{architecture}-v1"
    assert run.data.metrics["training/tokens_processed"] == 48
    latest = client.get_logged_model(run.data.tags["ventris.latest_checkpoint"])
    assert latest.status == "READY"
    assert latest.model_type == f"ventris-{architecture}-v1"
    assert run.outputs is not None
    assert latest.model_id in {output.model_id for output in run.outputs.model_outputs}
    assert Path(download_artifacts(f"models:/{latest.model_id}")) == checkpoint
    state, model = load_checkpoint(checkpoint, torch.device("cpu"), training)
    assert state["step"] == 3
    assert state["optimizer"]["state"]
    assert model.config.model_type == f"ventris-{architecture}-v1"
    assert (checkpoint / "tokenizer.json").is_file()
    models = client.search_logged_models([experiment.experiment_id])
    bundles = list(Path(local_training.mlflow_root).rglob(TRAINING_STATE_FILE))
    assert len(bundles) == len(models)
    assert len({bundle.stat().st_ino for bundle in bundles}) == len(bundles)


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
def test_evaluation_reloads_in_fresh_process_and_links_exact_weights(local_training, architecture):
    import json
    import os
    import subprocess
    import sys

    from ventris.evaluate import evaluate

    checkpoint = train_module.train(short_training(2), local_training, architecture=architecture)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("ventris")
    assert experiment is not None
    training_run = client.search_runs([experiment.experiment_id])[0]
    model_id = training_run.data.tags["ventris.latest_checkpoint"]
    reference = f"models:/{model_id}"
    first_id = evaluate(reference, mlflow_root=local_training.mlflow_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys; from ventris.evaluate import evaluate; "
            "print(json.dumps(evaluate(sys.argv[1], mlflow_root=sys.argv[2])))",
            reference,
            local_training.mlflow_root,
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    second_id = json.loads(completed.stdout.splitlines()[-1])
    assert first_id != second_id
    first, second = client.get_run(first_id), client.get_run(second_id)
    for evaluation in (first, second):
        assert evaluation.data.tags["ventris.checkpoint"] == model_id
        assert evaluation.data.params["fixture_sha256"]
        assert evaluation.data.params["context_length"] == "8"
        assert evaluation.inputs is not None
        assert evaluation.inputs.model_inputs[0].model_id == model_id
        assert evaluation.data.metrics["fixture/loss"] > 0
    assert first.data.metrics["fixture/loss"] == pytest.approx(
        second.data.metrics["fixture/loss"], abs=1e-5
    )
    assert client.get_logged_model(model_id).metrics
    assert (checkpoint / TRAINING_STATE_FILE).is_file()


def test_native_alias_selects_weights_for_generation_and_keeps_evaluation_inputs(
    local_training, monkeypatch
):
    from ventris.evaluate import evaluate
    from ventris.generate import generate

    train_module.train(short_training(2), local_training, architecture="vanilla")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("ventris")
    assert experiment is not None
    first_run = client.search_runs([experiment.experiment_id])[0]
    first_id = first_run.data.tags["ventris.latest_checkpoint"]
    client.create_registered_model("ventris-selected")
    version = client.create_model_version(
        "ventris-selected", f"models:/{first_id}", run_id=first_run.info.run_id
    )
    client.set_registered_model_alias("ventris-selected", "selected", version.version)
    alias = "models:/ventris-selected@selected"
    evaluation_id = evaluate(alias, mlflow_root=local_training.mlflow_root)
    generated = generate(alias, "token1", max_new_tokens=2, mlflow_root=local_training.mlflow_root)
    assert isinstance(generated, str)
    train_module.train(short_training(2), local_training, architecture="rope")
    second_run = client.search_runs(
        [experiment.experiment_id], filter_string="tags.`ventris.kind` = 'training'"
    )[0]
    second_id = second_run.data.tags["ventris.latest_checkpoint"]
    version = client.create_model_version(
        "ventris-selected", f"models:/{second_id}", run_id=second_run.info.run_id
    )
    client.set_registered_model_alias("ventris-selected", "selected", version.version)
    inputs = client.get_run(evaluation_id).inputs
    assert inputs is not None
    assert inputs.model_inputs[0].model_id == first_id
    assert (
        client.get_model_version_by_alias("ventris-selected", "selected").source
        == f"models:/{second_id}"
    )


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
def test_interrupted_run_continues_latest_and_checkpoint_fork_preserves_source(
    local_training, monkeypatch, architecture
):
    import wandb

    from tests.test_reporting import RecordingRun

    class InterruptedRun(RecordingRun):
        def log(self, metrics, step):
            super().log(metrics, step)
            if step == 2:
                raise RuntimeError("interrupted")

    monkeypatch.setattr(wandb, "init", lambda **kwargs: InterruptedRun())
    training = short_training(4)
    interrupted = replace(local_training, wandb_project="test")
    with pytest.raises(RuntimeError, match="interrupted"):
        train_module.train(training, interrupted, architecture=architecture)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("ventris")
    assert experiment is not None
    original = client.search_runs([experiment.experiment_id])[0]
    assert original.info.status == "FAILED"
    source_id = original.data.tags["ventris.latest_checkpoint"]
    source_path = Path(download_artifacts(f"models:/{source_id}"))
    original_bytes = (source_path / "model.safetensors").read_bytes()
    fork = train_module.train(training, local_training, fork_checkpoint=f"models:/{source_id}")
    fork_run = client.get_run(client.search_runs([experiment.experiment_id])[0].info.run_id)
    assert fork_run.info.run_id != original.info.run_id
    assert fork_run.data.tags["ventris.source_checkpoint"] == source_id
    assert fork_run.inputs is not None
    assert fork_run.inputs.model_inputs[0].model_id == source_id
    assert client.get_run(original.info.run_id).data.tags == original.data.tags
    assert (source_path / "model.safetensors").read_bytes() == original_bytes
    continued = train_module.train(
        training, replace(local_training, device_batch_size=1), continue_run=original.info.run_id
    )
    assert len(client.search_runs([experiment.experiment_id])) == 2
    assert client.get_run(original.info.run_id).info.status == "FINISHED"
    expected_state, expected_model = load_checkpoint(fork, torch.device("cpu"), training)
    actual_state, actual_model = load_checkpoint(continued, torch.device("cpu"), training)
    assert actual_state["step"] == expected_state["step"] == 4
    for name, value in expected_model.state_dict().items():
        torch.testing.assert_close(actual_model.state_dict()[name], value, atol=1e-6, rtol=1e-5)
    assert (source_path / "model.safetensors").read_bytes() == original_bytes


@pytest.mark.parametrize("use_mlflow", [False, True])
@pytest.mark.parametrize("use_wandb", [False, True])
def test_reporting_switches_are_independent_and_checkpoints_remain_usable(
    local_training, monkeypatch, tmp_path, use_mlflow, use_wandb
):
    import wandb

    from tests.test_reporting import RecordingRun

    recorded = RecordingRun()
    monkeypatch.setattr(wandb, "init", lambda **kwargs: recorded)
    run_config = replace(
        local_training,
        mlflow_root=local_training.mlflow_root if use_mlflow else None,
        wandb_project="test" if use_wandb else None,
    )
    checkpoint = train_module.train(short_training(3), run_config, architecture="vanilla")
    state, _ = load_checkpoint(checkpoint, torch.device("cpu"), short_training(3))
    assert state["step"] == 3
    assert bool(recorded.history) is use_wandb
    assert (Path(local_training.mlflow_root) / "mlflow.db").exists() is use_mlflow
    if not use_mlflow:
        assert checkpoint.name == "latest"
        assert checkpoint.is_symlink()
        assert (checkpoint.parent / "best").resolve() == checkpoint.resolve()
        assert len(list((tmp_path / "checkpoints").rglob(TRAINING_STATE_FILE))) == 1


def test_mlflow_write_failure_propagates(local_training, tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file")
    with pytest.raises(FileExistsError):
        train_module.train(
            short_training(2),
            replace(local_training, mlflow_root=str(blocked)),
            architecture="vanilla",
        )
    assert mlflow.active_run() is None


def test_retention_keeps_evaluated_registered_and_milestone_snapshots(local_training, monkeypatch):
    import wandb

    from tests.test_reporting import RecordingRun
    from ventris.evaluate import evaluate

    stop_at = 2

    class InterruptedRun(RecordingRun):
        def log(self, metrics, step):
            if step == stop_at:
                raise RuntimeError("interrupted")

    monkeypatch.setattr(wandb, "init", lambda **kwargs: InterruptedRun())
    run_config = replace(
        local_training,
        validation_interval_steps=1,
        milestone_interval_checkpoints=3,
        wandb_project="test",
    )
    training = short_training(6)
    with pytest.raises(RuntimeError, match="interrupted"):
        train_module.train(training, run_config, architecture="vanilla")
    client = MlflowClient()
    experiment = client.get_experiment_by_name("ventris")
    assert experiment is not None
    run = client.search_runs([experiment.experiment_id])[0]
    run_id = run.info.run_id
    evaluated_id = run.data.tags["ventris.latest_checkpoint"]
    evaluate(f"models:/{evaluated_id}", mlflow_root=local_training.mlflow_root)
    stop_at = 4
    with pytest.raises(RuntimeError, match="interrupted"):
        train_module.train(training, run_config, continue_run=run_id)
    selected_id = client.get_run(run_id).data.tags["ventris.latest_checkpoint"]
    client.create_registered_model("selected")
    version = client.create_model_version("selected", f"models:/{selected_id}", run_id=run_id)
    client.set_registered_model_alias("selected", "selected", version.version)
    train_module.train(training, replace(run_config, wandb_project=None), continue_run=run_id)
    models = client.search_logged_models([experiment.experiment_id])
    steps = {int(model.tags["ventris.step"]) for model in models}
    assert {2, 3, 4, 6} <= steps
    assert 0 not in steps and 1 not in steps and 5 not in steps
    for model in models:
        path = Path(download_artifacts(f"models:/{model.model_id}"))
        assert (path / TRAINING_STATE_FILE).is_file()
    assert len(list(Path(local_training.mlflow_root).rglob(TRAINING_STATE_FILE))) == len(models)
    assert client.get_logged_model(evaluated_id).status == "READY"
    assert client.get_logged_model(selected_id).status == "READY"


def test_fork_can_continue_before_improving_inherited_best(local_training, tmp_path, monkeypatch):
    import wandb

    from tests.test_reporting import RecordingRun
    from ventris.checkpoint import save_checkpoint
    from ventris.training_run import build_optimizer

    training = short_training(4)
    model = tiny_model()
    source = save_checkpoint(
        tmp_path / "legacy-milestone",
        model,
        build_optimizer(model, training.peak_learning_rate),
        2,
        training_config=training,
        run_config=local_training,
        tokenizer_dir=tmp_path,
        training_seconds_elapsed=1.0,
        best_validation_loss=0.1,
        best_validation_step=1,
    )

    class InterruptedRun(RecordingRun):
        def log(self, metrics, step):
            if step == 3:
                raise RuntimeError("interrupted")

    monkeypatch.setattr(wandb, "init", lambda **kwargs: InterruptedRun())
    config = replace(local_training, validation_interval_steps=1, wandb_project="test")
    with pytest.raises(RuntimeError, match="interrupted"):
        train_module.train(training, config, fork_checkpoint=source)
    client = MlflowClient()
    experiment = client.get_experiment_by_name("ventris")
    assert experiment is not None
    run = client.search_runs([experiment.experiment_id])[0]
    assert "ventris.best_checkpoint" not in run.data.tags
    assert run.data.tags["ventris.source_checkpoint"] == str(source)
    checkpoint = train_module.train(
        training, replace(config, wandb_project=None), continue_run=run.info.run_id
    )
    state, _ = load_checkpoint(checkpoint, torch.device("cpu"), training)
    assert state["step"] == 4
    assert state["best_validation_loss"] == 0.1
    assert state["best_validation_step"] == 1
