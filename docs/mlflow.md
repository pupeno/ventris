# Local MLflow workflow

Training writes directly to SQLite and local artifact files. No tracking server or account is needed. The default root is `data/mlflow/`; `--mlflow-root DIRECTORY` selects another store. Start the UI only when you want to inspect it:

```console
mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db --port 5000
```

Open http://127.0.0.1:5000. For another root, point the UI at that directory's `mlflow.db`. Artifact locations are absolute local paths. Moving stores between machines is outside this first integration.

## Training and storage

MLflow is on by default and W&B is off. The switches are independent:

```console
scripts/train.py --architecture rope
scripts/train.py --architecture rope --wandb-project ventris
scripts/train.py --architecture rope --no-mlflow --wandb-project ventris
scripts/train.py --architecture rope --no-mlflow
```

The trainer prints the MLflow run ID and an exact `models:/MODEL_ID` reference whenever it saves a checkpoint. Each saved checkpoint is a native MLflow LoggedModel, linked as a training-run output. Its directory contains the model weights, configuration, tokenizer, optimizer, and progress. Ventris writes the bundle directly to MLflow's local artifact location once, then marks it READY. It uses its existing loader rather than an MLflow serving flavor. Enabled reporting errors propagate and stop training normally.

Latest and best validation checkpoints are run tags (`ventris.latest_checkpoint` and `ventris.best_checkpoint`) containing exact model IDs. Checkpoint metrics and step tags are visible on the LoggedModel. Periodic milestones are retained every four validation checkpoints by default: 1,000 optimizer steps with the default validation interval. Superseded snapshots are removed unless they are latest, best, milestones, evaluated, used as a fork source, or registered. Registering any native model version retains its checkpoint, even after moving an alias.

A model ID identifies one saved snapshot. Training never overwrites its weights. Do not edit tracked bundles in place. The data directory remains excluded from backups under the existing project policy.

With MLflow disabled, complete bundles live under `data/checkpoints/`. `latest`, `best`, and milestone paths are symlinks to those snapshots, so references do not copy weights. These local paths still work for generation and forking.

## Continue an interrupted run or fork a checkpoint

Continue by run identity; Ventris chooses that run's latest checkpoint:

```console
scripts/train.py --continue-run RUN_ID
```

The same MLflow run is resumed. Repeat the original training settings if they differed from defaults. Architecture can be inferred from the checkpoint. Device batch size, validation settings, and reporting options can change. Each invocation records its runtime configuration separately.

Fork a chosen checkpoint into a new run:

```console
scripts/train.py --fork-checkpoint models:/MODEL_ID
scripts/train.py --fork-checkpoint models:/ventris-rope-v1@selected
scripts/train.py --fork-checkpoint data/checkpoints/LEGACY_RUN/step-001000
```

The fork records its source and preserves the existing optimizer, progress, schedule, seed, and training-configuration rules. Training configuration must match the saved checkpoint, including total steps. Forking a completed final checkpoint does not extend its schedule. An inherited best loss can refer to earlier source history; the fork acquires its own best checkpoint reference when it improves that loss.

Continuation requires an existing MLflow training run. Forking from a legacy local checkpoint starts new tracked history; it does not import the old run. There is no continuation from an arbitrarily selected older checkpoint.

## Select weights using the native registry

Use the printed run ID to find its latest model and give it a native registry alias. Run this from the project root with the virtual environment active:

```python
import mlflow
from mlflow import MlflowClient
from ventris.tracking import configure_tracking

configure_tracking()  # or configure_tracking("data/mlflow-smoke")
client = MlflowClient()
run = client.get_run("RUN_ID")
model_id = run.data.tags["ventris.latest_checkpoint"]
name = "ventris-rope-v1"
client.create_registered_model(name)  # once per registered name
version = client.create_model_version(name, source=f"models:/{model_id}", run_id=run.info.run_id)
client.set_registered_model_alias(name, "selected", version.version)
print(f"models:/{model_id}")
```

Register from `models:/MODEL_ID` so aliases preserve the LoggedModel identity. Registry revision numbers are not Ventris architecture Versions. The alias expresses a manual selection; best validation loss is a separate, per-training-run choice. Repeating the version creation and alias assignment selects another checkpoint without changing prior evaluation records.

```console
scripts/generate.py --checkpoint models:/ventris-rope-v1@selected --prompt "The purpose of education is"
scripts/evaluate.py --checkpoint models:/ventris-rope-v1@selected
scripts/evaluate.py --checkpoint models:/MODEL_ID
```

Pass the same `--mlflow-root` to generation and evaluation when using a nondefault store. Generation also accepts existing local checkpoint paths. Evaluation requires an exact tracked model reference or native alias; it does not import arbitrary local weights.

Each evaluation reloads the checkpoint in CPU FP32 and computes real next-token cross-entropy over a tiny fixed text fixture, split into context-length windows. Each target token is counted once. A separate evaluation run records the exact model input, native dataset input, full fixture SHA-256, window/reduction settings, text, token count, and loss. MLflow's dataset digest uses the first 32 hex characters because its native field is shorter. Alias resolution happens before recording the input, so moving an alias leaves previous records unchanged. This fixture tests the workflow; its loss is not a model-quality benchmark or PIQA.

## Short acceptance walkthrough

After preparing the normal tokenizer and training data, these are actual supported training commands. They use the normal model dimensions and require a supported CUDA/BF16 GPU:

```console
scripts/train.py --architecture vanilla --steps 2 --warmup-steps 1 --effective-batch-size 1 --device-batch-size 1 --validation-interval 1 --validation-tokens 1024 --milestone-interval 1 --mlflow-root data/mlflow-smoke
scripts/train.py --architecture rope --steps 2 --warmup-steps 1 --effective-batch-size 1 --device-batch-size 1 --validation-interval 1 --validation-tokens 1024 --milestone-interval 1 --mlflow-root data/mlflow-smoke
```

For each run, retrieve its latest model ID with the native API above. Evaluate both, then repeat one evaluation:

```console
scripts/evaluate.py --mlflow-root data/mlflow-smoke --checkpoint models:/VANILLA_MODEL_ID
scripts/evaluate.py --mlflow-root data/mlflow-smoke --checkpoint models:/ROPE_MODEL_ID
scripts/evaluate.py --mlflow-root data/mlflow-smoke --checkpoint models:/ROPE_MODEL_ID
```

Select one through the native registry and generate using its alias. Start the UI against `data/mlflow-smoke/mlflow.db`. Inspect the training histories, model outputs/artifacts, registry selection, separate evaluation runs, and model/dataset links. Confirm that the workflow feels useful; this human assessment is separate from automated tests.

`tests/test_mlflow_workflow.py` exercises tiny real Vanilla/RoPE training, fresh-process evaluation, native selection, retention, interruption/continuation, forking, and all four reporting combinations. W&B uses its existing recording double. Historical W&B import and PIQA remain separate work.
