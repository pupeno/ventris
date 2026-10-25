# Ventris

Ventris is an educational LLM meant to be understood end to end. Decisions in building it are meant to be modern without piling up complexity. Without being as complex as today's LLMs, it will also not have historic idiosyncrasies that can get in the way of understanding. Generally the style will not be defensive or production grade except where it is truly necessary or educational.

The implemented architectures use these default parameter counts:

| Architecture | Version | Parameters  |
| ------------ | ------- | ----------- |
| Vanilla      | v1      | 124,337,664 |
| RoPE         | v1      | 123,551,232 |

RoPE replaces learned position embeddings with rotary positions on queries and keys, removing 1,024 × 768 = 786,432 parameters. Both defaults otherwise use the same dimensions and context length. Counts include tied token/output embeddings once. Base and Instruct describe training variants and do not change architecture parameter counts. MLA is a planned architecture.

A version changes when an architecture changes in a backward-incompatible way. Models with the same architecture and version can have different parameter counts, for example by using different numbers of layers. The variant describes how the model was trained.

## Setup

Install the dependencies:

```console
uv sync
```

## Activate virtual environment

```console
source .venv/bin/activate
```

## Train the tokenizer

```console
scripts/train_tokenizer.py
```

It will automatically download the data it needs.

## Prepare data for training

```console
scripts/prepare_data.py
```

## Train the model

Training script is:

```
scripts/train.py
```

Get all the possible arguments with:

```
scripts/train.py --help
```

Train the vanilla architecture:

```console
scripts/train.py --architecture vanilla
```

Train the rope architecture:

```console
scripts/train.py --architecture rope
```

Process eight sequences per device batch:

```console
scripts/train.py --architecture vanilla --device-batch-size 8
```

Continue an interrupted tracked run from its latest checkpoint:

```console
scripts/train.py --continue-run RUN_ID
```

Fork a saved checkpoint into a new run, preserving its training state:

```console
scripts/train.py --fork-checkpoint models:/MODEL_ID
```

Train on every GPU in the machine with one process per GPU:

```console
torchrun --standalone --nproc-per-node=gpu scripts/train.py --architecture vanilla
```

## Generate text

Generate text from a checkpoint:

```console
scripts/generate.py \
  --checkpoint models:/MODEL_ID \
  --prompt "The purpose of education is"
```

## Evaluate on PIQA

Evaluate one model on all 1,838 labeled PIQA validation questions, zero-shot:

```console
scripts/evaluate.py --checkpoint models:/ventris-vanilla-v1@selected
scripts/evaluate.py --checkpoint models:/ventris-rope-v1@selected
scripts/evaluate.py --hf-model openai-community/gpt2 \
  --hf-revision 607a30d783dfa663caf39e06633721c8d4cfcd7e
```

Exact `models:/MODEL_ID` checkpoint references also work. Each invocation prints a new MLflow evaluation run ID. Compare `piqa/acc` and `piqa/acc_norm` (answer-character-normalized accuracy) in the `ventris` experiment. The artifacts contain aggregate results, every question's candidate scores and correct answer, and the resolved model/tokenizer, dataset, and execution configuration. GPT-2 weights stay in the Hugging Face cache; evaluations reuse their external MLflow model identity.

The dataset revision and harness are pinned. Evaluation uses each model's own tokenizer and context, without a chat template; this compares complete models on PIQA validation, not the test leaderboard or general model quality. See `scripts/evaluate.py --help` for device, precision, device batch size, and local tracking options.

## MLflow

Training tracks runs and checkpoints locally in `data/mlflow/` by default, without a running service. W&B is off by default. Use `--no-mlflow`, `--mlflow-root DIRECTORY`, or `--wandb-project PROJECT` to change reporting; see `scripts/train.py --help` for options.

Start the UI when needed:

```console
mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db --port 5000
```

Open http://localhost:5000 and select `ventris`. MLflow's built-in `Default` experiment can stay empty. For a browser on the native Linux Docker host, add `--host 0.0.0.0 --cors-allowed-origins http://CONTAINER_IP:5000` and open that address; `hostname -I` inside the container prints its IP.

Training prints run IDs and `models:/MODEL_ID` checkpoint references. Generation also accepts native registry aliases such as `models:/ventris-rope-v1@selected`.

## Fix formatting

```console
ruff format .
```

## Run all checks

```console
ruff format --check .
ruff check .
basedpyright
pytest -q
```
