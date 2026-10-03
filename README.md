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

Train Vanilla with the defaults:

```console
scripts/train.py
```

Explicitly select RoPE:

```console
scripts/train.py --architecture rope
```

Startup output and Weights & Biases configuration report the actual model identity (`ventris-vanilla-v1` or `ventris-rope-v1`) and measured parameter count.

Process eight sequences per device batch:

```console
scripts/train.py --device-batch-size 8
```

Continue training from a checkpoint; its saved configuration selects the architecture:

```console
scripts/train.py --resume-checkpoint data/checkpoints/2026-09-20_14-30-00/latest
```

Train on every GPU in the machine with one process per GPU:

```console
torchrun --standalone --nproc-per-node=gpu scripts/train.py
```

## Generate text

Generate text from a checkpoint:

```console
scripts/generate.py \
  --checkpoint data/checkpoints/2026-09-18_07-47-10/latest \
  --prompt "The purpose of education is"
```

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
