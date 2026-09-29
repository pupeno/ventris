# Ventris

Ventris is an educational LLM meant to be understood end to end. Decisions in building it are meant to be modern without piling up complexity. Without being as complex as today's LLMs, it will also not have historic idiosyncrasies that can get in the way of understanding. Generally the style will not be defensive or production grade except where it is truly necessary or educational.

The goal of this project is to potentially produce many

| Architecture | Version | Parameters | Variant  | 
| ------------ | ------- | ---------- | -------- | 
| Vanilla      | v1      | 124M       | Base     | 
| Vanilla      | v1      | 124M       | Instruct | 
| RoPE         | v1      | 124M       | Base     | 
| RoPE         | v1      | 124M       | Instruct | 
| MLA          | v1      | 124M       | Base     | 
| MLA          | v1      | 124M       | Instruct | 

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

Plain train with the defaults:

```console
scripts/train.py
```

Process four sequences per device batch:

```console
scripts/train.py --device-batch-size 8
```

Continue training from a checkpoint:

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
pyright .
pytest -q
```
