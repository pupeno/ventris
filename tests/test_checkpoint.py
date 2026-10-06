import json
from dataclasses import asdict, replace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from tests.helpers import tiny_model
from ventris.checkpoint import (
    TRAINING_STATE_FILE,
    load_checkpoint,
    save_checkpoint,
)
from ventris.common import RunConfig, TrainingConfig
from ventris.data import EOS_TEXT
from ventris.models import Model, load_model

TRAINING_CONFIG = TrainingConfig(
    steps=4,
    effective_batch_size=2,
    warmup_steps=1,
    peak_learning_rate=6e-4,
    minimum_learning_rate=6e-5,
    seed=7,
)
RUN_CONFIG = RunConfig(device_batch_size=2)


def optimizer(model: Model) -> torch.optim.AdamW:
    return torch.optim.AdamW(model.parameters(), lr=6e-4)


def tiny_tokenizer(path):
    vocab = {f"token{i}": i for i in range(31)}
    vocab[EOS_TEXT] = 31
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="token0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token=EOS_TEXT).save_pretrained(path)
    return path


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
def test_checkpoint_round_trip(tmp_path, architecture):
    torch.manual_seed(0)
    model = tiny_model(architecture)
    if architecture == "rope":
        model.config.rope_theta = 625.0
    input_ids = torch.randint(32, (2, 8))
    trained_optimizer = optimizer(model)
    model(input_ids).logits.square().mean().backward()
    trained_optimizer.step()
    trained_optimizer.zero_grad()
    expected = model(input_ids).logits

    path = save_checkpoint(
        tmp_path / "latest",
        model,
        trained_optimizer,
        step=3,
        training_config=TRAINING_CONFIG,
        run_config=RUN_CONFIG,
        tokenizer_dir=tiny_tokenizer(tmp_path / "tokenizer"),
        training_seconds_elapsed=12.5,
        best_validation_loss=1.25,
        best_validation_step=2,
    )
    training_state, restored = load_checkpoint(path, torch.device("cpu"), TRAINING_CONFIG)

    torch.testing.assert_close(restored(input_ids).logits, expected)
    torch.testing.assert_close(load_model(path)(input_ids).logits, expected)
    assert restored.generation_config.use_cache is False
    assert (
        restored.generate(  # pyright: ignore[reportAttributeAccessIssue]
            input_ids[:, :2], max_new_tokens=2
        ).shape
        == (2, 4)
    )
    loaded_tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    assert loaded_tokenizer.eos_token_id == 31
    assert loaded_tokenizer.model_max_length == model.config.max_position_embeddings
    assert (
        loaded_tokenizer.encode("token1 token2", add_special_tokens=False)
        == Tokenizer.from_file(str(tmp_path / "tokenizer" / "tokenizer.json"))
        .encode("token1 token2", add_special_tokens=False)
        .ids
    )
    assert (path / "config.json").is_file()
    assert (path / "model.safetensors").is_file()
    assert (path / "generation_config.json").is_file()
    assert (path / "tokenizer.json").is_file()
    saved_config = json.loads((path / "config.json").read_text())
    assert saved_config["hidden_size"] == 32
    assert saved_config["num_hidden_layers"] == 1
    assert saved_config["num_attention_heads"] == 4
    assert saved_config["intermediate_size"] == 64
    assert saved_config["max_position_embeddings"] == 8
    assert saved_config["model_type"] == f"ventris-{architecture}-v1"
    if architecture == "rope":
        assert saved_config["rope_theta"] == restored.config.rope_theta == 625.0
    restored_optimizer = optimizer(restored)
    restored_optimizer.load_state_dict(training_state["optimizer"])
    assert restored_optimizer.state
    for saved_moments, restored_moments in zip(
        trained_optimizer.state.values(), restored_optimizer.state.values(), strict=True
    ):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(restored_moments[key], saved_moments[key])
    assert training_state["step"] == 3
    assert training_state["training_config"] == asdict(TRAINING_CONFIG)
    assert training_state["run_config"] == asdict(RUN_CONFIG)
    assert training_state["training_seconds_elapsed"] == 12.5
    assert training_state["best_validation_loss"] == 1.25
    assert training_state["best_validation_step"] == 2


def test_load_checkpoint_rejects_missing_model_weights(tmp_path):
    model = tiny_model()
    path = tmp_path / "checkpoint"
    path.mkdir()
    model.config.save_pretrained(path)
    torch.save({"optimizer": optimizer(model).state_dict()}, path / TRAINING_STATE_FILE)

    with pytest.raises(OSError, match="model.safetensors"):
        load_checkpoint(path, torch.device("cpu"), TRAINING_CONFIG)


def test_missing_checkpoint_error_names_the_artifact_and_command(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"checkpoint.*scripts/train\.py"):
        load_checkpoint(tmp_path / "missing", torch.device("cpu"), TRAINING_CONFIG)


def test_load_checkpoint_rejects_changed_training_config(tmp_path):
    model = tiny_model()
    path = save_checkpoint(
        tmp_path / "latest",
        model,
        optimizer(model),
        step=1,
        training_config=TRAINING_CONFIG,
        run_config=RUN_CONFIG,
        tokenizer_dir=tiny_tokenizer(tmp_path / "tokenizer"),
        training_seconds_elapsed=0.0,
        best_validation_loss=1.0,
        best_validation_step=1,
    )

    with pytest.raises(ValueError, match="training config does not match checkpoint"):
        load_checkpoint(
            path,
            torch.device("cpu"),
            replace(TRAINING_CONFIG, peak_learning_rate=1e-3),
        )


def test_load_checkpoint_requires_training_config(tmp_path):
    model = tiny_model()
    path = save_checkpoint(
        tmp_path / "latest",
        model,
        optimizer(model),
        step=1,
        training_config=TRAINING_CONFIG,
        run_config=RUN_CONFIG,
        tokenizer_dir=tiny_tokenizer(tmp_path / "tokenizer"),
        training_seconds_elapsed=0.0,
        best_validation_loss=1.0,
        best_validation_step=1,
    )
    checkpoint = torch.load(path / TRAINING_STATE_FILE)
    del checkpoint["training_config"]
    torch.save(checkpoint, path / TRAINING_STATE_FILE)

    with pytest.raises(ValueError, match="training config does not match checkpoint"):
        load_checkpoint(path, torch.device("cpu"), TRAINING_CONFIG)


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
def test_model_only_directory_cannot_resume_training(tmp_path, architecture):
    tiny_model(architecture).save_pretrained(tmp_path)

    with pytest.raises(FileNotFoundError, match="training_state.pt"):
        load_checkpoint(tmp_path, torch.device("cpu"), TRAINING_CONFIG)
