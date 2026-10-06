import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from ventris.common import default_device
from ventris.data import EOS_TEXT
from ventris.generate import generate, generate_from_model
from ventris.models import Model
from ventris.models.rope import ModelConfig as RopeConfig
from ventris.models.rope import Ventris as RopeVentris
from ventris.models.vanilla import ModelConfig, Ventris


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {f"token{i}": i for i in range(255)}
    vocab["hello"] = 1
    vocab[EOS_TEXT] = 255
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="token0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token=EOS_TEXT, model_max_length=16
    )


def tiny_model(architecture: str = "vanilla", context_length: int = 16) -> Model:
    dimensions = dict(
        vocab_size=256,
        max_position_embeddings=context_length,
        num_hidden_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
    )
    if architecture == "rope":
        return RopeVentris(RopeConfig(**dimensions))
    return Ventris(ModelConfig(**dimensions))


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
def test_generate_loads_a_model_directory_without_training_state(tmp_path, architecture):
    tokenizer = tiny_tokenizer()
    tokenizer.save_pretrained(tmp_path)

    model = tiny_model(architecture)
    model.save_pretrained(tmp_path)

    result = generate(
        tmp_path,
        "hello",
        max_new_tokens=2,
        top_k=1,
    )

    model.to(default_device())  # pyright: ignore[reportArgumentType]
    assert result == generate_from_model(
        model,
        tokenizer,
        "hello",
        max_new_tokens=2,
        top_k=1,
    )


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
def test_generate_from_model_restores_training_mode(architecture):
    tokenizer = tiny_tokenizer()
    model = tiny_model(architecture)
    model.train()
    random_state = torch.random.get_rng_state()

    result = generate_from_model(
        model,
        tokenizer,
        "hello",
        max_new_tokens=2,
        top_k=1,
    )

    assert isinstance(result, str)
    assert model.training
    assert torch.equal(torch.random.get_rng_state(), random_state)


@pytest.mark.parametrize("architecture", ["vanilla", "rope"])
def test_saved_model_context_bounds_generation(tmp_path, architecture):
    tokenizer = tiny_tokenizer()
    tokenizer.save_pretrained(tmp_path)
    model = tiny_model(architecture, context_length=3)
    model.save_pretrained(tmp_path)

    with pytest.raises(ValueError, match="prompt fills the model context"):
        generate(tmp_path, "hello hello hello")
    result = generate(tmp_path, "hello", max_new_tokens=128, top_k=1)
    assert len(tokenizer.encode(result, add_special_tokens=False)) <= 2
