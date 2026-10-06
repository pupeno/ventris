import pytest
import torch

from ventris.models.rope import ModelConfig as RopeConfig
from ventris.models.rope import Ventris as RopeVentris
from ventris.models.vanilla import ModelConfig, Ventris


@pytest.fixture(params=["vanilla", "rope"])
def architecture(request):
    return request.param


def model_for(architecture, config=None):
    if architecture == "rope":
        return RopeVentris(RopeConfig(**config.shape_dict()) if config is not None else None)
    return Ventris(config)


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_position_embeddings=16,
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
    )


@pytest.mark.parametrize(
    "model_class,config_class", [(Ventris, ModelConfig), (RopeVentris, RopeConfig)]
)
def test_model_returns_one_logit_per_token_and_vocabulary_entry(model_class, config_class):
    model = model_class(config_class(**tiny_config().shape_dict()))
    tokens = torch.randint(32, (3, 12))

    assert model(tokens).logits.shape == (3, 12, 32)


def test_attention_cannot_see_future_tokens(architecture):
    model = model_for(architecture, tiny_config()).eval()
    first = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 9, 10]])

    with torch.inference_mode():
        first_logits = model(first).logits
        changed_logits = model(changed_future).logits

    torch.testing.assert_close(first_logits[:, :2], changed_logits[:, :2])


def test_token_embedding_is_also_the_output_projection(architecture):
    model = model_for(architecture, tiny_config())
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    model(torch.tensor([[1, 2]])).logits[0, -1, 7].backward()

    assert parameter_ids.count(id(model.token_embedding.weight)) == 1
    gradient = model.token_embedding.weight.grad
    assert gradient is not None
    assert gradient[7].abs().sum() > 0
    assert torch.isfinite(gradient).all()


def test_tied_embeddings_start_with_moderate_logits(architecture):
    torch.manual_seed(0)
    model = model_for(architecture, tiny_config())

    logits = model(torch.randint(32, (2, 16))).logits

    assert logits.std().item() < 2


def test_generate_recomputes_full_context_without_a_cache(architecture):
    model = model_for(architecture, tiny_config()).eval()
    prompt = torch.tensor([[1, 2, 3]])

    with torch.inference_mode():
        expected = prompt
        for _ in range(3):
            next_token = model(expected).logits[:, -1:].argmax(dim=-1)
            expected = torch.cat((expected, next_token), dim=-1)

        actual = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
            prompt, max_new_tokens=3, do_sample=False
        )

    torch.testing.assert_close(actual, expected)


def test_default_config_has_the_architecture_parameter_count(architecture):
    with torch.device("meta"):
        model = model_for(architecture)

    expected = {"vanilla": 124_337_664, "rope": 123_551_232}
    assert sum(parameter.numel() for parameter in model.parameters()) == expected[architecture]


@pytest.mark.parametrize("shape", [(2, 0), (2, 17), (2, 3, 4)])
def test_model_rejects_inputs_outside_its_sequence_contract(shape, architecture):
    model = model_for(architecture, tiny_config())

    with pytest.raises(ValueError, match="sequence|shape"):
        model(torch.zeros(shape, dtype=torch.long))


def test_model_forward_and_backward_are_finite(architecture):
    model = model_for(architecture, tiny_config())
    logits = model(torch.tensor([[1, 2, 3], [3, 2, 1]])).logits
    logits.square().mean().backward()

    assert torch.isfinite(logits).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
