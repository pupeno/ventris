import torch

from ventris.model import ModelConfig, Ventris

EXPECTED_PARAMETER_COUNT = 124_337_664


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_position_embeddings=16,
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
    )


def test_model_returns_one_logit_per_token_and_vocabulary_entry():
    model = Ventris(tiny_config())
    tokens = torch.randint(32, (3, 12))

    assert model(tokens).logits.shape == (3, 12, 32)


def test_attention_cannot_see_future_tokens():
    model = Ventris(tiny_config()).eval()
    first = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 9, 10]])

    with torch.inference_mode():
        first_logits = model(first).logits
        changed_logits = model(changed_future).logits

    torch.testing.assert_close(first_logits[:, :2], changed_logits[:, :2])


def test_token_embedding_is_also_the_output_projection():
    model = Ventris(tiny_config())
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    model(torch.tensor([[1, 2]])).logits[0, -1, 7].backward()

    assert parameter_ids.count(id(model.token_embedding.weight)) == 1
    gradient = model.token_embedding.weight.grad
    assert gradient is not None
    assert gradient[7].abs().sum() > 0


def test_tied_embeddings_start_with_moderate_logits():
    torch.manual_seed(0)
    model = Ventris(tiny_config())

    logits = model(torch.randint(32, (2, 16))).logits

    assert logits.std().item() < 2


def test_generate_recomputes_full_context_without_a_cache():
    model = Ventris(tiny_config()).eval()
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


def test_default_config_is_the_named_124m_model():
    model = Ventris()

    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETER_COUNT
