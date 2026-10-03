"""Independent rotary mathematics at the agreed numerical and attention seams."""

import cmath

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ventris.models.rope import rotate


def test_adjacent_rotation_matches_complex_reference_and_preserves_pair_norms():
    vectors = torch.arange(1, 49, dtype=torch.float32).reshape(2, 2, 3, 4)
    angles = torch.tensor([[0.0, 0.0], [1.0, 0.1], [2.0, 0.2]])

    actual = rotate(vectors, angles.cos(), angles.sin())

    expected = torch.empty_like(vectors)
    for sequence_index in range(2):
        for head in range(2):
            for position in range(3):
                for pair in range(2):
                    value = complex(
                        *vectors[sequence_index, head, position, 2 * pair : 2 * pair + 2]
                    )
                    rotated = value * cmath.exp(1j * float(angles[position, pair]))
                    expected[sequence_index, head, position, 2 * pair] = rotated.real
                    expected[sequence_index, head, position, 2 * pair + 1] = rotated.imag
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, :, 0], vectors[:, :, 0])
    torch.testing.assert_close(
        actual.reshape(2, 2, 3, 2, 2).square().sum(-1),
        vectors.reshape(2, 2, 3, 2, 2).square().sum(-1),
    )


@pytest.mark.parametrize("mixed_precision", [False, True])
@pytest.mark.parametrize("theta,frequencies", [(10_000.0, (1.0, 0.01)), (625.0, (1.0, 0.04))])
def test_model_rotates_full_queries_and_keys_but_leaves_values_unchanged(
    monkeypatch, mixed_precision, theta, frequencies
):
    import torch.nn.functional as functional

    from ventris.models.rope import ModelConfig, Ventris

    model = Ventris(
        ModelConfig(
            vocab_size=16,
            hidden_size=8,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=16,
            max_position_embeddings=4,
            rope_theta=theta,
        )
    )
    projected = []
    for name, projection in model.named_modules():
        if name.endswith((".query", ".key", ".value")):
            projection.register_forward_hook(lambda module, args, output: projected.append(output))
    attend = functional.scaled_dot_product_attention
    seen = []

    def inspect_attention(query, key, value, **kwargs):
        originals = [item.reshape(2, 3, 2, 4).transpose(1, 2) for item in projected[-3:]]
        for actual, original in zip((query, key), originals[:2], strict=True):
            expected = torch.empty_like(original)
            for position in range(3):
                for pair, frequency in enumerate(frequencies):
                    values = original[:, :, position, 2 * pair : 2 * pair + 2].double()
                    as_complex = torch.view_as_complex(values.contiguous())
                    rotated = as_complex * cmath.exp(1j * position * frequency)
                    expected[:, :, position, 2 * pair : 2 * pair + 2] = torch.view_as_real(rotated)
            torch.testing.assert_close(actual, expected)
        expected_dtype = torch.bfloat16 if mixed_precision else torch.float32
        assert query.dtype == key.dtype == value.dtype == expected_dtype
        torch.testing.assert_close(value, originals[2], rtol=0, atol=0)
        seen.append(value)
        return attend(query, key, value, **kwargs)

    monkeypatch.setattr(functional, "scaled_dot_product_attention", inspect_attention)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=mixed_precision):
        model(torch.tensor([[1, 2, 3], [3, 2, 1]]))
    assert len(seen) == 2


def test_rotary_dot_product_depends_on_relative_position():
    query = torch.tensor([[[[2.0, -1.0, 0.5, 3.0]]]])
    key = torch.tensor([[[[-0.5, 1.5, 2.0, -4.0]]]])
    frequencies = torch.tensor([[1.0, 0.01]])
    at_query = rotate(query, (2 * frequencies).cos(), (2 * frequencies).sin())
    at_key = rotate(key, (5 * frequencies).cos(), (5 * frequencies).sin())
    relative_key = rotate(key, (3 * frequencies).cos(), (3 * frequencies).sin())

    torch.testing.assert_close((at_query * at_key).sum(-1), (query * relative_key).sum(-1))


def test_rope_config_rejects_unusable_head_dimensions_and_frequency_bases():
    from ventris.models.rope import ModelConfig

    invalid = [
        {"hidden_size": 7, "num_attention_heads": 2},
        {"hidden_size": 6, "num_attention_heads": 2},
        {"hidden_size": 0},
        {"hidden_size": -8},
        {"num_attention_heads": 0},
        {"num_attention_heads": -2},
        *({"rope_theta": theta} for theta in (0, -1, float("nan"), float("inf"), -float("inf"))),
    ]
    for overrides in invalid:
        with pytest.raises(ValueError, match="head|rope_theta"):
            ModelConfig(**overrides)


def test_position_tables_and_rotation_use_fp32_outside_autocast():
    from ventris.models.rope import ModelConfig, Ventris

    observed = []

    class InspectTrigonometry(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func in (torch.ops.aten.cos.default, torch.ops.aten.sin.default):
                assert args[0].dtype == result.dtype == torch.float32
                assert args[0].device.type == "cpu"
                assert not torch.is_autocast_enabled("cpu")
                observed.append(result.shape)
            return result

    model = Ventris(
        ModelConfig(
            vocab_size=16,
            hidden_size=8,
            num_attention_heads=2,
            num_hidden_layers=2,
            intermediate_size=16,
            max_position_embeddings=4,
        )
    )
    with torch.autocast("cpu", dtype=torch.bfloat16), InspectTrigonometry():
        model(torch.tensor([[1, 2, 3]]))
    # One cosine and sine table serves every layer, batch element and head.
    assert observed == [torch.Size([3, 2]), torch.Size([3, 2])]

    arithmetic = []

    class InspectRotation(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func in (
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.sub.Tensor,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.stack.default,
            ):
                assert result.dtype == torch.float32
                assert not torch.is_autocast_enabled("cpu")
                arithmetic.append(func)
            return result

    projected = torch.tensor([[[[1.25, -2.5, 0.5, 3.0]]]], dtype=torch.bfloat16)
    angles = torch.tensor([[1.0, 0.01]])
    with torch.autocast("cpu", dtype=torch.bfloat16), InspectRotation():
        result = rotate(projected, angles.cos(), angles.sin())
    assert arithmetic
    assert result.dtype == torch.bfloat16
