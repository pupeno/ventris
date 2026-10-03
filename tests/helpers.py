"""Small models shared by checkpoint and training tests."""

from ventris.models.vanilla import ModelConfig, Ventris


def tiny_model() -> Ventris:
    return Ventris(
        ModelConfig(
            vocab_size=32,
            max_position_embeddings=8,
            num_hidden_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            intermediate_size=64,
        )
    )
