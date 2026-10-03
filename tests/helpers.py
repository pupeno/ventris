"""Small models shared by checkpoint and training tests."""

from ventris.models import Model
from ventris.models.rope import ModelConfig as RopeConfig
from ventris.models.rope import Ventris as RopeVentris
from ventris.models.vanilla import ModelConfig, Ventris


def tiny_model(architecture: str = "vanilla") -> Model:
    dimensions = dict(
        vocab_size=32,
        max_position_embeddings=8,
        num_hidden_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
    )
    if architecture == "rope":
        return RopeVentris(RopeConfig(**dimensions))
    return Ventris(ModelConfig(**dimensions))
