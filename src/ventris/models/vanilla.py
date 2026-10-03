"""A small, complete decoder-only Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class ModelConfig(PreTrainedConfig):
    """The dimensions of the Ventris architecture.

    Terminology used by other implementations:

    | Ventris                 | GPT-2      | HF GPT-2    | Llama       | HF Llama                |
    |-------------------------|------------|-------------|-------------|-------------------------|
    | vocab_size              | n_vocab    | vocab_size  | vocab_size  | vocab_size              |
    | max_position_embeddings | n_ctx      | n_positions | max_seq_len | max_position_embeddings |
    | num_hidden_layers       | n_layer    | n_layer     | n_layers    | num_hidden_layers       |
    | hidden_size             | n_embd     | n_embd      | dim         | hidden_size             |
    | num_attention_heads     | n_head     | n_head      | n_heads     | num_attention_heads     |
    | intermediate_size       | 4 * n_embd | n_inner     | hidden_dim  | intermediate_size       |
    """

    model_type = "ventris-vanilla-v1"

    def __init__(
        self,
        vocab_size: int = 50_257,
        max_position_embeddings: int = 1_024,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 2_048,
        **kwargs,
    ) -> None:
        if not kwargs.pop("tie_word_embeddings", True):
            raise ValueError("Ventris always ties its input and output embeddings")
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        kwargs["tie_word_embeddings"] = True
        super().__init__(**kwargs)
        self.use_cache = False

    def shape_dict(self) -> dict[str, int]:
        """Return the architecture fields used in training reports."""
        return {
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
        }

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads


class Ventris(PreTrainedModel, GenerationMixin):
    """The model. It maps token IDs to next-token logits."""

    config_class = ModelConfig
    _input_embed_layer = "token_embedding"

    def __init__(self, config: ModelConfig | None = None) -> None:
        config = config or ModelConfig()
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.final_norm = nn.RMSNorm(config.hidden_size)
        self.post_init()

    def forward(
        self, input_ids: torch.Tensor, use_cache: bool = False, return_dict: bool = True
    ) -> CausalLMOutput:
        if use_cache:
            raise ValueError("Ventris does not support KV caching")
        if not return_dict:
            raise ValueError("Ventris only returns CausalLMOutput")
        if input_ids.ndim != 2:
            raise ValueError("tokens must have shape (device batch, sequence)")
        sequence_length = input_ids.shape[1]
        if not 1 <= sequence_length <= self.config.max_position_embeddings:
            raise ValueError("sequence length must be between 1 and max_position_embeddings")

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states)

        # Reuse the token embeddings to score every token as the possible next token.
        logits = F.linear(self.final_norm(hidden_states), self.token_embedding.weight)
        return CausalLMOutput(logits=logits)  # pyright: ignore[reportArgumentType]


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.hidden_size)
        self.attention = Attention(config)
        self.mlp_norm = nn.RMSNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Each sublayer reads normalized states and adds its result to the residual stream.
        attention_input = self.attention_norm(hidden_states)
        attention_output = self.attention(attention_input)
        hidden_states = hidden_states + attention_output

        mlp_input = self.mlp_norm(hidden_states)
        mlp_output = self.mlp(mlp_input)

        return hidden_states + mlp_output


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        device_batch_size, length, _ = hidden_states.shape
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Split model width into heads, then put heads before the sequence axis:
        # (batch, sequence, model width) -> (batch, heads, sequence, head width).
        head_shape = (
            device_batch_size,
            length,
            self.config.num_attention_heads,
            self.config.head_dim,
        )
        query = query.view(head_shape).transpose(1, 2)
        key = key.view(head_shape).transpose(1, 2)
        value = value.view(head_shape).transpose(1, 2)

        # Each head attends independently, and the causal mask hides future tokens.
        output = F.scaled_dot_product_attention(query, key, value, is_causal=True)

        # Put the sequence axis back and join the heads into one hidden state.
        output = output.transpose(1, 2).reshape(device_batch_size, length, self.config.hidden_size)
        return self.output_projection(output)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_projection = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_projection = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # SwiGLU uses one learned expansion to gate another element by element.
        gate = F.silu(self.gate_projection(hidden_states))
        up = self.up_projection(hidden_states)
        return self.down_projection(gate * up)
