"""Sample text from a trained Ventris model."""

from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from ventris.common import default_device, evaluation_mode, mixed_precision
from ventris.data import load_tokenizer
from ventris.models import Model, load_model


def generate(
    checkpoint: Path,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 50,
    seed: int = 0,
) -> str:
    """Load a checkpoint and sample one continuation."""
    model = load_model(checkpoint)
    model.to(default_device())  # pyright: ignore[reportArgumentType]
    return generate_from_model(
        model,
        load_tokenizer(checkpoint),
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )


def generate_from_model(
    model: Model,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 50,
    seed: int = 0,
) -> str:
    """Sample one continuation from an in-memory model."""
    if max_new_tokens <= 0 or temperature <= 0 or top_k < 0:
        raise ValueError(
            "max_new_tokens and temperature must be positive; top_k cannot be negative"
        )

    eos_id = tokenizer.eos_token_id
    assert isinstance(eos_id, int)
    context = tokenizer.encode(prompt, add_special_tokens=False) or [eos_id]
    if len(context) >= model.config.max_position_embeddings:
        raise ValueError("the prompt fills the model context")

    target = next(model.parameters()).device
    input_ids = torch.tensor([context], device=target)
    available = model.config.max_position_embeddings - len(context)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        with evaluation_mode(model), mixed_precision(target):
            output = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                input_ids,
                max_new_tokens=min(max_new_tokens, available),
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
            )

    generated = output[0, len(context) :].tolist()
    if eos_id in generated:
        generated = generated[: generated.index(eos_id)]
    return tokenizer.decode(generated, skip_special_tokens=False)  # pyright: ignore[reportReturnType]
