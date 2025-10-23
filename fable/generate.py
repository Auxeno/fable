import random

import jax
import jax.numpy as jnp
from flax import nnx

from fable.data.tokenize import detokenize, load_tokenizer_config, tokenize
from fable.model import GPT


def predict_next_token(
    rng: jax.Array,
    graphdef: nnx.GraphDef,
    state: nnx.State,
    context: jax.Array,
    last_token_idx: int,
    temperature: float,
) -> tuple[jax.Array, jax.Array]:
    """
    Sample the next token ID from a GPT model given the current context.

    Parameters
    ----------
    rng : jax.Array
        RNG key used for stochastic sampling.
    graphdef : nnx.GraphDef
        Frozen graph definition produced by `nnx.split(model)`.
    state : nnx.State
        Trainable parameters paired with `graphdef`.
    context : jax.Array
        Batched token context of shape `(1, max_seq_len)` consumed by the model.
    last_token_idx : int
        Index of the most recent token available in the context.
    temperature : float
        Softmax temperature applied before categorical sampling.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        The sampled token ID (shape `()`), and the updated RNG key.
    """
    rng, key = jax.random.split(rng)
    model = nnx.merge(graphdef, state)

    # Predict next token for all tokens in sequence
    logits = model(context)

    # Index current token logits, which predict the next token distribution
    next_token_logits = logits[0, last_token_idx, :] / jnp.maximum(temperature, 1e-5)

    # Sample next token
    next_token = jax.random.categorical(key, next_token_logits)

    return next_token, rng


def generate_text(
    model: GPT,
    story_beginning: str,
    *,
    temperature: float = 1.0,
    max_output_tokens: int = 10_000,
    seed: int | None = None,
) -> None:
    """
    Stream autoregressive text from a GPT model starting from a prompt.

    Parameters
    ----------
    model : GPT
        The autoregressive model to sample from (will be switched to eval mode).
    story_beginning : str
        Prompt text to seed generation; must be shorter than the model context.
    temperature : float, optional
        Sampling temperature; lower values make predictions more greedy.
    max_output_tokens : int, optional
        Maximum number of tokens to emit after the prompt before stopping.
    seed : int | None, optional
        RNG seed, random seed generated when omitted.
    """
    # Infer training maximum sequence length from positional encodings
    max_seq_len = len(model.positional_encodings)

    # Retrieve EOT token
    tokenizer_config = load_tokenizer_config()
    eot_token = tokenizer_config["special_tokens"][tokenizer_config["eot_token"]]

    # Tokenize story beginning
    context_tokens: list[int] = tokenize(story_beginning, tokenizer_config)
    if not context_tokens:
        raise ValueError("`story_beginning` must contain at least one token.")
    if len(context_tokens) >= max_seq_len:
        raise ValueError(f"Prompt longer than model context length of {max_seq_len}.")

    # Create initial context, padding tokens beyond initial sequence with zeros
    context = jnp.zeros((1, max_seq_len), dtype=jnp.int32)
    context = context.at[0, : len(context_tokens)].set(jnp.array(context_tokens))
    last_token_idx = len(context_tokens) - 1

    # Seed RNG
    if seed is None:
        seed = random.randint(0, 1_000_000)
    rng = jax.random.PRNGKey(seed)

    # Set model to eval and split
    model.eval()
    graphdef, state = nnx.split(model)

    # JIT compile next token function
    next_token_fn = jax.jit(predict_next_token)

    # Print story beginning
    print(story_beginning, end="", flush=True)

    # Main inference loop
    for _ in range(max_output_tokens):
        # Predict next token for current sequence
        next_token, rng = next_token_fn(
            rng=rng,
            graphdef=graphdef,
            state=state,
            context=context,
            last_token_idx=last_token_idx,
            temperature=temperature,
        )

        # If next token is the EOT token, break for loop
        if next_token.item() == eot_token:
            break

        # Write into next position in context window
        if last_token_idx < max_seq_len - 1:
            context = context.at[0, last_token_idx + 1].set(next_token)
            last_token_idx += 1
        else:
            context = context.at[0, 0 : max_seq_len - 1].set(context[0, 1:max_seq_len])
            context = context.at[0, -1].set(next_token)

        # Detokenize sequence for printing
        new_text = detokenize([next_token.item()], tokenizer_config)

        # Print current prediction in-line
        print(new_text, end="", flush=True)

    print()
