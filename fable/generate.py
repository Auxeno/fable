import argparse
import random
import time

import jax
import jax.numpy as jnp
from flax import nnx

from fable.checkpoint import load
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
    next_token = jax.random.categorical(key, next_token_logits).astype(jnp.int8)

    return next_token, rng


def generate_text(
    start: str,
    *,
    model: GPT | None = None,
    checkpoint: str = "demo.ckpt",
    temperature: float = 0.6,
    max_output_tokens: int = 5_000,
    chars_per_second: float = 70.0,
    seed: int | None = None,
) -> None:
    """
    Stream autoregressive text from a GPT model starting from a prompt.

    Parameters
    ----------
    start : str
        Story beginning text to seed generation, must be shorter than the model context.
    model : GPT | None, optional
        The autoregressive model to sample from (will be switched to eval mode).
    checkpoint : str, optional
        Checkpoint filename used when `model` is not provided (default: demo.ckpt).
    temperature : float, optional
        Sampling temperature; lower values make predictions more greedy.
    max_output_tokens : int, optional
        Maximum number of tokens to emit after the prompt before stopping.
    chars_per_second : float, optional
        Target characters-per-second rate for console output throttling.
    seed : int | None, optional
        RNG seed, random seed generated when omitted.
    """
    if model is None:
        model = load(filename=checkpoint)

    # Infer training maximum sequence length from positional encodings
    max_seq_len = len(model.positional_encodings)

    # Retrieve EOT token
    tokenizer_config = load_tokenizer_config()
    eot_token = tokenizer_config["special_tokens"][tokenizer_config["eot_token"]]

    # Tokenise story beginning
    context_tokens: list[int] = tokenize(start, tokenizer_config)
    if not context_tokens:
        raise ValueError("`start` must contain at least one token.")
    if len(context_tokens) >= max_seq_len:
        raise ValueError(f"Prompt longer than model context length of {max_seq_len}.")

    # Create initial context, padding tokens beyond initial sequence with zeros
    context = jnp.zeros((1, max_seq_len), dtype=jnp.int8)
    context = context.at[0, : len(context_tokens)].set(
        jnp.asarray(context_tokens, dtype=jnp.int8)
    )
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
    print(start, end="", flush=True)

    # Configure pacing for console output
    seconds_per_char = 1.0 / chars_per_second
    next_print_time = time.monotonic()

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

        # Detokenise sequence for printing
        new_text = detokenize([next_token.item()], tokenizer_config)

        # Print current prediction in-line
        delay = next_print_time - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        print(new_text, end="", flush=True)
        next_print_time += seconds_per_char

    print()


def main() -> None:
    """Command-line entry point for text generation."""
    parser = argparse.ArgumentParser(
        description="Generate a short story from a Fable GPT checkpoint."
    )
    parser.add_argument(
        "--start",
        "-s",
        help="Prompt text to seed generation. Reads from stdin when omitted.",
    )
    parser.add_argument(
        "--checkpoint",
        default="demo.ckpt",
        help="Checkpoint filename stored under the checkpoints directory.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.6).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5_000,
        help="Maximum number of tokens to generate after the prompt (default: 5000).",
    )
    parser.add_argument(
        "--cps",
        type=float,
        default=70.0,
        help="Target characters-per-second rate for console output (default: 70).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling.",
    )

    args = parser.parse_args()

    start_text = args.start
    if start_text is None:
        try:
            start_text = input("Start: ").strip()
        except EOFError:
            start_text = ""

    if not start_text:
        raise SystemExit("A non-empty start is required to generate text.")

    generate_text(
        start=start_text,
        checkpoint=args.checkpoint,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
        seed=args.seed,
        chars_per_second=args.cps,
    )


if __name__ == "__main__":
    main()
