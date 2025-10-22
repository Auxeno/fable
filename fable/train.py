"""
Model training logic and utilities.
"""

from collections import deque

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from fable.config import GPTConfig
from fable.data import load_tokenized_tinystories
from fable.model import GPT
from fable.utils import train_progress


def sample_batch_indices(
    key: jax.Array,
    batch_size: int,
    seq_len: int,
    dataset_length: int,
) -> jax.Array:
    """
    Generate a batch of contiguous token indices.

    Parameters
    ----------
    key : jax.Array
        PRNG key for random number generation.
    batch_size : int
        Number of sequences in the batch.
    seq_len : int
        Length of each sequence.
    dataset_length : int
        Total number of tokens in the dataset.

    Returns
    -------
    indices : jax.Array
        Array of shape `(batch_size, seq_len)` containing token indices.
    """
    start = jax.random.randint(
        key,
        shape=(batch_size,),
        minval=0,
        maxval=dataset_length - seq_len + 1,
        dtype=jnp.int32,
    )
    indices = start[:, None] + jnp.arange(seq_len, dtype=jnp.int32)

    return indices


def build_batch(
    rng: jax.Array,
    tokens: jax.Array,
    batch_size: int,
    seq_len: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Sample a random batch of contiguous input-target token sequences.

    Parameters
    ----------
    rng : jax.Array
        PRNG key for random number generation.
    tokens : jax.Array
        Tokenized dataset as a 1-D array of integer token IDs.
    batch_size : int
        Number of sequences to sample per batch.
    seq_len : int
        Length of each sampled sequence.

    Returns
    -------
    inputs : jax.Array
        Input token sequences of shape `(batch_size, seq_len)`.
    targets : jax.Array
        Next-token targets of shape `(batch_size, seq_len)`.
    rng : jax.Array
        Updated PRNG key.
    """
    rng, key = jax.random.split(rng)

    indices = sample_batch_indices(
        key=key,
        batch_size=batch_size,
        seq_len=seq_len,
        dataset_length=len(tokens),
    )

    inputs = tokens[indices]
    targets = tokens[indices + 1]

    return inputs, targets, rng


def train_step(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    inputs: jax.Array,
    targets: jax.Array,
) -> tuple[jax.Array, nnx.State]:
    """
    Run a single optimization step.

    Parameters
    ----------
    graphdef : nnx.GraphDef
        Static definition for (model, optimizer, metrics, ...).
    state : nnx.State
        Dynamic state tree aligned with `graphdef`.
    inputs : jax.Array
        Token batch fed into the model, shape `(batch, seq_len)`.
    targets : jax.Array
        Teacher-forced next-token targets, shape `(batch, seq_len)`.

    Returns
    -------
    loss : jax.Array
        Scalar training loss for the current batch.
    new_state : nnx.State
        Updated state containing model and optimizer mutations.
    """
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model):
        logits = model(inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=targets,
        ).mean()
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    new_state = nnx.state((model, optimizer))

    return loss, new_state


def train(config: GPTConfig = GPTConfig()) -> tuple[GPT, nnx.Optimizer, nnx.State]:
    """
    Run a simple training loop over the TinyStories dataset.

    Parameters
    ----------
    config : GPTConfig, optional
        Model and optimization hyperparameters.

    Returns
    -------
    tuple[GPT, nnx.Optimizer, nnx.State]
        Trained model, optimizer, and latest NNX state tree.
    """
    rng, key_init = jax.random.split(jax.random.PRNGKey(config.seed))

    # Load dataset
    tokenized = load_tokenized_tinystories()
    train_tokens = tokenized["train"]
    dataset_length = len(train_tokens)

    # Model and optimizer setup
    model = GPT(
        num_layers=config.num_layers,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        dropout_rate=config.dropout_rate,
        rngs=nnx.Rngs(key_init),
    )
    optimizer = nnx.Optimizer(model, optax.adam(config.learning_rate), wrt=nnx.Param)
    graphdef, state = nnx.split((model, optimizer))

    # JIT compile training functions
    step_fn = jax.jit(train_step, static_argnums=(0,))
    batch_fn = jax.jit(build_batch, static_argnums=(2, 3))

    # Number of batches per training epoch, tokens expected to appear once per epoch
    num_batches = dataset_length // (config.batch_size * config.max_seq_len)

    # Track a rolling window of batch losses for printing
    losses = deque(maxlen=100)

    if config.verbose:
        print(f"Training GPT model for {config.num_epochs} epochs...")

    # Main training loop
    for epoch in range(config.num_epochs):
        # Progress bar
        desc = f"Epoch {epoch + 1}/{config.num_epochs}"
        with train_progress(num_batches, desc, enabled=config.verbose) as progress:
            for _ in progress:
                # Load batch of data
                inputs, targets, rng = batch_fn(
                    rng=rng,
                    tokens=train_tokens,
                    batch_size=config.batch_size,
                    seq_len=config.max_seq_len,
                )

                # Perform training step
                loss, state = step_fn(graphdef, state, inputs, targets)

                if config.verbose:
                    losses.append(loss.item())
                    progress.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

    if config.verbose:
        print(
            f"Training complete after {config.num_epochs} epochs "
            f"({config.num_epochs * num_batches} batches).\n"
        )

    nnx.update((model, optimizer), state)

    return model, optimizer, state
