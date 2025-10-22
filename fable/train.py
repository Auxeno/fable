"""
Model training logic and utilities.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from fable.config import GPTConfig
from fable.data.pipeline import load_tokenized_tinystories
from fable.model.gpt import GPT


def sample_batch_indices(
    rng: jax.Array,
    batch_size: int,
    seq_len: int,
    dataset_length: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Draw a batch of contiguous token indices from a dataset.

    Parameters
    ----------
    rng : jax.Array
        PRNG state for random number generation.
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
    rng, key = jax.random.split(rng)

    start = jax.random.randint(
        key,
        shape=(batch_size,),
        minval=0,
        maxval=dataset_length - seq_len + 1,
        dtype=jnp.int32,
    )
    indices = start[:, None] + jnp.arange(seq_len, dtype=jnp.int32)

    return indices, rng


def train_step(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    inputs: jax.Array,
    targets: jax.Array,
) -> nnx.State:
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

    grads = nnx.grad(loss_fn)(model)
    optimizer.update(model, grads)

    new_state = nnx.state((model, optimizer))

    return new_state


def train(config: GPTConfig) -> tuple[GPT, nnx.Optimizer, nnx.State]:
    """
    Run a simple training loop over the TinyStories dataset.

    Parameters
    ----------
    config : GPTConfig
        Model and optimization hyperparameters.

    Returns
    -------
    tuple[GPT, nnx.Optimizer, nnx.State]
        Trained model, optimizer, and latest NNX state tree.
    """

    tokenized = load_tokenized_tinystories()
    train_tokens = tokenized["train"]
    dataset_length = len(train_tokens)

    rng, key_init = jax.random.split(jax.random.PRNGKey(config.seed))
    model = GPT(
        num_layers=config.num_layers,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        dropout_rate=config.dropout_rate,
        rngs=nnx.Rngs(key_init),
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adam(config.learning_rate),
        wrt=nnx.Param,
    )

    graphdef, state = nnx.split((model, optimizer))

    # JIT compile train step
    step_fn = jax.jit(train_step)

    # Each token in dataset is expected to appear once per epoch
    batches_per_epoch = dataset_length // (config.batch_size * config.max_seq_len)

    for epoch in range(config.num_epochs):
        for step in range(batches_per_epoch):
            indices, rng = sample_batch_indices(
                rng=rng,
                batch_size=config.batch_size,
                seq_len=config.max_seq_len,
                dataset_length=dataset_length,
            )
            inputs = train_tokens[indices]
            targets = train_tokens[indices + 1]
            state = step_fn(graphdef, state, inputs, targets)

    nnx.update((model, optimizer), state)

    return model, optimizer, state
