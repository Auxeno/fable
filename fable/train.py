"""
Model training logic and utilities.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx


def generate_batch_indices(
    key: jax.Array,
    batch_size: int,
    seq_len: int,
    dataset_length: int,
) -> jax.Array:
    """
    Draw a batch of contiguous token indices from a dataset.

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

    start_indices = jax.random.randint(
        key,
        shape=(batch_size,),
        minval=0,
        maxval=dataset_length - seq_len + 1,
        dtype=jnp.int32,
    )
    slices = start_indices[:, None] + jnp.arange(seq_len, dtype=jnp.int32)

    return slices


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
