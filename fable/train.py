"""
Model training logic and utilities.
"""

import jax
import jax.numpy as jnp


def generate_batch_indices(
    key: jax.Array,
    batch_size: int,
    seq_len: int,
    dataset_length: int,
) -> jax.Array:
    """
    Draw a batch of contiguous token indices from a dataset.

    Returns
    -------
    indices: jax.Array
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
