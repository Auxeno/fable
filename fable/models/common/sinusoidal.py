"""
Sinusoidal positional encoding generator.
"""

import jax
import jax.numpy as jnp


def sinusoidal_embeddings(
    seq_len: int,
    embed_dim: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """
    Compute sinusoidal positional embeddings.

    Parameters
    ----------
    seq_len : int
        Maximum sequence length.
    embed_dim : int
        Dimensionality of embeddings (must be even).
    dtype : jnp.dtype, optional
        Array dtype for the resulting embeddings.

    Returns
    -------
    embeddings: jax.Array
        Array of shape `(seq_len, embed_dim)` containing sinusoidal encodings.

    Notes
    -----
    S = seq_len, E = embed_dim
    """
    # Relative positions (S, 1) and frequency scales for even dimensions (E / 2,)
    positions = jnp.arange(seq_len, dtype=dtype)[:, None]
    divisor = jnp.exp(
        jnp.arange(0, embed_dim, 2, dtype=dtype) * (-jnp.log(10_000.0) / embed_dim)
    )

    # Phase offsets for sine/cosine pairs (S, E/2)
    angles = positions * divisor
    sin, cos = jnp.sin(angles), jnp.cos(angles)

    # Allocate embedding matrix and write sine/cosine into alternating columns (S, E)
    embeddings = jnp.zeros((seq_len, embed_dim), dtype=dtype)
    embeddings = embeddings.at[:, 0::2].set(sin)
    embeddings = embeddings.at[:, 1::2].set(cos)

    return embeddings
