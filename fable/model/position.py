"""
Position encoding modules.
"""

import jax
import jax.numpy as jnp


def sinusoidal_embeddings(seq_len: int, embed_dim: int) -> jax.Array:
    """
    Compute sinusoidal positional embeddings.

    Parameters
    ----------
    seq_len : int
        Maximum sequence length.
    embed_dim : int
        Dimensionality of embeddings (must be even).

    Returns
    -------
    embeddings: jax.Array
        Array of shape (seq_len, embed_dim) containing sinusoidal encodings.
    """
    positions = jnp.arange(seq_len)[:, None]
    divisor = jnp.exp(jnp.arange(0, embed_dim, 2) * (-jnp.log(10_000.0) / embed_dim))

    angles = positions * divisor
    sin, cos = jnp.sin(angles), jnp.cos(angles)

    # Interleave sin and cos
    embeddings = jnp.zeros((seq_len, embed_dim), dtype=jnp.float32)
    embeddings = embeddings.at[:, 0::2].set(sin)
    embeddings = embeddings.at[:, 1::2].set(cos)

    return embeddings
