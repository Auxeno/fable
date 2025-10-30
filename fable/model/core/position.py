"""
Position encoding modules.
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
    s = seq_len, d = embed_dim
    """
    s, d = seq_len, embed_dim

    # Relative positions (s, 1) and frequency scales for even dimensions (d / 2,)
    positions = jnp.arange(s, dtype=dtype)[:, None]
    divisor = jnp.exp(jnp.arange(0, d, 2, dtype=dtype) * (-jnp.log(10_000.0) / d))

    # Phase offsets for sine/cosine pairs (s, d / 2)
    angles = positions * divisor
    sin, cos = jnp.sin(angles), jnp.cos(angles)

    # Allocate embedding matrix and write sine/cosine into alternating columns (s, d)
    embeddings = jnp.zeros((s, d), dtype=dtype)
    embeddings = embeddings.at[:, 0::2].set(sin)
    embeddings = embeddings.at[:, 1::2].set(cos)

    return embeddings
