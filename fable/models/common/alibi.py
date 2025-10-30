"""
ALiBi (Attention with Linear Biases) position bias.
"""

import jax
import jax.numpy as jnp


def alibi_bias(
    seq_len: int,
    num_heads: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """
    Compute ALiBi position bias matrix.

    Parameters
    ----------
    seq_len : int
        Sequence length.
    num_heads : int
        Number of attention heads (must be a power of 2).
    dtype : jnp.dtype, optional
        Array dtype for the resulting bias.

    Returns
    -------
    bias : jax.Array
        Array of shape `(num_heads, seq_len, seq_len)` containing
        per-head linear positional biases to add to attention logits.

    Notes
    -----
    H = num_heads, S = seq_len

    bias[h, i, j] = slope[h] * (j - i)
    """
    assert (num_heads & (num_heads - 1)) == 0, "`num_heads` must be a power of 2."

    # Geometric series of per-head slopes 0.5, 0.25, 0.125, ...  (H,)
    head_slopes = (0.5 ** jnp.arange(1, num_heads + 1)).astype(dtype)

    # Relative distances between tokens (i - j) shape: (S, S)
    positions = jnp.arange(seq_len, dtype=dtype)
    rel_pos = positions[None, :] - positions[:, None]

    # Broadcast slopes over sequence dims and multiply by relative distances (H, S, S)
    bias = head_slopes[:, None, None] * rel_pos[None, :, :]

    return bias
