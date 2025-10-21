"""
Attention mechanisms for sequence modeling.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class SelfAttention(nnx.Module):
    """
    Multi-head self-attention module.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of input embeddings.
    num_heads : int
        Number of attention heads.
    rngs : nnx.Rngs, optional
        Random number generator for parameter initialization.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        if not embed_dim % num_heads == 0:
            raise ValueError("`embed_dim` must be divisible by `num_heads`.")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Initialize projection matrices
        self.qkv_proj = nnx.Param(
            0.02 * rngs.normal(shape=(embed_dim, 3 * embed_dim), dtype=jnp.float32)
        )
        self.out_proj = nnx.Param(
            0.02 * rngs.normal(shape=(embed_dim, embed_dim), dtype=jnp.float32)
        )

    def __call__(self, x: jax.Array, causal: bool) -> jax.Array:
        """
        Apply self-attention to the input sequence.

        Parameters
        ----------
        x : jax.Array
            Input array of shape (batch_size, seq_len, embed_dim).
        causal : bool
            Whether to apply a causal mask to prevent attending to future tokens.
        """
        batch_size, seq_len, embed_dim = x.shape

        qkv = x @ self.qkv_proj
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, self.head_dim, 3)
        q, k, v = jnp.transpose(qkv, (4, 0, 2, 1, 3))
        attention_logits = (q @ k.swapaxes(-1, -2)) / jnp.sqrt(self.head_dim)

        causal_mask = jnp.where(
            causal,
            -1e9 * jnp.triu(jnp.ones_like(attention_logits), k=1),
            0.0,
        )
        attention_logits += causal_mask

        attention_weights = jax.nn.softmax(attention_logits, axis=-1)
        context = (attention_weights @ v).transpose(0, 2, 1, 3)

        outputs = context.reshape(batch_size, seq_len, embed_dim) @ self.out_proj

        return outputs
