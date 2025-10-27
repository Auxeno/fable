"""
Attention mechanisms for sequence modeling.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import truncated_normal


class MultiHeadAttention(nnx.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        init: Callable = truncated_normal(stddev=0.02),
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        assert dim % num_heads != 0, "`dim` must be divisible by `num_heads`."

        key_qkv, key_out = jax.random.split(rngs.params())
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Initialise input and output projection weight matrices
        self.qkv_proj = nnx.Param(init(key_qkv, (dim, 3 * dim), dtype=dtype))
        self.out_proj = nnx.Param(init(key_out, (dim, dim), dtype=dtype))

    def __call__(self, x: jax.Array, *, causal: bool) -> jax.Array:
        batch_size, seq_len, dim = x.shape

        # Project inputs into queries, keys and values (B, S, 3 * D)
        qkv = x @ self.qkv_proj

        # Separate attention heads from final dimension (B, S, N, H, 3)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, self.head_dim, 3)

        # Transpose and split into q, k, v vectors 3 * (B, N, S, H)
        q, k, v = jnp.transpose(qkv, (4, 0, 2, 1, 3))

        # Scaled dot-product attention logits (B, N, S, S)
        logits = (q @ k.swapaxes(-1, -2)) / jnp.sqrt(self.head_dim).astype(x.dtype)

        # Apply causal mask to prevent attending to future tokens (B, S, S)
        mask = jnp.where(
            causal,
            -1e9 * jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=x.dtype), k=1),
            jnp.zeros((1, 1, seq_len, seq_len), dtype=x.dtype),
        )
        logits += mask

        # Normalise attention weights across key positions (B, N, S, S)
        attention = jax.nn.softmax(logits, axis=-1)

        # Weigh value vectors by attention scores (B, N, S, H)
        x = (attention @ v).transpose(0, 2, 1, 3)

        # Fold heads in embedding dimension and project output (B, S, D)
        x = x.reshape(batch_size, seq_len, dim) @ self.out_proj

        return x
