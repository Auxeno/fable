"""
Attention mechanisms for sequence modeling.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import truncated_normal

from fable.models.common.alibi import alibi_bias


class MultiGroupAttention(nnx.Module):
    """Multi-group self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        init: Callable = truncated_normal(stddev=0.02),
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(params=0),
    ) -> None:
        assert dim % num_heads == 0, "`dim` must be divisible by `num_heads`."
        assert num_heads % num_groups == 0, (
            "`num_heads` must be divisible by `num_groups`"
        )

        key_q, key_k, key_v, key_out = jax.random.split(rngs.params(), 4)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Initialise input and output projection weight matrices
        self.q_proj = nnx.Param(init(key_q, (dim, dim), dtype=dtype))
        self.k_proj = nnx.Param(init(key_k, (dim, dim // num_heads), dtype=dtype))
        self.v_proj = nnx.Param(init(key_v, (dim, dim // num_heads), dtype=dtype))
        self.out_proj = nnx.Param(init(key_out, (dim, dim), dtype=dtype))

    def __call__(self, x: jax.Array, causal: bool = True) -> jax.Array:
        batch_size, seq_len, dim = x.shape

        # Project queries, separate attention heads from final dim (B, N, S, H)
        q = x @ self.q_proj
        q = q.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        # Project keys and values, add singleton head dim for broadcasting (B, 1, S, H)
        k = (x @ self.k_proj)[:, None, :, :]
        v = (x @ self.v_proj)[:, None, :, :]

        # Broadcast keys across queries (B, N, S, S)
        logits = (q @ k.swapaxes(-1, -2)) / jnp.sqrt(self.head_dim).astype(x.dtype)

        # Apply ALiBi biases to attention logits (B, N, S, S)
        logits += alibi_bias(seq_len, self.num_heads)[None, :, :, :]

        # Apply causal mask to prevent attending to future tokens (B, S, S)
        causal_mask = jnp.where(
            causal,
            -1e9 * jnp.triu(jnp.ones((1, 1, seq_len, seq_len), dtype=x.dtype), k=1),
            jnp.zeros((1, 1, seq_len, seq_len), dtype=x.dtype),
        )
        logits += causal_mask

        # Normalise attention weights across key positions (B, N, S, S)
        attention = jax.nn.softmax(logits, axis=-1)

        # Weigh value vectors by attention scores (B, N, S, H)
        x = (attention @ v).transpose(0, 2, 1, 3)

        # Fold heads in embedding dimension and project output (B, S, D)
        x = x.reshape(batch_size, seq_len, dim) @ self.out_proj

        return x
