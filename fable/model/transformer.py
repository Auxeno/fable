"""
Transformer building blocks and configuration utilities.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import truncated_normal

from fable.model.attention import MultiHeadAttention
from fable.model.feed_forward import FeedForward


class Transformer(nnx.Module):
    """Transformer decoder block module."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout_rate: float,
        *,
        hidden_mult: int = 4,
        use_bias: bool = False,
        init: Callable = truncated_normal(stddev=0.02),
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        self.layer_norm_1 = nnx.LayerNorm(dim, rngs=rngs)
        self.layer_norm_2 = nnx.LayerNorm(dim, rngs=rngs)

        self.dropout_1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.dropout_2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        self.attention = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            init=init,
            dtype=dtype,
            rngs=rngs,
        )

        self.feed_forward = FeedForward(
            dim=dim,
            hidden_mult=hidden_mult,
            use_bias=use_bias,
            init=init,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = self.layer_norm_1(x)
        x = self.attention(x, causal=True)
        x = self.dropout_1(x)
        x = x + residual
        x = self.layer_norm_2(x)

        residual = x
        x = self.feed_forward(x)
        x = self.dropout_2(x)
        x = x + residual

        return x
