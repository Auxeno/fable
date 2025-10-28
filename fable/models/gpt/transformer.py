"""
Transformer building blocks and configuration utilities.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import truncated_normal

from fable.models.gpt.attention import MultiHeadAttention
from fable.models.common.dropout import Dropout
from fable.models.gpt.feed_forward import FeedForward
from fable.models.common.layer_norm import LayerNorm


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
        rngs: nnx.Rngs = nnx.Rngs(params=0, dropout=1),
    ) -> None:
        self.layer_norm_1 = LayerNorm(dim, dtype=dtype)
        self.layer_norm_2 = LayerNorm(dim, dtype=dtype)

        self.dropout_1 = Dropout(rate=dropout_rate, rngs=rngs)
        self.dropout_2 = Dropout(rate=dropout_rate, rngs=rngs)

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
