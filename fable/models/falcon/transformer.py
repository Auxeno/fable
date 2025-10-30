"""
Transformer building blocks and configuration utilities.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import truncated_normal

from fable.models.falcon.attention import MultiQueryAttention
from fable.models.falcon.feed_forward import GatedFeedForward
from fable.models.common.rms_norm import RMSNorm


class Transformer(nnx.Module):
    """Transformer decoder block module."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        hidden_mult: int = 4,
        use_bias: bool = False,
        init: Callable = truncated_normal(stddev=0.02),
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(params=0),
    ) -> None:
        self.rms_norm = RMSNorm(dim, dtype=dtype)

        self.attention = MultiQueryAttention(
            dim=dim,
            num_heads=num_heads,
            init=init,
            dtype=dtype,
            rngs=rngs,
        )

        self.feed_forward = GatedFeedForward(
            dim=dim,
            hidden_mult=hidden_mult,
            use_bias=use_bias,
            init=init,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        g = self.attention(x)
        h = self.feed_forward(x)
        x = x + g + h
        x = self.rms_norm(x)

        return x
