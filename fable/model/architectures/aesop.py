from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import truncated_normal

from fable.config import AesopConfig
from fable.model.core.attention import GroupQueryAttention
from fable.model.core.feed_forward import SwiGLUFeedForward
from fable.model.core.normalize import RMSNorm


class Transformer(nnx.Module):
    """Modern-style transformer decoder block module."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        *,
        hidden_mult: int = 3,
        use_bias: bool = False,
        use_alibi: bool = True,
        init: Callable = truncated_normal(stddev=0.02),
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(params=0, dropout=1),
    ) -> None:
        self.rms_norm_1 = RMSNorm(dim, dtype=dtype)
        self.rms_norm_2 = RMSNorm(dim, dtype=dtype)

        self.attention = GroupQueryAttention(
            dim=dim,
            num_heads=num_heads,
            num_groups=num_groups,
            use_alibi=use_alibi,
            init=init,
            dtype=dtype,
            rngs=rngs,
        )

        self.feed_forward = SwiGLUFeedForward(
            dim=dim,
            hidden_mult=hidden_mult,
            use_bias=use_bias,
            init=init,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Attention stream
        x_norm_1 = self.rms_norm_1(x)
        x_att = self.attention(x_norm_1)

        # Feed-forward stream
        x_norm_2 = self.rms_norm_2(x)
        x_ff = self.feed_forward(x_norm_2)

        # Residual
        x = x + x_att + x_ff

        return x
