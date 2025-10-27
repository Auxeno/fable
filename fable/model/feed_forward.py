"""
Feedforward neural network modules.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import truncated_normal


class FeedForward(nnx.Module):
    """Standard two-layer feed-forward block."""

    def __init__(
        self,
        dim: int,
        hidden_mult: int = 4,
        use_bias: bool = False,
        init: Callable = truncated_normal(stddev=0.02),
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        key_up, key_down = jax.random.split(rngs.params())
        hidden_dim = hidden_mult * dim

        # Initialise weights
        self.w_up = nnx.Param(init(key_up, (dim, hidden_dim), dtype=dtype))
        self.w_down = nnx.Param(init(key_down, (hidden_dim, dim), dtype=dtype))

        # Initialise biases
        self.b_up = jnp.zeros((hidden_dim,), dtype=dtype)
        self.b_down = jnp.zeros((dim,), dtype=dtype)
        if use_bias:
            self.b_up = nnx.Param(self.b_up)
            self.b_down = nnx.Param(self.b_down)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x @ self.w_up + self.b_up
        x = jax.nn.gelu(x)
        x = x @ self.w_down + self.b_down
        return x
