from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import truncated_normal


class GatedFeedForward(nnx.Module):
    """Gated two-layer feed-forward block."""

    def __init__(
        self,
        dim: int,
        hidden_mult: int = 4,
        use_bias: bool = False,
        init: Callable = truncated_normal(stddev=0.02),
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(params=0),
    ) -> None:
        k_up, k_gate, k_down = jax.random.split(rngs.params(), 3)
        hidden = hidden_mult * dim

        # Initialise weights
        self.w_up = nnx.Param(init(k_up, (dim, hidden), dtype=dtype))
        self.w_gate = nnx.Param(init(k_gate, (dim, hidden), dtype=dtype))
        self.w_down = nnx.Param(init(k_down, (hidden, dim), dtype=dtype))

        # Initialise biases
        self.b_up = nnx.Param(jnp.zeros((hidden,), dtype=dtype)) if use_bias else 0.0
        self.b_gate = nnx.Param(jnp.zeros((hidden,), dtype=dtype)) if use_bias else 0.0
        self.b_down = nnx.Param(jnp.zeros((dim,), dtype=dtype)) if use_bias else 0.0

    def __call__(self, x: jax.Array) -> jax.Array:
        # Two parallel projections
        h = jax.nn.gelu(x @ self.w_up + self.b_up)
        g = jax.nn.sigmoid(x @ self.w_gate + self.b_gate)

        # Elementwise gating
        x = h * g

        # Project back down
        x = x @ self.w_down + self.b_down

        return x
