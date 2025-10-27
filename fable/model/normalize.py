"""
Normalisation layers used within Fable models.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class LayerNorm(nnx.Module):
    """Layer normalisation with optional scale and bias."""

    def __init__(
        self,
        dim: int,
        epsilon: float = 1e-5,
        use_scale: bool = True,
        use_bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.epsilon = epsilon

        # Learnable scale and bias parameters
        self.scale = nnx.Param(jnp.ones((dim,), dtype=dtype)) if use_scale else 1.0
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=dtype)) if use_bias else 0.0

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        x = x * self.scale
        x = x + self.bias
        return x
