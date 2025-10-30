import jax
import jax.numpy as jnp
from flax import nnx


class RMSNorm(nnx.Module):
    """Root-mean-square normalisation with optional scale."""

    def __init__(
        self,
        dim: int,
        epsilon: float = 1e-5,
        use_scale: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.epsilon = epsilon

        # Learnable scale parameters
        self.scale = nnx.Param(jnp.ones((dim,), dtype=dtype)) if use_scale else 1.0

    def __call__(self, x: jax.Array) -> jax.Array:
        rms = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(rms + self.epsilon)
        x = x * self.scale
        return x
