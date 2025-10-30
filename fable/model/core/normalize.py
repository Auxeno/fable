"""
Normalisation layers used within Fable models.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class LayerNorm(nnx.Module):
    """
    Layer normalisation with optional scale and bias.

    Parameters
    ----------
    dim : int
        Dimensionality of the feature axis to normalise.
    epsilon : float, optional
        Small constant added to the variance for numerical stability.
    use_scale : bool, optional
        When True a learnable scale parameter is applied.
    use_bias : bool, optional
        When True a learnable bias parameter is applied.
    dtype : jnp.dtype, optional
        Data type used for scale and bias parameters.
    """

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
        """
        Apply layer normalisation to the final dimension of the input.

        Parameters
        ----------
        x : jax.Array
            Input tensor of arbitrary leading shape with features on the last axis.

        Returns
        -------
        jax.Array
            Normalised tensor with the same shape and dtype as `x`.
        """
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.epsilon)
        x = x * self.scale + self.bias

        return x
