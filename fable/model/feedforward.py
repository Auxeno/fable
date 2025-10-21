"""
Feedforward neural network modules.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class FeedForward(nnx.Module):
    """
    Transformer-style feed-forward network module.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of input embeddings.
    rngs : nnx.Rngs, optional
        Random number generator for parameter initialization.
    """

    def __init__(self, embed_dim: int, rngs: nnx.Rngs = nnx.Rngs(0)) -> None:
        # Initialize weights and biases for two linear layers
        self.kernel_1 = nnx.Param(
            0.02 * rngs.normal(shape=(embed_dim, 4 * embed_dim), dtype=jnp.float32)
        )
        self.bias_1 = nnx.Param(jnp.zeros(shape=(4 * embed_dim,), dtype=jnp.float32))

        self.kernel_2 = nnx.Param(
            0.02 * rngs.normal(shape=(4 * embed_dim, embed_dim), dtype=jnp.float32)
        )
        self.bias_2 = nnx.Param(jnp.zeros(shape=(embed_dim,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply feed-forward network to the input sequence.

        Parameters
        ----------
        x : jax.Array
            Input array of shape (batch_size, seq_len, embed_dim).
        """
        x = x @ self.kernel_1 + self.bias_1
        x = jax.nn.gelu(x)
        x = x @ self.kernel_2 + self.bias_2

        return x
