"""
Feedforward neural network modules.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import truncated_normal


class FeedForward(nnx.Module):
    """
    Transformer-style feed-forward network module.

    Parameters
    ----------
    dim : int
        Dimensionality of input embeddings.
    hidden_mult : int
        Width multiplier for the hidden layer.
    use_bias : bool
        Whether to include bias terms in linear projections.
    init : Callable
        Initialiser for learnable parameters.
    dtype : jnp.dtype
        Data type for learnable parameters.
    rngs : nnx.Rngs, optional
        Random number generator for parameter initialisetion.
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: int = 4,
        use_bias: bool = False,
        init: Callable = truncated_normal(stddev=0.02),
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(params=0, dropout=1),
    ) -> None:
        key_up, key_down = jax.random.split(rngs.params())
        hidden = hidden_mult * dim

        # Initialise weights and biases
        self.w_up = nnx.Param(init(key_up, (dim, hidden), dtype=dtype))
        self.w_down = nnx.Param(init(key_down, (hidden, dim), dtype=dtype))
        self.b_up = nnx.Param(jnp.zeros((hidden,), dtype=dtype)) if use_bias else 0.0
        self.b_down = nnx.Param(jnp.zeros((dim,), dtype=dtype)) if use_bias else 0.0

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply feed-forward network to the input sequence.

        Parameters
        ----------
        x : jax.Array
            Input array of shape `(batch_size, seq_len, dim)`.

        Returns
        -------
        x : jax.Array
            Output array of shape `(batch_size, seq_len, dim)`.
        """
        x = x @ self.w_up + self.b_up
        x = jax.nn.gelu(x)
        x = x @ self.w_down + self.b_down

        return x


class SwiGLUFeedForward(nnx.Module):
    """
    Transformer-style SwiGLU feed-forward network module.

    Parameters
    ----------
    dim : int
        Dimensionality of input embeddings.
    hidden_mult : int
        Width multiplier for the hidden layer.
    use_bias : bool
        Whether to include bias terms in linear projections.
    init : Callable
        Initialiser for learnable parameters.
    dtype : jnp.dtype
        Data type for learnable parameters.
    rngs : nnx.Rngs, optional
        Random number generator for parameter initialisation.
    """

    def __init__(
        self,
        dim: int,
        hidden_mult: int = 3,
        use_bias: bool = False,
        init: Callable = truncated_normal(stddev=0.02),
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        key_up, key_gate, key_down = jax.random.split(rngs.params(), 3)
        hidden = hidden_mult * dim

        # Initialise weights and biases
        self.w_up = nnx.Param(init(key_up, (dim, hidden), dtype=dtype))
        self.w_gate = nnx.Param(init(key_gate, (dim, hidden), dtype=dtype))
        self.w_down = nnx.Param(init(key_down, (hidden, dim), dtype=dtype))
        self.b_up = nnx.Param(jnp.zeros((hidden,), dtype=dtype)) if use_bias else 0.0
        self.b_gate = nnx.Param(jnp.zeros((hidden,), dtype=dtype)) if use_bias else 0.0
        self.b_down = nnx.Param(jnp.zeros((dim,), dtype=dtype)) if use_bias else 0.0

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply SwiGLU feed-forward network to the input sequence.

        Parameters
        ----------
        x : jax.Array
            Input array of shape `(batch_size, seq_len, dim)`.

        Returns
        -------
        x : jax.Array
            Output array of shape `(batch_size, seq_len, dim)`.
        """
        # Two parallel projections
        h = x @ self.w_up + self.b_up
        z = x @ self.w_gate + self.b_gate
        g = z * jax.nn.sigmoid(z)

        # Elementwise gating
        x = h * g

        # Project back down
        x = x @ self.w_down + self.b_down
        return x
