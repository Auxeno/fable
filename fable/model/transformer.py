"""
Transformer building blocks and configuration utilities.
"""

import jax
from flax import nnx

from .attention import SelfAttention
from .feedforward import FeedForward


class TransformerDecoder(nnx.Module):
    """
    Transformer decoder block module.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of input embeddings.
    num_heads : int
        Number of attention heads.
    dropout_rate : float
        Dropout rate to apply after attention and feed-forward layers.
    rngs : nnx.Rngs, optional
        Random number generator for parameter initialization.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_rate: float,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        self.layer_norm_1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.layer_norm_2 = nnx.LayerNorm(embed_dim, rngs=rngs)

        self.dropout_1 = nnx.Dropout(rate=dropout_rate, rngs=rngs)
        self.dropout_2 = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        self.attention = SelfAttention(embed_dim, num_heads, rngs=rngs)

        self.feedforward = FeedForward(embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply transformer decoder block to the input sequence.

        Parameters
        ----------
        x : jax.Array
            Input array of shape (batch_size, seq_len, embed_dim).

        Returns
        -------
        x : jax.Array
            Output array of shape (batch_size, seq_len, embed_dim).
        """
        residual = x
        x = self.layer_norm_1(x)
        x = self.attention(x, causal=True)
        x = self.dropout_1(x)
        x = x + residual
        x = self.layer_norm_2(x)

        residual = x
        x = self.feedforward(x)
        x = self.dropout_2(x)
        x = x + residual

        return x
