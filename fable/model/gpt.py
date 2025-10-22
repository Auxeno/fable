"""
GPT-style autoregressive language model components.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from fable.model.position import sinusoidal_embeddings
from fable.model.transformer import TransformerDecoder


class GPT(nnx.Module):
    """
    GPT-style autoregressive language model.

    Parameters
    ----------
    num_layers : int
        Number of transformer decoder blocks.
    embed_dim : int
        Dimensionality of input embeddings.
    num_heads : int
        Number of attention heads.
    vocab_size : int
        Size of the vocabulary.
    max_seq_len : int
        Maximum sequence length.
    dropout_rate : float, optional
        Dropout rate to apply after embeddings.
    rngs : nnx.Rngs, optional
        Random number generator.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        vocab_size: int,
        max_seq_len: int,
        dropout_rate: float,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        self.positional_encodings = sinusoidal_embeddings(max_seq_len, embed_dim)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(embed_dim, rngs=rngs)

        # Learnable token projection matrix
        self.embedding_matrix = nnx.Param(
            0.02 * rngs.params.normal((vocab_size, embed_dim), dtype=jnp.float32)
        )

        self.transformer_blocks = nnx.List(
            TransformerDecoder(embed_dim, num_heads, dropout_rate, rngs=rngs)
            for _ in range(num_layers)
        )

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """
        Apply GPT model to input token sequences.

        Parameters
        ----------
        tokens : jax.Array
            Input array of shape `(batch_size, seq_len)` containing integer token IDs.

        Returns
        -------
        probs : jax.Array
            Output array of shape `(batch_size, seq_len, vocab_size)` containing next
            token probabilities.
        """
        x = self.embedding_matrix[tokens]

        x += self.positional_encodings

        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = x @ self.embedding_matrix.T

        probs = jax.nn.softmax(x, axis=-1)

        return probs
