"""
GPT-style autoregressive language model components.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from fable.config import GPTConfig
from fable.model.position import sinusoidal_embeddings
from fable.model.transformer import TransformerDecoder


class GPT(nnx.Module):
    """
    GPT-style autoregressive language model composed from a configuration.

    Parameters
    ----------
    config : GPTConfig
        Hyperparameter bundle describing the model architecture.
    rngs : nnx.Rngs, optional
        Random number generator collection.
    """

    def __init__(
        self,
        config: GPTConfig = GPTConfig(),
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        self.positional_encodings = sinusoidal_embeddings(
            config.max_seq_len, config.embed_dim
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(config.embed_dim, rngs=rngs)

        # Learnable token projection matrix
        self.embedding_matrix = nnx.Param(
            rngs.normal((config.vocab_size, config.embed_dim), dtype=jnp.float32) * 0.02
        )

        self.transformer_blocks = nnx.List(
            TransformerDecoder(
                config.embed_dim,
                config.num_heads,
                config.dropout_rate,
                rngs=rngs,
            )
            for _ in range(config.num_layers)
        )

        self.config = config

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """
        Apply GPT model to input token sequences.

        Parameters
        ----------
        tokens : jax.Array
            Input array of shape `(batch_size, seq_len)` containing integer token IDs.
            Sequences must already be padded or truncated to `config.max_seq_len`.

        Returns
        -------
        logits : jax.Array
            Output array of shape `(batch_size, seq_len, vocab_size)` containing next
            token probability logits.
        """
        x = self.embedding_matrix[tokens]
        x += self.positional_encodings
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.layer_norm(x)
        logits = x @ self.embedding_matrix.T

        return logits
