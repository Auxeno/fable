"""
GPT-style autoregressive language model components.
"""

import jax
from flax import nnx

from fable.config import GPTConfig
from fable.model.position import sinusoidal_embeddings
from fable.model.transformer import Transformer


class GPT(nnx.Module):
    """GPT-style autoregressive language model composed from a configuration."""

    def __init__(
        self,
        config: GPTConfig = GPTConfig(),
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        self.config = config

        self.positional_encodings = sinusoidal_embeddings(
            config.max_seq_len,
            config.embed_dim,
            dtype=config.param_dtype,
        )
        self.dropout = nnx.Dropout(config.dropout_rate, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(config.embed_dim, rngs=rngs)

        embed_key, blocks_key = jax.random.split(rngs.params())
        block_keys = jax.random.split(blocks_key, config.num_layers)

        # Learnable token projection matrix
        self.embedding_matrix = nnx.Param(
            config.init_fn(
                embed_key,
                (config.vocab_size, config.embed_dim),
                config.param_dtype,
            )
        )

        self.transformer_blocks = nnx.List(
            Transformer(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                dropout_rate=config.dropout_rate,
                hidden_mult=config.mlp_hidden_mult,
                use_bias=config.use_bias,
                init=config.init_fn,
                dtype=config.param_dtype,
                rngs=nnx.Rngs(key),
            )
            for key in block_keys
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
