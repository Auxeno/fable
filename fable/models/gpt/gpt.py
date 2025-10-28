"""
GPT-style autoregressive language model components.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import truncated_normal

from fable.config import GPTConfig
from fable.models.common.dropout import Dropout
from fable.models.common.layer_norm import LayerNorm
from fable.models.common.sinusoidal import sinusoidal_embeddings
from fable.models.gpt.transformer import Transformer


class GPT(nnx.Module):
    """
    GPT-style autoregressive language model composed from a configuration.

    Parameters
    ----------
    config : GPTConfig
        Hyperparameter bundle describing the model architecture.
    init : Callable
        Initialiser for learnable parameters.
    rngs : nnx.Rngs, optional
        Random number generator collection.
    """

    def __init__(
        self,
        config: GPTConfig = GPTConfig(),
        *,
        init: Callable = truncated_normal(stddev=0.02),
        rngs: nnx.Rngs = nnx.Rngs(params=0, dropout=0),
    ) -> None:
        self.config = config
        dtype = getattr(jnp, config.param_dtype)

        self.positional_encodings = sinusoidal_embeddings(
            config.max_seq_len,
            config.embed_dim,
            dtype=dtype,
        )
        self.dropout = Dropout(config.dropout_rate, rngs=rngs)
        self.layer_norm = LayerNorm(config.embed_dim, dtype=dtype)

        embed_key, blocks_key = jax.random.split(rngs.params())
        block_keys = jax.random.split(blocks_key, config.num_layers)

        # Learnable token projection matrix
        self.embedding_matrix = nnx.Param(
            init(
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
                init=init,
                dtype=dtype,
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
        x += self.positional_encodings[None, :, :]
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.layer_norm(x)
        logits = x @ self.embedding_matrix.T

        return logits
