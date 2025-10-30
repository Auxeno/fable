"""
GPT-style autoregressive language model components.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import truncated_normal

from fable.config import GPTConfig
from fable.model.core.attention import MultiHeadAttention
from fable.model.core.dropout import Dropout
from fable.model.core.feed_forward import FeedForward
from fable.model.core.normalize import LayerNorm
from fable.model.core.position import sinusoidal_embeddings


class Transformer(nnx.Module):
    """GPT-style transformer decoder block module."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout_rate: float,
        *,
        hidden_mult: int = 4,
        use_bias: bool = False,
        init: Callable = truncated_normal(stddev=0.02),
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(params=0, dropout=1),
    ) -> None:
        self.layer_norm_1 = LayerNorm(dim, dtype=dtype)
        self.layer_norm_2 = LayerNorm(dim, dtype=dtype)

        self.dropout_1 = Dropout(rate=dropout_rate, rngs=rngs)
        self.dropout_2 = Dropout(rate=dropout_rate, rngs=rngs)

        self.attention = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            init=init,
            dtype=dtype,
            rngs=rngs,
        )

        self.feed_forward = FeedForward(
            dim=dim,
            hidden_mult=hidden_mult,
            use_bias=use_bias,
            init=init,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = self.layer_norm_1(x)
        x = self.attention(x, causal=True)
        x = self.dropout_1(x)
        x = x + residual
        x = self.layer_norm_2(x)

        residual = x
        x = self.feed_forward(x)
        x = self.dropout_2(x)
        x = x + residual

        return x


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
