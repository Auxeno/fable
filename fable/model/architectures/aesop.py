from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import truncated_normal

from fable.config import AesopConfig
from fable.model.core.attention import GroupQueryAttention
from fable.model.core.feed_forward import SwiGLUFeedForward
from fable.model.core.normalize import RMSNorm


class Transformer(nnx.Module):
    """Modern-style transformer decoder block module."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        *,
        hidden_mult: int = 3,
        use_bias: bool = False,
        use_alibi: bool = True,
        init: Callable = truncated_normal(stddev=0.02),
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(params=0, dropout=1),
    ) -> None:
        self.rms_norm_1 = RMSNorm(dim, dtype=dtype)
        self.rms_norm_2 = RMSNorm(dim, dtype=dtype)

        self.attention = GroupQueryAttention(
            dim=dim,
            num_heads=num_heads,
            num_groups=num_groups,
            use_alibi=use_alibi,
            init=init,
            dtype=dtype,
            rngs=rngs,
        )

        self.feed_forward = SwiGLUFeedForward(
            dim=dim,
            hidden_mult=hidden_mult,
            use_bias=use_bias,
            init=init,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Attention stream
        x_norm_1 = self.rms_norm_1(x)
        x_att = self.attention(x_norm_1)

        # Feed-forward stream
        x_norm_2 = self.rms_norm_2(x)
        x_ff = self.feed_forward(x_norm_2)

        # Residual
        x = x + x_att + x_ff

        return x


class Aesop(nnx.Module):
    """
    Modern-style autoregressive language model composed from a configuration.

    Parameters
    ----------
    config : AesopConfig
        Hyperparameter bundle describing the model architecture.
    init : Callable
        Initialiser for learnable parameters.
    rngs : nnx.Rngs, optional
        Random number generator collection.
    """

    def __init__(
        self,
        config: AesopConfig = AesopConfig(),
        *,
        init: Callable = truncated_normal(stddev=0.02),
        rngs: nnx.Rngs = nnx.Rngs(params=0, dropout=0),
    ) -> None:
        self.config = config
        dtype = getattr(jnp, config.param_dtype)

        self.rms_norm = RMSNorm(config.embed_dim, dtype=dtype)

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
                num_groups=config.num_groups,
                hidden_mult=config.mlp_hidden_mult,
                use_bias=config.use_bias,
                use_alibi=config.use_alibi,
                init=init,
                dtype=dtype,
                rngs=nnx.Rngs(key),
            )
            for key in block_keys
        )

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """
        Apply Aesop model to input token sequences.

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
        for block in self.transformer_blocks:
            x = block(x)
        x = self.rms_norm(x)
        logits = x @ self.embedding_matrix.T

        return logits
