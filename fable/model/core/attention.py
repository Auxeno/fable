"""
Attention mechanisms for sequence modeling.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.nn.initializers import truncated_normal

from fable.model.core.position import alibi_bias


class MultiHeadAttention(nnx.Module):
    """
    Multi-head self-attention module.

    Parameters
    ----------
    dim : int
        Dimensionality of input embeddings.
    num_heads : int
        Number of attention heads.
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
        num_heads: int,
        use_alibi: bool = False,
        init: Callable = truncated_normal(stddev=0.02),
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(params=0, dropout=1),
    ) -> None:
        assert dim % num_heads == 0, "`dim` must be divisible by `num_heads`."

        key_qkv, key_out = jax.random.split(rngs.params())
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_alibi = use_alibi

        # Initialise input and output projection weight matrices
        self.qkv_proj = nnx.Param(init(key_qkv, (dim, 3 * dim), dtype=dtype))
        self.out_proj = nnx.Param(init(key_out, (dim, dim), dtype=dtype))

    def __call__(self, x: jax.Array, *, causal: bool) -> jax.Array:
        """
        Apply self-attention to the input sequence.

        Parameters
        ----------
        x : jax.Array
            Input array of shape (b, s, d).
        causal : bool
            Whether to apply a causal mask to prevent attending to future tokens.

        Returns
        -------
        outputs : jax.Array
            Attention layer output array of shape `(b, s, d)`.

        Notes
        -----
        b = batch_size, s = seq_len, d = dim, n = num_heads, h = head_dim
        """
        (b, s, d), n, h = x.shape, self.num_heads, self.head_dim

        # Project inputs into queries, keys and values (b, s, 3 * d)
        qkv = x @ self.qkv_proj

        # Separate attention heads from final dimension (b, s, n, h, 3)
        qkv = qkv.reshape(b, s, n, h, 3)

        # Transpose and split into q, k, v vectors 3 * (b, n, s, h)
        q, k, v = jnp.transpose(qkv, (4, 0, 2, 1, 3))

        # Scaled dot-product attention logits (b, n, s, s)
        logits = (q @ k.swapaxes(-1, -2)) / jnp.sqrt(h).astype(x.dtype)

        # Optionally add ALiBi positional bias (b, n, s, s)
        if self.use_alibi:
            logits += alibi_bias(s, n, dtype=logits.dtype)

        # Apply causal mask to prevent attending to future tokens (b, s, s)
        mask = jnp.where(
            causal,
            -1e9 * jnp.triu(jnp.ones((1, 1, s, s), dtype=x.dtype), k=1),
            jnp.zeros((1, 1, s, s), dtype=x.dtype),
        )
        logits += mask

        # Normalise attention weights across key positions (b, n, s, s)
        attention = jax.nn.softmax(logits, axis=-1)

        # Weigh value vectors by attention scores (b, n, s, h)
        x = (attention @ v).transpose(0, 2, 1, 3)

        # Fold heads in embedding dimension and project output (b, s, d)
        x = x.reshape(b, s, d) @ self.out_proj

        return x
