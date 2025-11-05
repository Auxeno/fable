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
    Multi-head self-attention module with causal attention masking.

    Parameters
    ----------
    dim : int
        Dimensionality of input embeddings.
    num_heads : int
        Number of attention heads.
    use_alibi : bool
        Whether to apply ALiBi positional bias.
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

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply self-attention to the input sequence.

        Parameters
        ----------
        x : jax.Array
            Input array of shape (b, s, d).

        Returns
        -------
        outputs : jax.Array
            Attention layer output array of shape `(b, s, d)`.

        Notes
        -----
        b = batch_size, s = seq_len, d = dim, n = num_heads, h = head_dim
        """
        b, s, d, n, h = *x.shape, self.num_heads, self.head_dim

        # Project inputs into queries, keys and values
        qkv = x @ self.qkv_proj  # (b, s, 3 * d)

        # Separate attention heads from final dimension
        qkv = qkv.reshape(b, s, n, h, 3)  # (b, s, n, h, 3)

        # Transpose and split into q, k, v vectors
        q, k, v = jnp.transpose(qkv, (4, 0, 2, 1, 3))  # 3 * (b, n, s, h)

        # Scaled dot-product attention logits
        logits = (q @ k.swapaxes(-1, -2)) / jnp.sqrt(h)  # (b, n, s, s)

        # Optionally add ALiBi positional bias
        if self.use_alibi:
            bias = alibi_bias(s, n, dtype=logits.dtype)  # (n, s, s)
            logits += bias  # (b, n, s, s)

        # Apply causal mask to prevent attending to future tokens
        mask = jnp.triu(jnp.full((s, s), -jnp.inf, dtype=logits.dtype), k=1)  # (s, s)
        logits += mask  # (b, n, s, s)

        # Normalise attention weights across key positions
        attention = jax.nn.softmax(logits, axis=-1)  # (b, n, s, s)

        # Weigh value vectors by attention scores
        x = (attention @ v).transpose(0, 2, 1, 3)  # (b, n, s, h)

        # Fold heads in embedding dimension and project output
        x = x.reshape(b, s, d) @ self.out_proj  # (b, s, d)

        return x


class GroupQueryAttention(nnx.Module):
    """
    Grouped-query self-attention module.

    Parameters
    ----------
    dim : int
        Dimensionality of input embeddings.
    num_heads : int
        Total number of query heads.
    num_groups : int
        Number of groups, each group shares one key/value projection.
    use_alibi : bool
        Whether to apply ALiBi positional bias.
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
        num_groups: int,
        use_alibi: bool = False,
        init: Callable = truncated_normal(stddev=0.02),
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(params=0, dropout=1),
    ) -> None:
        assert dim % num_heads == 0, "`dim` must be divisible by `num_heads`."
        assert num_heads % num_groups == 0, (
            "`num_heads` must be divisible by `num_groups`"
        )

        key_q, key_k, key_v, key_out = jax.random.split(rngs.params(), 4)
        self.head_dim = head_dim = dim // num_heads
        self.heads_per_group = heads_per_group = num_heads // num_groups
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.use_alibi = use_alibi

        # Initialise input and output projection weight matrices
        self.q_proj = nnx.Param(
            init(key_q, (num_groups, heads_per_group, dim, head_dim), dtype=dtype)
        )
        self.k_proj = nnx.Param(init(key_k, (num_groups, dim, head_dim), dtype=dtype))
        self.v_proj = nnx.Param(init(key_v, (num_groups, dim, head_dim), dtype=dtype))
        self.out_proj = nnx.Param(init(key_out, (dim, dim), dtype=dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply grouped-query attention to the input sequence.

        Parameters
        ----------
        x : jax.Array
            Input array of shape (b, s, d).

        Returns
        -------
        outputs : jax.Array
            Attention layer output array of shape (b, s, d).

        Notes
        -----
        b = batch_size, s = seq_len, d = dim, n = num_heads,
        h = head_dim, g = num_groups, m = heads_per_group
        """
        b, s, d, h, n = *x.shape, self.head_dim, self.num_heads
        g, m = self.num_groups, self.heads_per_group

        # Project inputs into per-group query, key, and value vectors
        q = jnp.einsum("bsd,gmdh->bgmsh", x, self.q_proj)  # (b, g, m, s, h)
        k = jnp.einsum("bsd,gdh->bgsh", x, self.k_proj)  # (b, g, s, h)
        v = jnp.einsum("bsd,gdh->bgsh", x, self.v_proj)  # (b, g, s, h)

        # Compute scaled dot-product attention using shared group keys
        logits = jnp.einsum("bgmsh,bgkh->bgmsk", q, k) / jnp.sqrt(h)  # (b, g, m, s, s)

        # Optionally add ALiBi positional bias per head
        if self.use_alibi:
            bias = alibi_bias(s, n, dtype=logits.dtype)  # (n, s, s)
            logits += bias.reshape(g, m, s, s)  # (b, g, m, s, s)

        # Mask out future tokens for causal attention
        mask = jnp.triu(jnp.full((s, s), -jnp.inf, dtype=logits.dtype), k=1)  # (s, s)
        logits += mask  # (b, g, m, s, s)

        # Normalise attention weights across key positions within each group
        attention = jax.nn.softmax(logits, axis=-1)  # (b, g, m, s, s)

        # Combine attention weights with shared values to get context per head
        x = jnp.einsum("bgmsk,bgkh->bsgmh", attention, v)  # (b, s, g, m, h)

        # Merge all groups and heads, then apply output projection
        x = x.reshape(b, s, d) @ self.out_proj  # (b, s, d)

        return x
