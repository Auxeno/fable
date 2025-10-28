"""
Dropout layers used within Fable models.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class Dropout(nnx.Module):
    """
    Randomly zeros activations during training to prevent overfitting.

    Parameters
    ----------
    rate : float
        Probability of dropping each activation (between 0 and 1).
    deterministic : bool, optional
        When True dropout is disabled. Defaults to False.
    rngs : nnx.Rngs, optional
        RNG collection used for sampling dropout masks.
    """

    def __init__(
        self,
        rate: float,
        *,
        deterministic: bool = False,
        rngs: nnx.Rngs = nnx.Rngs(params=0, dropout=1),
    ) -> None:
        assert 0.0 <= rate <= 1.0, "`rate` must be in closed interval [0, 1]."

        self.rate = rate
        self.deterministic = deterministic
        self.rngs = rngs

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool | None = None,
    ) -> jax.Array:
        """
        Apply dropout to the input tensor.

        Parameters
        ----------
        x : jax.Array
            Input tensor to regularise.
        deterministic : bool, optional
            Overrides the module's deterministic flag when provided.

        Returns
        -------
        jax.Array
            Tensor with randomly dropped activations.
        """
        if deterministic is None:
            deterministic = self.deterministic

        if deterministic or self.rate == 0.0:
            return x

        if self.rate == 1.0:
            return jnp.zeros_like(x)

        key = self.rngs.dropout()

        # Chance of keeping each activation
        keep_prob = scale = 1.0 - self.rate

        # Sample elementwise dropout mask
        mask = jax.random.bernoulli(key, keep_prob, x.shape)

        # Apply mask and rescale activations to maintain expected value
        return x * mask.astype(x.dtype) / scale
