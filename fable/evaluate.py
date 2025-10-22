"""
Model evaluation logic and utilities.
"""

import jax
import optax
from flax import nnx


def eval_step(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    inputs: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """
    Evaluate the model on a batch without applying parameter updates.

    Parameters
    ----------
    graphdef : nnx.GraphDef
        Static definition describing the model and auxiliary modules.
    state : nnx.State
        Mutable state aligned with `graphdef`.
    inputs : jax.Array
        Token batch fed into the model, shape `(batch, seq_len)`.
    targets : jax.Array
        Integer next-token labels, shape `(batch, seq_len)`.

    Returns
    -------
    loss : jax.Array
        Scalar mean cross-entropy loss for the batch.
    """
    model, _ = nnx.merge(graphdef, state)

    logits = model(inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=targets,
    ).mean()

    return loss
