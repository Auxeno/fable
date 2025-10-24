"""
Model training logic and utilities.
"""

from collections import deque

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from fable.checkpoint import save
from fable.config import GPTConfig
from fable.data import load_tokenized_tinystories
from fable.evaluate import eval_step
from fable.model import GPT
from fable.utils import train_progress


def sample_batch_indices(
    key: jax.Array,
    batch_size: int,
    seq_len: int,
    dataset_length: int,
) -> jax.Array:
    """
    Generate a batch of contiguous token indices.

    Parameters
    ----------
    key : jax.Array
        PRNG key for random number generation.
    batch_size : int
        Number of sequences in the batch.
    seq_len : int
        Length of each sequence.
    dataset_length : int
        Total number of tokens in the dataset.

    Returns
    -------
    indices : jax.Array
        Array of shape `(batch_size, seq_len)` containing token indices.
    """
    start = jax.random.randint(
        key,
        shape=(batch_size,),
        minval=0,
        maxval=dataset_length - seq_len,
        dtype=jnp.int32,
    )
    indices = start[:, None] + jnp.arange(seq_len, dtype=jnp.int32)

    return indices


def build_batch(
    rng: jax.Array,
    tokens: jax.Array,
    batch_size: int,
    seq_len: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Sample a random batch of contiguous input-target token sequences.

    Parameters
    ----------
    rng : jax.Array
        PRNG key for random number generation.
    tokens : jax.Array
        Tokenized dataset as a 1-D array of integer token IDs.
    batch_size : int
        Number of sequences to sample per batch.
    seq_len : int
        Length of each sampled sequence.

    Returns
    -------
    inputs : jax.Array
        Input token sequences of shape `(batch_size, seq_len)`.
    targets : jax.Array
        Next-token targets of shape `(batch_size, seq_len)`.
    rng : jax.Array
        Updated PRNG key.
    """
    rng, key = jax.random.split(rng)

    indices = sample_batch_indices(
        key=key,
        batch_size=batch_size,
        seq_len=seq_len,
        dataset_length=len(tokens),
    )

    inputs = tokens[indices]
    targets = tokens[indices + 1]

    return inputs, targets, rng


def train_step(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    inputs: jax.Array,
    targets: jax.Array,
) -> tuple[jax.Array, nnx.State]:
    """
    Run a single optimization step.

    Parameters
    ----------
    graphdef : nnx.GraphDef
        Static definition for `(model, optimizer, metrics, ...)`.
    state : nnx.State
        Dynamic state tree aligned with `graphdef`.
    inputs : jax.Array
        Token batch fed into the model, shape `(batch, seq_len)`.
    targets : jax.Array
        Teacher-forced next-token targets, shape `(batch, seq_len)`.

    Returns
    -------
    loss : jax.Array
        Scalar training loss for the current batch.
    new_state : nnx.State
        Updated state containing model and optimizer mutations.
    """
    model, optimizer = nnx.merge(graphdef, state)

    def loss_fn(model):
        logits = model(inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=targets,
        ).mean()
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    new_state = nnx.state((model, optimizer))

    return loss, new_state


def train(model: GPT | None = None) -> tuple[GPT, nnx.Optimizer, nnx.State]:
    """
    Run a simple training loop over the TinyStories dataset.

    Parameters
    ----------
    model : GPT, optional
        The model to train. When omitted a new instance is created from
        `GPTConfig()`.

    Returns
    -------
    tuple[GPT, nnx.Optimizer, nnx.State]
        Trained model, optimizer, and latest NNX state tree.
    """
    if model is None:
        # Initialise a model if not provided
        config = GPTConfig()
        rng, key_init = jax.random.split(jax.random.PRNGKey(config.seed))
        model = GPT(config=config, rngs=nnx.Rngs(key_init))
    else:
        # Else read training config from model
        config = model.config
        rng = jax.random.PRNGKey(config.seed)

    # Load dataset from disk
    tokenized = load_tokenized_tinystories()
    train_tokens, valid_tokens = tokenized["train"], tokenized["valid"]

    # Get dataset length, number of gradient steps per epoch, and total optimizer steps
    dataset_length = len(train_tokens)
    epoch_steps = dataset_length // (config.batch_size * config.max_seq_len)
    total_steps = epoch_steps * config.num_epochs

    # Compute checkpoint schedule if enabled
    checkpoint_steps = set()
    if config.enable_checkpointing and config.checkpoint_frequency > 0:
        checkpoint_steps = {
            (total_steps * i) // config.checkpoint_frequency
            for i in range(1, config.checkpoint_frequency + 1)
        }

    # Ensure model is in training mode
    model.train()

    # Initialize optimiser using linear warmup with cosine decay scheduler
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=int(0.01 * total_steps),
        decay_steps=total_steps,
        end_value=0.1 * config.learning_rate,
    )
    optimizer = nnx.Optimizer(model, tx=optax.adamw(learning_rate), wrt=nnx.Param)

    # Separate static and stateful components (NNX functional API)
    graphdef, state = nnx.split((model, optimizer))

    # JIT compile functions used in training loop
    batch_fn = jax.jit(build_batch, static_argnums=(2, 3))
    step_fn = jax.jit(train_step, static_argnums=(0,))
    eval_fn = jax.jit(eval_step, static_argnums=(0,))

    if config.verbose:
        print(f"Training GPT model for {config.num_epochs} epochs...")

    # Main training loop
    for epoch in range(config.num_epochs):
        # Track a rolling window of batch losses for printing
        train_losses, valid_losses = deque(maxlen=100), deque(maxlen=100)

        # Progress bar
        desc = f"Epoch {epoch + 1}/{config.num_epochs}"
        with train_progress(epoch_steps, desc, enabled=config.verbose) as progress:
            for step in progress:
                # Sample batch of training data
                inputs, targets, rng = batch_fn(
                    rng=rng,
                    tokens=train_tokens,
                    batch_size=config.batch_size,
                    seq_len=config.max_seq_len,
                )

                # Perform training step and record loss
                train_loss, state = step_fn(graphdef, state, inputs, targets)
                train_losses.append(train_loss.item())

                # Eval step every 100 train steps
                if step % 100 == 0:
                    # Deterministically sample batch of validation data
                    valid_inputs, valid_targets, _ = batch_fn(
                        rng=jax.random.PRNGKey(0),
                        tokens=valid_tokens,
                        batch_size=config.batch_size,
                        seq_len=config.max_seq_len,
                    )

                    # Record validation loss
                    valid_loss = eval_fn(graphdef, state, valid_inputs, valid_targets)
                    valid_losses.append(valid_loss.item())

                    # Show losses on progress bar
                    progress.set_postfix(
                        train_loss=f"{sum(train_losses) / len(train_losses):.3f}",
                        val_loss=f"{sum(valid_losses) / len(valid_losses):.3f}",
                    )

                # Save checkpoints at configured milestones
                if (global_step := step + 1 + epoch * epoch_steps) in checkpoint_steps:
                    model_snapshot, _ = nnx.merge(graphdef, state)
                    save(
                        nnx.state(model_snapshot),
                        config=config,
                        filename=f"model_step_{global_step}.ckpt",
                    )

    if config.verbose:
        print(
            f"Training complete after {config.num_epochs} epochs "
            f"({total_steps} batches).\n"
        )

    nnx.update((model, optimizer), state)

    return model, optimizer, state
