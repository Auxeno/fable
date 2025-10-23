"""
Configuration management for Fable experiments.
"""

from dataclasses import dataclass

from fable.data.tokenize import get_vocabulary_size


@dataclass(frozen=True)
class GPTConfig:
    num_layers: int = 6
    """Number of transformer decoder layers."""

    embed_dim: int = 384
    """Dimensionality of input embeddings."""

    num_heads: int = 6
    """Number of attention heads."""

    vocab_size: int = get_vocabulary_size()
    """Size of the vocabulary."""

    max_seq_len: int = 192
    """Maximum sequence length."""

    dropout_rate: float = 0.2
    """Dropout rate to apply after embeddings and transformer blocks."""

    learning_rate: float = 3e-4
    """Learning rate for the optimizer."""

    batch_size: int = 64
    """Batch size for training."""

    num_epochs: int = 5
    """Number of training epochs."""

    seed: int = 0
    """Initial RNG seed."""

    verbose: bool = True
    """Print training progress to console."""

    enable_checkpointing: bool = True
    """Persist checkpoints during training when True."""

    checkpoint_frequency: int = 10
    """Number of evenly spaced checkpoints to save across the full training run."""
