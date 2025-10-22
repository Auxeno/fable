"""
Configuration management for Fable experiments.
"""

from dataclasses import dataclass

from fable.data.tokenize import get_vocabulary_size


@dataclass(frozen=True)
class GPTConfig:
    num_layers: int = 4
    """Number of transformer decoder layers."""

    embed_dim: int = 128
    """Dimensionality of input embeddings."""

    num_heads: int = 4
    """Number of attention heads."""

    vocab_size: int = get_vocabulary_size()
    """Size of the vocabulary."""

    max_seq_len: int = 128
    """Maximum sequence length."""

    dropout_rate: float = 0.1
    """Dropout rate to apply after embeddings and transformer blocks."""

    learning_rate: float = 2.5e-4
    """Learning rate for the optimizer."""

    batch_size: int = 64
    """Batch size for training."""

    num_epochs: int = 5
    """Number of training epochs."""

    seed: int = 0
    """Initial RNG seed."""
