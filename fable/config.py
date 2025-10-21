"""
Configuration management for Fable experiments.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    num_layers: int = 4
    """Number of transformer decoder layers."""

    embed_dim: int = 128
    """Dimensionality of input embeddings."""

    num_heads: int = 4
    """Number of attention heads."""

    vocab_size: int = 128  # TODO: set from tokenizer
    """Size of the vocabulary."""

    max_seq_len: int = 512
    """Maximum sequence length."""

    dropout_rate: float = 0.1
    """Dropout rate to apply after embeddings and transformer blocks."""

    learning_rate: float = 1e-4
    """Learning rate for the optimizer."""

    batch_size: int = 32
    """Batch size for training."""

    num_epochs: int = 10
    """Number of training epochs."""
