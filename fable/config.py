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

    max_seq_len: int = 192
    """Maximum sequence length."""

    dropout_rate: float = 0.1
    """Dropout rate to apply after embeddings and transformer blocks."""

    mlp_hidden_mult: int = 4
    """Width multiplier for the feed-forward network hidden layer."""

    use_bias: bool = False
    """Whether linear projections include a bias term."""

    use_sinusoidal: bool = True
    """Use sinusoidal positional encodings when True."""

    use_alibi: bool = False
    """Add ALiBi positional bias in attention layers when True."""

    param_dtype: str = "bfloat16"
    """Default dtype for model parameters."""

    learning_rate: float = 5e-4
    """Learning rate for the optimiser."""

    batch_size: int = 64
    """Batch size for training."""

    num_epochs: int = 10
    """Number of training epochs."""

    seed: int = 0
    """Initial RNG seed."""

    verbose: bool = True
    """Print training progress to console."""

    enable_checkpointing: bool = True
    """Persist checkpoints during training when True."""

    checkpoint_frequency: int = 10
    """Number of evenly spaced checkpoints to save across the full training run."""


@dataclass(frozen=True)
class AesopConfig:
    num_layers: int = 4
    """Number of transformer decoder layers."""

    embed_dim: int = 128
    """Dimensionality of token embeddings."""

    num_heads: int = 4
    """Number of grouped-query attention heads (embed_dim must be divisible by this value)."""

    num_groups: int = 2
    """Number of grouped-query key/value projections (must divide `num_heads`)."""

    vocab_size: int = get_vocabulary_size()
    """Size of the tokenizer vocabulary."""

    max_seq_len: int = 192
    """Maximum sequence length seen during training/evaluation."""

    mlp_hidden_mult: int = 3
    """Width multiplier for SwiGLU feed-forward inner layer."""

    use_bias: bool = False
    """Whether feed-forward projections include bias terms."""

    use_alibi: bool = True
    """Add ALiBi positional bias in attention layers when True."""

    param_dtype: str = "bfloat16"
    """Default dtype for model parameters."""

    learning_rate: float = 5e-4
    """Learning rate for the optimiser."""

    batch_size: int = 64
    """Batch size for training."""

    num_epochs: int = 10
    """Number of training epochs."""

    seed: int = 0
    """Initial RNG seed."""

    verbose: bool = True
    """Print training progress to console."""

    enable_checkpointing: bool = True
    """Persist checkpoints during training when True."""

    checkpoint_frequency: int = 10
    """Number of evenly spaced checkpoints to save across the full training run."""
