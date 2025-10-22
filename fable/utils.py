"""
Utility helpers shared across Fable modules.

Includes TQDM progress bars for cleaner usage signatures.
"""

from tqdm import tqdm


def download_progress(
    total: int | None,
    desc: str | None = None,
    *,
    enabled: bool = True,
):
    """Return a tqdm progress bar for data download operations."""
    return tqdm(
        total=total,
        desc="Downloading" if desc is None else desc,
        unit="B",
        unit_scale=True,
        leave=True,
        disable=not enabled,
    )


def clean_progress(
    total: int | None,
    desc: str | None = None,
    *,
    enabled: bool = True,
):
    """Return a tqdm progress bar for data cleaning operations."""
    return tqdm(
        total=total,
        desc="Cleaning" if desc is None else desc,
        unit="B",
        unit_scale=True,
        leave=True,
        disable=not enabled,
    )


def tokenize_progress(
    total: int,
    desc: str | None = None,
    *,
    enabled: bool = True,
):
    """Return a tqdm progress bar for data tokenization."""
    return tqdm(
        total=total,
        desc="Tokenizing" if desc is None else desc,
        unit="B",
        unit_scale=True,
        leave=True,
        disable=not enabled,
    )


def train_progress(
    total: int,
    desc: str | None = None,
    *,
    enabled: bool = True,
):
    """Return a tqdm progress bar for model training iterations."""
    return tqdm(
        range(total),
        total=total,
        desc="Training" if desc is None else desc,
        unit="batch",
        leave=True,
        disable=not enabled,
    )
