"""
Fable top-level package namespace and exports.
"""

from importlib import metadata as importlib_metadata

from fable.checkpoint import load, save
from fable.config import GPTConfig
from fable.generate import generate_text
from fable.models import GPT
from fable.train import train

try:
    __version__ = importlib_metadata.version("fable")
except importlib_metadata.PackageNotFoundError:
    __version__ = "0.1.1"

__all__ = ["generate_text", "GPT", "GPTConfig", "load", "save", "train", "__version__"]
