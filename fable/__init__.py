"""
Fable top-level package namespace and exports.
"""

from fable.checkpoint import load, save
from fable.config import GPTConfig
from fable.generate import generate_text
from fable.model import GPT
from fable.train import train

__all__ = ["generate_text", "GPT", "GPTConfig", "load", "save", "train"]
