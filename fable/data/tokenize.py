"""
Tokenizer utilities for loading configuration and converting between text and tokens.
"""

import json
from pathlib import Path
from typing import Sequence


def load_tokenizer_config() -> dict:
    """
    Load the tokenizer configuration JSON as a dictionary.
    """
    path = Path(__file__).with_name("tokenizer-config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def tokenize(text: str, tokenizer_config: dict) -> list[int]:
    """
    Convert ``text`` into tokenizer IDs using the configured vocabulary.
    """
    raise NotImplementedError("Tokenization logic to be implemented.")


def detokenize(token_ids: Sequence[int], tokenizer_config: dict) -> str:
    """
    Convert tokenizer IDs back into their text representation.
    """
    raise NotImplementedError("Detokenization logic to be implemented.")
