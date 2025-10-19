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
    Tokenize a string of text into a sequence of integers IDs.
    """
    char_to_id: dict[str, int] = tokenizer_config["char_to_id"]

    token_ids: list[int] = []
    for char in text:
        try:
            token_ids.append(char_to_id[char])
        except KeyError as exc:
            raise ValueError(
                f"Character {char!r} missing from tokenizer vocabulary."
            ) from exc

    return token_ids


def detokenize(token_ids: Sequence[int], tokenizer_config: dict) -> str:
    """
    Convert a string of sequence of integer IDs back into text.
    """
    char_to_id: dict[str, int] = tokenizer_config["char_to_id"]
    id_to_char = {token_id: char for char, token_id in char_to_id.items()}

    characters: list[str] = []
    for token_id in token_ids:
        try:
            characters.append(id_to_char[token_id])
        except KeyError as exc:
            raise ValueError(
                f"Token ID {token_id} missing from tokenizer vocabulary."
            ) from exc

    return "".join(characters)
