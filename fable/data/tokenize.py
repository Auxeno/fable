"""
Tokenizer utilities for loading configuration and converting between text and tokens.
"""

import json
from pathlib import Path
from typing import Sequence


def load_tokenizer_config() -> dict:
    """
    Load the tokenizer configuration JSON as a dictionary.

    Returns
    -------
    dict
        The tokenizer config dictionary.
    """
    path = Path(__file__).with_name("tokenizer-config.json")
    return json.loads(path.read_text(encoding="utf-8"))


def get_vocabulary_size() -> int:
    """
    Compute the tokeniser vocabulary size from the JSON configuration.

    Returns
    -------
    int
        The total number of token IDs defined by the tokeniser.
    """
    config = load_tokenizer_config()
    special_tokens: dict[str, int] = config.get("special_tokens", {})
    return len(config["char_to_id"]) + len(special_tokens)


def tokenize(text: str, tokenizer_config: dict) -> list[int]:
    """
    Tokenise a string of text into a sequence of integers IDs.

    Parameters
    ----------
    text : str
        The input text to tokenise.
    tokenizer_config : dict
        The tokeniser config dictionary, as returned by `load_tokenizer_config()`.
    """
    # Build mapping from character to ID
    char_to_id: dict[str, int] = tokenizer_config["char_to_id"]
    special_tokens: dict[str, int] = tokenizer_config.get("special_tokens", {})

    # Order special tokens longest-first so overlapping prefixes match greedily.
    special_token_items = sorted(
        special_tokens.items(), key=lambda item: len(item[0]), reverse=True
    )

    token_ids: list[int] = []

    contains_special = any(token in text for token, _ in special_token_items)
    if not special_token_items or not contains_special:
        for char in text:
            try:
                token_ids.append(char_to_id[char])
            except KeyError as exc:
                raise ValueError(
                    f"Character {char!r} missing from tokeniser vocabulary."
                ) from exc
        return token_ids

    index = 0
    while index < len(text):
        matched_special = False

        for token_string, token_id in special_token_items:
            if text.startswith(token_string, index):
                token_ids.append(token_id)
                index += len(token_string)
                matched_special = True
                break

        if matched_special:
            continue

        char = text[index]
        try:
            token_ids.append(char_to_id[char])
        except KeyError as exc:
            raise ValueError(
                f"Character {char!r} missing from tokeniser vocabulary."
            ) from exc
        index += 1

    return token_ids


def detokenize(token_ids: Sequence[int], tokenizer_config: dict) -> str:
    """
    Convert a string of sequence of integer IDs back into text.

    Parameters
    ----------
    token_ids : Sequence[int]
        The sequence of token IDs to convert back into text.
    tokenizer_config : dict
        The tokeniser config dictionary, as returned by `load_tokenizer_config()`.
    """
    # Build reverse mapping from ID to character
    char_to_id: dict[str, int] = tokenizer_config["char_to_id"]
    special_tokens: dict[str, int] = tokenizer_config.get("special_tokens", {})

    id_to_symbol: dict[int, str] = {
        token_id: char for char, token_id in char_to_id.items()
    }
    id_to_symbol.update({token_id: token for token, token_id in special_tokens.items()})

    # Convert token IDs back to characters
    characters: list[str] = []
    for token_id in token_ids:
        try:
            characters.append(id_to_symbol[token_id])
        except KeyError as exc:
            raise ValueError(
                f"Token ID {token_id} missing from tokeniser vocabulary."
            ) from exc

    return "".join(characters)
