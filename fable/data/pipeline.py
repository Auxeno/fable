"""
Utilities for downloading and preparing the TinyStories dataset.
"""

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import jax.numpy as jnp

from fable.data.tokenize import load_tokenizer_config, tokenize
from fable.utils import clean_progress, download_progress, tokenize_progress


def download_tinystories(*, overwrite: bool = False, verbose: bool = True) -> None:
    """
    Download TinyStories splits from Hugging Face into the raw data directory.

    Parameters
    ----------
    overwrite : bool, optional
        Redownload files even if they already exist when `True`. Defaults to `False`.
    verbose : bool, optional
        Prints simple progress information while downloading. Defaults to `True`.

    Raises
    ------
    RuntimeError
        Raised when a split cannot be downloaded successfully.
    """
    # Create the target directory if it doesn't exist
    target_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    target_dir.mkdir(parents=True, exist_ok=True)

    display_root = Path(__file__).resolve().parents[3]

    if verbose:
        print("Downloading TinyStories dataset...")

    # Download train and validation splits from HuggingFace
    for split in ("train", "valid"):
        filename = f"tinystories-{split}.txt"
        destination = target_dir / filename
        url = (
            "https://huggingface.co/datasets/roneneldan/TinyStories/"
            "resolve/main/TinyStories-"
            f"{split}.txt"
        )

        if destination.exists() and not overwrite:
            if verbose:
                print(f"Skipping {split}, file already present.")
            continue

        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request) as response, destination.open("wb") as output_file:
                total_header = response.headers.get("Content-Length")
                total_bytes = int(total_header) if total_header is not None else None

                # Progress bar
                desc = f"Downloading `{filename}`"
                with download_progress(total_bytes, desc, enabled=verbose) as progress:
                    while True:
                        # 1 MiB chunks
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break

                        output_file.write(chunk)
                        progress.update(len(chunk))

        except (HTTPError, URLError) as exc:
            raise RuntimeError(
                f"Failed to download TinyStories '{split}' split from {url}."
            ) from exc

    if verbose:
        try:
            saved_path = target_dir.relative_to(display_root)
        except ValueError:
            saved_path = target_dir
        print(f"TinyStories dataset saved to `{saved_path.as_posix()}`.\n")


def clean_tinystories(*, verbose: bool = True) -> None:
    """
    Remove TinyStories stories containing chars not present in the tokenizer's alphabet.

    Parameters
    ----------
    verbose : bool, optional
        Prints simple progress information while cleaning. Defaults to `True`.
    """
    # Locate raw data directory
    repo_root = Path(__file__).resolve().parents[2]
    source_dir = repo_root / "data" / "raw"

    # Prepare clean data directory
    destination_dir = repo_root / "data" / "clean"
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer configuration
    config = load_tokenizer_config()
    char_to_id: dict[str, int] = config["char_to_id"]
    valid_characters: set[str] = set(char_to_id.keys())
    end_of_text_token: str = config["end_of_text_token"]
    special_tokens: dict[str, int] = config.get("special_tokens", {})

    if end_of_text_token not in special_tokens:
        raise KeyError(
            "Tokenizer config must register the end-of-text token as a special token."
        )

    if verbose:
        print("Cleaning TinyStories dataset...")

    for split in ("train", "valid"):
        source_file = source_dir / f"tinystories-{split}.txt"
        destination_file = destination_dir / f"tinystories-{split}.txt"

        kept = 0
        dropped = 0
        total_bytes = source_file.stat().st_size if source_file.exists() else None

        # Process line-by-line, accumulating until the end-of-text token is found
        with (
            source_file.open("r", encoding="utf-8") as src,
            destination_file.open("w", encoding="utf-8", newline="") as dst,
        ):
            # Buffer to accumulate lines for the current story
            line_buffer: list[str] = []

            # Progress bar
            desc = f"Cleaning `{source_file.name}`"
            with clean_progress(total_bytes, desc, enabled=verbose) as progress:
                # Read through the whole source file
                for line in src:
                    line_buffer.append(line)
                    progress.update(len(line.encode("utf-8")))

                    # Check if the end-of-text token is present in the current line
                    if end_of_text_token in line:
                        # Join the buffered lines to form the complete story
                        story = "".join(line_buffer)

                        # Write the story to the destination if valid, else drop it
                        if set(story).issubset(valid_characters):
                            dst.write(story)
                            kept += 1
                        else:
                            dropped += 1

                        line_buffer.clear()

        if verbose:
            print(f"Finished {split}: kept {kept} stories, dropped {dropped} stories.")

    if verbose:
        try:
            saved_path = destination_dir.relative_to(repo_root)
        except ValueError:
            saved_path = destination_dir
        print(f"TinyStories dataset cleaned and saved to `{saved_path.as_posix()}`.\n")


def tokenize_tinystories(*, verbose: bool = True) -> None:
    """
    Tokenize cleaned TinyStories text files and persist the encoded bytes to disk.

    Parameters
    ----------
    verbose : bool, optional
        Prints simple progress information while cleaning. Defaults to `True`.
    """
    repo_root = Path(__file__).resolve().parents[2]
    source_dir = repo_root / "data" / "clean"
    destination_dir = repo_root / "data" / "tokenized"
    destination_dir.mkdir(parents=True, exist_ok=True)

    config = load_tokenizer_config()
    end_of_text_token = config["end_of_text_token"]
    special_tokens: dict[str, int] = config.get("special_tokens", {})

    if end_of_text_token not in special_tokens:
        raise KeyError(
            "Tokenizer config must include the end-of-text token in `special_tokens`."
        )

    if verbose:
        print("Tokenizing TinyStories dataset...")

    for split in ("train", "valid"):
        source_file = source_dir / f"tinystories-{split}.txt"
        destination_file = destination_dir / f"tinystories-{split}.bin"

        if not source_file.exists():
            raise FileNotFoundError(
                f"Clean TinyStories split `{source_file.as_posix()}` not found. "
                "Run `clean_tinystories` before tokenizing."
            )

        total_bytes = source_file.stat().st_size
        with (
            source_file.open("r", encoding="utf-8") as src,
            destination_file.open("wb") as dst,
        ):
            # Progress bar
            desc = f"Tokenizing `{source_file.name}`"
            with tokenize_progress(total_bytes, desc, enabled=verbose) as progress:
                for line in src:
                    token_ids = tokenize(line, config)
                    dst.write(bytes(token_ids))
                    progress.update(len(line.encode("utf-8")))

        if verbose:
            print(
                f"Finished {split}: tokens written to `{destination_file.as_posix()}`."
            )

    if verbose:
        try:
            saved_path = destination_dir.relative_to(repo_root)
        except ValueError:
            saved_path = destination_dir
        print(f"TinyStories tokens saved to `{saved_path.as_posix()}`.\n")


def load_tokenized_tinystories() -> dict[str, jnp.ndarray]:
    """
    Load tokenized TinyStories splits from disk into JAX arrays.

    Returns
    -------
    dict[str, jnp.ndarray]
        A dictionary mapping split names to JAX arrays of token IDs.
    """
    repo_root = Path(__file__).resolve().parents[2]
    token_dir = repo_root / "data" / "tokenized"

    # Load tokenized splits
    arrays: dict[str, jnp.ndarray] = {}
    for split in ("train", "valid"):
        token_file = token_dir / f"tinystories-{split}.bin"
        if not token_file.exists():
            raise FileNotFoundError(
                f"Tokenized TinyStories split `{token_file.as_posix()}` not found. "
                "Run `tokenize_tinystories` before loading tokens."
            )

        # Load raw bytes as uint8 tokens, then lift into a JAX array
        arrays[split] = jnp.frombuffer(token_file.read_bytes(), dtype=jnp.uint8)

    return arrays


def prepare_tinystories_dataset(
    *,
    overwrite_download: bool = False,
    verbose: bool = True,
) -> None:
    """
    Run the full TinyStories data pipeline: download, clean, and tokenize.

    Parameters
    ----------
    overwrite_download : bool, optional
        Redownload files even if they already exist when `True`. Defaults to `False`.
    verbose : bool, optional
        Prints simple progress information while running the pipeline.
        Defaults to `True`.
    """
    download_tinystories(overwrite=overwrite_download, verbose=verbose)
    clean_tinystories(verbose=verbose)
    tokenize_tinystories(verbose=verbose)
