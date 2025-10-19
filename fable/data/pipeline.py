"""
Utilities for downloading and preparing the TinyStories dataset.
"""

import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_TINYSTORIES_URLS: dict[str, str] = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt",
    "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt",
}


def download_tinystories(
    raw_dir: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    """
    Download TinyStories splits from Hugging Face into the raw data directory.

    Parameters
    ----------
    raw_dir : str or Path, optional
        Destination directory for the raw dataset files. Defaults to `data/raw` under the repository root.
    overwrite : bool, optional
        Re-download files even if they already exist when `True`. Defaults to `False`.

    Raises
    ------
    RuntimeError
        Raised when a split cannot be downloaded successfully.
    """
    # Create the target directory if it doesn't exist
    if raw_dir is None:
        target_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    else:
        target_dir = Path(raw_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Download train and validation splits
    for split, url in _TINYSTORIES_URLS.items():
        filename = f"tinystories-{split}.txt"
        destination = target_dir / filename

        if destination.exists() and not overwrite:
            continue

        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request) as response, destination.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
        except (HTTPError, URLError) as exc:
            raise RuntimeError(
                f"Failed to download TinyStories '{split}' split from {url}."
            ) from exc
