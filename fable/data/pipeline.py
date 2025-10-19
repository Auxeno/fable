"""
Utilities for downloading and preparing the TinyStories dataset.
"""

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
    verbose: bool = True,
) -> None:
    """
    Download TinyStories splits from Hugging Face into the raw data directory.

    Parameters
    ----------
    raw_dir : str or Path, optional
        Destination directory for the raw dataset files. Defaults to ``data/raw`` under the repository root.
    overwrite : bool, optional
        Re-download files even if they already exist when ``True``. Defaults to ``False``.
    verbose : bool, optional
        Emit simple progress information while downloading. Defaults to ``True``.

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

    display_root = Path(__file__).resolve().parents[3]

    if verbose:
        print("Downloading TinyStories dataset...")

    # Download train and validation splits
    for split, url in _TINYSTORIES_URLS.items():
        filename = f"tinystories-{split}.txt"
        destination = target_dir / filename

        if destination.exists() and not overwrite:
            if verbose:
                print(f"Skipping {split}, file already present.")
            continue

        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request) as response, destination.open("wb") as output_file:
                if verbose:
                    try:
                        display_path = destination.parent.relative_to(display_root)
                    except ValueError:
                        display_path = destination.parent
                    prefix = f"Downloading `{filename}` to `{display_path.as_posix()}`"
                    print(prefix, end="", flush=True)

                total_header = response.headers.get("Content-Length")
                total_bytes = int(total_header) if total_header is not None else None
                downloaded = 0

                while True:
                    # 1 MiB chunks
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break

                    output_file.write(chunk)
                    downloaded += len(chunk)

                    if verbose:
                        if total_bytes:
                            fraction = min(downloaded / total_bytes, 1.0)
                            filled = int(fraction * 20)
                            bar = "█" * filled + "." * (20 - filled)
                            percent = fraction * 100
                            print(
                                f"\r{prefix} [{bar}] {percent:5.1f}%",
                                end="",
                                flush=True,
                            )
                        else:
                            mib = downloaded / (1 << 20)
                            print(
                                f"\r{prefix} {mib:6.1f} MiB",
                                end="",
                                flush=True,
                            )

                if verbose:
                    print()

        except (HTTPError, URLError) as exc:
            raise RuntimeError(
                f"Failed to download TinyStories '{split}' split from {url}."
            ) from exc

    if verbose:
        print("Finished downloading TinyStories dataset.")
