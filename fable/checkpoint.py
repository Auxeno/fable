import shutil
from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp
from absl import logging as absl_logging
from flax import nnx

from fable.config import GPTConfig

# Reduce verbosity of Orbax logging
absl_logging.set_verbosity(absl_logging.ERROR)


def save(
    state: nnx.State,
    *,
    filename: str = "model_state.ckpt",
    folder_name: str = "checkpoints",
    path: Path | None = None,
    rng: jax.Array | None = None,
    config: GPTConfig | None = None,
    overwrite: bool = True,
) -> None:
    """Persist the training state (and optional RNG/config) to disk."""
    if path is None:
        checkpoint_path = Path(folder_name) / filename
    else:
        checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.suffix == "":
        checkpoint_path = checkpoint_path.with_suffix(".ckpt")

    checkpoint_path = checkpoint_path.resolve()

    # Remove old checkpoint if needed
    if overwrite and checkpoint_path.exists():
        shutil.rmtree(checkpoint_path, ignore_errors=True)

    # Build payload
    payload: dict[str, Any] = {"state": state}
    if rng is not None:
        payload["rng"] = rng
    if config is not None:
        payload["config"] = config

    # Create and use AsyncCheckpointer with StandardCheckpointHandler
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckptr.save(checkpoint_path, args=ocp.args.StandardSave(payload))  # type: ignore
    ckptr.wait_until_finished()


def load(
    *,
    filename: str = "model_state.ckpt",
    folder_name: str = "checkpoints",
    path: Path | None = None,
) -> dict[str, Any]:
    """Load a checkpoint dictionary from disk."""
    if path is None:
        checkpoint_path = Path(folder_name) / filename
    else:
        checkpoint_path = Path(path)

    if checkpoint_path.suffix == "":
        checkpoint_path = checkpoint_path.with_suffix(".ckpt")

    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    restored = ckptr.restore(checkpoint_path)
    return restored
