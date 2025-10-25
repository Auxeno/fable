import shutil
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp
from absl import logging as absl_logging
from flax import nnx

from fable.config import GPTConfig
from fable.model import GPT

# Reduce verbosity of Orbax logging
absl_logging.set_verbosity(absl_logging.ERROR)


def save(
    model: GPT,
    *,
    filename: str = "model_state.ckpt",
    folder_name: Path | str = "checkpoints",
    path: Path | None = None,
    overwrite: bool = True,
) -> None:
    """
    Saves a GPT model to disk using Orbax (configuration stored for reloads).

    Parameters
    ----------
    model : GPT
        Model instance whose parameters should be persisted.
    filename : str, optional
        Checkpoint filename. Defaults to `"model_state.ckpt"`.
    folder_name : Path | str, optional
        Directory where checkpoints are stored. Defaults to the packaged
        `fable/checkpoints` directory.
    path : pathlib.Path, optional
        Full path to the checkpoint directory. Overrides `folder_name` and
        `filename` when provided.
    overwrite : bool, optional
        Whether to remove an existing checkpoint at the same path. Defaults to
        `True`.
    """
    if path is None:
        base_dir = Path(folder_name)
        if not base_dir.is_absolute():
            base_dir = (Path(__file__).resolve().parent / base_dir).resolve()
        checkpoint_path = base_dir / filename
    else:
        checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.suffix == "":
        checkpoint_path = checkpoint_path.with_suffix(".ckpt")

    checkpoint_path = checkpoint_path.resolve()

    # Remove old checkpoint if needed
    if overwrite and checkpoint_path.exists():
        shutil.rmtree(checkpoint_path, ignore_errors=True)

    # Build payload using a pure mapping so checkpoint round-trips stay portable.
    state = nnx.state(model)

    payload: dict[str, Any] = {"state": nnx.to_pure_dict(state)}
    payload["config"] = asdict(model.config)

    # Create and use AsyncCheckpointer with StandardCheckpointHandler
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckptr.save(checkpoint_path, args=ocp.args.StandardSave(payload))  # type: ignore
    ckptr.wait_until_finished()


def load(
    *,
    filename: str = "demo.ckpt",
    folder_name: Path | str = "checkpoints",
    path: Path | None = None,
) -> GPT:
    """
    Restore a GPT model from a saved checkpoint.

    Parameters
    ----------
    filename : str, optional
        Checkpoint filename. Defaults to `"demo.ckpt"`.
    folder_name : Path | str, optional
        Directory where checkpoints are stored. Defaults to the packaged
        `fable/checkpoints` directory.
    path : pathlib.Path, optional
        Full path to the checkpoint directory. Overrides `folder_name` and
        `filename` when provided.

    Returns
    -------
    GPT
        The restored model instance.

    Raises
    ------
    FileNotFoundError
        If the checkpoint path does not exist.
    KeyError
        If the checkpoint is missing the expected payload keys.
    TypeError
        If the stored state payload is not a mapping produced by
        `nnx.to_pure_dict`.
    """
    if path is None:
        base_dir = Path(folder_name)
        if not base_dir.is_absolute():
            base_dir = (Path(__file__).resolve().parent / base_dir).resolve()
        checkpoint_path = base_dir / filename
    else:
        checkpoint_path = Path(path)

    if checkpoint_path.suffix == "":
        checkpoint_path = checkpoint_path.with_suffix(".ckpt")

    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    restored = ckptr.restore(checkpoint_path)

    if "config" not in restored:
        raise KeyError("Checkpoint is missing a saved GPTConfig payload.")
    if "state" not in restored:
        raise KeyError("Checkpoint is missing the model state payload.")

    config = GPTConfig(**restored["config"])

    # Build a fresh model instance and hydrate it with the restored mapping
    model = GPT(config=config, rngs=nnx.Rngs(0))
    state_payload = restored["state"]
    if not isinstance(state_payload, Mapping):
        raise TypeError(
            "Checkpoint state must be a mapping produced by nnx.to_pure_dict; "
            f"got {type(state_payload)!r}"
        )

    # Hydrate the model with the checkpoint contents.
    model_state = nnx.state(model)
    nnx.replace_by_pure_dict(model_state, state_payload)  # type: ignore
    nnx.update(model, model_state)

    return model
