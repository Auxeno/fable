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
    state: nnx.State,
    *,
    config: GPTConfig,
    filename: str = "model_state.ckpt",
    folder_name: str = "checkpoints",
    path: Path | None = None,
    overwrite: bool = True,
) -> None:
    """Persist model state and configuration to disk."""
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

    # Build payload using an Orbax-friendly pure dict representation for the model state
    payload: dict[str, Any] = {"state": nnx.to_pure_dict(state)}
    payload["config"] = asdict(config)

    # Create and use AsyncCheckpointer with StandardCheckpointHandler
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckptr.save(checkpoint_path, args=ocp.args.StandardSave(payload))  # type: ignore
    ckptr.wait_until_finished()


def load(
    *,
    filename: str = "model_state.ckpt",
    folder_name: str = "checkpoints",
    path: Path | None = None,
) -> tuple[GPT, GPTConfig]:
    """Load a GPT model and its configuration from disk."""
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

    if "config" not in restored:
        raise KeyError("Checkpoint is missing a saved GPTConfig payload.")
    if "state" not in restored:
        raise KeyError("Checkpoint is missing the model state payload.")

    config = GPTConfig(**restored["config"])

    # Build a fresh model instance and hydrate it with the restored state, which
    # may be stored either as a pure mapping (new checkpoints) or as an
    # nnx.State (legacy checkpoints).
    model = GPT(config=config, rngs=nnx.Rngs(0))
    state_payload = restored["state"]
    if isinstance(state_payload, Mapping):
        model_state = nnx.state(model)
        nnx.replace_by_pure_dict(model_state, state_payload)  # type: ignore
        nnx.update(model, model_state)
    elif isinstance(state_payload, nnx.State):
        nnx.update(model, state_payload)
    else:
        raise TypeError(
            "Checkpoint state must be a mapping or nnx.State; "
            f"got {type(state_payload)!r}"
        )

    return model, config
