"""
Simple, portable GPT checkpoint save/load using NNX filters.
"""

import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, serialization

from fable.config import GPTConfig
from fable.model import GPT, Aesop


def save(model: GPT | Aesop, checkpoint_name: str = "model_state") -> None:
    path = Path("checkpoints") / checkpoint_name
    path.mkdir(parents=True, exist_ok=True)

    # Split parameters (nnx.Param) and ignore the rest with ...
    _, params, _ = nnx.split(model, nnx.Param, ...)
    pure = nnx.to_pure_dict(params)

    # Move parameters to CPU and convert to NumPy arrays for serialisation
    pure_cpu = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), pure)

    # Serialise and write to disk
    (path / "params.msgpack").write_bytes(serialization.to_bytes(pure_cpu))
    (path / "config.json").write_text(json.dumps(asdict(model.config), indent=2))


def load(checkpoint_name: str = "model_state") -> GPT | Aesop:
    user_path = Path("checkpoints") / checkpoint_name
    package_path = Path(__file__).resolve().parent / "checkpoints" / checkpoint_name

    # Check both user and package paths for the checkpoint
    if user_path.exists():
        path = user_path
    elif package_path.exists():
        path = package_path
    else:
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_name}' not found in {user_path} "
            f"or {package_path}."
        )

    # Build model skeleton from saved config
    config = json.loads((path / "config.json").read_text())
    model = GPT(config=GPTConfig(**config), rngs=nnx.Rngs(params=0, dropout=0))

    # Hydrate with saved parameters
    _, params, _ = nnx.split(model, nnx.Param, ...)
    pure = nnx.to_pure_dict(params)
    params_bytes = (path / "params.msgpack").read_bytes()
    loaded = serialization.from_bytes(pure, params_bytes)
    loaded = jax.tree.map(jnp.asarray, loaded)

    nnx.replace_by_pure_dict(params, loaded)
    nnx.update(model, params)

    return model
