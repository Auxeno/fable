"""
Simple, portable checkpoint save/load using NNX filters.
"""

import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, serialization

from fable.config import GPTConfig, AesopConfig
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
    checkpoint_config = {
        "model_type": model.__class__.__name__,
        "config": asdict(model.config),
    }
    (path / "config.json").write_text(json.dumps(checkpoint_config, indent=2))


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
    config_payload = json.loads((path / "config.json").read_text())
    if "model_type" in config_payload:
        model_type = config_payload["model_type"]
        config_dict = config_payload["config"]
    else:
        # Backwards compatibility for legacy checkpoints
        model_type = "GPT"
        config_dict = config_payload

    config_cls_map = {
        "GPT": GPTConfig,
        "Aesop": AesopConfig,
    }
    model_cls_map = {
        "GPT": GPT,
        "Aesop": Aesop,
    }

    if model_type not in config_cls_map:
        raise ValueError(
            f"Unknown model type '{model_type}' in checkpoint '{checkpoint_name}'."
        )

    config = config_cls_map[model_type](**config_dict)
    model = model_cls_map[model_type](config=config, rngs=nnx.Rngs(params=0, dropout=0))

    # Hydrate with saved parameters
    _, params, _ = nnx.split(model, nnx.Param, ...)
    pure = nnx.to_pure_dict(params)
    params_bytes = (path / "params.msgpack").read_bytes()
    loaded = serialization.from_bytes(pure, params_bytes)
    loaded = jax.tree.map(jnp.asarray, loaded)

    nnx.replace_by_pure_dict(params, loaded)
    nnx.update(model, params)

    return model
