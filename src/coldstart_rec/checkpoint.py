"""Checkpoint serialization with enough metadata for standalone inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .model import ColdStartSetEncoder, ModelConfig


def save_checkpoint(
    path: str | Path,
    model: ColdStartSetEncoder,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    epoch: int,
    dataset_fingerprint: str,
) -> None:
    """Save a CPU-portable model checkpoint."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_class": "ColdStartSetEncoder",
        "model_config": model.config.to_dict(),
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "config": dict(config),
        "metrics": dict(metrics),
        "epoch": int(epoch),
        "dataset_fingerprint": dataset_fingerprint,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def load_model_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[ColdStartSetEncoder, dict[str, Any]]:
    """Load a checkpoint and reconstruct its model from feature buffers."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    state = checkpoint["model_state"]
    model = ColdStartSetEncoder(
        movie_genres=state["movie_genres"],
        movie_years=state["movie_years"],
        config=ModelConfig.from_dict(checkpoint["model_config"]),
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint

