"""Reproducibility, device validation, and artifact helpers."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str, require_single_gpu: bool = True) -> torch.device:
    """Resolve the training device and reject accidental multi-GPU visibility."""
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        count = torch.cuda.device_count()
        if require_single_gpu and count != 1:
            raise RuntimeError(
                f"Exactly one visible GPU is required; torch sees {count}. "
                "Set CUDA_VISIBLE_DEVICES to a single index."
            )
        return torch.device("cuda:0")
    return torch.device(requested)


def write_json(value: Any, path: str | Path) -> None:
    """Write JSON atomically enough for local experiment artifacts."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(output)


def append_jsonl(value: Mapping[str, Any], path: str | Path) -> None:
    """Append one JSON log record."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n")


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def parameter_count(model: torch.nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

