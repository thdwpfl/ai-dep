"""Configuration loading, validation, and dotted-key overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and return a mutable dictionary."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    return value


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    """Write a configuration as stable YAML."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)


def apply_overrides(
    config: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a deep copy with dotted-key overrides applied."""
    result = copy.deepcopy(dict(config))
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target: dict[str, Any] = result
        for part in parts[:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override below non-mapping key {part!r}")
            target = child
        target[parts[-1]] = value
    return result


def canonical_json(config: Mapping[str, Any]) -> str:
    """Serialize a configuration deterministically for checkpoints/logs."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"))

