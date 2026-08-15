"""YAML configuration loading with recursive, deterministic inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(
    config_path: str | Path,
    _active_stack: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Load a YAML file and recursively merge its optional ``base_config``."""
    path = Path(config_path).expanduser().resolve()
    if path in _active_stack:
        chain = " -> ".join(str(item) for item in (*_active_stack, path))
        raise ValueError(f"circular base_config inheritance: {chain}")
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as stream:
        current = yaml.safe_load(stream) or {}
    if not isinstance(current, dict):
        raise TypeError(f"top-level YAML node must be a mapping: {path}")

    base_reference = current.pop("base_config", None)
    if base_reference is None:
        merged = current
    else:
        base_path = Path(base_reference)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base = load_config(base_path, (*_active_stack, path))
        merged = _deep_merge(base, current)

    merged["_config_file"] = str(path)
    return merged

