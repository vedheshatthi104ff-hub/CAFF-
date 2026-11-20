"""TOML configuration helpers (standard library only)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    config["_config_path"] = str(path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def resolve_project_path(value: str | Path, config: dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(config["_project_root"]) / path).resolve()

