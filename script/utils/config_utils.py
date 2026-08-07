"""Load mapping configuration without making YAML a core dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_mapping_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML file, with optional PyYAML fallback.

    JSON is a valid subset of YAML, so the shipped offline profiles need no
    dependency. Conventional YAML syntax is supported when the optional
    ``PyYAML`` package is installed.
    """

    config_path = Path(path).expanduser().resolve()
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"cannot read configuration file {config_path}: {exc}") from exc

    try:
        value = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ValueError(
                f"{config_path} is not JSON-compatible YAML and PyYAML is not "
                "installed; install the 'yaml' optional dependency"
            ) from json_error
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as exc:  # type: ignore[attr-defined]
            raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(value, dict):
        raise TypeError(f"configuration root in {config_path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"configuration keys in {config_path} must be strings")
    return value
