"""Explicit manipulation registry and construction boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any

from ._base import BaseManipulation
from .mock import MockManipulation


ManipulationBuilder = Callable[..., BaseManipulation]

MANIPULATION_REGISTRY: dict[str, ManipulationBuilder] = {
    "mock": MockManipulation,
}

# Compatibility with the spelling used by the developer guide while keeping a
# public registry for direct inspection in tests.
_MANIPULATION_REGISTRY = MANIPULATION_REGISTRY


def build_manipulation(config: Mapping[str, Any]) -> BaseManipulation:
    """Build manipulation from ``{"type": <registered name>, ...}``."""

    manipulation_type, options = _split_config(config)
    try:
        builder = MANIPULATION_REGISTRY[manipulation_type]
    except KeyError as exc:
        available = ", ".join(sorted(MANIPULATION_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown manipulation type {manipulation_type!r}; "
            f"available types: {available}"
        ) from exc
    try:
        inspect.signature(builder).bind(**options)
    except TypeError as exc:
        raise TypeError(
            f"invalid options for manipulation type {manipulation_type!r}: {exc}"
        ) from exc
    manipulation = builder(**options)
    if not isinstance(manipulation, BaseManipulation):
        raise TypeError(
            f"manipulation builder for type {manipulation_type!r} returned "
            f"{type(manipulation).__name__}, expected BaseManipulation"
        )
    return manipulation


def _split_config(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(config, Mapping):
        raise TypeError("manipulation config must be a mapping")
    options = dict(config)
    if "type" not in options:
        raise ValueError("manipulation config is missing required key 'type'")
    manipulation_type = options.pop("type")
    if not isinstance(manipulation_type, str) or not manipulation_type.strip():
        raise ValueError("manipulation config 'type' must be a non-empty string")
    return manipulation_type.strip(), options
