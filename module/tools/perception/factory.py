"""Explicit perception registry and profile-aware construction helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any

from ._base import BasePerceptionTool
from .mock import MockPerceptionTool


PerceptionBuilder = Callable[..., BasePerceptionTool]

PERCEPTION_REGISTRY: dict[str, PerceptionBuilder] = {
    "mock": MockPerceptionTool,
}

# Profiles are data, not hidden control flow.  Composition roots may provide a
# larger profile mapping and resolve it before calling build_perception().
PERCEPTION_PROFILES: dict[str, Mapping[str, Any]] = {
    "mock": {"type": "mock"},
}


def build_perception(config: Mapping[str, Any]) -> BasePerceptionTool:
    """Build perception from ``{"type": <registered name>, ...}``."""

    perception_type, options = _split_config(config)
    try:
        builder = PERCEPTION_REGISTRY[perception_type]
    except KeyError as exc:
        available = ", ".join(sorted(PERCEPTION_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown perception type {perception_type!r}; "
            f"available types: {available}"
        ) from exc
    try:
        inspect.signature(builder).bind(**options)
    except TypeError as exc:
        raise TypeError(
            f"invalid options for perception type {perception_type!r}: {exc}"
        ) from exc
    perception = builder(**options)
    if not isinstance(perception, BasePerceptionTool):
        raise TypeError(
            f"perception builder for type {perception_type!r} returned "
            f"{type(perception).__name__}, expected BasePerceptionTool"
        )
    return perception


def build_perception_profile(
    profile_name: str,
    *,
    profiles: Mapping[str, Mapping[str, Any]] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> BasePerceptionTool:
    """Resolve one explicit profile, merge overrides, and build perception."""

    if not isinstance(profile_name, str) or not profile_name.strip():
        raise ValueError("profile_name must be a non-empty string")
    selected_profiles = PERCEPTION_PROFILES if profiles is None else profiles
    if not isinstance(selected_profiles, Mapping):
        raise TypeError("profiles must be a mapping")
    try:
        profile = selected_profiles[profile_name]
    except KeyError as exc:
        available = ", ".join(sorted(str(name) for name in selected_profiles))
        raise ValueError(
            f"unknown perception profile {profile_name!r}; "
            f"available profiles: {available or '<none>'}"
        ) from exc
    if not isinstance(profile, Mapping):
        raise TypeError(f"perception profile {profile_name!r} must be a mapping")
    if overrides is not None and not isinstance(overrides, Mapping):
        raise TypeError("overrides must be a mapping or None")
    return build_perception({**dict(profile), **dict(overrides or {})})


def _split_config(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(config, Mapping):
        raise TypeError("perception config must be a mapping")
    options = dict(config)
    if "type" not in options:
        raise ValueError("perception config is missing required key 'type'")
    perception_type = options.pop("type")
    if not isinstance(perception_type, str) or not perception_type.strip():
        raise ValueError("perception config 'type' must be a non-empty string")
    return perception_type.strip(), options
