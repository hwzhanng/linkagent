"""Explicit environment registry and construction boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any

from ._base import BaseEnv
from .mock import MockEnv


EnvBuilder = Callable[..., BaseEnv]

ENV_REGISTRY: dict[str, EnvBuilder] = {
    "mock": MockEnv,
}


def build_env(config: Mapping[str, Any]) -> BaseEnv:
    """Build an environment from ``{"type": <registered name>, ...}``.

    The mapping is copied and never mutated.  Unknown types and constructor
    options fail at this boundary with the selected type in the error message.
    """

    env_type, options = _split_config(config)
    try:
        builder = ENV_REGISTRY[env_type]
    except KeyError as exc:
        available = ", ".join(sorted(ENV_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown env type {env_type!r}; available types: {available}"
        ) from exc

    try:
        inspect.signature(builder).bind(**options)
    except TypeError as exc:
        raise TypeError(f"invalid options for env type {env_type!r}: {exc}") from exc
    env = builder(**options)
    if not isinstance(env, BaseEnv):
        raise TypeError(
            f"env builder for type {env_type!r} returned {type(env).__name__}, "
            "expected BaseEnv"
        )
    return env


def _split_config(config: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(config, Mapping):
        raise TypeError("env config must be a mapping")
    options = dict(config)
    if "type" not in options:
        raise ValueError("env config is missing required key 'type'")
    env_type = options.pop("type")
    if not isinstance(env_type, str) or not env_type.strip():
        raise ValueError("env config 'type' must be a non-empty string")
    return env_type.strip(), options
