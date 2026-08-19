"""Lazily loaded Env abstraction, mock, and explicit factory."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["BaseEnv", "ENV_REGISTRY", "MockEnv", "build_env"]

_EXPORTS = {
    "BaseEnv": ("module.tools.env._base", "BaseEnv"),
    "ENV_REGISTRY": ("module.tools.env.factory", "ENV_REGISTRY"),
    "build_env": ("module.tools.env.factory", "build_env"),
    "MockEnv": ("module.tools.env.mock", "MockEnv"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
