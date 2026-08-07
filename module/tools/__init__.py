"""Lazily loaded capability boundaries selected by Tasks and composition."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BaseEnv",
    "BaseManipulation",
    "BasePerceptionTool",
    "MockEnv",
    "MockManipulation",
    "MockPerceptionTool",
    "build_env",
    "build_manipulation",
    "build_perception",
]


_EXPORTS = {
    "BaseEnv": ("module.tools.env", "BaseEnv"),
    "MockEnv": ("module.tools.env", "MockEnv"),
    "build_env": ("module.tools.env", "build_env"),
    "BaseManipulation": ("module.tools.manipulation", "BaseManipulation"),
    "MockManipulation": ("module.tools.manipulation", "MockManipulation"),
    "build_manipulation": ("module.tools.manipulation", "build_manipulation"),
    "BasePerceptionTool": ("module.tools.perception", "BasePerceptionTool"),
    "MockPerceptionTool": ("module.tools.perception", "MockPerceptionTool"),
    "build_perception": ("module.tools.perception", "build_perception"),
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
