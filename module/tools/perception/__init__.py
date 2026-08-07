"""Lazily loaded perception abstraction, mock, and explicit factories."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BasePerceptionTool",
    "MockPerceptionTool",
    "PERCEPTION_PROFILES",
    "PERCEPTION_REGISTRY",
    "build_perception",
    "build_perception_profile",
]


_EXPORTS = {
    "BasePerceptionTool": ("module.tools.perception._base", "BasePerceptionTool"),
    "PERCEPTION_PROFILES": ("module.tools.perception.factory", "PERCEPTION_PROFILES"),
    "PERCEPTION_REGISTRY": ("module.tools.perception.factory", "PERCEPTION_REGISTRY"),
    "build_perception": ("module.tools.perception.factory", "build_perception"),
    "build_perception_profile": (
        "module.tools.perception.factory",
        "build_perception_profile",
    ),
    "MockPerceptionTool": ("module.tools.perception.mock", "MockPerceptionTool"),
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
