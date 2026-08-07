"""Lazily loaded manipulation abstraction, mock, and explicit factory."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BaseManipulation",
    "MANIPULATION_REGISTRY",
    "MockManipulation",
    "build_manipulation",
]


_EXPORTS = {
    "BaseManipulation": ("module.tools.manipulation._base", "BaseManipulation"),
    "MANIPULATION_REGISTRY": (
        "module.tools.manipulation.factory",
        "MANIPULATION_REGISTRY",
    ),
    "build_manipulation": (
        "module.tools.manipulation.factory",
        "build_manipulation",
    ),
    "MockManipulation": (
        "module.tools.manipulation.mock",
        "MockManipulation",
    ),
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
