"""Explicit registries for C1 tomato-picking task components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any

from module.robotics import ManualClock, MockPickWorld

from .contracts import MotionSafetyPort, TomatoGraspPlanner, TomatoPerceptionPort
from .mock_components import (
    MockMotionSafetyPort,
    MockPickVerifier,
    MockTomatoGraspPlanner,
    MockTomatoPerceptionPort,
)


PERCEPTION_REGISTRY: dict[str, Callable[..., TomatoPerceptionPort]] = {
    "mock": MockTomatoPerceptionPort,
}
PLANNER_REGISTRY: dict[str, Callable[..., TomatoGraspPlanner]] = {
    "mock": MockTomatoGraspPlanner,
}
SAFETY_REGISTRY: dict[str, Callable[..., MotionSafetyPort]] = {
    "mock": MockMotionSafetyPort,
}
VERIFIER_REGISTRY: dict[str, Callable[..., MockPickVerifier]] = {
    "mock": MockPickVerifier,
}


def validate_component_config(
    config: Mapping[str, Any],
    *,
    label: str,
    registry: Mapping[str, Callable[..., Any]],
    injected_names: tuple[str, ...],
) -> None:
    if not isinstance(config, Mapping):
        raise TypeError(f"{label} config must be a mapping")
    options = dict(config)
    component_type = options.pop("type", None)
    if not isinstance(component_type, str) or not component_type.strip():
        raise ValueError(f"{label}.type must be a non-empty string")
    try:
        builder = registry[component_type.strip()]
    except KeyError as exc:
        choices = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(f"unknown {label} type {component_type!r}; registered: {choices}") from exc
    injected = {name: object() for name in injected_names}
    try:
        inspect.signature(builder).bind(**injected, **options)
    except TypeError as exc:
        raise TypeError(f"invalid options for {label} type {component_type!r}: {exc}") from exc


def _build(
    config: Mapping[str, Any],
    *,
    label: str,
    registry: Mapping[str, Callable[..., Any]],
    injected: Mapping[str, Any],
) -> Any:
    validate_component_config(
        config,
        label=label,
        registry=registry,
        injected_names=tuple(injected),
    )
    options = dict(config)
    builder = registry[options.pop("type").strip()]
    return builder(**dict(injected), **options)


def build_perception(config: Mapping[str, Any], *, clock: ManualClock) -> TomatoPerceptionPort:
    result = _build(config, label="perception", registry=PERCEPTION_REGISTRY, injected={"clock": clock})
    if not isinstance(result, TomatoPerceptionPort):
        raise TypeError("perception builder returned an invalid port")
    return result


def build_planner(config: Mapping[str, Any], *, clock: ManualClock) -> TomatoGraspPlanner:
    result = _build(config, label="planner", registry=PLANNER_REGISTRY, injected={"clock": clock})
    if not isinstance(result, TomatoGraspPlanner):
        raise TypeError("planner builder returned an invalid port")
    return result


def build_safety(config: Mapping[str, Any], *, clock: ManualClock) -> MotionSafetyPort:
    result = _build(config, label="safety", registry=SAFETY_REGISTRY, injected={"clock": clock})
    if not isinstance(result, MotionSafetyPort):
        raise TypeError("safety builder returned an invalid port")
    return result


def build_verifier(config: Mapping[str, Any], *, world: MockPickWorld) -> MockPickVerifier:
    return _build(config, label="verifier", registry=VERIFIER_REGISTRY, injected={"world": world})


__all__ = [
    "PERCEPTION_REGISTRY",
    "PLANNER_REGISTRY",
    "SAFETY_REGISTRY",
    "VERIFIER_REGISTRY",
    "build_perception",
    "build_planner",
    "build_safety",
    "build_verifier",
    "validate_component_config",
]
