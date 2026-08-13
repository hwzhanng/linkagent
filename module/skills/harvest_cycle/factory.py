"""Explicit registries for harvest observation and verification ports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any

from module.robotics import RoboticsBundle
from module.robotics import Clock
from module.skills.tomato_vision import (
    ReplayTomatoVisionObservationPort,
    WorkerTomatoVisionObservationPort,
)

from .mock import (
    MockClearanceScenePort,
    MockDroppedPayloadObservationPort,
    MockPlacementVerifierPort,
    MockTargetObservationPort,
    ObservationClearanceScenePort,
)
from .ports import (
    ClearanceScenePort,
    DroppedPayloadObservationPort,
    PlacementVerifierPort,
    TargetObservationPort,
)


TargetObservationBuilder = Callable[..., TargetObservationPort]
PlacementVerifierBuilder = Callable[..., PlacementVerifierPort]
ClearanceSceneBuilder = Callable[..., ClearanceScenePort]
DroppedPayloadObservationBuilder = Callable[..., DroppedPayloadObservationPort]

TARGET_OBSERVATION_REGISTRY: dict[str, TargetObservationBuilder] = {
    "mock": MockTargetObservationPort,
    "tomato_vision_replay": ReplayTomatoVisionObservationPort,
    "tomato_vision_worker": WorkerTomatoVisionObservationPort,
}
PLACEMENT_VERIFIER_REGISTRY: dict[str, PlacementVerifierBuilder] = {
    "mock": MockPlacementVerifierPort,
}
CLEARANCE_SCENE_REGISTRY: dict[str, ClearanceSceneBuilder] = {
    "mock": MockClearanceScenePort,
    "observation_pointcloud": ObservationClearanceScenePort,
}
DROPPED_PAYLOAD_OBSERVATION_REGISTRY: dict[
    str, DroppedPayloadObservationBuilder
] = {
    "mock": MockDroppedPayloadObservationPort,
}


def _build(
    config: Mapping[str, Any],
    *,
    label: str,
    registry: Mapping[str, Callable[..., object]],
    injected: Mapping[str, object],
) -> object:
    if not isinstance(config, Mapping):
        raise TypeError(f"{label} config must be a mapping")
    options = dict(config)
    kind = options.pop("type", None)
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"{label}.type must be a non-empty string")
    try:
        builder = registry[kind.strip()]
    except KeyError as exc:
        choices = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(f"unknown {label} type {kind!r}; registered: {choices}") from exc
    try:
        inspect.signature(builder).bind(**injected, **options)
    except TypeError as exc:
        raise TypeError(f"invalid options for {label} type {kind!r}: {exc}") from exc
    return builder(**injected, **options)


def build_target_observation_port(
    config: Mapping[str, Any], *, clock: Clock
) -> TargetObservationPort:
    port = _build(
        config,
        label="target_observation",
        registry=TARGET_OBSERVATION_REGISTRY,
        injected={"clock": clock},
    )
    if not isinstance(port, TargetObservationPort):
        raise TypeError("target observation builder must return a TargetObservationPort")
    return port


def build_placement_verifier_port(
    config: Mapping[str, Any], *, robotics: RoboticsBundle
) -> PlacementVerifierPort:
    port = _build(
        config,
        label="placement_verifier",
        registry=PLACEMENT_VERIFIER_REGISTRY,
        injected={"robotics": robotics},
    )
    if not isinstance(port, PlacementVerifierPort):
        raise TypeError("placement verifier builder must return a PlacementVerifierPort")
    return port


def build_clearance_scene_port(
    config: Mapping[str, Any], *, clock: Clock
) -> ClearanceScenePort:
    port = _build(
        config,
        label="clearance_scene",
        registry=CLEARANCE_SCENE_REGISTRY,
        injected={"clock": clock},
    )
    if not isinstance(port, ClearanceScenePort):
        raise TypeError("clearance scene builder must return a ClearanceScenePort")
    return port


def build_dropped_payload_observation_port(
    config: Mapping[str, Any], *, clock: Clock
) -> DroppedPayloadObservationPort:
    port = _build(
        config,
        label="dropped_payload_observation",
        registry=DROPPED_PAYLOAD_OBSERVATION_REGISTRY,
        injected={"clock": clock},
    )
    if not isinstance(port, DroppedPayloadObservationPort):
        raise TypeError(
            "dropped payload observation builder must return a "
            "DroppedPayloadObservationPort"
        )
    return port


def validate_harvest_port_config(
    config: object,
    *,
    label: str,
    registry: Mapping[str, Callable[..., object]],
    injected_names: tuple[str, ...] = (),
) -> None:
    if not isinstance(config, Mapping):
        raise TypeError(f"{label} config must be a mapping")
    options = dict(config)
    kind = options.pop("type", None)
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"{label}.type must be a non-empty string")
    try:
        builder = registry[kind.strip()]
    except KeyError as exc:
        choices = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(f"unknown {label} type {kind!r}; registered: {choices}") from exc
    injected = {name: object() for name in injected_names}
    try:
        inspect.signature(builder).bind(**injected, **options)
    except TypeError as exc:
        raise TypeError(f"invalid options for {label} type {kind!r}: {exc}") from exc


__all__ = [
    "CLEARANCE_SCENE_REGISTRY",
    "DROPPED_PAYLOAD_OBSERVATION_REGISTRY",
    "PLACEMENT_VERIFIER_REGISTRY",
    "TARGET_OBSERVATION_REGISTRY",
    "build_clearance_scene_port",
    "build_dropped_payload_observation_port",
    "build_placement_verifier_port",
    "build_target_observation_port",
    "validate_harvest_port_config",
]
