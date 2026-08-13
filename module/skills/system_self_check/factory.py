"""Explicit registries for read-only system-health ports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any

from module.robotics import Clock

from .camera import MockCameraHealthPort, OrbbecViewerCameraHealthPort
from .emergency_stop import MockEmergencyStopHealthPort
from .ports import CameraHealthPort, EmergencyStopHealthPort, VisionWorkerHealthPort
from .vision_worker import LoopbackVisionWorkerHealthPort, MockVisionWorkerHealthPort


CAMERA_HEALTH_REGISTRY: dict[str, Callable[..., CameraHealthPort]] = {
    "mock": MockCameraHealthPort,
    "orbbec_viewer": OrbbecViewerCameraHealthPort,
}
EMERGENCY_STOP_HEALTH_REGISTRY: dict[
    str, Callable[..., EmergencyStopHealthPort]
] = {"mock": MockEmergencyStopHealthPort}
VISION_WORKER_HEALTH_REGISTRY: dict[
    str, Callable[..., VisionWorkerHealthPort]
] = {
    "mock": MockVisionWorkerHealthPort,
    "loopback": LoopbackVisionWorkerHealthPort,
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


def build_camera_health_port(
    config: Mapping[str, Any], *, clock: Clock
) -> CameraHealthPort:
    port = _build(
        config,
        label="camera_health",
        registry=CAMERA_HEALTH_REGISTRY,
        injected={"clock": clock},
    )
    if not isinstance(port, CameraHealthPort):
        raise TypeError("camera health builder must return a CameraHealthPort")
    return port


def build_emergency_stop_health_port(
    config: Mapping[str, Any], *, clock: Clock
) -> EmergencyStopHealthPort:
    port = _build(
        config,
        label="emergency_stop_health",
        registry=EMERGENCY_STOP_HEALTH_REGISTRY,
        injected={"clock": clock},
    )
    if not isinstance(port, EmergencyStopHealthPort):
        raise TypeError(
            "emergency-stop health builder must return an EmergencyStopHealthPort"
        )
    return port


def build_vision_worker_health_port(
    config: Mapping[str, Any], *, clock: Clock
) -> VisionWorkerHealthPort:
    port = _build(
        config,
        label="vision_worker_health",
        registry=VISION_WORKER_HEALTH_REGISTRY,
        injected={"clock": clock},
    )
    if not isinstance(port, VisionWorkerHealthPort):
        raise TypeError(
            "vision worker health builder must return a VisionWorkerHealthPort"
        )
    return port


def validate_health_port_config(
    config: object,
    *,
    label: str,
    registry: Mapping[str, Callable[..., object]],
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
    try:
        inspect.signature(builder).bind(clock=object(), **options)
    except TypeError as exc:
        raise TypeError(f"invalid options for {label} type {kind!r}: {exc}") from exc


__all__ = [
    "CAMERA_HEALTH_REGISTRY",
    "EMERGENCY_STOP_HEALTH_REGISTRY",
    "VISION_WORKER_HEALTH_REGISTRY",
    "build_camera_health_port",
    "build_emergency_stop_health_port",
    "build_vision_worker_health_port",
    "validate_health_port_config",
]
