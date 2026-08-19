"""Explicit C1 robotics registries and shared-bundle composition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any

from proto import ArmSide

from .mock import (
    MockArmControlPort,
    MockHandControlPort,
    MockRoboticsBundle,
    MockTactilePort,
    build_mock_robotics_bundle,
)
from .bundle import RoboticsBundle
from .shadow import build_linkbot_g20_shadow_bundle
from .fake_gateway import build_fake_gateway_robotics_bundle
from .hardware import build_linkbot_g20_hardware_bundle


ARM_PORT_REGISTRY: dict[str, type[MockArmControlPort]] = {"mock": MockArmControlPort}
HAND_PORT_REGISTRY: dict[str, type[MockHandControlPort]] = {"mock": MockHandControlPort}
TACTILE_PORT_REGISTRY: dict[str, type[MockTactilePort]] = {"mock": MockTactilePort}

RoboticsBuilder = Callable[..., RoboticsBundle]
ROBOTICS_REGISTRY: dict[str, RoboticsBuilder] = {
    "fake_gateway": build_fake_gateway_robotics_bundle,
    "linkbot_g20_shadow": build_linkbot_g20_shadow_bundle,
    "linkbot_g20_hardware": build_linkbot_g20_hardware_bundle,
    "mock": build_mock_robotics_bundle,
}


def build_robotics_bundle(
    config: Mapping[str, Any],
    *,
    side: ArmSide,
) -> RoboticsBundle:
    """Build one reviewed same-side robotics bundle from explicit config."""

    if not isinstance(config, Mapping):
        raise TypeError("robotics config must be a mapping")
    options = dict(config)
    robotics_type = options.pop("type", None)
    if not isinstance(robotics_type, str) or not robotics_type.strip():
        raise ValueError("robotics.type must be a non-empty string")
    try:
        builder = ROBOTICS_REGISTRY[robotics_type.strip()]
    except KeyError as exc:
        choices = ", ".join(sorted(ROBOTICS_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown robotics type {robotics_type!r}; registered types: {choices}"
        ) from exc
    try:
        inspect.signature(builder).bind(side=side, **options)
    except TypeError as exc:
        raise TypeError(
            f"invalid options for robotics type {robotics_type!r}: {exc}"
        ) from exc
    bundle = builder(side=side, **options)
    if not isinstance(bundle, RoboticsBundle):
        raise TypeError("robotics builder must return a RoboticsBundle")
    if bundle.side is not side:
        raise ValueError("robotics builder returned a bundle for the wrong side")
    return bundle


__all__ = [
    "ARM_PORT_REGISTRY",
    "HAND_PORT_REGISTRY",
    "ROBOTICS_REGISTRY",
    "TACTILE_PORT_REGISTRY",
    "build_robotics_bundle",
]
