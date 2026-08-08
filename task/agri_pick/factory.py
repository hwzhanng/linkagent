"""Explicit registries for C1 tomato-picking task components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any

from module.robotics import Clock, RoboticsBundle
from module.skills.tomato_pick import PickVerifier

from .contracts import MotionSafetyPort, TomatoGraspPlanner, TomatoPerceptionPort
from .mock_components import (
    MockMotionSafetyPort,
    MockPickVerifier,
    MockTomatoGraspPlanner,
    MockTomatoPerceptionPort,
)
from .verification import (
    IndependentPickVerifier,
    PickEvidenceSource,
    PickVerifierThresholds,
    RecordedPickEvidenceSource,
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
def _build_mock_verifier(*, robotics: RoboticsBundle) -> MockPickVerifier:
    world = robotics.verifier_context().get("mock_world")
    if world is None:
        raise TypeError("mock verifier requires a mock_world verifier resource")
    return MockPickVerifier(world)


def _build_independent_verifier(
    *,
    robotics: RoboticsBundle,
    profile_id: str,
    tactile_calibration_id: str,
    evidence_file: str | None = None,
    stable_hold_s: float = 1.0,
    minimum_contact_fingers: int = 2,
    minimum_normal_force_n: float = 0.60,
    maximum_normal_force_n: float = 12.0,
    maximum_slip_ratio: float = 0.75,
    hardware_validated: bool = False,
) -> IndependentPickVerifier:
    if evidence_file is not None:
        source: PickEvidenceSource = RecordedPickEvidenceSource(evidence_file)
    else:
        candidate = robotics.verifier_context().get("pick_evidence_source")
        if not isinstance(candidate, PickEvidenceSource):
            raise TypeError(
                "independent verifier requires evidence_file or bundle "
                "pick_evidence_source"
            )
        source = candidate
    return IndependentPickVerifier(
        source,
        PickVerifierThresholds(
            profile_id=profile_id,
            tactile_calibration_id=tactile_calibration_id,
            stable_hold_s=stable_hold_s,
            minimum_contact_fingers=minimum_contact_fingers,
            minimum_normal_force_n=minimum_normal_force_n,
            maximum_normal_force_n=maximum_normal_force_n,
            maximum_slip_ratio=maximum_slip_ratio,
            hardware_validated=hardware_validated,
        ),
    )


VERIFIER_REGISTRY: dict[str, Callable[..., PickVerifier]] = {
    "independent": _build_independent_verifier,
    "mock": _build_mock_verifier,
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


def build_perception(config: Mapping[str, Any], *, clock: Clock) -> TomatoPerceptionPort:
    result = _build(config, label="perception", registry=PERCEPTION_REGISTRY, injected={"clock": clock})
    if not isinstance(result, TomatoPerceptionPort):
        raise TypeError("perception builder returned an invalid port")
    return result


def build_planner(config: Mapping[str, Any], *, clock: Clock) -> TomatoGraspPlanner:
    result = _build(config, label="planner", registry=PLANNER_REGISTRY, injected={"clock": clock})
    if not isinstance(result, TomatoGraspPlanner):
        raise TypeError("planner builder returned an invalid port")
    return result


def build_safety(config: Mapping[str, Any], *, clock: Clock) -> MotionSafetyPort:
    result = _build(config, label="safety", registry=SAFETY_REGISTRY, injected={"clock": clock})
    if not isinstance(result, MotionSafetyPort):
        raise TypeError("safety builder returned an invalid port")
    return result


def build_verifier(config: Mapping[str, Any], *, robotics: RoboticsBundle) -> PickVerifier:
    result = _build(
        config,
        label="verifier",
        registry=VERIFIER_REGISTRY,
        injected={"robotics": robotics},
    )
    if not isinstance(result, PickVerifier):
        raise TypeError("verifier builder returned an invalid PickVerifier")
    return result


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
