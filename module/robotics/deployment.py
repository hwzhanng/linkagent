"""Validated offline deployment slots for the future dual-arm/dual-hand rig."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from proto import ArmSide


def _optional_name(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be null or a non-empty string")
    return value.strip()


def _name(value: object, *, field_name: str) -> str:
    prepared = _optional_name(value, field_name=field_name)
    if prepared is None:
        raise ValueError(f"{field_name} must be a non-empty string")
    return prepared


def _bitrate(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("bitrate must be an integer > 0")
    return value


@dataclass(frozen=True, slots=True)
class ArmEndpointConfig:
    side: ArmSide
    can_interface: str | None
    profile_path: str
    base_frame_id: str = "world"
    bitrate: int = 1_000_000
    validated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.side, ArmSide):
            object.__setattr__(self, "side", ArmSide(self.side))
        object.__setattr__(
            self,
            "can_interface",
            _optional_name(self.can_interface, field_name="can_interface"),
        )
        profile = _name(self.profile_path, field_name="profile_path")
        if PurePosixPath(profile).is_absolute():
            raise ValueError("profile_path must be relative to the repository")
        object.__setattr__(self, "profile_path", profile)
        object.__setattr__(
            self,
            "base_frame_id",
            _name(self.base_frame_id, field_name="base_frame_id"),
        )
        object.__setattr__(self, "bitrate", _bitrate(self.bitrate))
        if not isinstance(self.validated, bool):
            raise TypeError("validated must be a bool")


@dataclass(frozen=True, slots=True)
class HandEndpointConfig:
    side: ArmSide
    can_interface: str | None
    can_id: int | None
    calibration_id: str
    bitrate: int = 1_000_000
    validated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.side, ArmSide):
            object.__setattr__(self, "side", ArmSide(self.side))
        object.__setattr__(
            self,
            "can_interface",
            _optional_name(self.can_interface, field_name="can_interface"),
        )
        if self.can_id is not None and (
            isinstance(self.can_id, bool)
            or not isinstance(self.can_id, int)
            or not 0 <= self.can_id <= 0x7FF
        ):
            raise ValueError("can_id must be null or an 11-bit integer")
        object.__setattr__(
            self,
            "calibration_id",
            _name(self.calibration_id, field_name="calibration_id"),
        )
        object.__setattr__(self, "bitrate", _bitrate(self.bitrate))
        if not isinstance(self.validated, bool):
            raise TypeError("validated must be a bool")


@dataclass(frozen=True, slots=True)
class BimanualDeploymentConfig:
    """Exactly two arm slots and two G20 slots; hardware mapping may be pending."""

    arms: tuple[ArmEndpointConfig, ArmEndpointConfig]
    hands: tuple[HandEndpointConfig, HandEndpointConfig]
    schema_version: int = 1
    topology_validated: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not isinstance(self.topology_validated, bool):
            raise TypeError("topology_validated must be a bool")
        arms = tuple(self.arms)
        hands = tuple(self.hands)
        if len(arms) != 2 or not all(isinstance(item, ArmEndpointConfig) for item in arms):
            raise TypeError("arms must contain exactly two ArmEndpointConfig values")
        if len(hands) != 2 or not all(isinstance(item, HandEndpointConfig) for item in hands):
            raise TypeError("hands must contain exactly two HandEndpointConfig values")
        expected = {ArmSide.LEFT, ArmSide.RIGHT}
        if {item.side for item in arms} != expected:
            raise ValueError("arms must contain one left and one right endpoint")
        if {item.side for item in hands} != expected:
            raise ValueError("hands must contain one left and one right endpoint")
        object.__setattr__(self, "arms", tuple(sorted(arms, key=lambda item: item.side.value)))
        object.__setattr__(self, "hands", tuple(sorted(hands, key=lambda item: item.side.value)))

    @property
    def hardware_ready(self) -> bool:
        return not self.missing_assignments()

    def missing_assignments(self) -> tuple[str, ...]:
        missing: list[str] = []
        for endpoint in self.arms:
            if endpoint.can_interface is None:
                missing.append(f"arms.{endpoint.side.value}.can_interface")
            if not endpoint.validated:
                missing.append(f"arms.{endpoint.side.value}.validated")
        for endpoint in self.hands:
            if endpoint.can_interface is None:
                missing.append(f"hands.{endpoint.side.value}.can_interface")
            if endpoint.can_id is None:
                missing.append(f"hands.{endpoint.side.value}.can_id")
            if not endpoint.validated:
                missing.append(f"hands.{endpoint.side.value}.validated")
        if not self.topology_validated:
            missing.append("topology_validated")
        return tuple(missing)

    def require_hardware_ready(self) -> None:
        missing = self.missing_assignments()
        if missing:
            raise RuntimeError(
                "hardware deployment is incomplete: " + ", ".join(missing)
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BimanualDeploymentConfig":
        if not isinstance(value, Mapping):
            raise TypeError("deployment config must be a mapping")
        allowed = {"schema_version", "topology_validated", "arms", "hands"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown deployment fields: {', '.join(unknown)}")
        arms = _endpoint_mapping(value.get("arms"), label="arms")
        hands = _endpoint_mapping(value.get("hands"), label="hands")
        arm_values = tuple(
            ArmEndpointConfig(side=side, **dict(arms[side.value]))
            for side in (ArmSide.LEFT, ArmSide.RIGHT)
        )
        hand_values = tuple(
            HandEndpointConfig(side=side, **dict(hands[side.value]))
            for side in (ArmSide.LEFT, ArmSide.RIGHT)
        )
        return cls(
            arms=arm_values,
            hands=hand_values,
            schema_version=value.get("schema_version", 1),
            topology_validated=value.get("topology_validated", False),
        )


def _endpoint_mapping(value: object, *, label: str) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if set(value) != {"left", "right"}:
        raise ValueError(f"{label} must contain exactly left and right")
    for side, endpoint in value.items():
        if not isinstance(endpoint, Mapping):
            raise TypeError(f"{label}.{side} must be a mapping")
    return value  # type: ignore[return-value]


__all__ = [
    "ArmEndpointConfig",
    "BimanualDeploymentConfig",
    "HandEndpointConfig",
]
