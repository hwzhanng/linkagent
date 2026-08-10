"""Frozen DTOs for read-only harvest-system admission checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ._base_proto import _freeze_mapping
from .tomato_pick import ArmSide


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be an integer >= 0")
    return value


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of strings") from exc
    return tuple(_non_empty(item, field_name=f"{field_name} item") for item in items)


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _non_empty(value, field_name=field_name)


class CheckState(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SystemSelfCheckRequest:
    execution_id: str
    arm_side: ArmSide

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_id",
            _non_empty(self.execution_id, field_name="execution_id"),
        )
        if not isinstance(self.arm_side, ArmSide):
            try:
                object.__setattr__(self, "arm_side", ArmSide(self.arm_side))
            except (TypeError, ValueError) as exc:
                raise ValueError("arm_side must be left or right") from exc


@dataclass(frozen=True, slots=True)
class CameraHealthSnapshot:
    backend_name: str
    sdk_available: bool
    device_connected: bool
    rgb_live: bool
    depth_live: bool
    simulated: bool
    sample_monotonic_ns: int
    calibration_id: str | None = None
    device_id: str | None = None
    evidence_sources: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend_name", _non_empty(self.backend_name, field_name="backend_name")
        )
        for field_name in (
            "sdk_available",
            "device_connected",
            "rgb_live",
            "depth_live",
            "simulated",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        object.__setattr__(
            self,
            "sample_monotonic_ns",
            _non_negative_int(
                self.sample_monotonic_ns, field_name="sample_monotonic_ns"
            ),
        )
        object.__setattr__(
            self,
            "calibration_id",
            _optional_string(self.calibration_id, field_name="calibration_id"),
        )
        object.__setattr__(
            self,
            "device_id",
            _optional_string(self.device_id, field_name="device_id"),
        )
        object.__setattr__(
            self,
            "evidence_sources",
            _string_tuple(self.evidence_sources, field_name="evidence_sources"),
        )
        object.__setattr__(
            self, "details", _freeze_mapping(self.details, field_name="details")
        )


@dataclass(frozen=True, slots=True)
class EmergencyStopHealthSnapshot:
    backend_name: str
    available: bool
    healthy: bool | None
    simulated: bool
    sample_monotonic_ns: int
    evidence_sources: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend_name", _non_empty(self.backend_name, field_name="backend_name")
        )
        if not isinstance(self.available, bool) or not isinstance(self.simulated, bool):
            raise TypeError("available and simulated must be bools")
        if self.healthy is not None and not isinstance(self.healthy, bool):
            raise TypeError("healthy must be a bool or None")
        object.__setattr__(
            self,
            "sample_monotonic_ns",
            _non_negative_int(
                self.sample_monotonic_ns, field_name="sample_monotonic_ns"
            ),
        )
        object.__setattr__(
            self,
            "evidence_sources",
            _string_tuple(self.evidence_sources, field_name="evidence_sources"),
        )
        object.__setattr__(
            self, "details", _freeze_mapping(self.details, field_name="details")
        )


@dataclass(frozen=True, slots=True)
class VisionWorkerHealthSnapshot:
    worker_id: str
    backend_id: str
    ready: bool
    simulated: bool
    sample_monotonic_ns: int
    evidence_sources: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_id", _non_empty(self.worker_id, field_name="worker_id"))
        object.__setattr__(self, "backend_id", _non_empty(self.backend_id, field_name="backend_id"))
        if not isinstance(self.ready, bool) or not isinstance(self.simulated, bool):
            raise TypeError("ready and simulated must be bools")
        object.__setattr__(
            self,
            "sample_monotonic_ns",
            _non_negative_int(
                self.sample_monotonic_ns, field_name="sample_monotonic_ns"
            ),
        )
        object.__setattr__(
            self,
            "evidence_sources",
            _string_tuple(self.evidence_sources, field_name="evidence_sources"),
        )
        object.__setattr__(
            self, "details", _freeze_mapping(self.details, field_name="details")
        )


@dataclass(frozen=True, slots=True)
class CapabilityCheck:
    name: str
    state: CheckState
    critical: bool
    message: str
    evidence_sources: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, field_name="name"))
        if not isinstance(self.state, CheckState):
            try:
                object.__setattr__(self, "state", CheckState(self.state))
            except (TypeError, ValueError) as exc:
                choices = ", ".join(item.value for item in CheckState)
                raise ValueError(f"state must be one of: {choices}") from exc
        if not isinstance(self.critical, bool):
            raise TypeError("critical must be a bool")
        object.__setattr__(
            self, "message", _non_empty(self.message, field_name="message")
        )
        object.__setattr__(
            self,
            "evidence_sources",
            _string_tuple(self.evidence_sources, field_name="evidence_sources"),
        )
        object.__setattr__(
            self, "details", _freeze_mapping(self.details, field_name="details")
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "state": self.state.value,
            "critical": self.critical,
            "message": self.message,
            "evidence_sources": self.evidence_sources,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class SystemSelfCheckReport:
    execution_id: str
    execution_mode: str
    checked_monotonic_ns: int
    checks: tuple[CapabilityCheck, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_id",
            _non_empty(self.execution_id, field_name="execution_id"),
        )
        object.__setattr__(
            self,
            "execution_mode",
            _non_empty(self.execution_mode, field_name="execution_mode"),
        )
        object.__setattr__(
            self,
            "checked_monotonic_ns",
            _non_negative_int(
                self.checked_monotonic_ns, field_name="checked_monotonic_ns"
            ),
        )
        checks = tuple(self.checks)
        if not checks or not all(isinstance(item, CapabilityCheck) for item in checks):
            raise TypeError("checks must contain at least one CapabilityCheck")
        names = tuple(item.name for item in checks)
        if len(names) != len(set(names)):
            raise ValueError("check names must be unique")
        object.__setattr__(self, "checks", checks)

    @property
    def admission_allowed(self) -> bool:
        return all(
            check.state is CheckState.PASS
            for check in self.checks
            if check.critical
        )

    @property
    def failed_check_names(self) -> tuple[str, ...]:
        return tuple(
            check.name
            for check in self.checks
            if check.critical and check.state is not CheckState.PASS
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_id": self.execution_id,
            "execution_mode": self.execution_mode,
            "checked_monotonic_ns": self.checked_monotonic_ns,
            "admission_allowed": self.admission_allowed,
            "failed_check_names": self.failed_check_names,
            "checks": tuple(check.as_dict() for check in self.checks),
        }


__all__ = [
    "CameraHealthSnapshot",
    "CapabilityCheck",
    "CheckState",
    "EmergencyStopHealthSnapshot",
    "SystemSelfCheckReport",
    "SystemSelfCheckRequest",
]
