"""Backend-neutral data transfer objects shared by the runtime and adapters.

The DTOs intentionally do not import NumPy or a robotics SDK.  Sensor payloads
remain opaque at this boundary; their units are expressed by field names while
backend-specific interpretation stays inside adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping


def _freeze_mapping(
    value: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    """Return a shallow, read-only copy of a string-keyed mapping."""

    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    invalid_keys = [key for key in value if not isinstance(key, str)]
    if invalid_keys:
        raise TypeError(f"{field_name} keys must be strings")
    return MappingProxyType(dict(value))


def _finite_float(value: object, *, field_name: str) -> float:
    """Convert a real numeric value to a finite float, rejecting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _finite_float_tuple(
    value: object,
    *,
    field_name: str,
    length: int | None = None,
) -> tuple[float, ...]:
    """Validate and copy a finite numeric sequence."""

    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence of real numbers")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be a sequence of real numbers"
        ) from exc
    if length is not None and len(items) != length:
        raise ValueError(f"{field_name} must contain exactly {length} values")
    return tuple(
        _finite_float(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(items)
    )


@dataclass(frozen=True, slots=True)
class Observation:
    """One environment sample.

    ``depth_m`` is expressed in metres and ``joint_positions_rad`` in radians.
    Other sensor objects are opaque, SDK-free payloads.  ``metadata`` is the
    extension point used by deterministic mocks and must not contain live SDK
    clients or other resource owners.
    """

    sequence: int = 0
    timestamp_s: float | None = None
    rgb: object | None = None
    depth_m: object | None = None
    camera_intrinsics: object | None = None
    camera_extrinsics: object | None = None
    joint_positions_rad: tuple[float, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be an integer >= 0")
        if self.timestamp_s is not None:
            timestamp_s = _finite_float(
                self.timestamp_s,
                field_name="timestamp_s",
            )
            if timestamp_s < 0.0:
                raise ValueError("timestamp_s must be >= 0")
            object.__setattr__(self, "timestamp_s", timestamp_s)
        if self.joint_positions_rad is not None:
            object.__setattr__(
                self,
                "joint_positions_rad",
                _finite_float_tuple(
                    self.joint_positions_rad,
                    field_name="joint_positions_rad",
                ),
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, field_name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class Action:
    """A generic environment command selected by a Task-owned capability."""

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    action_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(
            self,
            "parameters",
            _freeze_mapping(self.parameters, field_name="parameters"),
        )
        if self.action_id is not None:
            if not isinstance(self.action_id, str) or not self.action_id.strip():
                raise ValueError("action_id must be a non-empty string when set")
            object.__setattr__(self, "action_id", self.action_id.strip())


@dataclass(frozen=True, slots=True)
class Result:
    """The verifiable outcome of an environment action."""

    success: bool
    message: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        object.__setattr__(
            self,
            "data",
            _freeze_mapping(self.data, field_name="data"),
        )
