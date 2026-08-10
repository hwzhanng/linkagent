"""Backend-neutral contracts shared by every atomic execution skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable


_MAX_SUMMARY_ITEMS = 32
_MAX_SUMMARY_TEXT = 512
_MAX_SUMMARY_DEPTH = 5


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _enum(value: object, enum_type: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except (TypeError, ValueError) as exc:
        choices = ", ".join(str(item.value) for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {choices}") from exc


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of strings") from exc
    return tuple(_non_empty(item, field_name=f"{field_name} item") for item in items)


def _bounded_summary(value: object, *, depth: int = 0) -> object:
    """Return a bounded JSON-like summary without retaining opaque payloads."""

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else MappingProxyType({"type": "nonfinite"})
    if isinstance(value, str):
        return value[:_MAX_SUMMARY_TEXT]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return MappingProxyType({"type": "binary", "size": len(value)})
    if depth >= _MAX_SUMMARY_DEPTH:
        return MappingProxyType({"type": type(value).__name__})
    if isinstance(value, Mapping):
        items = list(value.items())[:_MAX_SUMMARY_ITEMS]
        return MappingProxyType(
            {
                str(key)[:128]: _bounded_summary(item, depth=depth + 1)
                for key, item in items
            }
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(
            _bounded_summary(item, depth=depth + 1)
            for item in list(value)[:_MAX_SUMMARY_ITEMS]
        )
    return MappingProxyType({"type": type(value).__name__})


def bounded_summary_mapping(value: Mapping[str, object] | None) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("summary must be a mapping")
    summarized = _bounded_summary(value)
    assert isinstance(summarized, Mapping)
    return summarized


class AtomicSkillStatus(str, Enum):
    SUCCEEDED = "succeeded"
    RETRYABLE_FAILURE = "retryable_failure"
    SAFETY_ABORT = "safety_abort"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class AtomicSkillFailureCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    PRECONDITION_FAILED = "precondition_failed"
    TARGET_STALE = "target_stale"
    TARGET_OCCLUDED = "target_occluded"
    DEPTH_UNAVAILABLE = "depth_unavailable"
    PERCEPTION_UNAVAILABLE = "perception_unavailable"
    NO_HARVESTABLE_TARGET = "no_harvestable_target"
    CANDIDATE_SET_STALE = "candidate_set_stale"
    REACHABILITY_UNKNOWN = "reachability_unknown"
    VIEWPOINT_INSUFFICIENT = "viewpoint_insufficient"
    VIEWPOINTS_EXHAUSTED = "viewpoints_exhausted"
    VIEWPOINT_UNREACHABLE = "viewpoint_unreachable"
    VIEW_POSE_NOT_REACHED = "view_pose_not_reached"
    CLEARANCE_CONTEXT_MISSING = "clearance_context_missing"
    CLEARANCE_SCENE_STALE = "clearance_scene_stale"
    CLEARANCE_BLOCKED = "clearance_blocked"
    CLEARANCE_IK_UNREACHABLE = "clearance_ik_unreachable"
    SAFETY_REJECTED = "safety_rejected"
    CONTROL_BUSY = "control_busy"
    COMMAND_FAILED = "command_failed"
    PAYLOAD_UNSTABLE = "payload_unstable"
    PAYLOAD_DROPPED = "payload_dropped"
    TRANSPORT_POSE_NOT_REACHED = "transport_pose_not_reached"
    RESET_HAND_NOT_EMPTY = "reset_hand_not_empty"
    RESET_POSE_NOT_REACHED = "reset_pose_not_reached"
    CONTAINER_UNAVAILABLE = "container_unavailable"
    PLACEMENT_UNCONFIRMED = "placement_unconfirmed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class SkillTraceEventKind(str, Enum):
    STARTED = "started"
    CHECKPOINT = "checkpoint"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AtomicSkillDescriptor:
    skill_id: str
    version: str
    description: str
    effectful: bool
    required_resources: tuple[str, ...] = ()
    hardware_approved: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", _non_empty(self.skill_id, field_name="skill_id"))
        object.__setattr__(self, "version", _non_empty(self.version, field_name="version"))
        object.__setattr__(
            self, "description", _non_empty(self.description, field_name="description")
        )
        if not isinstance(self.effectful, bool) or not isinstance(
            self.hardware_approved, bool
        ):
            raise TypeError("effectful and hardware_approved must be bools")
        resources = _string_tuple(
            self.required_resources,
            field_name="required_resources",
        )
        if len(set(resources)) != len(resources):
            raise ValueError("required_resources must not contain duplicates")
        object.__setattr__(self, "required_resources", resources)


@dataclass(frozen=True, slots=True)
class SkillTraceEvent:
    sequence: int
    phase: str
    kind: SkillTraceEventKind
    monotonic_ns: int
    elapsed_ms: float
    message: str = ""
    input_summary: Mapping[str, object] = field(default_factory=dict)
    output_summary: Mapping[str, object] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be an integer >= 0")
        object.__setattr__(self, "phase", _non_empty(self.phase, field_name="phase"))
        object.__setattr__(
            self,
            "kind",
            _enum(self.kind, SkillTraceEventKind, field_name="kind"),
        )
        if isinstance(self.monotonic_ns, bool) or not isinstance(self.monotonic_ns, int):
            raise ValueError("monotonic_ns must be an integer")
        if self.monotonic_ns < 0:
            raise ValueError("monotonic_ns must be >= 0")
        if isinstance(self.elapsed_ms, bool) or not isinstance(self.elapsed_ms, (int, float)):
            raise TypeError("elapsed_ms must be a real number")
        if float(self.elapsed_ms) < 0.0:
            raise ValueError("elapsed_ms must be >= 0")
        object.__setattr__(self, "elapsed_ms", float(self.elapsed_ms))
        object.__setattr__(self, "message", str(self.message)[:_MAX_SUMMARY_TEXT])
        object.__setattr__(
            self, "input_summary", bounded_summary_mapping(self.input_summary)
        )
        object.__setattr__(
            self, "output_summary", bounded_summary_mapping(self.output_summary)
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _string_tuple(self.evidence_refs, field_name="evidence_refs"),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "phase": self.phase,
            "kind": self.kind.value,
            "monotonic_ns": self.monotonic_ns,
            "elapsed_ms": self.elapsed_ms,
            "message": self.message,
            "input_summary": dict(self.input_summary),
            "output_summary": dict(self.output_summary),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class AtomicSkillResult:
    execution_id: str
    skill_id: str
    status: AtomicSkillStatus
    success: bool
    duration_s: float
    failure_code: AtomicSkillFailureCode | None = None
    message: str = ""
    output: Mapping[str, object] = field(default_factory=dict)
    trace: tuple[SkillTraceEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "execution_id", _non_empty(self.execution_id, field_name="execution_id")
        )
        object.__setattr__(self, "skill_id", _non_empty(self.skill_id, field_name="skill_id"))
        object.__setattr__(
            self,
            "status",
            _enum(self.status, AtomicSkillStatus, field_name="status"),
        )
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        if self.failure_code is not None:
            object.__setattr__(
                self,
                "failure_code",
                _enum(
                    self.failure_code,
                    AtomicSkillFailureCode,
                    field_name="failure_code",
                ),
            )
        if isinstance(self.duration_s, bool) or not isinstance(self.duration_s, (int, float)):
            raise TypeError("duration_s must be a real number")
        if float(self.duration_s) < 0.0:
            raise ValueError("duration_s must be >= 0")
        object.__setattr__(self, "duration_s", float(self.duration_s))
        if self.success != (self.status is AtomicSkillStatus.SUCCEEDED):
            raise ValueError("success must agree with status")
        if self.success and self.failure_code is not None:
            raise ValueError("successful result cannot have failure_code")
        if not self.success and self.failure_code is None:
            raise ValueError("unsuccessful result requires failure_code")
        object.__setattr__(self, "message", str(self.message)[:_MAX_SUMMARY_TEXT])
        object.__setattr__(self, "output", bounded_summary_mapping(self.output))
        object.__setattr__(self, "trace", tuple(self.trace))
        if not all(isinstance(event, SkillTraceEvent) for event in self.trace):
            raise TypeError("trace must contain SkillTraceEvent values")
        if self.trace:
            sequences = tuple(event.sequence for event in self.trace)
            if sequences != tuple(range(len(sequences))):
                raise ValueError("trace sequence must start at zero and be contiguous")

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_id": self.execution_id,
            "skill_id": self.skill_id,
            "status": self.status.value,
            "success": self.success,
            "duration_s": self.duration_s,
            "failure_code": None if self.failure_code is None else self.failure_code.value,
            "message": self.message,
            "output": dict(self.output),
            "trace": [event.as_dict() for event in self.trace],
        }


@runtime_checkable
class AtomicSkill(Protocol):
    @property
    def descriptor(self) -> AtomicSkillDescriptor:
        """Return the immutable skill descriptor."""

    def run(self, request: object) -> AtomicSkillResult:
        """Run one idempotent execution identified by the request execution_id."""

    def cancel(self, execution_id: str, *, reason: str = "cancelled") -> AtomicSkillResult:
        """Cancel an execution or return its already-terminal result."""

    def close(self) -> None:
        """Release owned resources. Must be idempotent."""


__all__ = [
    "AtomicSkill",
    "AtomicSkillDescriptor",
    "AtomicSkillFailureCode",
    "AtomicSkillResult",
    "AtomicSkillStatus",
    "SkillTraceEvent",
    "SkillTraceEventKind",
    "bounded_summary_mapping",
]
