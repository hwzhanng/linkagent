"""Frozen backend-neutral DTOs for the single-tomato picking contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

from ._base_proto import _finite_float, _finite_float_tuple, _freeze_mapping
from .manipulation import Pose


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be an integer >= 0")
    return value


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be an integer > 0")
    return value


def _bounded_float(
    value: object,
    *,
    field_name: str,
    lower: float,
    upper: float,
) -> float:
    converted = _finite_float(value, field_name=field_name)
    if not lower <= converted <= upper:
        raise ValueError(f"{field_name} must be between {lower} and {upper}")
    return converted


def _optional_bounded_float(
    value: object | None,
    *,
    field_name: str,
    lower: float,
    upper: float,
) -> float | None:
    if value is None:
        return None
    return _bounded_float(
        value,
        field_name=field_name,
        lower=lower,
        upper=upper,
    )


def _enum(value: object, enum_type: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except (TypeError, ValueError) as exc:
        choices = ", ".join(str(item.value) for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {choices}") from exc


class ArmSide(str, Enum):
    """Selected arm and same-side hand."""

    LEFT = "left"
    RIGHT = "right"


class ExecutionState(str, Enum):
    """Lifecycle shared by mutable device commands."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class EvidenceState(str, Enum):
    """Tri-state verification result; unknown is never successful."""

    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class PickPhase(str, Enum):
    """Policy-active phases in fixed order."""

    CONTACT = "contact"
    GRASP = "grasp"
    DETACH = "detach"
    HOLD = "hold"


class SkillStatus(str, Enum):
    """Terminal status returned by TomatoPickSkill."""

    SUCCEEDED = "succeeded"
    RETRYABLE_FAILURE = "retryable_failure"
    SAFETY_ABORT = "safety_abort"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class PickFailureCode(str, Enum):
    """Stable failure taxonomy exposed above the skill boundary."""

    PRECONDITION_FAILED = "precondition_failed"
    TARGET_STALE = "target_stale"
    PLAN_STALE = "plan_stale"
    NO_CONTACT = "no_contact"
    UNSTABLE_GRASP = "unstable_grasp"
    DETACH_NOT_CONFIRMED = "detach_not_confirmed"
    SLIP_DETECTED = "slip_detected"
    DROPPED = "dropped"
    EXCESS_FORCE = "excess_force"
    ARM_STATE_STALE = "arm_state_stale"
    HAND_STATE_STALE = "hand_state_stale"
    TACTILE_STALE = "tactile_stale"
    OUT_OF_WORKSPACE = "out_of_workspace"
    JOINT_LIMIT = "joint_limit"
    COLLISION_UNSAFE = "collision_unsafe"
    CONTROL_BUSY = "control_busy"
    DEVICE_FAULT = "device_fault"
    COMMAND_REJECTED = "command_rejected"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    INTERNAL_ERROR = "internal_error"


class CollisionStatus(str, Enum):
    """Collision conclusion without pretending that IK proves safety."""

    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"
    SIMULATED_SAFE = "simulated_safe"


@dataclass(frozen=True, slots=True)
class StateStamp:
    """Sequence, monotonic times, and validity for one device state."""

    source_sequence: int
    sample_monotonic_ns: int
    received_monotonic_ns: int
    valid: bool = True
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_sequence",
            _non_negative_int(self.source_sequence, field_name="source_sequence"),
        )
        sample = _non_negative_int(
            self.sample_monotonic_ns,
            field_name="sample_monotonic_ns",
        )
        received = _non_negative_int(
            self.received_monotonic_ns,
            field_name="received_monotonic_ns",
        )
        if received < sample:
            raise ValueError("received_monotonic_ns must be >= sample_monotonic_ns")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")
        if self.error_code is not None:
            object.__setattr__(
                self,
                "error_code",
                _non_empty(self.error_code, field_name="error_code"),
            )


@dataclass(frozen=True, slots=True)
class ExecutionHandle:
    """Accepted command identity; submission is not task success."""

    command_id: str
    owner_id: str
    state: ExecutionState
    submitted_monotonic_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "command_id", _non_empty(self.command_id, field_name="command_id")
        )
        object.__setattr__(
            self, "owner_id", _non_empty(self.owner_id, field_name="owner_id")
        )
        object.__setattr__(
            self,
            "state",
            _enum(self.state, ExecutionState, field_name="state"),
        )
        object.__setattr__(
            self,
            "submitted_monotonic_ns",
            _non_negative_int(
                self.submitted_monotonic_ns,
                field_name="submitted_monotonic_ns",
            ),
        )


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Terminal or current controller result for a command ID."""

    command_id: str
    state: ExecutionState
    message: str = ""
    error_code: str | None = None
    started_monotonic_ns: int | None = None
    finished_monotonic_ns: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "command_id", _non_empty(self.command_id, field_name="command_id")
        )
        object.__setattr__(
            self,
            "state",
            _enum(self.state, ExecutionState, field_name="state"),
        )
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if len(self.message) > 1_024:
            raise ValueError("message must contain at most 1024 characters")
        if self.error_code is not None:
            object.__setattr__(
                self,
                "error_code",
                _non_empty(self.error_code, field_name="error_code"),
            )
        for field_name in ("started_monotonic_ns", "finished_monotonic_ns"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _non_negative_int(value, field_name=field_name),
                )
        if (
            self.started_monotonic_ns is not None
            and self.finished_monotonic_ns is not None
            and self.finished_monotonic_ns < self.started_monotonic_ns
        ):
            raise ValueError("finished_monotonic_ns must not precede start")


@dataclass(frozen=True, slots=True)
class ArmState:
    """One seven-joint arm state in SDK-neutral units.

    ``ee_twist`` is a spatial ``(linear xyz, angular xyz)`` vector expressed
    in ``ee_pose.frame_id``.  The feature builder rotates it into grasp axes;
    SDK gateways must not return a body-frame twist here.
    """

    side: ArmSide
    joint_positions_rad: tuple[float, ...]
    joint_velocities_rad_s: tuple[float, ...]
    ee_pose: Pose
    stamp: StateStamp
    ee_twist: tuple[float, ...] = (0.0,) * 6
    enabled: bool = True
    run_state: str = "idle"
    velocity_source: str = "measured"

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _enum(self.side, ArmSide, field_name="side"))
        object.__setattr__(
            self,
            "joint_positions_rad",
            _finite_float_tuple(
                self.joint_positions_rad,
                field_name="joint_positions_rad",
                length=7,
            ),
        )
        object.__setattr__(
            self,
            "joint_velocities_rad_s",
            _finite_float_tuple(
                self.joint_velocities_rad_s,
                field_name="joint_velocities_rad_s",
                length=7,
            ),
        )
        object.__setattr__(
            self,
            "ee_twist",
            _finite_float_tuple(self.ee_twist, field_name="ee_twist", length=6),
        )
        if not isinstance(self.ee_pose, Pose):
            raise TypeError("ee_pose must be a Pose")
        if not isinstance(self.stamp, StateStamp):
            raise TypeError("stamp must be a StateStamp")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        object.__setattr__(
            self, "run_state", _non_empty(self.run_state, field_name="run_state")
        )
        source = _non_empty(self.velocity_source, field_name="velocity_source")
        if source not in {"measured", "estimated"}:
            raise ValueError("velocity_source must be measured or estimated")
        object.__setattr__(self, "velocity_source", source)


@dataclass(frozen=True, slots=True)
class HandSynergy:
    """Canonical hand target independent of G20 raw and URDF joint counts."""

    closure: float
    thumb_opposition: float
    finger_spread: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "closure",
            _bounded_float(
                self.closure,
                field_name="closure",
                lower=0.0,
                upper=1.0,
            ),
        )
        for field_name in ("thumb_opposition", "finger_spread"):
            object.__setattr__(
                self,
                field_name,
                _bounded_float(
                    getattr(self, field_name),
                    field_name=field_name,
                    lower=-1.0,
                    upper=1.0,
                ),
            )

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.closure, self.thumb_opposition, self.finger_spread)


@dataclass(frozen=True, slots=True)
class HandState:
    """G20 state with the 16 independently feedback-bearing joint channels.

    The legacy three-value synergy summary remains a task-level diagnostic;
    learned policies consume the per-joint position and velocity fields.
    """

    side: ArmSide
    synergy: HandSynergy
    target_synergy: HandSynergy
    tracking_error: tuple[float, ...]
    stamp: StateStamp
    logical_raw: tuple[int, ...] | None = None
    temperatures_c: tuple[float, ...] = ()
    fault_codes: tuple[int, ...] = ()
    joint_positions_rad: tuple[float, ...] = (0.0,) * 16
    joint_velocities_rad_s: tuple[float, ...] = (0.0,) * 16
    target_joint_positions_rad: tuple[float, ...] = (0.0,) * 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _enum(self.side, ArmSide, field_name="side"))
        if not isinstance(self.synergy, HandSynergy):
            raise TypeError("synergy must be a HandSynergy")
        if not isinstance(self.target_synergy, HandSynergy):
            raise TypeError("target_synergy must be a HandSynergy")
        object.__setattr__(
            self,
            "tracking_error",
            _finite_float_tuple(
                self.tracking_error,
                field_name="tracking_error",
                length=3,
            ),
        )
        if not isinstance(self.stamp, StateStamp):
            raise TypeError("stamp must be a StateStamp")
        if self.logical_raw is not None:
            values = tuple(self.logical_raw)
            if len(values) != 20 or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 255
                for value in values
            ):
                raise ValueError("logical_raw must contain exactly 20 integers in 0..255")
            object.__setattr__(self, "logical_raw", values)
        object.__setattr__(
            self,
            "temperatures_c",
            _finite_float_tuple(
                self.temperatures_c,
                field_name="temperatures_c",
            ),
        )
        faults = tuple(self.fault_codes)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in faults):
            raise ValueError("fault_codes must contain integers >= 0")
        object.__setattr__(self, "fault_codes", faults)
        for field_name in (
            "joint_positions_rad",
            "joint_velocities_rad_s",
            "target_joint_positions_rad",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_float_tuple(
                    getattr(self, field_name),
                    field_name=field_name,
                    length=16,
                ),
            )


def _matrix_weight(value: object) -> tuple[tuple[int, ...], ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("matrix_weight_raw must be a 12x6 integer matrix")
    try:
        rows = tuple(tuple(row) for row in value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("matrix_weight_raw must be a 12x6 integer matrix") from exc
    if len(rows) != 12 or any(len(row) != 6 for row in rows):
        raise ValueError("matrix_weight_raw must have shape 12x6")
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or not 0 <= item <= 255
        for row in rows
        for item in row
    ):
        raise ValueError("matrix_weight_raw values must be integers in 0..255")
    return rows


@dataclass(frozen=True, slots=True)
class FingerTactile:
    """One finger's force channels and raw spatial-weight matrix."""

    normal_force_n: float | None = None
    tangential_force_n: float | None = None
    tangential_direction_raw: float | None = None
    matrix_weight_raw: tuple[tuple[int, ...], ...] | None = None
    approach_raw: float | None = None
    force_sample_monotonic_ns: int | None = None
    matrix_sample_monotonic_ns: int | None = None
    force_valid: bool = False
    matrix_valid: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normal_force_n",
            _optional_bounded_float(
                self.normal_force_n,
                field_name="normal_force_n",
                lower=0.0,
                upper=20.0,
            ),
        )
        object.__setattr__(
            self,
            "tangential_force_n",
            _optional_bounded_float(
                self.tangential_force_n,
                field_name="tangential_force_n",
                lower=0.0,
                upper=20.0,
            ),
        )
        object.__setattr__(
            self,
            "tangential_direction_raw",
            _optional_bounded_float(
                self.tangential_direction_raw,
                field_name="tangential_direction_raw",
                lower=0.0,
                upper=255.0,
            ),
        )
        if self.matrix_weight_raw is not None:
            object.__setattr__(
                self,
                "matrix_weight_raw",
                _matrix_weight(self.matrix_weight_raw),
            )
        if self.approach_raw is not None:
            object.__setattr__(
                self,
                "approach_raw",
                _finite_float(self.approach_raw, field_name="approach_raw"),
            )
        for field_name in (
            "force_sample_monotonic_ns",
            "matrix_sample_monotonic_ns",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _non_negative_int(value, field_name=field_name),
                )
        if not isinstance(self.force_valid, bool) or not isinstance(
            self.matrix_valid, bool
        ):
            raise TypeError("force_valid and matrix_valid must be bools")
        if self.force_valid and (
            self.normal_force_n is None
            or self.tangential_force_n is None
            or self.tangential_direction_raw is None
            or self.force_sample_monotonic_ns is None
        ):
            raise ValueError("valid force data requires force values, direction, and time")
        if self.matrix_valid and (
            self.matrix_weight_raw is None
            or self.matrix_sample_monotonic_ns is None
        ):
            raise ValueError("valid matrix data requires a matrix and sample time")


@dataclass(frozen=True, slots=True)
class TactileFrame:
    """Five asynchronous G20 finger samples in canonical finger order."""

    side: ArmSide
    fingers: tuple[FingerTactile, ...]
    calibration_id: str
    feature_extractor_id: str
    received_monotonic_ns: int
    source_sequence: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _enum(self.side, ArmSide, field_name="side"))
        fingers = tuple(self.fingers)
        if len(fingers) != 5 or not all(
            isinstance(item, FingerTactile) for item in fingers
        ):
            raise TypeError("fingers must contain exactly five FingerTactile values")
        object.__setattr__(self, "fingers", fingers)
        object.__setattr__(
            self,
            "calibration_id",
            _non_empty(self.calibration_id, field_name="calibration_id"),
        )
        object.__setattr__(
            self,
            "feature_extractor_id",
            _non_empty(self.feature_extractor_id, field_name="feature_extractor_id"),
        )
        object.__setattr__(
            self,
            "received_monotonic_ns",
            _non_negative_int(
                self.received_monotonic_ns,
                field_name="received_monotonic_ns",
            ),
        )
        object.__setattr__(
            self,
            "source_sequence",
            _non_negative_int(self.source_sequence, field_name="source_sequence"),
        )


@dataclass(frozen=True, slots=True)
class TomatoTarget:
    """Deployable oriented-ellipsoid target fitted from RGB-D perception.

    ``semi_axes_m`` is ``(a_x, a_y, a_z)`` in ``center_pose``'s local frame.
    ``stem_direction_xyz`` retains its legacy field name but is frozen as the
    unit pull axis pointing from the plant/stem toward the fruit (away from the
    plant), expressed in ``center_pose.frame_id``.  The grasp-frame z+ axis
    must align with it.
    Images, masks, and point clouds stay behind the perception port; policies
    receive this compact three-dimensional geometry instead of privileged sim
    state or a raw point cloud.
    """

    target_id: str
    center_pose: Pose
    semi_axes_m: tuple[float, ...]
    ripeness_score: float
    pick_confidence: float
    occlusion: float
    stem_direction_xyz: tuple[float, ...]
    observation_sequence: int
    sample_monotonic_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_id", _non_empty(self.target_id, field_name="target_id")
        )
        if not isinstance(self.center_pose, Pose):
            raise TypeError("center_pose must be a Pose")
        semi_axes = _finite_float_tuple(
            self.semi_axes_m,
            field_name="semi_axes_m",
            length=3,
        )
        if any(value <= 0.0 for value in semi_axes):
            raise ValueError("semi_axes_m values must be > 0")
        object.__setattr__(self, "semi_axes_m", semi_axes)
        for field_name in ("ripeness_score", "pick_confidence", "occlusion"):
            object.__setattr__(
                self,
                field_name,
                _bounded_float(
                    getattr(self, field_name),
                    field_name=field_name,
                    lower=0.0,
                    upper=1.0,
                ),
            )
        direction = _finite_float_tuple(
            self.stem_direction_xyz,
            field_name="stem_direction_xyz",
            length=3,
        )
        norm = math.sqrt(sum(value * value for value in direction))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("stem_direction_xyz must be normalized")
        object.__setattr__(self, "stem_direction_xyz", direction)
        object.__setattr__(
            self,
            "observation_sequence",
            _non_negative_int(
                self.observation_sequence,
                field_name="observation_sequence",
            ),
        )
        object.__setattr__(
            self,
            "sample_monotonic_ns",
            _non_negative_int(
                self.sample_monotonic_ns,
                field_name="sample_monotonic_ns",
            ),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "center_pose": self.center_pose.as_dict(),
            "semi_axes_m": list(self.semi_axes_m),
            "ripeness_score": self.ripeness_score,
            "pick_confidence": self.pick_confidence,
            "occlusion": self.occlusion,
            "observation_sequence": self.observation_sequence,
        }


@dataclass(frozen=True, slots=True)
class TomatoDetectionResult:
    """Target summaries tied to one environment observation."""

    targets: tuple[TomatoTarget, ...]
    observation_sequence: int
    model: str
    sample_monotonic_ns: int

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        if not all(isinstance(target, TomatoTarget) for target in targets):
            raise TypeError("targets must contain only TomatoTarget values")
        ids = [target.target_id for target in targets]
        if len(ids) != len(set(ids)):
            raise ValueError("target_id values must be unique in one detection result")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(
            self,
            "observation_sequence",
            _non_negative_int(
                self.observation_sequence,
                field_name="observation_sequence",
            ),
        )
        object.__setattr__(self, "model", _non_empty(self.model, field_name="model"))
        object.__setattr__(
            self,
            "sample_monotonic_ns",
            _non_negative_int(
                self.sample_monotonic_ns,
                field_name="sample_monotonic_ns",
            ),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "observation_sequence": self.observation_sequence,
            "model": self.model,
            "targets": [target.summary() for target in self.targets],
        }


@dataclass(frozen=True, slots=True)
class TomatoGraspPlan:
    """Immutable grasp-frame and bounded local-skill plan."""

    grasp_plan_id: str
    target_id: str
    source_observation_sequence: int
    arm_side: ArmSide
    grasp_frame: Pose
    pregrasp_pose: Pose
    contact_pose: Pose
    retreat_pose: Pose
    max_residual_translation_m: tuple[float, ...]
    max_residual_rotation_rad: tuple[float, ...]
    mapping_calibration_id: str
    planner_version: str
    expires_monotonic_ns: int
    pull_axis_in_grasp_frame: tuple[float, ...] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        for field_name in (
            "grasp_plan_id",
            "target_id",
            "mapping_calibration_id",
            "planner_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_empty(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "source_observation_sequence",
            _non_negative_int(
                self.source_observation_sequence,
                field_name="source_observation_sequence",
            ),
        )
        object.__setattr__(
            self,
            "arm_side",
            _enum(self.arm_side, ArmSide, field_name="arm_side"),
        )
        poses = (
            self.grasp_frame,
            self.pregrasp_pose,
            self.contact_pose,
            self.retreat_pose,
        )
        if not all(isinstance(pose, Pose) for pose in poses):
            raise TypeError("grasp_frame and motion poses must be Pose values")
        frames = {pose.frame_id for pose in poses}
        if len(frames) != 1:
            raise ValueError("all plan poses must use the same frame_id")
        residual = _finite_float_tuple(
            self.max_residual_translation_m,
            field_name="max_residual_translation_m",
            length=3,
        )
        if any(value <= 0.0 for value in residual):
            raise ValueError("max_residual_translation_m values must be > 0")
        object.__setattr__(self, "max_residual_translation_m", residual)
        rotation = _finite_float_tuple(
            self.max_residual_rotation_rad,
            field_name="max_residual_rotation_rad",
            length=3,
        )
        if any(value <= 0.0 for value in rotation):
            raise ValueError("max_residual_rotation_rad values must be > 0")
        object.__setattr__(self, "max_residual_rotation_rad", rotation)
        axis = _finite_float_tuple(
            self.pull_axis_in_grasp_frame,
            field_name="pull_axis_in_grasp_frame",
            length=3,
        )
        if any(abs(value - expected) > 1e-8 for value, expected in zip(axis, (0, 0, 1))):
            raise ValueError("pull_axis_in_grasp_frame is frozen as (0,0,1)")
        object.__setattr__(self, "pull_axis_in_grasp_frame", axis)
        object.__setattr__(
            self,
            "expires_monotonic_ns",
            _positive_int(
                self.expires_monotonic_ns,
                field_name="expires_monotonic_ns",
            ),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "grasp_plan_id": self.grasp_plan_id,
            "target_id": self.target_id,
            "source_observation_sequence": self.source_observation_sequence,
            "arm_side": self.arm_side.value,
            "pregrasp_pose": self.pregrasp_pose.as_dict(),
            "planner_version": self.planner_version,
            "expires_monotonic_ns": self.expires_monotonic_ns,
        }


@dataclass(frozen=True, slots=True)
class SafetyAssessment:
    """Plan- and state-bound reachability and constrained-workspace decision."""

    assessment_id: str
    grasp_plan_id: str
    observation_sequence: int
    arm_state_sequence: int
    reachable: bool
    joint_limits_ok: bool
    path_sampled_ok: bool
    collision_status: CollisionStatus
    allowed_workspace_xyzxyz_m: tuple[float, ...]
    expires_monotonic_ns: int
    reason: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("assessment_id", "grasp_plan_id"):
            object.__setattr__(
                self,
                field_name,
                _non_empty(getattr(self, field_name), field_name=field_name),
            )
        for field_name in ("observation_sequence", "arm_state_sequence"):
            object.__setattr__(
                self,
                field_name,
                _non_negative_int(getattr(self, field_name), field_name=field_name),
            )
        for field_name in ("reachable", "joint_limits_ok", "path_sampled_ok"):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        object.__setattr__(
            self,
            "collision_status",
            _enum(
                self.collision_status,
                CollisionStatus,
                field_name="collision_status",
            ),
        )
        workspace = _finite_float_tuple(
            self.allowed_workspace_xyzxyz_m,
            field_name="allowed_workspace_xyzxyz_m",
            length=6,
        )
        if any(workspace[index] > workspace[index + 3] for index in range(3)):
            raise ValueError("workspace minima must not exceed maxima")
        object.__setattr__(self, "allowed_workspace_xyzxyz_m", workspace)
        object.__setattr__(
            self,
            "expires_monotonic_ns",
            _positive_int(
                self.expires_monotonic_ns,
                field_name="expires_monotonic_ns",
            ),
        )
        if self.reason is not None:
            object.__setattr__(
                self, "reason", _non_empty(self.reason, field_name="reason")
            )

    @property
    def approved(self) -> bool:
        return (
            self.reachable
            and self.joint_limits_ok
            and self.path_sampled_ok
            and self.collision_status
            in {CollisionStatus.SAFE, CollisionStatus.SIMULATED_SAFE}
        )


@dataclass(frozen=True, slots=True)
class TomatoPickVerification:
    """Independent evidence for detachment and retained/no-slip success."""

    execution_id: str
    detached: EvidenceState
    retained_without_slip: EvidenceState
    dropped: bool
    stable_window_s: float
    evidence_sources: tuple[str, ...]
    verifier_version: str
    valid: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_id",
            _non_empty(self.execution_id, field_name="execution_id"),
        )
        for field_name in ("detached", "retained_without_slip"):
            object.__setattr__(
                self,
                field_name,
                _enum(getattr(self, field_name), EvidenceState, field_name=field_name),
            )
        if not isinstance(self.dropped, bool) or not isinstance(self.valid, bool):
            raise TypeError("dropped and valid must be bools")
        stable = _finite_float(self.stable_window_s, field_name="stable_window_s")
        if stable < 0.0:
            raise ValueError("stable_window_s must be >= 0")
        object.__setattr__(self, "stable_window_s", stable)
        sources = tuple(
            _non_empty(source, field_name="evidence_sources[]")
            for source in self.evidence_sources
        )
        object.__setattr__(self, "evidence_sources", sources)
        object.__setattr__(
            self,
            "verifier_version",
            _non_empty(self.verifier_version, field_name="verifier_version"),
        )
        if self.reason is not None:
            object.__setattr__(
                self, "reason", _non_empty(self.reason, field_name="reason")
            )

    @property
    def successful(self) -> bool:
        return (
            self.valid
            and self.detached is EvidenceState.CONFIRMED
            and self.retained_without_slip is EvidenceState.CONFIRMED
            and not self.dropped
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "detached": self.detached.value,
            "retained_without_slip": self.retained_without_slip.value,
            "dropped": self.dropped,
            "stable_window_s": self.stable_window_s,
            "evidence_sources": self.evidence_sources,
            "verifier_version": self.verifier_version,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TomatoPickSkillRequest:
    """Small call into the local skill runner."""

    execution_id: str
    target_id: str
    grasp_plan_id: str
    arm_side: ArmSide
    policy_id: str
    max_duration_s: float

    def __post_init__(self) -> None:
        for field_name in (
            "execution_id",
            "target_id",
            "grasp_plan_id",
            "policy_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_empty(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "arm_side",
            _enum(self.arm_side, ArmSide, field_name="arm_side"),
        )
        duration = _finite_float(self.max_duration_s, field_name="max_duration_s")
        if duration <= 0.0:
            raise ValueError("max_duration_s must be > 0")
        object.__setattr__(self, "max_duration_s", duration)


@dataclass(frozen=True, slots=True)
class TomatoPickSkillResult:
    """Terminal skill result whose success must agree with independent evidence."""

    execution_id: str
    target_id: str
    grasp_plan_id: str
    status: SkillStatus
    final_phase: PickPhase
    success: bool
    verification: TomatoPickVerification
    duration_s: float
    policy_id: str
    failure_code: PickFailureCode | None = None
    peak_normal_force_n: tuple[float | None, ...] = (None,) * 5
    peak_tangential_force_n: tuple[float | None, ...] = (None,) * 5
    final_arm_sequence: int | None = None
    final_hand_sequence: int | None = None
    final_tactile_sequence: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "execution_id",
            "target_id",
            "grasp_plan_id",
            "policy_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_empty(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, SkillStatus, field_name="status"),
        )
        object.__setattr__(
            self,
            "final_phase",
            _enum(self.final_phase, PickPhase, field_name="final_phase"),
        )
        if not isinstance(self.success, bool):
            raise TypeError("success must be a bool")
        if not isinstance(self.verification, TomatoPickVerification):
            raise TypeError("verification must be a TomatoPickVerification")
        duration = _finite_float(self.duration_s, field_name="duration_s")
        if duration < 0.0:
            raise ValueError("duration_s must be >= 0")
        object.__setattr__(self, "duration_s", duration)
        if self.failure_code is not None:
            object.__setattr__(
                self,
                "failure_code",
                _enum(
                    self.failure_code,
                    PickFailureCode,
                    field_name="failure_code",
                ),
            )
        for field_name in ("peak_normal_force_n", "peak_tangential_force_n"):
            raw = tuple(getattr(self, field_name))
            if len(raw) != 5:
                raise ValueError(f"{field_name} must contain exactly five values")
            values = tuple(
                _optional_bounded_float(
                    value,
                    field_name=f"{field_name}[{index}]",
                    lower=0.0,
                    upper=20.0,
                )
                for index, value in enumerate(raw)
            )
            object.__setattr__(self, field_name, values)
        for field_name in (
            "final_arm_sequence",
            "final_hand_sequence",
            "final_tactile_sequence",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _non_negative_int(value, field_name=field_name),
                )

        expected_success = (
            self.status is SkillStatus.SUCCEEDED
            and self.verification.successful
        )
        if self.success != expected_success:
            raise ValueError("success must agree with status and verification")
        if self.success and self.failure_code is not None:
            raise ValueError("a successful skill result cannot have failure_code")
        if not self.success and self.failure_code is None:
            raise ValueError("an unsuccessful skill result requires failure_code")

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "target_id": self.target_id,
            "grasp_plan_id": self.grasp_plan_id,
            "status": self.status.value,
            "final_phase": self.final_phase.value,
            "success": self.success,
            "verification": self.verification.as_dict(),
            "failure_code": (
                self.failure_code.value if self.failure_code is not None else None
            ),
            "duration_s": self.duration_s,
            "peak_normal_force_n": self.peak_normal_force_n,
            "peak_tangential_force_n": self.peak_tangential_force_n,
            "policy_id": self.policy_id,
        }


__all__ = [
    "ArmSide",
    "ArmState",
    "CollisionStatus",
    "CommandResult",
    "EvidenceState",
    "ExecutionHandle",
    "ExecutionState",
    "FingerTactile",
    "HandState",
    "HandSynergy",
    "PickFailureCode",
    "PickPhase",
    "SafetyAssessment",
    "SkillStatus",
    "StateStamp",
    "TactileFrame",
    "TomatoDetectionResult",
    "TomatoGraspPlan",
    "TomatoPickSkillRequest",
    "TomatoPickSkillResult",
    "TomatoPickVerification",
    "TomatoTarget",
]
