"""Bounded motion or shadow planning to a reviewed active-view waypoint."""

from __future__ import annotations

import math
from collections.abc import Sequence

from module.robotics import (
    ControlBusyError,
    LinearMoveCommand,
    RoboticsBundle,
)
from proto import (
    ActiveViewRepositionEvidence,
    ActiveViewRepositionRequest,
    ArmSide,
    ExecutionState,
    Pose,
)

from module.skills.core import (
    AtomicSkillBase,
    AtomicSkillDescriptor,
    AtomicSkillFailureCode,
    AtomicSkillResult,
    AtomicSkillStatus,
    SkillTraceEventKind,
    SkillTraceRecorder,
)


def _non_empty(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field_name} must be finite and > 0")
    return float(value)


def _vector3(value: object, *, field_name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a three-number sequence")
    try:
        items = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a three-number sequence") from exc
    if len(items) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    result: list[float] = []
    for index, item in enumerate(items):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{field_name}[{index}] must be finite")
        result.append(float(item))
    return result[0], result[1], result[2]


def _distance(left: Pose, right: Pose) -> float:
    return math.sqrt(
        sum(
            (a - b) ** 2
            for a, b in zip(left.position_m, right.position_m, strict=True)
        )
    )


def _orientation_error(left: Pose, right: Pose) -> float:
    dot = abs(
        sum(
            a * b
            for a, b in zip(
                left.quaternion_wxyz, right.quaternion_wxyz, strict=True
            )
        )
    )
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


class ActiveViewRepositionSkill(AtomicSkillBase):
    """Move the arm once to a reviewed de-occlusion pose under a fixed camera."""

    descriptor = AtomicSkillDescriptor(
        skill_id="active_view_reposition",
        version="1.0.0",
        description=(
            "Move the configured arm by one reviewed Cartesian de-occlusion "
            "offset under a fixed camera; never accepts an Agent-provided pose."
        ),
        effectful=True,
        required_resources=("arm", "control_lease", "active_view_profile"),
        hardware_approved=False,
    )

    def __init__(
        self,
        *,
        robotics: RoboticsBundle,
        execution_mode: str,
        view_profile_id: str,
        view_arm_side: ArmSide | str,
        view_offsets_world_m: Sequence[Sequence[float]],
        workspace_min_m: Sequence[float],
        workspace_max_m: Sequence[float],
        max_offset_m: float = 0.10,
        speed: float = 0.05,
        acceleration: float = 0.10,
        settle_time_s: float = 0.05,
        position_tolerance_m: float = 0.005,
        orientation_tolerance_rad: float = 0.03,
    ) -> None:
        super().__init__(clock=robotics.clock)
        if not isinstance(robotics, RoboticsBundle):
            raise TypeError("robotics must be a RoboticsBundle")
        if execution_mode not in {
            "offline_mock",
            "simulation",
            "shadow",
            "hardware",
        }:
            raise ValueError(
                "execution_mode must be offline_mock, simulation, shadow, or hardware"
            )
        if (
            execution_mode == "offline_mock"
            and robotics.verifier_context().get("mock_world") is None
        ):
            raise RuntimeError(
                "offline_mock active-view motion requires an explicit mock_world backend"
            )
        try:
            side = (
                view_arm_side
                if isinstance(view_arm_side, ArmSide)
                else ArmSide(view_arm_side)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("view_arm_side must be left or right") from exc
        if isinstance(view_offsets_world_m, (str, bytes, bytearray)):
            raise TypeError("view_offsets_world_m must be a sequence")
        offsets = tuple(
            _vector3(value, field_name=f"view_offsets_world_m[{index}]")
            for index, value in enumerate(view_offsets_world_m)
        )
        if not 1 <= len(offsets) <= 16:
            raise ValueError("view_offsets_world_m must contain between 1 and 16 offsets")
        maximum_offset = _positive(max_offset_m, field_name="max_offset_m")
        for offset in offsets:
            norm = math.sqrt(sum(value * value for value in offset))
            if norm <= 0.0 or norm > maximum_offset:
                raise ValueError(
                    "every active-view offset must be non-zero and within max_offset_m"
                )
        minimum = _vector3(workspace_min_m, field_name="workspace_min_m")
        maximum = _vector3(workspace_max_m, field_name="workspace_max_m")
        if any(low >= high for low, high in zip(minimum, maximum, strict=True)):
            raise ValueError("workspace_min_m must be strictly below workspace_max_m")
        self._robotics = robotics
        self._execution_mode = execution_mode
        self._profile_id = _non_empty(
            view_profile_id, field_name="view_profile_id"
        )
        self._side = side
        self._offsets = offsets
        self._workspace_min = minimum
        self._workspace_max = maximum
        self._speed = _positive(speed, field_name="speed")
        self._acceleration = _positive(acceleration, field_name="acceleration")
        self._settle_time_s = _positive(settle_time_s, field_name="settle_time_s")
        self._position_tolerance = _positive(
            position_tolerance_m, field_name="position_tolerance_m"
        )
        self._orientation_tolerance = _positive(
            orientation_tolerance_rad, field_name="orientation_tolerance_rad"
        )

    @property
    def execution_mode(self) -> str:
        return self._execution_mode

    @property
    def viewpoint_count(self) -> int:
        return len(self._offsets)

    def run(self, request: object) -> AtomicSkillResult:
        if not isinstance(request, ActiveViewRepositionRequest):
            raise TypeError("request must be an ActiveViewRepositionRequest")
        cached = self._cached(request.execution_id)
        if cached is not None:
            return cached
        trace = self._recorder(request.execution_id)
        trace.record(
            "viewpoint",
            SkillTraceEventKind.STARTED,
            input_summary={
                "target_id": request.target.target_id,
                "target_observation_sequence": request.target.observation_sequence,
                "arm_side": request.arm_side.value,
                "viewpoint_index": request.viewpoint_index,
                "view_profile_id": self._profile_id,
                "execution_mode": self._execution_mode,
            },
            evidence_refs=(f"active-view-profile:{self._profile_id}",),
        )
        if request.arm_side is not self._side:
            return self._failed(
                trace,
                AtomicSkillFailureCode.INVALID_REQUEST,
                f"active-view profile is bound to {self._side.value} arm",
            )
        if request.viewpoint_index >= len(self._offsets):
            return self._failed(
                trace,
                AtomicSkillFailureCode.VIEWPOINTS_EXHAUSTED,
                "active-view profile has no remaining viewpoint",
            )
        offset = self._offsets[request.viewpoint_index]
        requested_pose = Pose(
            tuple(
                value + delta
                for value, delta in zip(
                    request.reference_pose.position_m, offset, strict=True
                )
            ),
            request.reference_pose.quaternion_wxyz,
            request.reference_pose.frame_id,
        )
        if any(
            value < low or value > high
            for value, low, high in zip(
                requested_pose.position_m,
                self._workspace_min,
                self._workspace_max,
                strict=True,
            )
        ):
            return self._failed(
                trace,
                AtomicSkillFailureCode.SAFETY_REJECTED,
                "active-view pose is outside the configured workspace",
            )
        arm_state = self._robotics.arm.get_state(request.arm_side)
        if not arm_state.enabled or not arm_state.stamp.valid:
            return self._failed(
                trace,
                AtomicSkillFailureCode.SAFETY_REJECTED,
                "arm state is disabled or invalid before active-view motion",
            )
        ik = self._robotics.arm.ik(
            request.arm_side,
            requested_pose,
            arm_state.joint_positions_rad,
        )
        if not ik.success:
            return self._failed(
                trace,
                AtomicSkillFailureCode.VIEWPOINT_UNREACHABLE,
                "active-view pose failed inverse kinematics",
            )
        trace.record(
            "motion_safety",
            SkillTraceEventKind.CHECKPOINT,
            output_summary={
                "ik_reachable": True,
                "workspace_checked": True,
                "collision_proven": False,
                "hardware_approved": False,
            },
            evidence_refs=(f"active-view-profile:{self._profile_id}",),
        )
        if self._execution_mode == "shadow":
            evidence = self._evidence(
                request,
                requested_pose=requested_pose,
                actual_pose=arm_state.ee_pose,
                command_id=None,
                motion_executed=False,
            )
            trace.record(
                "shadow",
                SkillTraceEventKind.SUCCEEDED,
                message="active-view pose planned without robot motion",
                output_summary=evidence.as_dict(),
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.SUCCEEDED,
                message="active-view shadow plan completed",
                output=evidence.as_dict(),
            )
        try:
            lease = self._robotics.leases.acquire(
                owner_id=f"active-view:{request.execution_id}",
                side=request.arm_side,
                ttl_s=request.max_duration_s + self._settle_time_s + 0.5,
            )
        except ControlBusyError as exc:
            return self._failed(
                trace, AtomicSkillFailureCode.CONTROL_BUSY, str(exc), safety_abort=True
            )
        command_id = f"active-view-{request.execution_id}"
        try:
            handle = self._robotics.arm.move_linear(
                lease,
                LinearMoveCommand(
                    command_id=command_id,
                    side=request.arm_side,
                    target_pose=requested_pose,
                    speed=self._speed,
                    acceleration=self._acceleration,
                    timeout_s=request.max_duration_s,
                ),
            )
        finally:
            self._robotics.leases.release(lease)
        if handle.state is not ExecutionState.SUCCEEDED:
            return self._failed(
                trace,
                AtomicSkillFailureCode.COMMAND_FAILED,
                "controller rejected active-view MoveL command",
                safety_abort=True,
            )
        self._clock.sleep(self._settle_time_s)
        actual_pose = self._robotics.arm.get_state(request.arm_side).ee_pose
        evidence = self._evidence(
            request,
            requested_pose=requested_pose,
            actual_pose=actual_pose,
            command_id=command_id,
            motion_executed=True,
        )
        if (
            evidence.position_error_m > self._position_tolerance
            or evidence.orientation_error_rad > self._orientation_tolerance
        ):
            trace.record(
                "pose_readback",
                SkillTraceEventKind.FAILED,
                message="active-view actual pose missed the configured tolerance",
                output_summary=evidence.as_dict(),
            )
            return self._result(
                trace,
                status=AtomicSkillStatus.SAFETY_ABORT,
                failure_code=AtomicSkillFailureCode.VIEW_POSE_NOT_REACHED,
                message="active-view actual pose missed the configured tolerance",
                output=evidence.as_dict(),
            )
        trace.record(
            "pose_readback",
            SkillTraceEventKind.SUCCEEDED,
            message="active-view MoveL reached the reviewed viewpoint",
            output_summary=evidence.as_dict(),
        )
        return self._result(
            trace,
            status=AtomicSkillStatus.SUCCEEDED,
            message="active-view motion completed",
            output=evidence.as_dict(),
        )

    def _evidence(
        self,
        request: ActiveViewRepositionRequest,
        *,
        requested_pose: Pose,
        actual_pose: Pose,
        command_id: str | None,
        motion_executed: bool,
    ) -> ActiveViewRepositionEvidence:
        return ActiveViewRepositionEvidence(
            target_id=request.target.target_id,
            viewpoint_id=f"{self._profile_id}:view-{request.viewpoint_index}",
            profile_id=self._profile_id,
            arm_side=request.arm_side,
            requested_pose=requested_pose,
            actual_pose=actual_pose,
            motion_executed=motion_executed,
            shadow_only=self._execution_mode == "shadow",
            simulated_safety_only=self._execution_mode != "hardware",
            ik_reachable=True,
            position_error_m=_distance(actual_pose, requested_pose),
            orientation_error_rad=_orientation_error(actual_pose, requested_pose),
            command_id=command_id,
        )

    def _failed(
        self,
        trace: SkillTraceRecorder,
        code: AtomicSkillFailureCode,
        message: str,
        *,
        safety_abort: bool = False,
    ) -> AtomicSkillResult:
        trace.record("active_view", SkillTraceEventKind.FAILED, message=message)
        return self._result(
            trace,
            status=(
                AtomicSkillStatus.SAFETY_ABORT
                if safety_abort
                else AtomicSkillStatus.RETRYABLE_FAILURE
            ),
            failure_code=code,
            message=message,
        )


__all__ = ["ActiveViewRepositionSkill"]
