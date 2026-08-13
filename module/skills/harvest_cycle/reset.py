"""Return the robot to a clean observation state after one verified cycle."""

from __future__ import annotations

import math

from module.robotics import ControlBusyError, PoseMoveCommand, RoboticsBundle
from proto import ExecutionState, ResetForNextTargetRequest

from module.skills.core import (
    AtomicSkillBase,
    AtomicSkillDescriptor,
    AtomicSkillFailureCode,
    AtomicSkillResult,
    AtomicSkillStatus,
    SkillTraceEventKind,
    SkillTraceRecorder,
)


class ResetForNextTargetSkill(AtomicSkillBase):
    descriptor = AtomicSkillDescriptor(
        skill_id="reset_for_next_target",
        version="1.0.0",
        description="Return an empty hand to the configured observation pose for the next harvest cycle.",
        effectful=True,
        required_resources=("arm_control", "hand_state", "control_lease"),
        hardware_approved=False,
    )

    def __init__(
        self,
        *,
        robotics: RoboticsBundle,
        speed: float = 0.10,
        acceleration: float = 0.15,
        max_arrival_error_m: float = 0.01,
    ) -> None:
        super().__init__(clock=robotics.clock)
        if not isinstance(robotics, RoboticsBundle):
            raise TypeError("robotics must be a RoboticsBundle")
        for name, value in {
            "speed": speed,
            "acceleration": acceleration,
            "max_arrival_error_m": max_arrival_error_m,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be > 0")
        self._robotics = robotics
        self._speed = float(speed)
        self._acceleration = float(acceleration)
        self._max_arrival_error_m = float(max_arrival_error_m)

    def run(self, request: object) -> AtomicSkillResult:
        if not isinstance(request, ResetForNextTargetRequest):
            raise TypeError("request must be a ResetForNextTargetRequest")
        cached = self._cached(request.execution_id)
        if cached is not None:
            return cached
        trace = self._recorder(request.execution_id)
        trace.record(
            "reset",
            SkillTraceEventKind.STARTED,
            input_summary={
                "completed_cycle_id": request.completed_cycle_id,
                "arm_side": request.arm_side.value,
                "reset_pose": request.reset_pose.as_dict(),
            },
        )
        hand = self._robotics.hand.get_state(request.arm_side)
        if hand.synergy.closure > 0.05 or hand.fault_codes:
            return self._failed(
                trace,
                AtomicSkillFailureCode.RESET_HAND_NOT_EMPTY,
                "next-target reset requires a confirmed empty open hand",
            )
        arm = self._robotics.arm.get_state(request.arm_side)
        if arm.ee_pose.frame_id != request.reset_pose.frame_id:
            return self._failed(
                trace,
                AtomicSkillFailureCode.PRECONDITION_FAILED,
                "arm and reset pose frames do not match",
            )
        ik = self._robotics.arm.ik(
            request.arm_side, request.reset_pose, arm.joint_positions_rad
        )
        if not ik.success:
            return self._failed(
                trace,
                AtomicSkillFailureCode.RESET_POSE_NOT_REACHED,
                "configured next-target reset pose is not IK reachable",
            )
        lease = None
        try:
            lease = self._robotics.leases.acquire(
                owner_id=request.execution_id,
                side=request.arm_side,
                ttl_s=request.max_duration_s,
            )
            handle = self._robotics.arm.move_pose(
                lease,
                PoseMoveCommand(
                    command_id=f"{request.execution_id}-arm",
                    side=request.arm_side,
                    target_pose=request.reset_pose,
                    speed=self._speed,
                    acceleration=self._acceleration,
                    timeout_s=request.max_duration_s,
                ),
            )
            if handle.state is not ExecutionState.SUCCEEDED:
                return self._failed(
                    trace,
                    AtomicSkillFailureCode.COMMAND_FAILED,
                    "next-target reset arm command failed",
                    status=AtomicSkillStatus.INFRASTRUCTURE_ERROR,
                )
        except ControlBusyError as exc:
            return self._failed(trace, AtomicSkillFailureCode.CONTROL_BUSY, str(exc))
        except (RuntimeError, ValueError) as exc:
            return self._failed(
                trace,
                AtomicSkillFailureCode.COMMAND_FAILED,
                str(exc),
                status=AtomicSkillStatus.INFRASTRUCTURE_ERROR,
            )
        finally:
            if lease is not None:
                self._robotics.leases.release(lease)
        actual = self._robotics.arm.get_state(request.arm_side).ee_pose
        arrival_error = math.dist(actual.position_m, request.reset_pose.position_m)
        if arrival_error > self._max_arrival_error_m:
            return self._failed(
                trace,
                AtomicSkillFailureCode.RESET_POSE_NOT_REACHED,
                "next-target reset pose was not reached",
                output={"arrival_error_m": arrival_error, "command_id": handle.command_id},
            )
        trace.record(
            "reset",
            SkillTraceEventKind.SUCCEEDED,
            output_summary={
                "completed_cycle_id": request.completed_cycle_id,
                "command_id": handle.command_id,
                "arrival_error_m": arrival_error,
                "hand_empty": True,
            },
            evidence_refs=(f"arm_command:{handle.command_id}", "hand_state"),
        )
        return self._result(
            trace,
            status=AtomicSkillStatus.SUCCEEDED,
            message="robot is ready to observe the next target",
            output={
                "completed_cycle_id": request.completed_cycle_id,
                "command_id": handle.command_id,
                "arrival_error_m": arrival_error,
                "hand_empty": True,
                "next_observation_required": True,
            },
        )

    def _failed(
        self,
        trace: SkillTraceRecorder,
        code: AtomicSkillFailureCode,
        message: str,
        *,
        status: AtomicSkillStatus = AtomicSkillStatus.RETRYABLE_FAILURE,
        output: dict[str, object] | None = None,
    ) -> AtomicSkillResult:
        trace.record("reset", SkillTraceEventKind.FAILED, message=message)
        return self._result(
            trace,
            status=status,
            failure_code=code,
            message=message,
            output={} if output is None else output,
        )


__all__ = ["ResetForNextTargetSkill"]
