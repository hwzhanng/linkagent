"""Atomic, fail-closed motion into the PPO training-distribution handoff."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from module.robotics import ControlBusyError, PoseMoveCommand, RoboticsBundle
from module.skills.tomato_pick import evaluate_ppo_ready
from proto import (
    ExecutionState,
    MoveToPickReadyRequest,
    SafetyAssessment,
    TomatoGraspPlan,
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


@runtime_checkable
class PickReadySafetyPort(Protocol):
    def assess_pregrasp(self, plan: TomatoGraspPlan, arm_state: object) -> SafetyAssessment: ...
    def assess_skill_envelope(self, plan: TomatoGraspPlan, arm_state: object) -> SafetyAssessment: ...


class MoveToPickReadySkill(AtomicSkillBase):
    descriptor = AtomicSkillDescriptor(
        skill_id="move_to_pick_ready",
        version="1.0.0",
        description="Reach and prove a palm/fruit state inside the PPO reset distribution.",
        effectful=True,
        required_resources=("arm", "control_lease", "motion_safety"),
        hardware_approved=False,
    )

    def __init__(
        self,
        *,
        robotics: RoboticsBundle,
        safety: PickReadySafetyPort,
        execution_mode: str = "offline_mock",
        speed: float = 0.20,
        acceleration: float = 0.30,
        max_target_age_s: float = 0.5,
    ) -> None:
        super().__init__(clock=robotics.clock)
        if not isinstance(robotics, RoboticsBundle):
            raise TypeError("robotics must be a RoboticsBundle")
        if not isinstance(safety, PickReadySafetyPort):
            raise TypeError("safety must implement PickReadySafetyPort")
        if execution_mode not in {"offline_mock", "simulation", "shadow"}:
            raise ValueError("execution_mode must be offline_mock, simulation, or shadow")
        if (
            execution_mode == "offline_mock"
            and robotics.verifier_context().get("mock_world") is None
        ):
            raise RuntimeError(
                "offline_mock ready motion requires an explicit mock_world backend"
            )
        for name, value in {
            "speed": speed,
            "acceleration": acceleration,
            "max_target_age_s": max_target_age_s,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be > 0")
        self._robotics = robotics
        self._safety = safety
        self._execution_mode = execution_mode
        self._speed = float(speed)
        self._acceleration = float(acceleration)
        self._max_target_age_s = float(max_target_age_s)
        self._assessments: dict[str, SafetyAssessment] = {}

    def assessment_for(self, grasp_plan_id: str) -> SafetyAssessment:
        try:
            return self._assessments[grasp_plan_id]
        except KeyError as exc:
            raise KeyError(f"no successful ready assessment for {grasp_plan_id!r}") from exc

    def run(self, request: object) -> AtomicSkillResult:
        if not isinstance(request, MoveToPickReadyRequest):
            raise TypeError("request must be a MoveToPickReadyRequest")
        cached = self._cached(request.execution_id)
        if cached is not None:
            return cached
        trace = self._recorder(request.execution_id)
        trace.record(
            "admission",
            SkillTraceEventKind.STARTED,
            input_summary={
                "grasp_plan_id": request.plan.grasp_plan_id,
                "target_id": request.target.target_id,
                "execution_mode": self._execution_mode,
            },
        )
        now_ns = self._clock.monotonic_ns()
        if request.plan.planner_version != "ppo_training_distribution_ready_v1":
            return self._failed(
                trace,
                AtomicSkillFailureCode.PRECONDITION_FAILED,
                "ready motion requires a training-distribution grasp plan",
            )
        if now_ns >= request.plan.expires_monotonic_ns:
            return self._failed(
                trace,
                AtomicSkillFailureCode.TARGET_STALE,
                "grasp plan expired before ready motion",
            )
        target_age_s = max(0.0, (now_ns - request.target.sample_monotonic_ns) / 1.0e9)
        if target_age_s > self._max_target_age_s:
            return self._failed(
                trace,
                AtomicSkillFailureCode.TARGET_STALE,
                "visual target is too old for ready motion",
            )
        arm_state = self._robotics.arm.get_state(request.plan.arm_side)
        pregrasp = self._safety.assess_pregrasp(request.plan, arm_state)
        if not pregrasp.approved:
            return self._failed(
                trace,
                AtomicSkillFailureCode.SAFETY_REJECTED,
                pregrasp.reason or "pregrasp safety rejected the plan",
            )
        trace.record(
            "motion_safety",
            SkillTraceEventKind.CHECKPOINT,
            output_summary={"assessment_id": pregrasp.assessment_id},
        )
        command_id: str | None = None
        actual_pose = request.plan.pregrasp_pose
        if self._execution_mode in {"offline_mock", "simulation"}:
            try:
                lease = self._robotics.leases.acquire(
                    owner_id=f"pick-ready:{request.execution_id}",
                    side=request.plan.arm_side,
                    ttl_s=request.max_duration_s + 0.5,
                )
            except ControlBusyError as exc:
                return self._failed(
                    trace, AtomicSkillFailureCode.CONTROL_BUSY, str(exc)
                )
            try:
                handle = self._robotics.arm.move_pose(
                    lease,
                    PoseMoveCommand(
                        command_id=f"ready-{request.plan.grasp_plan_id}",
                        side=request.plan.arm_side,
                        target_pose=request.plan.pregrasp_pose,
                        speed=self._speed,
                        acceleration=self._acceleration,
                        timeout_s=request.max_duration_s,
                    ),
                )
            finally:
                self._robotics.leases.release(lease)
            command_id = handle.command_id
            if handle.state is not ExecutionState.SUCCEEDED:
                return self._failed(
                    trace,
                    AtomicSkillFailureCode.COMMAND_FAILED,
                    "controller rejected move_to_pick_ready",
                )
            actual_pose = self._robotics.arm.get_state(request.plan.arm_side).ee_pose
            trace.record(
                "motion",
                SkillTraceEventKind.CHECKPOINT,
                output_summary={"command_id": command_id, "state": handle.state.value},
            )

        envelope = self._safety.assess_skill_envelope(
            request.plan,
            self._robotics.arm.get_state(request.plan.arm_side),
        )
        if not envelope.approved:
            return self._failed(
                trace,
                AtomicSkillFailureCode.SAFETY_REJECTED,
                envelope.reason or "PPO skill envelope was rejected",
            )
        evidence = evaluate_ppo_ready(
            target=request.target,
            plan=request.plan,
            actual_palm_pose=actual_pose,
        )
        if not bool(evidence["admitted"]):
            checks = dict(evidence["checks"])  # type: ignore[arg-type]
            failed = ", ".join(name for name, passed in checks.items() if not passed)
            return self._failed(
                trace,
                AtomicSkillFailureCode.PRECONDITION_FAILED,
                f"PPO training-distribution admission failed: {failed}",
            )
        self._assessments[request.plan.grasp_plan_id] = envelope
        trace.record(
            "ppo_handoff",
            SkillTraceEventKind.SUCCEEDED,
            output_summary={
                "shadow_only": self._execution_mode == "shadow",
                "fruit_center_in_palm_m": evidence["fruit_center_in_palm_m"],
                "fruit_semi_axes_m": evidence["fruit_semi_axes_m"],
                "pull_axis_in_palm": evidence["pull_axis_in_palm"],
            },
        )
        return self._result(
            trace,
            status=AtomicSkillStatus.SUCCEEDED,
            message="PPO ready-position admitted",
            output={
                "grasp_plan_id": request.plan.grasp_plan_id,
                "command_id": command_id,
                "execution_mode": self._execution_mode,
                "shadow_only": self._execution_mode == "shadow",
                "pregrasp_safety_assessment_id": pregrasp.assessment_id,
                "skill_envelope_assessment_id": envelope.assessment_id,
                "ppo_ready_evidence": evidence,
            },
        )

    def _failed(
        self,
        trace: SkillTraceRecorder,
        code: AtomicSkillFailureCode,
        message: str,
    ) -> AtomicSkillResult:
        trace.record("admission", SkillTraceEventKind.FAILED, message=message)
        return self._result(
            trace,
            status=(
                AtomicSkillStatus.SAFETY_ABORT
                if code in {AtomicSkillFailureCode.SAFETY_REJECTED, AtomicSkillFailureCode.CONTROL_BUSY}
                else AtomicSkillStatus.RETRYABLE_FAILURE
            ),
            failure_code=code,
            message=message,
        )


__all__ = ["MoveToPickReadySkill", "PickReadySafetyPort"]
