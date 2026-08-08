"""Application service behind the five model-visible tomato-picking tools."""

from __future__ import annotations

from typing import Any

from module.robotics import ControlBusyError, PoseMoveCommand, RoboticsBundle
from module.skills.tomato_pick import (
    TomatoPickPolicy,
    PickVerifier,
    TomatoPickSkillContext,
    TomatoPickSkillRunner,
)
from proto import (
    ArmSide,
    ExecutionState,
    Observation,
    PickFailureCode,
    SafetyAssessment,
    TomatoDetectionResult,
    TomatoGraspPlan,
    TomatoPickSkillRequest,
    TomatoPickSkillResult,
    TomatoTarget,
)

from .contracts import MotionSafetyPort, TomatoGraspPlanner, TomatoPerceptionPort
from .state import AgriPickState


class TomatoPickServiceError(RuntimeError):
    """Expected workflow failure safe to return through an Agent tool."""


class TomatoPickService:
    """Own state, IDs, idempotency, and device/skill orchestration for C1."""

    def __init__(
        self,
        *,
        robotics: RoboticsBundle,
        perception: TomatoPerceptionPort,
        planner: TomatoGraspPlanner,
        safety: MotionSafetyPort,
        verifier: PickVerifier,
        policy: TomatoPickPolicy,
        default_arm_side: ArmSide,
        max_skill_duration_s: float,
    ) -> None:
        if not isinstance(robotics, RoboticsBundle):
            raise TypeError("robotics must implement RoboticsBundle")
        if not isinstance(perception, TomatoPerceptionPort):
            raise TypeError("perception must implement TomatoPerceptionPort")
        if not isinstance(planner, TomatoGraspPlanner):
            raise TypeError("planner must implement TomatoGraspPlanner")
        if not isinstance(safety, MotionSafetyPort):
            raise TypeError("safety must implement MotionSafetyPort")
        if not isinstance(verifier, PickVerifier):
            raise TypeError("verifier must implement PickVerifier")
        if not isinstance(policy, TomatoPickPolicy):
            raise TypeError("policy must implement TomatoPickPolicy")
        if not isinstance(default_arm_side, ArmSide):
            default_arm_side = ArmSide(default_arm_side)
        if (
            isinstance(max_skill_duration_s, bool)
            or not isinstance(max_skill_duration_s, (int, float))
            or max_skill_duration_s <= 0
        ):
            raise ValueError("max_skill_duration_s must be > 0")
        self.robotics = robotics
        self.perception = perception
        self.planner = planner
        self.safety = safety
        self.verifier = verifier
        self.runner = TomatoPickSkillRunner(policy)
        self.default_arm_side = default_arm_side
        self.max_skill_duration_s = float(max_skill_duration_s)
        self.state = AgriPickState.OBSERVE
        self.targets: dict[str, TomatoTarget] = {}
        self.last_detection: TomatoDetectionResult | None = None
        self.plans: dict[str, TomatoGraspPlan] = {}
        self.assessments: dict[str, SafetyAssessment] = {}
        self.executions: dict[str, TomatoPickSkillResult] = {}
        self._execution_by_plan: dict[str, str] = {}
        self._verified_execution_id: str | None = None
        self._closed = False

    @property
    def completed(self) -> bool:
        return self.state is AgriPickState.COMPLETE and self._verified_execution_id is not None

    def detect_tomatoes(self, observation: Observation) -> dict[str, Any]:
        self._require_open()
        result: TomatoDetectionResult = self.perception.detect(observation)
        self.last_detection = result
        self.targets = {target.target_id: target for target in result.targets}
        self.state = AgriPickState.PLAN if self.targets else AgriPickState.OBSERVE
        return result.summary()

    def plan_tomato_grasp(self, target_id: str, arm_side: str = "auto") -> dict[str, Any]:
        self._require_open()
        target_id = self._identifier(target_id, "target_id")
        try:
            target = self.targets[target_id]
        except KeyError as exc:
            raise TomatoPickServiceError(
                f"unknown target_id {target_id!r}; call detect_tomatoes first"
            ) from exc
        if arm_side == "auto":
            side = self.default_arm_side
        else:
            try:
                side = ArmSide(arm_side)
            except (TypeError, ValueError) as exc:
                raise TomatoPickServiceError("arm_side must be auto, left, or right") from exc
        if side is not self.default_arm_side:
            raise TomatoPickServiceError(
                f"robotics bundle is configured for {self.default_arm_side.value!r}, not {side.value!r}"
            )
        arm_state = self.robotics.arm.get_state(side)
        hand_state = self.robotics.hand.get_state(side)
        plan = self.planner.plan(target, arm_state, hand_state)
        self.plans[plan.grasp_plan_id] = plan
        self.state = AgriPickState.PREGRASP
        return plan.summary()

    def move_to_pregrasp(self, grasp_plan_id: str) -> dict[str, Any]:
        self._require_open()
        plan = self._plan(grasp_plan_id)
        now = self.robotics.clock.monotonic_ns()
        if now >= plan.expires_monotonic_ns:
            raise TomatoPickServiceError("grasp plan expired; plan the grasp again")
        arm_state = self.robotics.arm.get_state(plan.arm_side)
        assessment = self.safety.assess_pregrasp(plan, arm_state)
        self.assessments[plan.grasp_plan_id] = assessment
        if not assessment.approved:
            self.state = AgriPickState.ABORTED
            raise TomatoPickServiceError(
                f"pregrasp rejected by motion safety: {assessment.reason or 'unsafe'}"
            )
        pregrasp_assessment = assessment
        try:
            lease = self.robotics.leases.acquire(
                owner_id=f"pregrasp:{plan.grasp_plan_id}",
                side=plan.arm_side,
                ttl_s=2.0,
            )
        except ControlBusyError as exc:
            raise TomatoPickServiceError(str(exc)) from exc
        try:
            handle = self.robotics.arm.move_pose(
                lease,
                PoseMoveCommand(
                    command_id=f"pregrasp-{plan.grasp_plan_id}",
                    side=plan.arm_side,
                    target_pose=plan.pregrasp_pose,
                    speed=0.20,
                    acceleration=0.30,
                    timeout_s=1.5,
                ),
            )
        finally:
            self.robotics.leases.release(lease)
        if handle.state is not ExecutionState.SUCCEEDED:
            self.state = AgriPickState.ABORTED
            raise TomatoPickServiceError("controller rejected move_to_pregrasp")
        assessment = self.safety.assess_skill_envelope(
            plan,
            self.robotics.arm.get_state(plan.arm_side),
        )
        self.assessments[plan.grasp_plan_id] = assessment
        if not assessment.approved:
            self.state = AgriPickState.ABORTED
            raise TomatoPickServiceError(
                f"skill envelope rejected by motion safety: {assessment.reason or 'unsafe'}"
            )
        self.state = AgriPickState.PICK
        return {
            "grasp_plan_id": plan.grasp_plan_id,
            "command_id": handle.command_id,
            "state": handle.state.value,
            "pregrasp_safety_assessment_id": pregrasp_assessment.assessment_id,
            "skill_envelope_assessment_id": assessment.assessment_id,
        }

    def execute_tomato_pick(self, grasp_plan_id: str) -> dict[str, Any]:
        self._require_open()
        plan = self._plan(grasp_plan_id)
        existing_id = self._execution_by_plan.get(plan.grasp_plan_id)
        if existing_id is not None:
            return self.executions[existing_id].as_dict()
        try:
            assessment = self.assessments[plan.grasp_plan_id]
        except KeyError as exc:
            raise TomatoPickServiceError("call move_to_pregrasp before executing the skill") from exc
        if self.state is not AgriPickState.PICK:
            raise TomatoPickServiceError(
                f"pick cannot start from workflow state {self.state.value!r}"
            )
        target = self.targets[plan.target_id]
        execution_id = f"exec-{plan.grasp_plan_id}"
        request = TomatoPickSkillRequest(
            execution_id=execution_id,
            target_id=target.target_id,
            grasp_plan_id=plan.grasp_plan_id,
            arm_side=plan.arm_side,
            policy_id=self.runner.manifest.policy_id,
            max_duration_s=self.max_skill_duration_s,
        )
        context = TomatoPickSkillContext(
            target=target,
            plan=plan,
            safety=assessment,
            perception=self.last_detection,
            arm=self.robotics.arm,
            hand=self.robotics.hand,
            tactile=self.robotics.tactile,
            leases=self.robotics.leases,
            verifier=self.verifier,
            clock=self.robotics.clock,
        )
        result = self.runner.run(request, context)
        self.executions[execution_id] = result
        self._execution_by_plan[plan.grasp_plan_id] = execution_id
        self.state = AgriPickState.VERIFY
        return result.as_dict()

    def verify_tomato_pick(self, execution_id: str) -> dict[str, Any]:
        self._require_open()
        execution_id = self._identifier(execution_id, "execution_id")
        try:
            result = self.executions[execution_id]
        except KeyError as exc:
            raise TomatoPickServiceError(
                f"unknown execution_id {execution_id!r}; execute the skill first"
            ) from exc
        if result.success and result.verification.successful:
            self._verified_execution_id = execution_id
            self.state = AgriPickState.COMPLETE
        else:
            self.state = (
                AgriPickState.PLAN
                if result.failure_code
                in {
                    PickFailureCode.NO_CONTACT,
                    PickFailureCode.UNSTABLE_GRASP,
                    PickFailureCode.DETACH_NOT_CONFIRMED,
                    PickFailureCode.SLIP_DETECTED,
                    PickFailureCode.DROPPED,
                }
                else AgriPickState.ABORTED
            )
        return {
            "execution_id": execution_id,
            "success": result.success and result.verification.successful,
            "workflow_state": self.state.value,
            "verification": result.verification.as_dict(),
            "failure_code": result.failure_code.value if result.failure_code else None,
        }

    def status_summary(self) -> dict[str, Any]:
        return {
            "workflow_state": self.state.value,
            "detected_target_ids": tuple(self.targets),
            "plan_ids": tuple(self.plans),
            "execution_ids": tuple(self.executions),
            "verified_execution_id": self._verified_execution_id,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        try:
            self.perception.close()
        except BaseException as exc:
            errors.append(exc)
        try:
            self.robotics.close()
        except BaseException as exc:
            errors.append(exc)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("agri-pick service cleanup failed", errors)

    def _plan(self, grasp_plan_id: str) -> TomatoGraspPlan:
        grasp_plan_id = self._identifier(grasp_plan_id, "grasp_plan_id")
        try:
            return self.plans[grasp_plan_id]
        except KeyError as exc:
            raise TomatoPickServiceError(
                f"unknown grasp_plan_id {grasp_plan_id!r}; plan the grasp first"
            ) from exc

    @staticmethod
    def _identifier(value: object, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise TomatoPickServiceError(f"{field_name} must be a non-empty string")
        return value.strip()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("TomatoPickService is closed")


__all__ = ["TomatoPickService", "TomatoPickServiceError"]
