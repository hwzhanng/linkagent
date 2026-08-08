"""Adapters for externally supplied perception, planning, and safety gateways.

Construction is dependency-injected and side-effect free.  Camera/ROS/vendor
clients remain outside the task and can be replaced by recordings in tests.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from proto import (
    ArmState,
    HandState,
    Observation,
    SafetyAssessment,
    TomatoDetectionResult,
    TomatoGraspPlan,
    TomatoTarget,
)


@runtime_checkable
class TomatoPerceptionGateway(Protocol):
    def detect_tomatoes(self, observation: Observation) -> TomatoDetectionResult: ...
    def close(self) -> None: ...


@runtime_checkable
class TomatoPlanningGateway(Protocol):
    def plan_tomato_grasp(
        self, target: TomatoTarget, arm_state: ArmState, hand_state: HandState
    ) -> TomatoGraspPlan: ...


@runtime_checkable
class MotionSafetyGateway(Protocol):
    def assess(
        self, stage: str, plan: TomatoGraspPlan, arm_state: ArmState
    ) -> SafetyAssessment: ...


class GatewayTomatoPerceptionPort:
    def __init__(self, gateway: TomatoPerceptionGateway) -> None:
        if not isinstance(gateway, TomatoPerceptionGateway):
            raise TypeError("gateway must implement TomatoPerceptionGateway")
        self.gateway = gateway

    def detect(self, observation: Observation) -> TomatoDetectionResult:
        result = self.gateway.detect_tomatoes(observation)
        if not isinstance(result, TomatoDetectionResult):
            raise TypeError("perception gateway returned an invalid result")
        return result

    def close(self) -> None:
        self.gateway.close()


class GatewayTomatoGraspPlanner:
    def __init__(self, gateway: TomatoPlanningGateway) -> None:
        if not isinstance(gateway, TomatoPlanningGateway):
            raise TypeError("gateway must implement TomatoPlanningGateway")
        self.gateway = gateway

    def plan(
        self, target: TomatoTarget, arm_state: ArmState, hand_state: HandState
    ) -> TomatoGraspPlan:
        result = self.gateway.plan_tomato_grasp(target, arm_state, hand_state)
        if not isinstance(result, TomatoGraspPlan):
            raise TypeError("planning gateway returned an invalid plan")
        return result


class GatewayMotionSafetyPort:
    def __init__(self, gateway: MotionSafetyGateway) -> None:
        if not isinstance(gateway, MotionSafetyGateway):
            raise TypeError("gateway must implement MotionSafetyGateway")
        self.gateway = gateway

    def assess_pregrasp(
        self, plan: TomatoGraspPlan, arm_state: ArmState
    ) -> SafetyAssessment:
        return self._assess("pregrasp", plan, arm_state)

    def assess_skill_envelope(
        self, plan: TomatoGraspPlan, arm_state: ArmState
    ) -> SafetyAssessment:
        return self._assess("skill_envelope", plan, arm_state)

    def _assess(
        self, stage: str, plan: TomatoGraspPlan, arm_state: ArmState
    ) -> SafetyAssessment:
        result = self.gateway.assess(stage, plan, arm_state)
        if not isinstance(result, SafetyAssessment):
            raise TypeError("motion-safety gateway returned an invalid assessment")
        if result.grasp_plan_id != plan.grasp_plan_id:
            raise ValueError("motion-safety result belongs to another grasp plan")
        return result


__all__ = [
    "GatewayMotionSafetyPort",
    "GatewayTomatoGraspPlanner",
    "GatewayTomatoPerceptionPort",
    "MotionSafetyGateway",
    "TomatoPerceptionGateway",
    "TomatoPlanningGateway",
]
