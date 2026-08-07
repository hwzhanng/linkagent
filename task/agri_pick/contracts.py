"""Task-local ports for tomato perception, grasp planning, and safety."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from proto import (
    ArmSide,
    ArmState,
    HandState,
    Observation,
    SafetyAssessment,
    TomatoDetectionResult,
    TomatoGraspPlan,
    TomatoTarget,
)


@runtime_checkable
class TomatoPerceptionPort(Protocol):
    def detect(self, observation: Observation) -> TomatoDetectionResult: ...

    def close(self) -> None: ...


@runtime_checkable
class TomatoGraspPlanner(Protocol):
    def plan(
        self,
        target: TomatoTarget,
        arm_state: ArmState,
        hand_state: HandState,
    ) -> TomatoGraspPlan: ...


@runtime_checkable
class MotionSafetyPort(Protocol):
    def assess_pregrasp(
        self,
        plan: TomatoGraspPlan,
        arm_state: ArmState,
    ) -> SafetyAssessment: ...

    def assess_skill_envelope(
        self,
        plan: TomatoGraspPlan,
        arm_state: ArmState,
    ) -> SafetyAssessment: ...


__all__ = ["MotionSafetyPort", "TomatoGraspPlanner", "TomatoPerceptionPort"]
