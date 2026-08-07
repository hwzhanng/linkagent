"""Deterministic C1 task components backed only by configured mock data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from module.robotics import ManualClock, MockPickOutcome, MockPickWorld
from module.skills.tomato_pick import VerifierSession
from proto import (
    ArmSide,
    ArmState,
    CollisionStatus,
    EvidenceState,
    HandState,
    Observation,
    Pose,
    SafetyAssessment,
    TactileFrame,
    TomatoDetectionResult,
    TomatoGraspPlan,
    TomatoPickVerification,
    TomatoTarget,
)


def _pose(value: object, *, field_name: str) -> Pose:
    if isinstance(value, Pose):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a Pose or mapping")
    try:
        return Pose(**dict(value))
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"invalid {field_name}: {exc}") from exc


class MockTomatoPerceptionPort:
    """Read small tomato summaries from ``Observation.metadata.tomatoes``."""

    def __init__(
        self,
        clock: ManualClock,
        *,
        model: str = "mock_tomato_detector_v1",
        force_stale: bool = False,
    ) -> None:
        if not isinstance(clock, ManualClock):
            raise TypeError("clock must be a ManualClock")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(force_stale, bool):
            raise TypeError("force_stale must be a bool")
        self._clock = clock
        self._model = model.strip()
        self._force_stale = force_stale
        self._closed = False

    def detect(self, observation: Observation) -> TomatoDetectionResult:
        if self._closed:
            raise RuntimeError("perception port is closed")
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        raw_targets = observation.metadata.get("tomatoes", ())
        if isinstance(raw_targets, (str, bytes, bytearray, Mapping)):
            raise TypeError("observation.metadata.tomatoes must be a list")
        try:
            values = tuple(raw_targets)
        except TypeError as exc:
            raise TypeError("observation.metadata.tomatoes must be a list") from exc
        if self._force_stale:
            self._clock.advance_s(6.0)
        now = self._clock.monotonic_ns()
        targets: list[TomatoTarget] = []
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise TypeError(f"tomatoes[{index}] must be a mapping")
            options = dict(value)
            options["center_pose"] = _pose(
                options.get("center_pose"), field_name=f"tomatoes[{index}].center_pose"
            )
            options.setdefault("observation_sequence", observation.sequence)
            sample_ns = now - 6_000_000_000 if self._force_stale else now
            options.setdefault("sample_monotonic_ns", sample_ns)
            targets.append(TomatoTarget(**options))
        return TomatoDetectionResult(
            targets=tuple(targets),
            observation_sequence=observation.sequence,
            model=self._model,
            sample_monotonic_ns=now,
        )

    def close(self) -> None:
        self._closed = True


class MockTomatoGraspPlanner:
    """Create a deterministic grasp frame and pregrasp for one tomato."""

    def __init__(
        self,
        clock: ManualClock,
        *,
        mapping_calibration_id: str = "g20_eigengrasp_5d_v3_thumb_roll_fixed",
        plan_ttl_s: float = 5.0,
        force_stale: bool = False,
    ) -> None:
        if not isinstance(clock, ManualClock):
            raise TypeError("clock must be a ManualClock")
        if not isinstance(mapping_calibration_id, str) or not mapping_calibration_id:
            raise ValueError("mapping_calibration_id must be a non-empty string")
        if isinstance(plan_ttl_s, bool) or not isinstance(plan_ttl_s, (int, float)) or plan_ttl_s <= 0:
            raise ValueError("plan_ttl_s must be > 0")
        if not isinstance(force_stale, bool):
            raise TypeError("force_stale must be a bool")
        self._clock = clock
        self._mapping_id = mapping_calibration_id
        self._ttl_s = float(plan_ttl_s)
        self._force_stale = force_stale

    def plan(
        self,
        target: TomatoTarget,
        arm_state: ArmState,
        hand_state: HandState,
    ) -> TomatoGraspPlan:
        if not isinstance(target, TomatoTarget):
            raise TypeError("target must be a TomatoTarget")
        if not isinstance(arm_state, ArmState) or not isinstance(hand_state, HandState):
            raise TypeError("arm_state and hand_state must use frozen DTOs")
        if arm_state.side is not hand_state.side:
            raise ValueError("arm_state and hand_state must use the same side")
        arm_side = arm_state.side
        x, y, z = target.center_pose.position_m
        q = target.center_pose.quaternion_wxyz
        frame = target.center_pose.frame_id
        # C1 uses a fixed mock convention. Real perception will later supply a
        # calibrated grasp frame while preserving this exact DTO.
        grasp = Pose((x, y, z), q, frame)
        # The high-level move hands off inside the policy's residual workspace;
        # the CONTACT phase then owns the final 5 mm approach.
        pregrasp = Pose((x, y, z + 0.015), q, frame)
        contact = Pose((x, y, z + 0.01), q, frame)
        retreat = Pose((x, y, z + 0.12), q, frame)
        now = self._clock.monotonic_ns()
        expires = max(1, now - 1) if self._force_stale else max(1, now + int(self._ttl_s * 1e9))
        return TomatoGraspPlan(
            grasp_plan_id=(
                f"plan-{target.target_id}-{arm_side.value}-{target.observation_sequence}"
            ),
            target_id=target.target_id,
            source_observation_sequence=target.observation_sequence,
            arm_side=arm_side,
            grasp_frame=grasp,
            pregrasp_pose=pregrasp,
            contact_pose=contact,
            retreat_pose=retreat,
            max_residual_translation_m=(0.02, 0.02, 0.04),
            max_residual_rotation_rad=(0.60, 0.60, 1.20),
            mapping_calibration_id=self._mapping_id,
            planner_version="mock_tomato_grasp_planner_v1",
            expires_monotonic_ns=expires,
        )


class MockMotionSafetyPort:
    """C1 reachability/workspace gate; explicitly simulated, not collision IK."""

    def __init__(
        self,
        clock: ManualClock,
        *,
        force_unsafe: bool = False,
        workspace_xyzxyz_m: tuple[float, ...] = (-1.5, -1.5, 0.0, 1.5, 1.5, 2.0),
    ) -> None:
        if not isinstance(clock, ManualClock):
            raise TypeError("clock must be a ManualClock")
        if not isinstance(force_unsafe, bool):
            raise TypeError("force_unsafe must be a bool")
        if len(tuple(workspace_xyzxyz_m)) != 6:
            raise ValueError("workspace_xyzxyz_m must have six values")
        self._clock = clock
        self._force_unsafe = force_unsafe
        self._workspace = tuple(float(value) for value in workspace_xyzxyz_m)

    def assess_pregrasp(
        self,
        plan: TomatoGraspPlan,
        arm_state: ArmState,
    ) -> SafetyAssessment:
        return self._assess(plan, arm_state, stage="pregrasp")

    def assess_skill_envelope(
        self,
        plan: TomatoGraspPlan,
        arm_state: ArmState,
    ) -> SafetyAssessment:
        return self._assess(plan, arm_state, stage="skill")

    def _assess(
        self,
        plan: TomatoGraspPlan,
        arm_state: ArmState,
        *,
        stage: str,
    ) -> SafetyAssessment:
        if not isinstance(plan, TomatoGraspPlan) or not isinstance(arm_state, ArmState):
            raise TypeError("plan and arm_state must use frozen DTOs")
        now = self._clock.monotonic_ns()
        inside = all(
            self._workspace[index] <= plan.pregrasp_pose.position_m[index] <= self._workspace[index + 3]
            for index in range(3)
        )
        approved = inside and not self._force_unsafe
        return SafetyAssessment(
            assessment_id=f"safety-{stage}-{plan.grasp_plan_id}-{arm_state.stamp.source_sequence}",
            grasp_plan_id=plan.grasp_plan_id,
            observation_sequence=plan.source_observation_sequence,
            arm_state_sequence=arm_state.stamp.source_sequence,
            reachable=approved,
            joint_limits_ok=approved,
            path_sampled_ok=approved,
            collision_status=(
                CollisionStatus.SIMULATED_SAFE if approved else CollisionStatus.UNSAFE
            ),
            allowed_workspace_xyzxyz_m=self._workspace,
            expires_monotonic_ns=max(1, now + 2_000_000_000),
            reason=None if approved else "mock plan is outside the allowed workspace",
        )


@dataclass(slots=True)
class _MockVerifierState:
    last: TomatoPickVerification


class MockPickVerifier:
    """Independent mock evidence reader; it never consumes policy outputs."""

    def __init__(self, world: MockPickWorld) -> None:
        if not isinstance(world, MockPickWorld):
            raise TypeError("world must be a MockPickWorld")
        self._world = world
        self._sessions: dict[str, _MockVerifierState] = {}

    def begin(
        self,
        execution_id: str,
        target: TomatoTarget,
        plan: TomatoGraspPlan,
    ) -> VerifierSession:
        del target, plan
        session = VerifierSession(execution_id)
        verification = self._verification(execution_id)
        self._sessions[execution_id] = _MockVerifierState(verification)
        return session

    def update(
        self,
        session: VerifierSession,
        perception: TomatoDetectionResult | None,
        arm: ArmState,
        hand: HandState,
        tactile: TactileFrame,
    ) -> TomatoPickVerification:
        del perception, arm, hand, tactile
        self._require_session(session)
        self._world.update_verifier()
        verification = self._verification(session.execution_id)
        self._sessions[session.execution_id].last = verification
        return verification

    def finish(self, session: VerifierSession) -> TomatoPickVerification:
        state = self._require_session(session)
        return state.last

    def _require_session(self, session: VerifierSession) -> _MockVerifierState:
        if not isinstance(session, VerifierSession):
            raise TypeError("session must be a VerifierSession")
        try:
            return self._sessions[session.execution_id]
        except KeyError as exc:
            raise KeyError(f"unknown verifier session {session.execution_id!r}") from exc

    def _verification(self, execution_id: str) -> TomatoPickVerification:
        detached = (
            EvidenceState.CONFIRMED if self._world.detached else EvidenceState.UNKNOWN
        )
        if self._world.dropped or self._world.slip:
            retained = EvidenceState.REJECTED
        elif self._world.retained_stably:
            retained = EvidenceState.CONFIRMED
        else:
            retained = EvidenceState.UNKNOWN
        return TomatoPickVerification(
            execution_id=execution_id,
            detached=detached,
            retained_without_slip=retained,
            dropped=self._world.dropped,
            stable_window_s=self._world.hold_checks / 20.0,
            evidence_sources=("mock_world_detachment", "mock_g20_tactile"),
            verifier_version="mock_pick_verifier_v1",
            valid=True,
            reason=None,
        )


__all__ = [
    "MockMotionSafetyPort",
    "MockPickVerifier",
    "MockTomatoGraspPlanner",
    "MockTomatoPerceptionPort",
]
