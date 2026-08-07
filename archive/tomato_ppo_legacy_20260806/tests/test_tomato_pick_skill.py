"""C1 four-phase runner success, failure, idempotency, and soak tests."""

from __future__ import annotations

from dataclasses import replace
import unittest

from module.robotics import MockPickOutcome, build_robotics_bundle
from module.skills.tomato_pick import (
    MockTomatoPickPolicy,
    TomatoPickSkillContext,
    TomatoPickSkillRunner,
)
from proto import (
    ArmSide,
    PickFailureCode,
    Pose,
    SkillStatus,
    TomatoPickSkillRequest,
    TomatoDetectionResult,
    TomatoTarget,
)
from task.agri_pick.mock_components import (
    MockMotionSafetyPort,
    MockPickVerifier,
    MockTomatoGraspPlanner,
)


def _skill_case(outcome: MockPickOutcome):
    bundle = build_robotics_bundle(
        {"type": "mock", "scenario": {"outcome": outcome.value}},
        side=ArmSide.RIGHT,
    )
    now = bundle.clock.monotonic_ns()
    target = TomatoTarget(
        target_id="tomato-1",
        center_pose=Pose((0.45, -0.25, 0.72), (1.0, 0.0, 0.0, 0.0), "world"),
        semi_axes_m=(0.035, 0.032, 0.038),
        ripeness_score=0.9,
        pick_confidence=0.95,
        occlusion=0.1,
        stem_direction_xyz=(0.0, 0.0, -1.0),
        observation_sequence=0,
        sample_monotonic_ns=now,
    )
    if outcome is MockPickOutcome.TARGET_STALE:
        bundle.clock.advance_s(6.0)
    planner = MockTomatoGraspPlanner(
        bundle.clock,
        force_stale=outcome is MockPickOutcome.PLAN_STALE,
    )
    arm_state = bundle.arm.get_state(ArmSide.RIGHT)
    hand_state = bundle.hand.get_state(ArmSide.RIGHT)
    plan = planner.plan(target, arm_state, hand_state)
    safety = MockMotionSafetyPort(
        bundle.clock,
        force_unsafe=outcome is MockPickOutcome.COLLISION_UNSAFE,
    ).assess_skill_envelope(plan, arm_state)
    policy = MockTomatoPickPolicy()
    runner = TomatoPickSkillRunner(policy)
    request = TomatoPickSkillRequest(
        execution_id="exec-1",
        target_id=target.target_id,
        grasp_plan_id=plan.grasp_plan_id,
        arm_side=ArmSide.RIGHT,
        policy_id=policy.manifest.policy_id,
        max_duration_s=4.0,
    )
    context = TomatoPickSkillContext(
        target=target,
        plan=plan,
        safety=safety,
        perception=TomatoDetectionResult((target,), 0, "test", target.sample_monotonic_ns),
        arm=bundle.arm,
        hand=bundle.hand,
        tactile=bundle.tactile,
        leases=bundle.leases,
        verifier=MockPickVerifier(bundle.world),
        clock=bundle.clock,
    )
    return bundle, runner, request, context


class TomatoPickSkillRunnerTests(unittest.TestCase):
    def test_success_requires_detached_and_retained_evidence(self) -> None:
        bundle, runner, request, context = _skill_case(MockPickOutcome.SUCCESS)
        try:
            result = runner.run(request, context)
            self.assertTrue(result.success)
            self.assertIs(result.status, SkillStatus.SUCCEEDED)
            self.assertTrue(result.verification.successful)
            self.assertTrue(bundle.world.detached)
            self.assertTrue(bundle.world.retained_stably)
            self.assertFalse(bundle.world.slip)
            self.assertFalse(bundle.world.dropped)
        finally:
            bundle.close()

    def test_failure_injections_map_to_stable_codes(self) -> None:
        expected = {
            MockPickOutcome.NO_CONTACT: PickFailureCode.NO_CONTACT,
            MockPickOutcome.UNSTABLE_GRASP: PickFailureCode.UNSTABLE_GRASP,
            MockPickOutcome.DETACH_FAILURE: PickFailureCode.DETACH_NOT_CONFIRMED,
            MockPickOutcome.SLIP: PickFailureCode.SLIP_DETECTED,
            MockPickOutcome.DROPPED: PickFailureCode.DROPPED,
            MockPickOutcome.EXCESS_FORCE: PickFailureCode.EXCESS_FORCE,
            MockPickOutcome.ARM_STATE_STALE: PickFailureCode.ARM_STATE_STALE,
            MockPickOutcome.HAND_STATE_STALE: PickFailureCode.HAND_STATE_STALE,
            MockPickOutcome.TACTILE_STALE: PickFailureCode.TACTILE_STALE,
            MockPickOutcome.COMMAND_REJECTED: PickFailureCode.COMMAND_REJECTED,
            MockPickOutcome.DEVICE_FAULT: PickFailureCode.DEVICE_FAULT,
            MockPickOutcome.TARGET_STALE: PickFailureCode.TARGET_STALE,
            MockPickOutcome.PLAN_STALE: PickFailureCode.PLAN_STALE,
            MockPickOutcome.COLLISION_UNSAFE: PickFailureCode.COLLISION_UNSAFE,
        }
        for outcome, code in expected.items():
            with self.subTest(outcome=outcome.value):
                bundle, runner, request, context = _skill_case(outcome)
                try:
                    result = runner.run(request, context)
                    self.assertFalse(result.success)
                    self.assertIs(result.failure_code, code)
                finally:
                    bundle.close()

    def test_execution_id_is_at_most_once(self) -> None:
        bundle, runner, request, context = _skill_case(MockPickOutcome.SUCCESS)
        try:
            first = runner.run(request, context)
            counts = (
                bundle.world.contact_attempts,
                bundle.world.grasp_attempts,
                bundle.world.detach_attempts,
                bundle.world.hold_checks,
            )
            second = runner.run(request, context)
            self.assertIs(first, second)
            self.assertEqual(
                counts,
                (
                    bundle.world.contact_attempts,
                    bundle.world.grasp_attempts,
                    bundle.world.detach_attempts,
                    bundle.world.hold_checks,
                ),
            )
        finally:
            bundle.close()

    def test_pre_requested_cancellation_stops_before_first_command(self) -> None:
        bundle, runner, request, context = _skill_case(MockPickOutcome.SUCCESS)
        try:
            runner.request_cancel(request.execution_id, "operator_cancel")
            result = runner.run(request, context)
            self.assertIs(result.status, SkillStatus.CANCELLED)
            self.assertIs(result.failure_code, PickFailureCode.CANCELLED)
            self.assertEqual(bundle.world.contact_attempts, 0)
        finally:
            bundle.close()

    def test_duration_budget_returns_distinct_timeout(self) -> None:
        bundle, runner, request, context = _skill_case(MockPickOutcome.SUCCESS)
        try:
            result = runner.run(replace(request, max_duration_s=0.01), context)
            self.assertIs(result.status, SkillStatus.TIMED_OUT)
            self.assertIs(result.failure_code, PickFailureCode.TIMEOUT)
        finally:
            bundle.close()

    def test_existing_control_owner_is_not_preempted(self) -> None:
        bundle, runner, request, context = _skill_case(MockPickOutcome.SUCCESS)
        lease = bundle.leases.acquire(
            owner_id="other-controller",
            side=ArmSide.RIGHT,
            ttl_s=5.0,
        )
        try:
            result = runner.run(request, context)
            self.assertFalse(result.success)
            self.assertIs(result.failure_code, PickFailureCode.CONTROL_BUSY)
            self.assertEqual(bundle.world.contact_attempts, 0)
        finally:
            bundle.leases.release(lease)
            bundle.close()

    def test_one_hundred_deterministic_scenarios(self) -> None:
        outcomes = (
            MockPickOutcome.SUCCESS,
            MockPickOutcome.NO_CONTACT,
            MockPickOutcome.UNSTABLE_GRASP,
            MockPickOutcome.DETACH_FAILURE,
            MockPickOutcome.SLIP,
            MockPickOutcome.DROPPED,
            MockPickOutcome.EXCESS_FORCE,
            MockPickOutcome.ARM_STATE_STALE,
            MockPickOutcome.HAND_STATE_STALE,
            MockPickOutcome.TACTILE_STALE,
        )
        completed = 0
        for repetition in range(10):
            for outcome in outcomes:
                with self.subTest(repetition=repetition, outcome=outcome.value):
                    bundle, runner, request, context = _skill_case(outcome)
                    try:
                        result = runner.run(request, context)
                        self.assertEqual(result.success, outcome is MockPickOutcome.SUCCESS)
                        completed += 1
                    finally:
                        bundle.close()
        self.assertEqual(completed, 100)


if __name__ == "__main__":
    unittest.main()
