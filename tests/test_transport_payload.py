"""Closed-loop tests for transport from post-pick retreat to container pre-place."""

from __future__ import annotations

import unittest

from module.agent import RecoverableToolError
from module.runtime import RuntimeContext
from module.tools.env import MockEnv
from proto import HarvestCycleState, Pose
from task import build_task
from tests.test_harvest_cycle_task import (
    _observation,
    _run_through_retreat,
    _task_config,
    _tool_map,
)


RETREAT_ID = "retreat-exec-plan-tomato-1-right-0"
TRANSPORT_ID = "transport-payload-exec-plan-tomato-1-right-0"


def _retreated_task(*, transport_config: dict[str, object] | None = None):
    task = build_task(_task_config(transport_payload=transport_config))
    env = MockEnv(observations=[_observation()])
    context = RuntimeContext(env)
    context.update_observation(env.reset())
    tools = _tool_map(task, context)
    _run_through_retreat(tools)
    return task, env, tools


class TransportPayloadTests(unittest.TestCase):
    def test_two_segment_transport_samples_retention_and_reaches_preplace(self) -> None:
        task, env, tools = _retreated_task()
        try:
            self.assertIs(task.service.state, HarvestCycleState.TRANSPORT)  # type: ignore[attr-defined]
            world = task.service.pick_service.robotics.verifier_context()["mock_world"]  # type: ignore[attr-defined]
            before_sequence = world.arm_sequence
            result = tools["transport_payload"].handler(  # type: ignore[attr-defined]
                retreat_execution_id=RETREAT_ID
            )
            evidence = result["output"]["transport_evidence"]
            self.assertTrue(result["output"]["place_authorized"])
            self.assertEqual(result["transport_execution_id"], TRANSPORT_ID)
            self.assertEqual(evidence["retention_sample_count"], 3)
            self.assertEqual(len(evidence["command_ids"]), 2)
            self.assertGreaterEqual(evidence["minimum_contact_fingers"], 3)
            self.assertEqual(
                world.ee_pose,
                Pose((0.35, 0.40, 0.77), (1.0, 0.0, 0.0, 0.0), "world"),
            )
            self.assertEqual(world.arm_sequence, before_sequence + 2)
            self.assertGreaterEqual(world.hand_synergy.closure, 0.15)
            repeated = tools["transport_payload"].handler(  # type: ignore[attr-defined]
                retreat_execution_id=RETREAT_ID
            )
            self.assertEqual(repeated["execution_id"], result["execution_id"])
            self.assertEqual(world.arm_sequence, before_sequence + 2)
        finally:
            task.close()
            env.close()

    def test_place_is_locked_until_transport_succeeds(self) -> None:
        task, env, tools = _retreated_task()
        try:
            world = task.service.pick_service.robotics.verifier_context()["mock_world"]  # type: ignore[attr-defined]
            before = (world.ee_pose, world.arm_sequence, world.hand_synergy)
            with self.assertRaisesRegex(RecoverableToolError, "transport first"):
                tools["place_tomato"].handler(  # type: ignore[attr-defined]
                    transport_execution_id=TRANSPORT_ID
                )
            self.assertEqual(before, (world.ee_pose, world.arm_sequence, world.hand_synergy))
        finally:
            task.close()
            env.close()

    def test_drop_after_lift_routes_localization_and_never_authorizes_place(self) -> None:
        task, env, tools = _retreated_task(
            transport_config={
                "type": "transport_payload",
                "force_drop_after_segment": 1,
            }
        )
        try:
            world = task.service.pick_service.robotics.verifier_context()["mock_world"]  # type: ignore[attr-defined]
            before_sequence = world.arm_sequence
            with self.assertRaisesRegex(RecoverableToolError, "payload_dropped"):
                tools["transport_payload"].handler(retreat_execution_id=RETREAT_ID)  # type: ignore[attr-defined]
            self.assertIs(
                task.service.state,  # type: ignore[attr-defined]
                HarvestCycleState.DROPPED_PAYLOAD_RECOVERY,
            )
            self.assertTrue(world.dropped)
            self.assertEqual(world.arm_sequence, before_sequence + 1)
            result = task.service.atomic_results[TRANSPORT_ID]  # type: ignore[attr-defined]
            self.assertFalse(result.output["place_authorized"])
            self.assertEqual(result.output["failed_segment"], "lift")
            localization = tools["localize_dropped_payload"].handler()  # type: ignore[attr-defined]
            self.assertEqual(
                localization["output"]["disposition"],
                "dedicated_recovery_required",
            )
            self.assertFalse(localization["output"]["motion_authorized"])
            self.assertEqual(world.arm_sequence, before_sequence + 1)
            with self.assertRaisesRegex(RecoverableToolError, "transport first"):
                tools["place_tomato"].handler(transport_execution_id=TRANSPORT_ID)  # type: ignore[attr-defined]
        finally:
            task.close()
            env.close()

    def test_unstable_grip_at_preplace_blocks_release(self) -> None:
        task, env, tools = _retreated_task(
            transport_config={
                "type": "transport_payload",
                "force_payload_unstable_after_segment": 2,
            }
        )
        try:
            world = task.service.pick_service.robotics.verifier_context()["mock_world"]  # type: ignore[attr-defined]
            with self.assertRaisesRegex(RecoverableToolError, "payload_unstable"):
                tools["transport_payload"].handler(retreat_execution_id=RETREAT_ID)  # type: ignore[attr-defined]
            self.assertGreaterEqual(world.hand_synergy.closure, 0.15)
            self.assertFalse(world.dropped)
            result = task.service.atomic_results[TRANSPORT_ID]  # type: ignore[attr-defined]
            self.assertEqual(result.output["failed_segment"], "preplace")
            self.assertFalse(result.output["place_authorized"])
        finally:
            task.close()
            env.close()

    def test_place_rejects_motion_if_arm_left_the_bound_preplace_pose(self) -> None:
        task, env, tools = _retreated_task()
        try:
            tools["transport_payload"].handler(retreat_execution_id=RETREAT_ID)  # type: ignore[attr-defined]
            world = task.service.pick_service.robotics.verifier_context()["mock_world"]  # type: ignore[attr-defined]
            world.set_pose(Pose((0.0, 0.0, 0.8), (1.0, 0.0, 0.0, 0.0), "world"))
            before = (world.arm_sequence, world.hand_synergy)
            with self.assertRaisesRegex(RecoverableToolError, "pre-place pose"):
                tools["place_tomato"].handler(transport_execution_id=TRANSPORT_ID)  # type: ignore[attr-defined]
            self.assertEqual(before, (world.arm_sequence, world.hand_synergy))
        finally:
            task.close()
            env.close()


if __name__ == "__main__":
    unittest.main()
