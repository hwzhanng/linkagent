"""Cycle-boundary tests for returning to observation and harvesting again."""

from __future__ import annotations

from dataclasses import replace
import unittest

from module.agent import RecoverableToolError
from module.runtime import RuntimeContext
from module.tools.env import MockEnv
from proto import HandSynergy, HarvestCycleState, Pose
from task import build_task
from tests.test_harvest_cycle_task import (
    _observation,
    _run_through_retreat,
    _task_config,
    _tool_map,
)


def _finish_first_cycle(tools: dict[str, object]) -> dict[str, object]:
    _run_through_retreat(tools)
    transport = tools["transport_payload"].handler(  # type: ignore[attr-defined]
        retreat_execution_id="retreat-exec-plan-tomato-1-right-0"
    )
    place = tools["place_tomato"].handler(  # type: ignore[attr-defined]
        transport_execution_id=transport["transport_execution_id"]
    )
    return tools["verify_harvest_cycle"].handler(  # type: ignore[attr-defined]
        placement_id=place["placement_id"]
    )


def _finish_second_cycle(tools: dict[str, object]) -> dict[str, object]:
    plan_id = "plan-tomato-1-right-1"
    execution_id = f"exec-{plan_id}"
    tools["detect_tomatoes"].handler()  # type: ignore[attr-defined]
    tools["select_harvest_target"].handler()  # type: ignore[attr-defined]
    tools["reobserve_tomato"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
    tools["plan_tomato_grasp"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
    tools["check_pregrasp_clearance"].handler(grasp_plan_id=plan_id)  # type: ignore[attr-defined]
    tools["move_to_pick_ready"].handler(grasp_plan_id=plan_id)  # type: ignore[attr-defined]
    tools["execute_tomato_pick"].handler(grasp_plan_id=plan_id)  # type: ignore[attr-defined]
    tools["retreat_with_tomato"].handler(execution_id=execution_id)  # type: ignore[attr-defined]
    transport = tools["transport_payload"].handler(  # type: ignore[attr-defined]
        retreat_execution_id=f"retreat-{execution_id}"
    )
    place = tools["place_tomato"].handler(  # type: ignore[attr-defined]
        transport_execution_id=transport["transport_execution_id"]
    )
    return tools["verify_harvest_cycle"].handler(  # type: ignore[attr-defined]
        placement_id=place["placement_id"]
    )


class ResetForNextTargetTests(unittest.TestCase):
    def test_verified_cycle_resets_to_observe_and_a_second_cycle_completes(self) -> None:
        first_observation = _observation()
        task = build_task(_task_config())
        env = MockEnv(observations=[first_observation])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        tools = _tool_map(task, context)
        try:
            first = _finish_first_cycle(tools)
            self.assertTrue(first["success"])
            self.assertTrue(task.service.completed)  # type: ignore[attr-defined]
            reset = tools["reset_for_next_target"].handler()  # type: ignore[attr-defined]
            self.assertTrue(reset["success"])
            self.assertTrue(reset["output"]["next_observation_required"])
            self.assertEqual(reset["workflow_state"], HarvestCycleState.OBSERVE.value)
            self.assertFalse(task.service.completed)  # type: ignore[attr-defined]
            world = task.service.pick_service.robotics.verifier_context()["mock_world"]  # type: ignore[attr-defined]
            self.assertEqual(
                world.ee_pose,
                Pose((0.0, 0.0, 0.80), (1.0, 0.0, 0.0, 0.0), "world"),
            )
            self.assertLessEqual(world.hand_synergy.closure, 0.05)
            self.assertEqual(task.service.pick_service.targets, {})  # type: ignore[attr-defined]
            self.assertEqual(task.service.pick_service.plans, {})  # type: ignore[attr-defined]
            self.assertEqual(task.service.pick_service.executions, {})  # type: ignore[attr-defined]

            context.update_observation(replace(first_observation, sequence=1))
            second = _finish_second_cycle(tools)
            self.assertTrue(second["success"])
            self.assertTrue(task.service.completed)  # type: ignore[attr-defined]
            self.assertEqual(len(task.service.verifications), 2)  # type: ignore[attr-defined]
            self.assertEqual(  # type: ignore[attr-defined]
                set(task.service.verifications),
                {
                    "cycle-exec-plan-tomato-1-right-0",
                    "cycle-exec-plan-tomato-1-right-1",
                },
            )
        finally:
            task.close()
            env.close()

    def test_reset_before_verified_completion_is_rejected_without_motion(self) -> None:
        task = build_task(_task_config())
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        tools = _tool_map(task, context)
        try:
            world = task.service.pick_service.robotics.verifier_context()["mock_world"]  # type: ignore[attr-defined]
            before = (world.ee_pose, world.arm_sequence)
            with self.assertRaisesRegex(RecoverableToolError, "successfully verified"):
                tools["reset_for_next_target"].handler()  # type: ignore[attr-defined]
            self.assertEqual(before, (world.ee_pose, world.arm_sequence))
        finally:
            task.close()
            env.close()

    def test_nonempty_hand_blocks_reset_and_a_new_attempt_can_recover(self) -> None:
        task = build_task(_task_config())
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        tools = _tool_map(task, context)
        try:
            _finish_first_cycle(tools)
            world = task.service.pick_service.robotics.verifier_context()["mock_world"]  # type: ignore[attr-defined]
            world.set_synergy(HandSynergy(0.3, 0.0, 0.0))
            before_sequence = world.arm_sequence
            with self.assertRaisesRegex(RecoverableToolError, "reset_hand_not_empty"):
                tools["reset_for_next_target"].handler()  # type: ignore[attr-defined]
            self.assertEqual(world.arm_sequence, before_sequence)
            self.assertTrue(task.service.completed)  # type: ignore[attr-defined]
            world.set_synergy(HandSynergy(0.0, 0.0, 0.0))
            reset = tools["reset_for_next_target"].handler()  # type: ignore[attr-defined]
            self.assertTrue(reset["success"])
            self.assertEqual(reset["workflow_state"], "observe")
        finally:
            task.close()
            env.close()


if __name__ == "__main__":
    unittest.main()
