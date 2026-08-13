"""Closed-loop tests for the read-only pregrasp point-cloud clearance atom."""

from __future__ import annotations

from copy import deepcopy
import unittest

from module.agent import RecoverableToolError
from module.runtime import RuntimeContext
from module.tools.env import MockEnv
from proto import HarvestCycleState, Observation
from task import build_task
from tests.test_harvest_cycle_task import _observation, _task_config, _tool_map


PLAN_ID = "plan-tomato-1-right-0"


def _planned_task(*, clearance_scene: dict[str, object] | None = None, observation: Observation | None = None):
    task = build_task(_task_config(clearance_scene=clearance_scene))
    sample = _observation() if observation is None else observation
    env = MockEnv(observations=[sample])
    context = RuntimeContext(env)
    context.update_observation(env.reset())
    tools = _tool_map(task, context)
    tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
    tools["detect_tomatoes"].handler()  # type: ignore[attr-defined]
    tools["select_harvest_target"].handler()  # type: ignore[attr-defined]
    tools["reobserve_tomato"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
    plan = tools["plan_tomato_grasp"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
    return task, env, tools, plan


class PregraspClearanceTests(unittest.TestCase):
    def test_clear_scene_is_read_only_and_releases_the_exact_plan(self) -> None:
        task, env, tools, plan = _planned_task()
        try:
            self.assertEqual(plan["workflow_state"], HarvestCycleState.CLEARANCE.value)
            world = task.service.pick_service.robotics.verifier_context()["mock_world"]  # type: ignore[attr-defined]
            before = (world.ee_pose, world.arm_sequence)
            result = tools["check_pregrasp_clearance"].handler(  # type: ignore[attr-defined]
                grasp_plan_id=PLAN_ID
            )
            self.assertTrue(result["success"])
            self.assertTrue(result["output"]["motion_authorized"])
            evidence = result["output"]["clearance_evidence"]
            self.assertTrue(evidence["collision_free"])
            self.assertEqual(evidence["checked_segments"], ("transit", "approach", "finger_corridor"))
            self.assertEqual(before, (world.ee_pose, world.arm_sequence))
            ready = tools["move_to_pick_ready"].handler(grasp_plan_id=PLAN_ID)  # type: ignore[attr-defined]
            self.assertTrue(ready["ppo_execution_allowed"])
            self.assertGreater(world.arm_sequence, before[1])
        finally:
            task.close()
            env.close()

    def test_blocked_transit_never_sends_motion_and_ready_stays_locked(self) -> None:
        scene = {
            "type": "mock",
            "obstacle_points_world_m": [
                [0.188, -0.098, 0.607],
                [0.90, 0.80, 1.30],
                [-0.30, -0.70, 0.25],
            ],
        }
        task, env, tools, _ = _planned_task(clearance_scene=scene)
        try:
            world = task.service.pick_service.robotics.verifier_context()["mock_world"]  # type: ignore[attr-defined]
            before = (world.ee_pose, world.arm_sequence)
            with self.assertRaisesRegex(RecoverableToolError, "clearance_blocked"):
                tools["check_pregrasp_clearance"].handler(grasp_plan_id=PLAN_ID)  # type: ignore[attr-defined]
            self.assertEqual(before, (world.ee_pose, world.arm_sequence))
            self.assertIs(task.service.state, HarvestCycleState.CLEARANCE)  # type: ignore[attr-defined]
            with self.assertRaisesRegex(RecoverableToolError, "exact grasp plan"):
                tools["move_to_pick_ready"].handler(grasp_plan_id=PLAN_ID)  # type: ignore[attr-defined]
            self.assertEqual(before, (world.ee_pose, world.arm_sequence))
            result = task.service.atomic_results[f"clearance-{PLAN_ID}-0"]  # type: ignore[attr-defined]
            evidence = result.output["clearance_evidence"]
            self.assertEqual(evidence["blocked_segment"], "transit")
        finally:
            task.close()
            env.close()

    def test_target_surface_is_excluded_but_a_nearby_branch_is_not(self) -> None:
        cases = (
            (
                "fruit_only",
                [[0.45, -0.25, 0.72], [0.90, 0.80, 1.30], [-0.30, -0.70, 0.25]],
                True,
            ),
            (
                "near_branch",
                [[0.45, -0.25, 0.72], [0.40, -0.205, 0.55], [0.90, 0.80, 1.30]],
                False,
            ),
        )
        for name, points, expected_clear in cases:
            with self.subTest(name=name):
                task, env, tools, _ = _planned_task(
                    clearance_scene={"type": "mock", "obstacle_points_world_m": points}
                )
                try:
                    if expected_clear:
                        result = tools["check_pregrasp_clearance"].handler(grasp_plan_id=PLAN_ID)  # type: ignore[attr-defined]
                        self.assertEqual(
                            result["output"]["clearance_evidence"]["ignored_target_point_count"],
                            1,
                        )
                    else:
                        with self.assertRaisesRegex(RecoverableToolError, "clearance_blocked"):
                            tools["check_pregrasp_clearance"].handler(grasp_plan_id=PLAN_ID)  # type: ignore[attr-defined]
                finally:
                    task.close()
                    env.close()

    def test_missing_stale_and_low_coverage_scenes_have_stable_failures(self) -> None:
        cases = (
            ({"type": "mock", "force_missing": True}, "clearance_context_missing"),
            ({"type": "mock", "sample_age_s": 0.6}, "clearance_scene_stale"),
            ({"type": "mock", "coverage_ratio": 0.2}, "clearance_context_missing"),
        )
        for scene, expected in cases:
            with self.subTest(scene=scene):
                task, env, tools, _ = _planned_task(clearance_scene=scene)
                try:
                    with self.assertRaisesRegex(RecoverableToolError, expected):
                        tools["check_pregrasp_clearance"].handler(grasp_plan_id=PLAN_ID)  # type: ignore[attr-defined]
                finally:
                    task.close()
                    env.close()

    def test_observation_pointcloud_port_uses_runtime_observation_not_agent_arguments(self) -> None:
        original = _observation()
        metadata = deepcopy(dict(original.metadata))
        metadata["pregrasp_clearance"] = {
            "scene_id": "depth-scene-0",
            "sample_monotonic_ns": 1_000_000_000,
            "points_world_m": [
                [0.90, 0.80, 1.30],
                [0.85, 0.75, 1.25],
                [-0.30, -0.70, 0.25],
            ],
            "coverage_ratio": 0.95,
            "calibration_id": "future-depth-camera-calibration",
            "frame_id": "world",
            "simulated": True,
            "evidence_sources": ["observation.aligned_depth_pointcloud"],
        }
        observation = Observation(sequence=0, metadata=metadata)
        task, env, tools, _ = _planned_task(
            clearance_scene={"type": "observation_pointcloud"},
            observation=observation,
        )
        try:
            schema = tools["check_pregrasp_clearance"].as_dict()["parameters"]  # type: ignore[attr-defined]
            self.assertEqual(set(schema["properties"]), {"grasp_plan_id"})
            result = tools["check_pregrasp_clearance"].handler(grasp_plan_id=PLAN_ID)  # type: ignore[attr-defined]
            evidence = result["output"]["clearance_evidence"]
            self.assertEqual(evidence["scene_id"], "depth-scene-0")
            self.assertIn("observation.aligned_depth_pointcloud", evidence["evidence_sources"])
        finally:
            task.close()
            env.close()


if __name__ == "__main__":
    unittest.main()
