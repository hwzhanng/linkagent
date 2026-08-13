"""Vision authority, RGB-D quality, identity, and PPO-ready admission tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest

from module.agent import RecoverableToolError
from module.robotics import ManualClock
from module.runtime import RuntimeContext
from module.skills.tomato_vision import (
    ReplayTomatoVisionObservationPort,
    build_replay_vision_pipeline,
)
from module.tools.env import MockEnv
from proto import Observation, TargetReobserveRequest
from task import build_task
from tests.test_harvest_cycle_task import _observation, _task_config, _tool_map


def _mutated_observation(mutator: object) -> Observation:
    original = _observation()
    metadata = deepcopy(dict(original.metadata))
    mutator(metadata["tomato_vision"])  # type: ignore[operator]
    return Observation(sequence=original.sequence, metadata=metadata)


class TomatoVisionPipelineTests(unittest.TestCase):
    def test_selected_yolo_mask_depth_produces_metric_ppo_target(self) -> None:
        pipeline = build_replay_vision_pipeline(clock=ManualClock())
        try:
            result = pipeline.observe(_observation())
            target = result.geometry.target
            self.assertEqual(target.target_id, "tomato-1")
            self.assertEqual(result.selection.selected_instance_id, "mask-1")
            self.assertAlmostEqual(target.center_pose.position_m[0], 0.45, places=5)
            self.assertAlmostEqual(target.center_pose.position_m[1], -0.25, places=5)
            self.assertAlmostEqual(target.center_pose.position_m[2], 0.72, places=5)
            self.assertTrue(all(0.025 <= axis <= 0.035 for axis in target.semi_axes_m))
            self.assertAlmostEqual(target.stem_direction_xyz[1], -1.0, places=6)
            self.assertEqual(result.geometry.point_count, 6)
            self.assertEqual(result.geometry.inlier_count, 6)
            self.assertLess(result.geometry.fit_residual_m, 0.001)
        finally:
            pipeline.close()

    def test_unaligned_or_unsynchronized_depth_fails_closed(self) -> None:
        cases = (
            lambda vision: vision["camera"].update({"depth_aligned_to_rgb": False}),
            lambda vision: vision["camera"].update(
                {
                    "rgb_sample_monotonic_ns": 1_000_000_000,
                    "depth_sample_monotonic_ns": 1_100_000_000,
                }
            ),
        )
        for mutate in cases:
            with self.subTest(mutate=mutate):
                pipeline = build_replay_vision_pipeline(clock=ManualClock())
                try:
                    with self.assertRaisesRegex(ValueError, "aligned|skew"):
                        pipeline.observe(_mutated_observation(mutate))
                finally:
                    pipeline.close()

    def test_deployment_calibration_and_device_identity_are_bound(self) -> None:
        cases = (
            {"expected_calibration_id": "wrong-calibration"},
            {"expected_device_id": "wrong-device"},
        )
        for options in cases:
            with self.subTest(options=options):
                pipeline = build_replay_vision_pipeline(
                    clock=ManualClock(), **options
                )
                try:
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        pipeline.observe(_observation())
                finally:
                    pipeline.close()

    def test_vlm_point_outside_yolo_mask_is_rejected(self) -> None:
        observation = _mutated_observation(
            lambda vision: vision["vlm_selection"].update({"point_xy": [10, 10]})
        )
        pipeline = build_replay_vision_pipeline(clock=ManualClock())
        try:
            with self.assertRaisesRegex(ValueError, "exactly one.*mask"):
                pipeline.observe(observation)
        finally:
            pipeline.close()

    def test_pinned_reobservation_rejects_vlm_identity_switch(self) -> None:
        def mutate(vision: dict[str, object]) -> None:
            second = deepcopy(vision["instances"][0])  # type: ignore[index]
            second["instance_id"] = "mask-2"
            second["track_id"] = "tomato-2"
            vision["instances"].append(second)  # type: ignore[union-attr]
            vision["vlm_selection"].update(  # type: ignore[union-attr]
                {"selected_instance_id": "mask-2"}
            )

        port = ReplayTomatoVisionObservationPort(clock=ManualClock())
        try:
            with self.assertRaisesRegex(ValueError, "changed target identity"):
                port.reobserve(
                    TargetReobserveRequest(
                        execution_id="identity-switch",
                        target_id="tomato-1",
                        observation=_mutated_observation(mutate),
                    )
                )
        finally:
            port.close()

    def test_reobservation_requires_a_strictly_new_camera_sample_by_default(self) -> None:
        config = _task_config()
        config["atomic_skills"]["target_reobserve"] = {  # type: ignore[index]
            "type": "target_reobserve",
            "require_depth": True,
            "require_new_sample": True,
        }
        task = build_task(config)
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        tools = _tool_map(task, context)
        try:
            tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            tools["detect_tomatoes"].handler()  # type: ignore[attr-defined]
            tools["select_harvest_target"].handler()  # type: ignore[attr-defined]
            with self.assertRaisesRegex(RecoverableToolError, "fresh camera sample"):
                tools["reobserve_tomato"].handler(  # type: ignore[attr-defined]
                    target_id="tomato-1"
                )
        finally:
            task.close()
            env.close()

    def test_depth_samples_outside_selected_mask_are_rejected(self) -> None:
        def mutate(vision: dict[str, object]) -> None:
            vision["instances"][0].update(  # type: ignore[index]
                {"mask_pixels_xy": [[1, 1], [2, 2]]}
            )
            vision["vlm_selection"].update({"point_xy": [1, 1]})  # type: ignore[union-attr]

        observation = _mutated_observation(mutate)
        pipeline = build_replay_vision_pipeline(clock=ManualClock())
        try:
            with self.assertRaisesRegex(ValueError, "insufficient valid aligned depth"):
                pipeline.observe(observation)
        finally:
            pipeline.close()

    def test_dense_mask_depth_coverage_cannot_be_hidden_by_valid_sampling(self) -> None:
        def mutate(vision: dict[str, object]) -> None:
            vision["instances"][0].update(  # type: ignore[index]
                {"mask_area_px": 100, "valid_depth_pixel_count": 6}
            )

        pipeline = build_replay_vision_pipeline(clock=ManualClock())
        try:
            with self.assertRaisesRegex(ValueError, "insufficient valid aligned depth"):
                pipeline.observe(_mutated_observation(mutate))
        finally:
            pipeline.close()

    def test_geometry_axis_and_fit_quality_limits_fail_closed(self) -> None:
        cases = (
            ({"max_axis_m": 0.029}, "axes are outside"),
            ({"max_fit_residual_m": 1.0e-10}, "fit residual"),
        )
        for options, expected in cases:
            with self.subTest(options=options):
                pipeline = build_replay_vision_pipeline(
                    clock=ManualClock(), **options
                )
                try:
                    with self.assertRaisesRegex(ValueError, expected):
                        pipeline.observe(_observation())
                finally:
                    pipeline.close()

    def test_shadow_ready_skill_submits_no_motion_and_cannot_start_ppo(self) -> None:
        config = _task_config()
        config["atomic_skills"]["move_to_pick_ready"] = {  # type: ignore[index]
            "type": "move_to_pick_ready",
            "execution_mode": "shadow",
        }
        task = build_task(config)
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        tools = _tool_map(task, context)
        try:
            world = task.service.pick_service.robotics.verifier_context()["mock_world"]  # type: ignore[attr-defined]
            tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            tools["detect_tomatoes"].handler()  # type: ignore[attr-defined]
            tools["select_harvest_target"].handler()  # type: ignore[attr-defined]
            tools["reobserve_tomato"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
            tools["plan_tomato_grasp"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
            tools["check_pregrasp_clearance"].handler(  # type: ignore[attr-defined]
                grasp_plan_id="plan-tomato-1-right-0"
            )
            before = (world.ee_pose, world.arm_sequence)
            ready = tools["move_to_pick_ready"].handler(  # type: ignore[attr-defined]
                grasp_plan_id="plan-tomato-1-right-0"
            )
            self.assertTrue(ready["success"])
            self.assertFalse(ready["ppo_execution_allowed"])
            self.assertTrue(ready["output"]["shadow_only"])
            self.assertEqual(before, (world.ee_pose, world.arm_sequence))
            with self.assertRaisesRegex(RecoverableToolError, "move_to_pick_ready"):
                tools["execute_tomato_pick"].handler(  # type: ignore[attr-defined]
                    grasp_plan_id="plan-tomato-1-right-0"
                )
        finally:
            task.close()
            env.close()

    def test_effectful_ready_skill_returns_the_exact_nine_visual_inputs(self) -> None:
        config = _task_config()
        config["robotics"]["scenario"]["stable_hold_ticks"] = 8  # type: ignore[index]
        task = build_task(config)
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        tools = _tool_map(task, context)
        try:
            tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            tools["detect_tomatoes"].handler()  # type: ignore[attr-defined]
            tools["select_harvest_target"].handler()  # type: ignore[attr-defined]
            tools["reobserve_tomato"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
            tools["plan_tomato_grasp"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
            tools["check_pregrasp_clearance"].handler(  # type: ignore[attr-defined]
                grasp_plan_id="plan-tomato-1-right-0"
            )
            ready = tools["move_to_pick_ready"].handler(  # type: ignore[attr-defined]
                grasp_plan_id="plan-tomato-1-right-0"
            )
            evidence = ready["output"]["ppo_ready_evidence"]
            self.assertTrue(evidence["admitted"])
            self.assertEqual(len(evidence["fruit_center_in_palm_m"]), 3)
            self.assertEqual(len(evidence["fruit_semi_axes_m"]), 3)
            self.assertEqual(len(evidence["pull_axis_in_palm"]), 3)
            self.assertTrue(ready["ppo_execution_allowed"])
            execution = tools["execute_tomato_pick"].handler(  # type: ignore[attr-defined]
                grasp_plan_id="plan-tomato-1-right-0"
            )
            self.assertTrue(execution["success"])
            self.assertGreater(execution["duration_s"], 0.5)
        finally:
            task.close()
            env.close()

    def test_ready_skill_rejects_stale_or_out_of_distribution_geometry(self) -> None:
        for failure in ("stale", "axis"):
            with self.subTest(failure=failure):
                task = build_task(_task_config())
                env = MockEnv(observations=[_observation()])
                context = RuntimeContext(env)
                context.update_observation(env.reset())
                tools = _tool_map(task, context)
                try:
                    tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
                    tools["detect_tomatoes"].handler()  # type: ignore[attr-defined]
                    tools["select_harvest_target"].handler()  # type: ignore[attr-defined]
                    tools["reobserve_tomato"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
                    tools["plan_tomato_grasp"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
                    tools["check_pregrasp_clearance"].handler(  # type: ignore[attr-defined]
                        grasp_plan_id="plan-tomato-1-right-0"
                    )
                    if failure == "stale":
                        task.service.pick_service.robotics.clock.sleep(0.6)  # type: ignore[attr-defined]
                        expected = "target_stale"
                    else:
                        target = task.service.pick_service.targets["tomato-1"]  # type: ignore[attr-defined]
                        task.service.pick_service.targets["tomato-1"] = replace(  # type: ignore[attr-defined]
                            target, semi_axes_m=(0.040, 0.040, 0.040)
                        )
                        expected = "fruit_axis_training_range"
                    with self.assertRaisesRegex(RecoverableToolError, expected):
                        tools["move_to_pick_ready"].handler(  # type: ignore[attr-defined]
                            grasp_plan_id="plan-tomato-1-right-0"
                        )
                finally:
                    task.close()
                    env.close()


if __name__ == "__main__":
    unittest.main()
