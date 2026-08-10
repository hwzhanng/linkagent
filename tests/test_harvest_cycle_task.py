"""Offline end-to-end and recovery tests for the harvest-cycle Task."""

from __future__ import annotations

from pathlib import Path
import unittest

from module.agent import RecoverableToolError
from module.runtime import CompletionReason, RuntimeContext
from module.tools.env import MockEnv
from proto import HarvestCycleState, Observation, Pose
from script.run import build_runtime_from_file
from task import build_task


ROOT = Path(__file__).resolve().parents[1]


def _observation() -> Observation:
    return Observation(
        sequence=0,
        metadata={
            "tomato_vision": {
                "camera": {
                    "device_id": "orbbec-replay-001",
                    "depth_aligned_to_rgb": True,
                    "evidence_sources": ["replay:rgb", "replay:aligned-depth"],
                    "intrinsics": {
                        "width_px": 640,
                        "height_px": 480,
                        "fx_px": 600.0,
                        "fy_px": 600.0,
                        "cx_px": 320.0,
                        "cy_px": 240.0,
                        "calibration_id": "orbbec_gemini_pro_mock_calibration_v1",
                    },
                    "camera_pose_in_world": {
                        "position_m": [0.35, -0.25, -0.08],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "frame_id": "world",
                    },
                },
                "instances": [
                    {
                        "instance_id": "mask-1",
                        "track_id": "tomato-1",
                        "bbox_xyxy_px": [365.0, 210.0, 425.0, 270.0],
                        "mask_pixels_xy": [
                            [395, 240],
                            [372, 240],
                            [418, 240],
                            [395, 218],
                            [395, 262],
                        ],
                        "detector_confidence": 0.97,
                        "ripeness_score": 0.92,
                        "occlusion_ratio": 0.08,
                        "depth_samples_uvm": [
                            [395.01, 240.0, 0.82977],
                            [395.01, 240.0, 0.77023],
                            [372.43, 240.0, 0.80372],
                            [417.77, 240.0, 0.79628],
                            [395.0, 217.5, 0.8],
                            [395.0, 262.5, 0.8],
                        ],
                        "stem_point_camera_m": [0.1, 0.05, 0.8],
                    }
                ],
                "vlm_selection": {
                    "selected_instance_id": "mask-1",
                    "confidence": 0.95,
                    "reason": "ripe, reachable, and minimally occluded numbered candidate",
                    "point_xy": [395, 240],
                    "selector_model": "replay_set_of_mark_vlm_v1",
                },
            },
            "tomatoes": [
                {
                    "target_id": "tomato-1",
                    "center_pose": {
                        "position_m": [0.45, -0.25, 0.72],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "frame_id": "world",
                    },
                    "semi_axes_m": [0.035, 0.032, 0.038],
                    "ripeness_score": 0.92,
                    "pick_confidence": 0.96,
                    "occlusion": 0.08,
                    "stem_direction_xyz": [0.0, 0.0, -1.0],
                }
            ]
        },
    )


def _task_config(
    *,
    outcome: str = "success",
    target_observation: dict[str, object] | None = None,
    clearance_scene: dict[str, object] | None = None,
    dropped_payload_observation: dict[str, object] | None = None,
    placement_verifier: dict[str, object] | None = None,
    payload_retreat: dict[str, object] | None = None,
    transport_payload: dict[str, object] | None = None,
    container_available: bool = True,
    camera_health: dict[str, object] | None = None,
    vision_worker_health: dict[str, object] | None = None,
    emergency_stop_health: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "type": "harvest_cycle",
        "default_arm_side": "right",
        "max_skill_duration_s": 4.0,
        "max_atomic_duration_s": 8.0,
        "robotics": {"type": "mock", "scenario": {"outcome": outcome}},
        "skill": {"type": "mock"},
        "perception": {
            "type": "tomato_vision_replay",
            "expected_calibration_id": "orbbec_gemini_pro_mock_calibration_v1",
            "expected_device_id": "orbbec-replay-001",
        },
        "planner": {"type": "training_distribution"},
        "safety": {"type": "mock"},
        "verifier": {"type": "mock"},
        "atomic_skills": {
            "system_self_check": {
                "type": "system_self_check",
                "execution_mode": "offline_mock",
            },
            "active_view_reposition": {
                "type": "active_view_reposition",
                "execution_mode": "offline_mock",
                "view_profile_id": "offline-test-fixed-top-arm-parallax-v1",
                "view_arm_side": "right",
                "view_offsets_world_m": [
                    [0.0, 0.05, 0.0],
                    [0.0, -0.05, 0.0],
                    [0.0, 0.0, 0.05],
                ],
                "workspace_min_m": [-0.20, -0.80, 0.20],
                "workspace_max_m": [0.80, 0.80, 1.40],
            },
            "select_harvest_target": {
                "type": "select_harvest_target",
                "reachability_profile_id": "offline-test-right-arm-workspace-v1",
                "reachability_arm_side": "right",
                "workspace_min_m": [0.20, -0.60, 0.30],
                "workspace_max_m": [0.80, 0.20, 1.20],
            },
            "target_reobserve": {
                "type": "target_reobserve",
                "require_depth": True,
                "require_new_sample": False,
            },
            "pregrasp_clearance_check": {
                "type": "pregrasp_clearance_check",
            },
            "move_to_pick_ready": {
                "type": "move_to_pick_ready",
                "execution_mode": "offline_mock",
            },
            "safe_retract": {"type": "safe_retract"},
            "payload_retreat": payload_retreat or {"type": "payload_retreat"},
            "transport_payload": transport_payload or {"type": "transport_payload"},
            "payload_place": {"type": "payload_place"},
            "localize_dropped_payload": {
                "type": "localize_dropped_payload",
                "recovery_workspace_xyzxyz_m": [
                    -0.20,
                    -0.80,
                    0.0,
                    0.80,
                    0.80,
                    0.50,
                ],
            },
            "reset_for_next_target": {"type": "reset_for_next_target"},
        },
        "camera_health": camera_health or {"type": "mock"},
        "vision_worker_health": vision_worker_health or {"type": "mock"},
        "emergency_stop_health": emergency_stop_health or {"type": "mock"},
        "target_observation": target_observation
        or {
            "type": "tomato_vision_replay",
            "expected_calibration_id": "orbbec_gemini_pro_mock_calibration_v1",
            "expected_device_id": "orbbec-replay-001",
        },
        "clearance_scene": clearance_scene or {"type": "mock"},
        "dropped_payload_observation": dropped_payload_observation
        or {"type": "mock"},
        "placement_verifier": placement_verifier or {"type": "mock"},
        "container": {
            "container_id": "harvest-bin-1",
            "placement_pose": {
                "position_m": [0.35, 0.40, 0.65],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "frame_id": "world",
            },
            "available": container_available,
            "capacity_remaining": 20,
        },
        "safe_pose": {
            "position_m": [0.0, 0.0, 0.80],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "frame_id": "world",
        },
    }


def _tool_map(task: object, context: RuntimeContext) -> dict[str, object]:
    return {tool.name: tool for tool in task.build_tools(context)}  # type: ignore[attr-defined]


def _run_through_retreat(tools: dict[str, object]) -> None:
    tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
    tools["detect_tomatoes"].handler()  # type: ignore[attr-defined]
    tools["select_harvest_target"].handler()  # type: ignore[attr-defined]
    tools["reobserve_tomato"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
    tools["plan_tomato_grasp"].handler(target_id="tomato-1", arm_side="auto")  # type: ignore[attr-defined]
    tools["check_pregrasp_clearance"].handler(grasp_plan_id="plan-tomato-1-right-0")  # type: ignore[attr-defined]
    tools["move_to_pick_ready"].handler(grasp_plan_id="plan-tomato-1-right-0")  # type: ignore[attr-defined]
    tools["execute_tomato_pick"].handler(grasp_plan_id="plan-tomato-1-right-0")  # type: ignore[attr-defined]
    tools["retreat_with_tomato"].handler(execution_id="exec-plan-tomato-1-right-0")  # type: ignore[attr-defined]


class HarvestCycleTaskTests(unittest.TestCase):
    def test_task_exposes_exactly_sixteen_bounded_tools(self) -> None:
        task = build_task(_task_config())
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        try:
            tools = task.build_tools(context)
            self.assertEqual(
                tuple(tool.name for tool in tools),
                (
                    "check_harvest_system",
                    "detect_tomatoes",
                    "select_harvest_target",
                    "active_view_reposition",
                    "reobserve_tomato",
                    "plan_tomato_grasp",
                    "check_pregrasp_clearance",
                    "move_to_pick_ready",
                    "execute_tomato_pick",
                    "retreat_with_tomato",
                    "transport_payload",
                    "place_tomato",
                    "localize_dropped_payload",
                    "verify_harvest_cycle",
                    "reset_for_next_target",
                    "recover_harvest_cycle",
                ),
            )
            for tool in tools:
                schema = tool.as_dict()["parameters"]
                self.assertFalse(schema["additionalProperties"])
                serialized = repr(schema)
                self.assertNotIn("joint_positions", serialized)
                self.assertNotIn("tactile", serialized)
                self.assertNotIn("camera_intrinsics", serialized)
        finally:
            task.close()
            env.close()

    def test_shipped_config_completes_one_offline_harvest_cycle(self) -> None:
        runtime = build_runtime_from_file(ROOT / "config" / "harvest_cycle_mock.yaml")
        report = runtime.run(seed=7)
        self.assertTrue(report.completed)
        self.assertIs(report.reason, CompletionReason.TASK_COMPLETE)
        self.assertEqual(report.control_steps, 1)

    def test_success_requires_independent_end_to_end_evidence(self) -> None:
        task = build_task(_task_config())
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        tools = _tool_map(task, context)
        try:
            _run_through_retreat(tools)
            transport = tools["transport_payload"].handler(  # type: ignore[attr-defined]
                retreat_execution_id="retreat-exec-plan-tomato-1-right-0"
            )
            place = tools["place_tomato"].handler(  # type: ignore[attr-defined]
                transport_execution_id=transport["transport_execution_id"]
            )
            verification = tools["verify_harvest_cycle"].handler(  # type: ignore[attr-defined]
                placement_id=place["placement_id"]
            )
            self.assertTrue(verification["success"])
            self.assertTrue(verification["pick_verified"])
            self.assertTrue(verification["retreat_verified"])
            self.assertTrue(verification["transport_verified"])
            self.assertTrue(verification["placed_in_container"])
            self.assertTrue(verification["release_confirmed"])
            self.assertTrue(verification["hand_empty"])
            self.assertGreaterEqual(len(verification["evidence_sources"]), 4)
            self.assertTrue(task.service.completed)  # type: ignore[attr-defined]
            fake = {"messages": [{"role": "tool", "status": "success"}]}
            self.assertTrue(task.is_complete(context, fake))  # type: ignore[attr-defined]
        finally:
            task.close()
            env.close()

    def test_fake_transcript_cannot_complete_before_service_evidence(self) -> None:
        task = build_task(_task_config())
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        try:
            self.assertFalse(
                task.is_complete(  # type: ignore[attr-defined]
                    context,
                    {"messages": [{"name": "verify_harvest_cycle", "status": "success"}]},
                )
            )
        finally:
            task.close()
            env.close()

    def test_target_must_be_reobserved_before_planning(self) -> None:
        task = build_task(_task_config())
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        tools = _tool_map(task, context)
        try:
            tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            tools["detect_tomatoes"].handler()  # type: ignore[attr-defined]
            tools["select_harvest_target"].handler()  # type: ignore[attr-defined]
            with self.assertRaisesRegex(RecoverableToolError, "must be re-observed"):
                tools["plan_tomato_grasp"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
        finally:
            task.close()
            env.close()

    def test_reobservation_requires_the_pinned_selection(self) -> None:
        task = build_task(_task_config())
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        tools = _tool_map(task, context)
        try:
            tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            tools["detect_tomatoes"].handler()  # type: ignore[attr-defined]
            with self.assertRaisesRegex(RecoverableToolError, "select_harvest_target"):
                tools["reobserve_tomato"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
            selected = tools["select_harvest_target"].handler()  # type: ignore[attr-defined]
            self.assertEqual(selected["output"]["selected_target_id"], "tomato-1")
            self.assertFalse(selected["output"]["motion_authorized"])
            self.assertEqual(selected["workflow_state"], "reobserve")
            with self.assertRaisesRegex(RecoverableToolError, "pinned selected target"):
                tools["reobserve_tomato"].handler(target_id="tomato-2")  # type: ignore[attr-defined]
        finally:
            task.close()
            env.close()

    def test_missing_reobservation_returns_stable_failure(self) -> None:
        task = build_task(
            _task_config(target_observation={"type": "mock", "force_missing": True})
        )
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        tools = _tool_map(task, context)
        try:
            tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            tools["detect_tomatoes"].handler()  # type: ignore[attr-defined]
            tools["select_harvest_target"].handler()  # type: ignore[attr-defined]
            with self.assertRaisesRegex(RecoverableToolError, "target_stale"):
                tools["reobserve_tomato"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
            self.assertIs(task.service.state, HarvestCycleState.REOBSERVE)  # type: ignore[attr-defined]
        finally:
            task.close()
            env.close()

    def test_unconfirmed_container_evidence_does_not_complete(self) -> None:
        task = build_task(
            _task_config(
                placement_verifier={"type": "mock", "force_unconfirmed": True}
            )
        )
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        tools = _tool_map(task, context)
        try:
            _run_through_retreat(tools)
            transport = tools["transport_payload"].handler(  # type: ignore[attr-defined]
                retreat_execution_id="retreat-exec-plan-tomato-1-right-0"
            )
            place = tools["place_tomato"].handler(  # type: ignore[attr-defined]
                transport_execution_id=transport["transport_execution_id"]
            )
            verification = tools["verify_harvest_cycle"].handler(  # type: ignore[attr-defined]
                placement_id=place["placement_id"]
            )
            self.assertFalse(verification["success"])
            self.assertFalse(task.service.completed)  # type: ignore[attr-defined]
            self.assertIs(task.service.state, HarvestCycleState.ABORTED)  # type: ignore[attr-defined]
        finally:
            task.close()
            env.close()

    def test_unstable_retreat_can_use_safe_retract_recovery(self) -> None:
        task = build_task(
            _task_config(
                payload_retreat={"type": "payload_retreat", "force_payload_unstable": True}
            )
        )
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
            tools["move_to_pick_ready"].handler(grasp_plan_id="plan-tomato-1-right-0")  # type: ignore[attr-defined]
            tools["execute_tomato_pick"].handler(grasp_plan_id="plan-tomato-1-right-0")  # type: ignore[attr-defined]
            with self.assertRaisesRegex(RecoverableToolError, "payload_unstable"):
                tools["retreat_with_tomato"].handler(  # type: ignore[attr-defined]
                    execution_id="exec-plan-tomato-1-right-0"
                )
            recovery = tools["recover_harvest_cycle"].handler(  # type: ignore[attr-defined]
                reason="payload retention lost"
            )
            self.assertTrue(recovery["success"])
            arm = task.service.pick_service.robotics.arm.get_state(  # type: ignore[attr-defined]
                task.service.pick_service.default_arm_side  # type: ignore[attr-defined]
            )
            self.assertEqual(
                arm.ee_pose,
                Pose((0.0, 0.0, 0.80), (1.0, 0.0, 0.0, 0.0), "world"),
            )
            self.assertFalse(task.service.completed)  # type: ignore[attr-defined]
        finally:
            task.close()
            env.close()

    def test_unavailable_container_is_fail_closed(self) -> None:
        task = build_task(_task_config(container_available=False))
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        tools = _tool_map(task, context)
        try:
            with self.assertRaisesRegex(RecoverableToolError, "container"):
                tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            self.assertIs(task.service.state, HarvestCycleState.INIT)  # type: ignore[attr-defined]
        finally:
            task.close()
            env.close()

    def test_reachability_profile_must_match_the_task_arm(self) -> None:
        config = _task_config()
        config["atomic_skills"]["select_harvest_target"][  # type: ignore[index]
            "reachability_arm_side"
        ] = "left"
        with self.assertRaisesRegex(ValueError, "must match default_arm_side"):
            build_task(config)


if __name__ == "__main__":
    unittest.main()
