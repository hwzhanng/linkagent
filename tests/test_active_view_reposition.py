"""Closed-loop simulation tests for bounded active-view recovery."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from module.agent import RecoverableToolError
from module.robotics import ManualClock, build_robotics_bundle
from module.runtime import CompletionReason, RuntimeContext
from module.skills.harvest_cycle import ActiveViewRepositionSkill
from module.tools.env import MockEnv
from proto import (
    ActiveViewRepositionRequest,
    ArmSide,
    Observation,
    Pose,
    TomatoTarget,
)
from task import build_task
from tests.test_harvest_cycle_task import _observation, _task_config, _tool_map


def _target(clock: ManualClock) -> TomatoTarget:
    return TomatoTarget(
        target_id="tomato-1",
        center_pose=Pose(
            (0.45, -0.25, 0.72), (1.0, 0.0, 0.0, 0.0), "world"
        ),
        semi_axes_m=(0.035, 0.032, 0.038),
        ripeness_score=0.92,
        pick_confidence=0.95,
        occlusion=0.80,
        stem_direction_xyz=(0.0, 0.0, -1.0),
        observation_sequence=0,
        sample_monotonic_ns=clock.monotonic_ns(),
    )


def _active_skill(*, mode: str = "offline_mock") -> tuple[object, ActiveViewRepositionSkill]:
    bundle = build_robotics_bundle(
        {"type": "mock", "scenario": {"outcome": "success"}},
        side=ArmSide.RIGHT,
    )
    skill = ActiveViewRepositionSkill(
        robotics=bundle,
        execution_mode=mode,
        view_profile_id="test-wrist-view-lattice-v1",
        view_arm_side="right",
        view_offsets_world_m=((0.0, 0.05, 0.0), (0.0, -0.05, 0.0)),
        workspace_min_m=(-0.20, -0.80, 0.20),
        workspace_max_m=(0.80, 0.80, 1.40),
    )
    return bundle, skill


def _occluded_observation() -> Observation:
    original = _observation()
    metadata = deepcopy(dict(original.metadata))
    metadata["tomato_vision"]["instances"][0]["occlusion_ratio"] = 0.80  # type: ignore[index]
    metadata["tomatoes"][0]["stem_direction_xyz"] = [0.0, 0.0, 1.0]  # type: ignore[index]
    metadata["tomatoes"][0]["semi_axes_m"] = [0.030, 0.030, 0.030]  # type: ignore[index]
    return Observation(
        sequence=original.sequence,
        timestamp_s=original.timestamp_s,
        metadata=metadata,
    )


class ActiveViewRepositionSkillTests(unittest.TestCase):
    def test_offline_motion_is_linear_bounded_and_idempotent(self) -> None:
        bundle, skill = _active_skill()
        reference = bundle.arm.get_state(ArmSide.RIGHT).ee_pose  # type: ignore[attr-defined]
        clock = bundle.clock  # type: ignore[attr-defined]
        request = ActiveViewRepositionRequest(
            execution_id="active-view-one",
            target=_target(clock),
            arm_side=ArmSide.RIGHT,
            reference_pose=reference,
            viewpoint_index=0,
        )
        try:
            first = skill.run(request)
            second = skill.run(request)
            self.assertIs(first, second)
            self.assertTrue(first.success)
            self.assertTrue(first.output["motion_executed"])
            self.assertTrue(first.output["simulated_safety_only"])
            self.assertFalse(first.output["shadow_only"])
            actual = bundle.arm.get_state(ArmSide.RIGHT).ee_pose  # type: ignore[attr-defined]
            self.assertEqual(actual.position_m, (0.0, 0.05, 0.7))
            self.assertEqual(first.output["position_error_m"], 0.0)
            world = bundle.verifier_context()["mock_world"]  # type: ignore[attr-defined]
            self.assertEqual(world.arm_sequence, 1)
        finally:
            skill.close()
            bundle.close()  # type: ignore[attr-defined]

    def test_shadow_returns_a_plan_without_motion(self) -> None:
        bundle, skill = _active_skill(mode="shadow")
        reference = bundle.arm.get_state(ArmSide.RIGHT).ee_pose  # type: ignore[attr-defined]
        try:
            result = skill.run(
                ActiveViewRepositionRequest(
                    execution_id="active-view-shadow",
                    target=_target(bundle.clock),  # type: ignore[attr-defined]
                    arm_side=ArmSide.RIGHT,
                    reference_pose=reference,
                    viewpoint_index=0,
                )
            )
            self.assertTrue(result.success)
            self.assertTrue(result.output["shadow_only"])
            self.assertFalse(result.output["motion_executed"])
            self.assertEqual(
                bundle.arm.get_state(ArmSide.RIGHT).ee_pose, reference  # type: ignore[attr-defined]
            )
        finally:
            skill.close()
            bundle.close()  # type: ignore[attr-defined]

    def test_unknown_viewpoint_index_fails_without_motion(self) -> None:
        bundle, skill = _active_skill()
        reference = bundle.arm.get_state(ArmSide.RIGHT).ee_pose  # type: ignore[attr-defined]
        try:
            result = skill.run(
                ActiveViewRepositionRequest(
                    execution_id="active-view-exhausted",
                    target=_target(bundle.clock),  # type: ignore[attr-defined]
                    arm_side=ArmSide.RIGHT,
                    reference_pose=reference,
                    viewpoint_index=2,
                )
            )
            self.assertFalse(result.success)
            self.assertEqual(result.failure_code.value, "viewpoints_exhausted")
            world = bundle.verifier_context()["mock_world"]  # type: ignore[attr-defined]
            self.assertEqual(world.arm_sequence, 0)
        finally:
            skill.close()
            bundle.close()  # type: ignore[attr-defined]


class ActiveViewHarvestCycleTests(unittest.TestCase):
    def test_shipped_active_view_config_completes_one_recovery_cycle(self) -> None:
        from script.run import build_runtime_from_file

        root = Path(__file__).resolve().parents[1]
        runtime = build_runtime_from_file(
            root / "config" / "harvest_cycle_active_view_mock.yaml"
        )
        report = runtime.run(seed=7)
        self.assertTrue(report.completed)
        self.assertIs(report.reason, CompletionReason.TASK_COMPLETE)

    def test_occluded_target_recovers_and_completes_the_entire_cycle(self) -> None:
        config = _task_config(
            target_observation={
                "type": "mock",
                "occlusion_sequence": [0.10, 0.10],
                "sample_advance_s": 0.01,
            }
        )
        config["atomic_skills"]["target_reobserve"] = {  # type: ignore[index]
            "type": "target_reobserve",
            "require_depth": False,
            "require_new_sample": True,
        }
        config["atomic_skills"]["select_harvest_target"][  # type: ignore[index]
            "min_pick_confidence"
        ] = 0.10
        observation = _occluded_observation()
        task = build_task(config)
        env = MockEnv(observations=[observation])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        tools = _tool_map(task, context)
        try:
            tools["check_harvest_system"].handler()  # type: ignore[attr-defined]
            tools["detect_tomatoes"].handler()  # type: ignore[attr-defined]
            with self.assertRaisesRegex(
                RecoverableToolError, "active_view_required"
            ):
                tools["select_harvest_target"].handler()  # type: ignore[attr-defined]
            active = tools["active_view_reposition"].handler()  # type: ignore[attr-defined]
            self.assertTrue(active["success"])
            self.assertTrue(active["observation_refreshed"])
            self.assertGreater(active["visibility_improvement"], 0.6)
            selected = tools["select_harvest_target"].handler()  # type: ignore[attr-defined]
            self.assertEqual(selected["output"]["selected_target_id"], "tomato-1")
            tools["reobserve_tomato"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
            tools["plan_tomato_grasp"].handler(target_id="tomato-1")  # type: ignore[attr-defined]
            tools["check_pregrasp_clearance"].handler(  # type: ignore[attr-defined]
                grasp_plan_id="plan-tomato-1-right-0"
            )
            tools["move_to_pick_ready"].handler(  # type: ignore[attr-defined]
                grasp_plan_id="plan-tomato-1-right-0"
            )
            tools["execute_tomato_pick"].handler(  # type: ignore[attr-defined]
                grasp_plan_id="plan-tomato-1-right-0"
            )
            tools["retreat_with_tomato"].handler(  # type: ignore[attr-defined]
                execution_id="exec-plan-tomato-1-right-0"
            )
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
            self.assertTrue(task.service.completed)  # type: ignore[attr-defined]
        finally:
            task.close()
            env.close()


if __name__ == "__main__":
    unittest.main()
