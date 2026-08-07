"""Offline integration tests for the five-tool agricultural Task."""

from __future__ import annotations

from pathlib import Path
import unittest

from module.runtime import CompletionReason, RuntimeContext
from module.tools.env import MockEnv
from proto import Observation
from script.run import build_runtime, build_runtime_from_file
from task import build_task


ROOT = Path(__file__).resolve().parents[1]


def _task_config(outcome: str = "success") -> dict[str, object]:
    return {
        "type": "agri_pick",
        "default_arm_side": "right",
        "max_skill_duration_s": 4.0,
        "robotics": {"type": "mock", "scenario": {"outcome": outcome}},
        "skill": {"type": "mock"},
        "perception": {"type": "mock"},
        "planner": {"type": "mock"},
        "safety": {"type": "mock"},
        "verifier": {"type": "mock"},
    }


def _observation() -> Observation:
    return Observation(
        sequence=0,
        metadata={
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


class AgriPickTaskTests(unittest.TestCase):
    def test_task_exposes_exactly_five_small_tools(self) -> None:
        task = build_task(_task_config())
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        try:
            tools = task.build_tools(context)
            self.assertEqual(
                tuple(tool.name for tool in tools),
                (
                    "detect_tomatoes",
                    "plan_tomato_grasp",
                    "move_to_pregrasp",
                    "execute_tomato_pick",
                    "verify_tomato_pick",
                ),
            )
            for tool in tools:
                schema = tool.as_dict()["parameters"]
                self.assertFalse(schema["additionalProperties"])
                serialized = repr(schema)
                self.assertNotIn("joint_positions", serialized)
                self.assertNotIn("logical_raw", serialized)
                self.assertNotIn("tactile", serialized)
        finally:
            task.close()
            env.close()

    def test_shipped_c1_config_completes_one_offline_pick(self) -> None:
        runtime = build_runtime_from_file(ROOT / "config" / "agri_pick_mock.yaml")
        report = runtime.run(seed=3)
        self.assertTrue(report.completed)
        self.assertIs(report.reason, CompletionReason.TASK_COMPLETE)
        self.assertEqual(report.control_steps, 1)

    def test_completion_uses_service_evidence_not_a_fake_transcript(self) -> None:
        task = build_task(_task_config())
        env = MockEnv(observations=[_observation()])
        context = RuntimeContext(env)
        context.update_observation(env.reset())
        try:
            fake = {"messages": [{"role": "tool", "name": "verify_tomato_pick", "status": "success"}]}
            self.assertFalse(task.is_complete(context, fake))
        finally:
            task.close()
            env.close()

    def test_failed_skill_does_not_complete_task(self) -> None:
        root = {
            "models_file": "_model.yaml",
            "perception_file": "_perception.yaml",
            "agent": {
                "model": "offline_agri_pick",
                "max_iterations": 8,
                "max_tool_calls": 5,
                "max_tool_calls_per_turn": 1,
            },
            "env": {"type": "mock", "observations": [_observation()]},
            "task": _task_config("no_contact"),
            "runtime": {"max_steps": 1},
        }
        runtime = build_runtime(root, config_dir=ROOT / "config")
        report = runtime.run()
        self.assertFalse(report.completed)

    def test_nested_task_config_is_rejected_before_environment_construction(self) -> None:
        root = {
            "models_file": "_model.yaml",
            "perception_file": "_perception.yaml",
            "agent": {"model": "offline_agri_pick"},
            "env": {"type": "mock", "observations": [_observation()]},
            "task": _task_config(),
            "runtime": {"max_steps": 1},
        }
        bad = dict(root)
        bad["task"] = _task_config()
        bad["task"]["planner"] = {"type": "missing"}
        with self.assertRaisesRegex(ValueError, "unknown planner type"):
            build_runtime(bad, config_dir=ROOT / "config")


if __name__ == "__main__":
    unittest.main()
