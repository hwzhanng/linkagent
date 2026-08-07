"""Offline contract tests for neutral DTOs and mock capabilities.

These tests intentionally use only the Python standard library.  They exercise
the boundaries that real SDK adapters must preserve without importing or
connecting to any external backend.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
import unittest
from unittest.mock import patch

from module.tools.env import BaseEnv, ENV_REGISTRY, MockEnv, build_env
from module.tools.manipulation import (
    BaseManipulation,
    MANIPULATION_REGISTRY,
    MockManipulation,
    build_manipulation,
)
from module.tools.perception import (
    BasePerceptionTool,
    MockPerceptionTool,
    PERCEPTION_REGISTRY,
    build_perception,
    build_perception_profile,
)
from proto import (
    Action,
    GraspCandidate,
    Observation,
    Pose,
    Result,
    VisionInstance,
    VisionResult,
)


class DtoContractTests(unittest.TestCase):
    def test_base_dtos_validate_and_normalize_input(self) -> None:
        observation = Observation(
            sequence=3,
            timestamp_s=4,
            joint_positions_rad=[0, 1.5],
        )
        self.assertEqual(observation.timestamp_s, 4.0)
        self.assertEqual(observation.joint_positions_rad, (0.0, 1.5))

        action = Action("  move_to_pose  ", {"speed": 0.2}, "  command-1  ")
        self.assertEqual(action.name, "move_to_pose")
        self.assertEqual(action.action_id, "command-1")

        for bad_sequence in (-1, True, 1.5, "1"):
            with self.subTest(sequence=bad_sequence):
                with self.assertRaises(ValueError):
                    Observation(sequence=bad_sequence)  # type: ignore[arg-type]
        for bad_timestamp in (-0.1, math.inf, math.nan, True, "now"):
            with self.subTest(timestamp_s=bad_timestamp):
                with self.assertRaises((TypeError, ValueError)):
                    Observation(timestamp_s=bad_timestamp)  # type: ignore[arg-type]
        for bad_joints in ((0.0, math.nan), "0,1", (False,)):
            with self.subTest(joint_positions_rad=bad_joints):
                with self.assertRaises((TypeError, ValueError)):
                    Observation(  # type: ignore[arg-type]
                        joint_positions_rad=bad_joints
                    )
        for bad_name in ("", "   ", 7):
            with self.subTest(action_name=bad_name):
                with self.assertRaises(ValueError):
                    Action(bad_name)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Result(success=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Result(success=True, message=object())  # type: ignore[arg-type]

    def test_perception_dtos_reject_invalid_geometry_and_members(self) -> None:
        instance = VisionInstance(
            label="  cup  ",
            score=1,
            bbox_xyxy=[1, 2, 10, 20],
            instance_id=4,
        )
        self.assertEqual(instance.label, "cup")
        self.assertEqual(instance.score, 1.0)
        self.assertEqual(instance.bbox_xyxy, (1.0, 2.0, 10.0, 20.0))

        for kwargs in (
            {"label": "", "score": 0.5},
            {"label": "cup", "score": -0.01},
            {"label": "cup", "score": 1.01},
            {"label": "cup", "score": math.nan},
            {"label": "cup", "score": True},
            {"label": "cup", "score": 0.5, "bbox_xyxy": (0, 0, 1)},
            {"label": "cup", "score": 0.5, "bbox_xyxy": (2, 0, 1, 1)},
            {"label": "cup", "score": 0.5, "instance_id": False},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    VisionInstance(**kwargs)  # type: ignore[arg-type]

        result = VisionResult(
            model="  mock-v1  ",
            image_size=(640, 480),
            instances=[instance],
            elapsed_s=1,
            observation_sequence=2,
        )
        self.assertEqual(result.model, "mock-v1")
        self.assertEqual(result.instances, (instance,))
        self.assertEqual(result.elapsed_s, 1.0)

        for kwargs in (
            {"model": ""},
            {"model": "mock", "image_size": (640, 0)},
            {"model": "mock", "image_size": (640,)},
            {"model": "mock", "image_size": (True, 480)},
            {"model": "mock", "instances": [object()]},
            {"model": "mock", "elapsed_s": -0.1},
            {"model": "mock", "observation_sequence": -1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    VisionResult(**kwargs)  # type: ignore[arg-type]

    def test_manipulation_dtos_enforce_pose_contract(self) -> None:
        pregrasp = Pose((0, 0, 0.2), (1, 0, 0, 0), "  camera  ")
        grasp = Pose((0, 0, 0.1), (1.0, 0, 0, 0), "camera")
        candidate = GraspCandidate(
            pregrasp,
            grasp,
            score=2,
            model_score=1,
            forward_alignment=-1,
        )
        self.assertEqual(pregrasp.position_m, (0.0, 0.0, 0.2))
        self.assertEqual(pregrasp.frame_id, "camera")
        self.assertEqual(candidate.score, 2.0)
        self.assertEqual(candidate.forward_alignment, -1.0)

        for kwargs in (
            {
                "position_m": (0, 0),
                "quaternion_wxyz": (1, 0, 0, 0),
                "frame_id": "camera",
            },
            {
                "position_m": (0, 0, 0),
                "quaternion_wxyz": (2, 0, 0, 0),
                "frame_id": "camera",
            },
            {
                "position_m": (0, math.inf, 0),
                "quaternion_wxyz": (1, 0, 0, 0),
                "frame_id": "camera",
            },
            {
                "position_m": (0, 0, 0),
                "quaternion_wxyz": (1, 0, 0, 0),
                "frame_id": " ",
            },
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    Pose(**kwargs)  # type: ignore[arg-type]

        other_frame = Pose((0, 0, 0), (1, 0, 0, 0), "world")
        with self.assertRaises(ValueError):
            GraspCandidate(pregrasp, other_frame, score=0.5)
        with self.assertRaises(ValueError):
            GraspCandidate(pregrasp, grasp, score=0.5, forward_alignment=1.1)
        with self.assertRaises(TypeError):
            GraspCandidate(object(), grasp, score=0.5)  # type: ignore[arg-type]

    def test_mapping_fields_are_read_only_copies(self) -> None:
        source = {"source": "camera"}
        observation = Observation(metadata=source)
        source["late_mutation"] = True
        self.assertEqual(dict(observation.metadata), {"source": "camera"})
        with self.assertRaises(TypeError):
            observation.metadata["new"] = 1  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            observation.sequence = 9  # type: ignore[misc]

        mapping_owners = (
            Action("move", {"speed": 1}).parameters,
            Result(True, data={"attempt": 1}).data,
            VisionInstance("cup", 0.5, metadata={"track": 2}).metadata,
            VisionResult("mock", metadata={"source": "test"}).metadata,
            GraspCandidate(
                Pose((0, 0, 0), (1, 0, 0, 0), "world"),
                Pose((0, 0, 0), (1, 0, 0, 0), "world"),
                0.5,
                metadata={"planner": "mock"},
            ).metadata,
        )
        for mapping in mapping_owners:
            with self.subTest(mapping=dict(mapping)):
                with self.assertRaises(TypeError):
                    mapping["new"] = object()  # type: ignore[index]

        with self.assertRaises(TypeError):
            Observation(metadata={1: "not-a-string-key"})  # type: ignore[dict-item]
        with self.assertRaises(TypeError):
            Action("move", parameters=[("speed", 1)])  # type: ignore[arg-type]

    def test_masks_are_omitted_from_default_serialization(self) -> None:
        opaque_mask = object()
        instance = VisionInstance(
            "cup",
            0.9,
            bbox_xyxy=(1, 2, 3, 4),
            mask=opaque_mask,
        )
        instance_payload = instance.as_dict()
        self.assertNotIn("mask", instance_payload)
        self.assertTrue(instance_payload["has_mask"])
        json.dumps(instance_payload)
        self.assertIs(instance.as_dict(include_mask=True)["mask"], opaque_mask)

        result = VisionResult("mock", instances=(instance,))
        result_payload = result.as_dict()
        self.assertNotIn("mask", result_payload["instances"][0])
        json.dumps(result_payload)
        self.assertIs(
            result.as_dict(include_masks=True)["instances"][0]["mask"],
            opaque_mask,
        )


class MockEnvTests(unittest.TestCase):
    def test_reset_returns_first_frame_and_observations_advance(self) -> None:
        first = Observation(sequence=3, metadata={"frame": "first"})
        second = Observation(sequence=8, metadata={"frame": "second"})
        env = MockEnv(observations=(first, second))

        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.get_obs()
        self.assertIs(env.reset(seed=42), first)
        self.assertIs(env.get_obs(), second)

        repeated_one = env.get_obs()
        repeated_two = env.get_obs()
        self.assertEqual(repeated_one.sequence, 9)
        self.assertEqual(repeated_two.sequence, 10)
        self.assertEqual(repeated_two.metadata["frame"], "second")
        self.assertIs(env.reset(), first)

        with self.assertRaises(TypeError):
            env.reset(seed=True)

    def test_actions_report_success_rejection_and_infrastructure_failure(self) -> None:
        env = MockEnv(
            fail_actions={"reject"},
            raise_actions={"explode"},
            failure_message="blocked by mock",
        )
        accepted = Action("accept", {"target": "cup"})
        rejected = Action("reject")
        exploding = Action("explode")

        success = env.take_action(accepted)
        self.assertTrue(success.success)
        self.assertEqual(success.message, "action accepted")
        self.assertEqual(dict(success.data), {"action_name": "accept", "attempt": 1})

        refusal = env.take_action(rejected)
        self.assertFalse(refusal.success)
        self.assertEqual(refusal.message, "blocked by mock")
        self.assertEqual(refusal.data["attempt"], 2)

        with self.assertRaisesRegex(RuntimeError, "explode"):
            env.take_action(exploding)
        self.assertEqual(env.actions, (accepted, rejected, exploding))
        with self.assertRaises(TypeError):
            env.take_action("accept")  # type: ignore[arg-type]

    def test_fail_next_actions_are_reset_with_the_replay(self) -> None:
        env = MockEnv(fail_next_actions=1)
        first_failure = env.take_action(Action("move"))
        self.assertFalse(first_failure.success)
        self.assertTrue(env.take_action(Action("move")).success)

        env.reset()
        self.assertFalse(env.take_action(Action("move")).success)
        self.assertEqual(len(env.actions), 3)

    def test_close_is_idempotent_and_guards_all_operations(self) -> None:
        env = MockEnv()
        env.close()
        env.close()
        self.assertTrue(env.closed)

        for operation in (
            env.reset,
            env.get_obs,
            lambda: env.take_action(Action("move")),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    operation()

    def test_constructor_and_factory_fail_at_explicit_boundaries(self) -> None:
        with self.assertRaises(ValueError):
            MockEnv(observations=[])
        with self.assertRaises(ValueError):
            MockEnv(observations=[{"sequence": 2}, {"sequence": 2}])
        with self.assertRaises(TypeError):
            MockEnv(observations={"sequence": 0})  # type: ignore[arg-type]

        config = {"type": "mock", "fail_next_actions": 1}
        env = build_env(config)
        self.assertIsInstance(env, BaseEnv)
        self.assertEqual(config, {"type": "mock", "fail_next_actions": 1})

        for bad_config in (None, [], "mock"):
            with self.subTest(config=bad_config):
                with self.assertRaisesRegex(TypeError, "config must be a mapping"):
                    build_env(bad_config)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "missing required key 'type'"):
            build_env({})
        for bad_type in (None, 7, "   "):
            with self.subTest(env_type=bad_type):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    build_env({"type": bad_type})  # type: ignore[dict-item]
        with self.assertRaisesRegex(ValueError, "unknown env type.*mock"):
            build_env({"type": "not-registered"})
        with self.assertRaisesRegex(TypeError, "invalid options for env type 'mock'"):
            build_env({"type": "mock", "unknown_option": True})

        with patch.dict(ENV_REGISTRY, {"invalid-return": lambda: object()}):
            with self.assertRaisesRegex(TypeError, "expected BaseEnv"):
                build_env({"type": "invalid-return"})


class MockPerceptionTests(unittest.TestCase):
    @staticmethod
    def _observation() -> Observation:
        return Observation(
            sequence=7,
            metadata={
                "image_size": (320, 240),
                "detections": [
                    {
                        "label": "cup",
                        "score": 0.9,
                        "bbox_xyxy": (1, 2, 30, 40),
                        "instance_id": "cup-high",
                    },
                    VisionInstance("bottle", 0.8, instance_id="bottle"),
                    VisionInstance("cup", 0.5, instance_id="cup-threshold"),
                    VisionInstance("cup", 0.49, instance_id="cup-low"),
                ],
            },
        )

    def test_label_and_threshold_filters_preserve_source_order(self) -> None:
        tool = MockPerceptionTool(model="mock-detector")
        options = {"labels": ["cup"], "threshold": 0.5}
        result = tool.run(self._observation(), options=options)

        self.assertEqual(options, {"labels": ["cup"], "threshold": 0.5})
        self.assertEqual(result.model, "mock-detector")
        self.assertEqual(result.image_size, (320, 240))
        self.assertEqual(result.observation_sequence, 7)
        self.assertGreaterEqual(result.elapsed_s, 0.0)
        self.assertEqual(
            [instance.instance_id for instance in result.instances],
            ["cup-high", "cup-threshold"],
        )
        self.assertTrue(all(isinstance(item, VisionInstance) for item in result.instances))

        selected = tool.run(
            self._observation(),
            labels=("bottle", "cup"),
            min_score=0.8,
            max_instances=1,
        )
        self.assertEqual([item.instance_id for item in selected.instances], ["cup-high"])

    def test_unknown_duplicate_and_invalid_options_are_rejected(self) -> None:
        tool = MockPerceptionTool()
        observation = self._observation()
        with self.assertRaisesRegex(TypeError, "unsupported perception option.*mystery"):
            tool.run(observation, mystery=True)
        with self.assertRaisesRegex(TypeError, "only one of min_score and threshold"):
            tool.run(observation, min_score=0.4, threshold=0.5)
        with self.assertRaisesRegex(TypeError, "provided more than once.*threshold"):
            tool.run(observation, options={"threshold": 0.5}, threshold=0.5)
        with self.assertRaises(TypeError):
            tool.run(observation, threshold="high")
        with self.assertRaises(ValueError):
            tool.run(observation, threshold=1.1)
        with self.assertRaises(ValueError):
            tool.run(observation, labels=[""])

    def test_missing_detections_and_nonmatching_filters_return_empty_result(self) -> None:
        tool = MockPerceptionTool()
        missing = tool.run(Observation(sequence=1))
        filtered = tool.run(self._observation(), labels="plate")
        limited = tool.run(self._observation(), max_instances=0)

        self.assertEqual(missing.instances, ())
        self.assertEqual(missing.observation_sequence, 1)
        self.assertEqual(filtered.instances, ())
        self.assertEqual(limited.instances, ())

    def test_close_is_idempotent_and_failure_is_not_converted_to_empty(self) -> None:
        tool = MockPerceptionTool(fail=True, failure_message="camera unavailable")
        with self.assertRaisesRegex(RuntimeError, "camera unavailable"):
            tool.run(Observation())

        healthy = MockPerceptionTool()
        healthy.close()
        healthy.close()
        self.assertTrue(healthy.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            healthy.run(Observation())

        class FailingClosePerception(MockPerceptionTool):
            def __init__(self) -> None:
                super().__init__()
                self.close_count = 0

            def _close(self) -> None:
                self.close_count += 1
                raise RuntimeError("partial perception cleanup failure")

        failing_close = FailingClosePerception()
        with self.assertRaisesRegex(RuntimeError, "partial perception cleanup"):
            failing_close.close()
        failing_close.close()
        self.assertTrue(failing_close.closed)
        self.assertEqual(failing_close.close_count, 1)

    def test_factory_reports_config_profile_option_and_return_type_errors(self) -> None:
        config = {"type": "mock", "model": "detector"}
        tool = build_perception(config)
        self.assertIsInstance(tool, BasePerceptionTool)
        self.assertEqual(config, {"type": "mock", "model": "detector"})

        for bad_config in (None, [], "mock"):
            with self.subTest(config=bad_config):
                with self.assertRaisesRegex(TypeError, "config must be a mapping"):
                    build_perception(bad_config)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "missing required key 'type'"):
            build_perception({})
        for bad_type in (None, 7, "   "):
            with self.subTest(perception_type=bad_type):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    build_perception({"type": bad_type})  # type: ignore[dict-item]
        with self.assertRaisesRegex(ValueError, "unknown perception type.*mock"):
            build_perception({"type": "not-registered"})
        with self.assertRaisesRegex(
            TypeError, "invalid options for perception type 'mock'"
        ):
            build_perception({"type": "mock", "unknown_option": True})

        with patch.dict(PERCEPTION_REGISTRY, {"invalid-return": lambda: object()}):
            with self.assertRaisesRegex(TypeError, "expected BasePerceptionTool"):
                build_perception({"type": "invalid-return"})

        with self.assertRaisesRegex(ValueError, "unknown perception profile"):
            build_perception_profile("missing", profiles={"valid": {"type": "mock"}})
        with self.assertRaisesRegex(TypeError, "profiles must be a mapping"):
            build_perception_profile("mock", profiles=[])  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "overrides must be a mapping"):
            build_perception_profile("mock", overrides=[])  # type: ignore[arg-type]


class MockManipulationTests(unittest.TestCase):
    @staticmethod
    def _observation() -> Observation:
        actual_candidate = GraspCandidate(
            Pose((0.1, 0.0, 0.2), (1, 0, 0, 0), "camera"),
            Pose((0.1, 0.0, 0.1), (1, 0, 0, 0), "camera"),
            score=0.7,
            metadata={"kind": "dto"},
        )
        return Observation(
            sequence=11,
            metadata={
                "grasp_candidates": [
                    {
                        "position_m": (0.2, 0.0, 0.1),
                        "quaternion_wxyz": (1, 0, 0, 0),
                        "frame_id": "camera",
                        "pregrasp_position_m": (0.2, 0.0, 0.3),
                        "score": 0.4,
                        "model_score": 0.35,
                    },
                    {
                        "pregrasp_tcp_pose": {
                            "position_m": (0.3, 0.0, 0.3),
                            "quaternion_wxyz": (1, 0, 0, 0),
                            "frame_id": "camera",
                        },
                        "grasp_tcp_pose": {
                            "position_m": (0.3, 0.0, 0.1),
                            "quaternion_wxyz": (1, 0, 0, 0),
                            "frame_id": "camera",
                        },
                        "score": 0.9,
                        "forward_alignment": 0.8,
                        "metadata": {"kind": "nested"},
                    },
                    actual_candidate,
                ]
            },
        )

    def test_parses_both_mapping_forms_and_sorts_scores_descending(self) -> None:
        backend = MockManipulation()
        candidates = backend.rank_grasp_candidates(self._observation())

        self.assertEqual([item.score for item in candidates], [0.9, 0.7, 0.4])
        self.assertTrue(all(isinstance(item, GraspCandidate) for item in candidates))
        self.assertTrue(
            all(
                isinstance(item.pregrasp_tcp_pose, Pose)
                and isinstance(item.grasp_tcp_pose, Pose)
                for item in candidates
            )
        )
        self.assertEqual(candidates[0].metadata["kind"], "nested")
        self.assertEqual(candidates[2].pregrasp_tcp_pose.position_m, (0.2, 0.0, 0.3))
        self.assertEqual(candidates[2].grasp_tcp_pose.position_m, (0.2, 0.0, 0.1))

    def test_filters_caps_and_does_not_mutate_options(self) -> None:
        backend = MockManipulation()
        options = {"min_score": 0.7, "max_candidates": 1}
        candidates = backend.rank_grasp_candidates(
            self._observation(),
            options=options,
        )

        self.assertEqual(options, {"min_score": 0.7, "max_candidates": 1})
        self.assertEqual([item.score for item in candidates], [0.9])
        self.assertEqual(
            backend.rank_grasp_candidates(self._observation(), max_candidates=0),
            (),
        )
        self.assertEqual(backend.rank_grasp_candidates(Observation()), ())

    def test_invalid_candidate_and_options_have_contextual_errors(self) -> None:
        backend = MockManipulation()
        malformed = Observation(
            metadata={
                "grasp_candidates": [
                    {
                        "position_m": (0, 0, 0),
                        "quaternion_wxyz": (1, 0, 0, 0),
                        "frame_id": "world",
                        "score": 0.5,
                        "vendor_object": object(),
                    }
                ]
            }
        )
        with self.assertRaisesRegex(TypeError, r"grasp_candidates\[0\].*vendor_object"):
            backend.rank_grasp_candidates(malformed)
        with self.assertRaisesRegex(TypeError, "unsupported manipulation option.*mystery"):
            backend.rank_grasp_candidates(Observation(), mystery=True)
        with self.assertRaisesRegex(TypeError, "provided more than once.*min_score"):
            backend.rank_grasp_candidates(
                Observation(),
                options={"min_score": 0.2},
                min_score=0.3,
            )
        with self.assertRaises(TypeError):
            backend.rank_grasp_candidates(Observation(), min_score="high")
        with self.assertRaises(ValueError):
            backend.rank_grasp_candidates(Observation(), max_candidates=-1)

    def test_close_is_idempotent_and_closed_backend_cannot_run(self) -> None:
        class CountingMockManipulation(MockManipulation):
            def __init__(self) -> None:
                super().__init__()
                self.close_count = 0

            def _close(self) -> None:
                self.close_count += 1

        backend = CountingMockManipulation()
        backend.close()
        backend.close()

        self.assertTrue(backend.closed)
        self.assertEqual(backend.close_count, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            backend.rank_grasp_candidates(Observation())

        class FailingCloseManipulation(MockManipulation):
            def __init__(self) -> None:
                super().__init__()
                self.close_count = 0

            def _close(self) -> None:
                self.close_count += 1
                raise RuntimeError("partial manipulation cleanup failure")

        failing_close = FailingCloseManipulation()
        with self.assertRaisesRegex(RuntimeError, "partial manipulation cleanup"):
            failing_close.close()
        failing_close.close()
        self.assertTrue(failing_close.closed)
        self.assertEqual(failing_close.close_count, 1)

    def test_failure_and_factory_errors_are_explicit(self) -> None:
        failing = MockManipulation(
            fail=True,
            failure_message="planner unavailable",
        )
        with self.assertRaisesRegex(RuntimeError, "planner unavailable"):
            failing.rank_grasp_candidates(Observation())

        config = {"type": "mock", "metadata_key": "grasps"}
        backend = build_manipulation(config)
        self.assertIsInstance(backend, BaseManipulation)
        self.assertEqual(config, {"type": "mock", "metadata_key": "grasps"})

        for bad_config in (None, [], "mock"):
            with self.subTest(config=bad_config):
                with self.assertRaisesRegex(TypeError, "config must be a mapping"):
                    build_manipulation(bad_config)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "missing required key 'type'"):
            build_manipulation({})
        for bad_type in (None, 7, "   "):
            with self.subTest(manipulation_type=bad_type):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    build_manipulation({"type": bad_type})  # type: ignore[dict-item]
        with self.assertRaisesRegex(ValueError, "unknown manipulation type.*mock"):
            build_manipulation({"type": "not-registered"})
        with self.assertRaisesRegex(
            TypeError, "invalid options for manipulation type 'mock'"
        ):
            build_manipulation({"type": "mock", "unknown_option": True})

        with patch.dict(MANIPULATION_REGISTRY, {"invalid-return": lambda: object()}):
            with self.assertRaisesRegex(TypeError, "expected BaseManipulation"):
                build_manipulation({"type": "invalid-return"})


if __name__ == "__main__":
    unittest.main()
