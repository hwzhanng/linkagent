"""Independent verifier gates for Isaac/recorded evidence."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from module.robotics import MockPickScenario, build_mock_robotics_bundle
from proto import ArmSide, EvidenceState, Pose, TomatoGraspPlan, TomatoTarget
from task.agri_pick.verification import (
    IndependentPickVerifier,
    InMemoryPickEvidenceSource,
    PickEvidenceSnapshot,
    PickVerifierThresholds,
    RecordedPickEvidenceSource,
)


def _target_plan(now: int) -> tuple[TomatoTarget, TomatoGraspPlan]:
    pose = Pose((0.4, -0.2, 0.7), (1.0, 0.0, 0.0, 0.0), "world")
    target = TomatoTarget(
        target_id="tomato-1",
        center_pose=pose,
        semi_axes_m=(0.03, 0.03, 0.035),
        ripeness_score=0.9,
        pick_confidence=0.9,
        occlusion=0.1,
        stem_direction_xyz=(0.0, 0.0, -1.0),
        observation_sequence=1,
        sample_monotonic_ns=max(1, now),
    )
    plan = TomatoGraspPlan(
        grasp_plan_id="plan-1",
        target_id=target.target_id,
        source_observation_sequence=1,
        arm_side=ArmSide.RIGHT,
        grasp_frame=pose,
        pregrasp_pose=pose,
        contact_pose=pose,
        retreat_pose=pose,
        max_residual_translation_m=(0.02, 0.02, 0.04),
        max_residual_rotation_rad=(0.6, 0.6, 1.2),
        mapping_calibration_id="g20_eigengrasp_5d_v3_thumb_roll_fixed",
        planner_version="test",
        expires_monotonic_ns=max(2, now + 2_000_000_000),
    )
    return target, plan


class IndependentVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = build_mock_robotics_bundle(
            side=ArmSide.RIGHT,
            scenario=MockPickScenario(),
        )
        self.bundle.world.contact = True
        self.bundle.world.grasped = True
        self.thresholds = PickVerifierThresholds(
            "isaac_recording_v1",
            "mock_tactile_calibration_v1",
            stable_hold_s=1.0,
        )

    def tearDown(self) -> None:
        self.bundle.close()

    def test_success_requires_external_detach_and_full_stable_window(self) -> None:
        samples = [
            PickEvidenceSnapshot(1, 1, False, False, None, "isaac"),
            PickEvidenceSnapshot(500_000_001, 2, True, False, True, "isaac"),
            PickEvidenceSnapshot(1_000_000_001, 3, True, False, True, "isaac"),
            PickEvidenceSnapshot(1_500_000_001, 4, True, False, True, "isaac"),
        ]
        verifier = IndependentPickVerifier(
            InMemoryPickEvidenceSource(samples, source_id="isaac"), self.thresholds
        )
        target, plan = _target_plan(1)
        session = verifier.begin("exec-1", target, plan)
        states = []
        for _ in samples:
            states.append(
                verifier.update(
                    session,
                    None,
                    self.bundle.arm.get_state(ArmSide.RIGHT),
                    self.bundle.hand.get_state(ArmSide.RIGHT),
                    self.bundle.tactile.get_frame(ArmSide.RIGHT),
                )
            )
        self.assertIs(states[0].detached, EvidenceState.UNKNOWN)
        self.assertFalse(states[2].successful)
        self.assertTrue(states[3].successful)
        self.assertIn("isaac", states[3].evidence_sources)
        self.assertTrue(verifier.finish(session).successful)

    def test_drop_and_slip_are_rejected(self) -> None:
        dropped = PickEvidenceSnapshot(1, 1, True, True, False, "recording")
        verifier = IndependentPickVerifier(InMemoryPickEvidenceSource([dropped]), self.thresholds)
        target, plan = _target_plan(1)
        session = verifier.begin("exec-drop", target, plan)
        result = verifier.update(
            session,
            None,
            self.bundle.arm.get_state(ArmSide.RIGHT),
            self.bundle.hand.get_state(ArmSide.RIGHT),
            self.bundle.tactile.get_frame(ArmSide.RIGHT),
        )
        self.assertTrue(result.dropped)
        self.assertIs(result.retained_without_slip, EvidenceState.REJECTED)
        self.assertFalse(result.successful)

    def test_calibrated_slip_and_excess_force_are_independently_rejected(self) -> None:
        detached = PickEvidenceSnapshot(1, 1, True, False, True, "isaac")
        for outcome, expected_reason in (
            ("slip", "slip"),
            ("excess_force", "exceeded limit"),
        ):
            with self.subTest(outcome=outcome):
                bundle = build_mock_robotics_bundle(
                    side=ArmSide.RIGHT,
                    scenario=MockPickScenario(outcome=outcome),
                )
                try:
                    bundle.world.contact = True
                    bundle.world.grasped = True
                    bundle.world.slip = outcome == "slip"
                    verifier = IndependentPickVerifier(
                        InMemoryPickEvidenceSource([detached]), self.thresholds
                    )
                    target, plan = _target_plan(1)
                    session = verifier.begin(f"exec-{outcome}", target, plan)
                    result = verifier.update(
                        session,
                        None,
                        bundle.arm.get_state(ArmSide.RIGHT),
                        bundle.hand.get_state(ArmSide.RIGHT),
                        bundle.tactile.get_frame(ArmSide.RIGHT),
                    )
                    self.assertIs(result.retained_without_slip, EvidenceState.REJECTED)
                    self.assertIn(expected_reason, result.reason or "")
                    self.assertFalse(result.successful)
                finally:
                    bundle.close()

    def test_recorded_evidence_file_uses_the_same_verifier_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "isaac_evidence.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_id": "isaac_recording",
                        "samples": [
                            {
                                "sample_monotonic_ns": 1,
                                "source_sequence": 1,
                                "detached": True,
                                "dropped": False,
                                "retained": True,
                                "source_id": "isaac_recording",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            source = RecordedPickEvidenceSource(path)
            source.begin("exec-recorded", "tomato-1")
            sample = source.sample("exec-recorded")
            self.assertTrue(sample.detached)
            self.assertEqual(source.source_id, "isaac_recording")
            source.finish("exec-recorded")

    def test_unvalidated_real_threshold_profile_refuses_release(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "lacks real-data validation"):
            self.thresholds.require_hardware_validated()


if __name__ == "__main__":
    unittest.main()
