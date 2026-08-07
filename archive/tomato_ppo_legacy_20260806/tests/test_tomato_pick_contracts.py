"""Golden tests for the frozen C1 robot and policy interfaces."""

from __future__ import annotations

from dataclasses import replace
import unittest

from module.robotics import (
    ControlBusyError,
    ControlLeaseManager,
    ManualClock,
    MockHandSynergyMapper,
    build_robotics_bundle,
)
from module.skills.tomato_pick import (
    ACTION_DIM,
    OBSERVATION_DIM,
    PolicyManifest,
    TactileFeatureExtractor,
    TomatoPickAction,
    TomatoPickObservation,
    contact_ready,
    grasp_ready,
    supervise_action_values,
)
from module.skills.tomato_pick.geometry import pose_error_in_grasp_frame, twist_in_grasp_frame
from proto import ArmSide, FingerTactile, HandSynergy, PickPhase, Pose, TactileFrame


class FrozenPolicyContractTests(unittest.TestCase):
    def test_progress_labels_never_rewrite_policy_actions(self) -> None:
        action = tuple((-0.8 + 1.6 * index / 21.0) for index in range(22))
        for phase in PickPhase:
            self.assertEqual(supervise_action_values(action, phase), action)
        with self.assertRaises(ValueError):
            supervise_action_values((0.0,) * 21 + (1.01,), PickPhase.CONTACT)

    def test_deployable_progress_checks_are_available_for_metrics(self) -> None:
        support = (0.0, 0.6, 0.6, 0.0, 0.0)
        contact_only_enclosure = (0.6, 0.6, 0.6, 0.0, 0.0)
        load_bearing_enclosure = (4.5, 1.0, 1.0, 0.0, 0.0)
        self.assertTrue(contact_ready(support))
        self.assertFalse(contact_ready((0.8, 0.0, 0.0, 0.0, 0.0)))
        self.assertFalse(grasp_ready(support, closure=0.5))
        self.assertFalse(grasp_ready(contact_only_enclosure, closure=0.5))
        self.assertFalse(grasp_ready(load_bearing_enclosure, closure=0.14))
        self.assertTrue(grasp_ready(load_bearing_enclosure, closure=0.15))
        self.assertFalse(
            grasp_ready(
                load_bearing_enclosure,
                closure=0.15,
                thumb_opposition=-0.16,
            )
        )
        self.assertFalse(
            grasp_ready(
                load_bearing_enclosure,
                closure=0.15,
                finger_spread=0.33,
            )
        )

    def test_pose_error_and_twist_use_grasp_frame_spatial_convention(self) -> None:
        root_half = 2.0**-0.5
        grasp = Pose((0.0, 0.0, 0.0), (root_half, 0.0, 0.0, root_half), "world")
        reference = Pose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), "world")
        current = Pose(
            (1.0, 0.0, 0.0),
            (0.9950041652780258, 0.09983341664682815, 0.0, 0.0),
            "world",
        )
        error = pose_error_in_grasp_frame(
            reference=reference,
            current=current,
            grasp_frame=grasp,
        )
        expected = (0.0, -1.0, 0.0, 0.0, -0.2, 0.0)
        for actual, wanted in zip(error, expected):
            self.assertAlmostEqual(actual, wanted, places=6)
        twist = twist_in_grasp_frame((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), grasp_frame=grasp)
        for actual, wanted in zip(twist, (0.0, -1.0, 0.0, 1.0, 0.0, 0.0)):
            self.assertAlmostEqual(actual, wanted, places=6)

    def test_observation_golden_layout_is_exactly_113_values(self) -> None:
        observation = TomatoPickObservation.pack(
            target_pose_error=(1,) * 6,
            ee_twist=(2,) * 6,
            arm_joint_position_norm=(3,) * 7,
            arm_joint_velocity_norm=(4,) * 7,
            hand_joint_position_norm=(5,) * 16,
            hand_joint_velocity_norm=(6,) * 16,
            finger_pressure_norm=(7,) * 5,
            finger_pressure_rate_norm=(8,) * 5,
            gravity_direction=(9,) * 3,
            previous_action=(10,) * 22,
            target_arm_hand_valid=(11,) * 3,
            force_valid=(12,) * 5,
            target_arm_hand_age_norm=(13,) * 3,
            force_age_norm=(14,) * 5,
            fruit_axes_norm=(15,) * 3,
            episode_progress=16,
        )
        self.assertEqual(OBSERVATION_DIM, 113)
        self.assertEqual(len(observation.values), 113)
        expected = (
            (0, 6, 1.0), (6, 12, 2.0), (12, 19, 3.0),
            (19, 26, 4.0), (26, 42, 5.0), (42, 58, 6.0),
            (58, 63, 7.0), (63, 68, 8.0), (68, 71, 9.0),
            (71, 93, 10.0), (93, 96, 11.0), (96, 101, 12.0),
            (101, 104, 13.0), (104, 109, 14.0), (109, 112, 15.0),
            (112, 113, 16.0),
        )
        for start, stop, value in expected:
            self.assertEqual(observation.block(start, stop), (value,) * (stop - start))

    def test_action_and_manifest_reject_interface_drift(self) -> None:
        self.assertEqual(ACTION_DIM, 22)
        self.assertEqual(TomatoPickAction.zero().values, (0.0,) * 22)
        with self.assertRaisesRegex(ValueError, "exactly 22"):
            TomatoPickAction((0.0,) * 21)
        with self.assertRaisesRegex(ValueError, r"\[-1,1\]"):
            TomatoPickAction((0.0,) * 21 + (1.1,))
        manifest = PolicyManifest.mock_default()
        self.assertEqual(manifest.policy_rate_hz, 20)
        self.assertEqual(manifest.phase_step_limits, (30, 30, 40, 20))
        self.assertEqual(manifest.fruit_axis_lower_m, (0.025, 0.025, 0.025))
        self.assertEqual(manifest.fruit_axis_upper_m, (0.035, 0.035, 0.035))
        with self.assertRaisesRegex(ValueError, "observation_layout_id"):
            replace(manifest, observation_layout_id="changed")


class RobotShapeContractTests(unittest.TestCase):
    def test_actor_tactile_uses_only_shared_pressure_and_rate(self) -> None:
        finger = FingerTactile(
            normal_force_n=4.0,
            tangential_force_n=8.0,
            tangential_direction_raw=127.0,
            matrix_weight_raw=((255,) * 6,) * 12,
            force_sample_monotonic_ns=1,
            matrix_sample_monotonic_ns=1,
            force_valid=True,
            matrix_valid=True,
        )
        frame = TactileFrame(
            side=ArmSide.RIGHT,
            fingers=(finger,) * 5,
            calibration_id="test",
            feature_extractor_id="g20_pressure_features",
            received_monotonic_ns=1,
            source_sequence=0,
        )
        pressure, pressure_rate, *health = TactileFeatureExtractor().extract(
            frame, now_ns=1, manifest=PolicyManifest.mock_default()
        )
        self.assertEqual(pressure, (0.2,) * 5)
        self.assertEqual(pressure_rate, (0.2,) * 5)
        self.assertTrue(health[-1])

    def test_arm_hand_tactile_shapes_match_linkbot_and_g20(self) -> None:
        bundle = build_robotics_bundle({"type": "mock"}, side=ArmSide.RIGHT)
        try:
            arm = bundle.arm.get_state(ArmSide.RIGHT)
            hand = bundle.hand.get_state(ArmSide.RIGHT)
            tactile = bundle.tactile.get_frame(ArmSide.RIGHT)
            self.assertEqual(len(arm.joint_positions_rad), 7)
            self.assertEqual(len(arm.joint_velocities_rad_s), 7)
            self.assertEqual(len(hand.logical_raw or ()), 20)
            self.assertEqual(len(hand.joint_positions_rad), 16)
            self.assertEqual(len(hand.joint_velocities_rad_s), 16)
            self.assertEqual(len(tactile.fingers), 5)
            self.assertTrue(all(len(finger.matrix_weight_raw or ()) == 12 for finger in tactile.fingers))
            self.assertTrue(
                all(len(row) == 6 for finger in tactile.fingers for row in (finger.matrix_weight_raw or ()))
            )
            with self.assertRaisesRegex(ValueError, "exactly 7"):
                replace(arm, joint_positions_rad=(0.0,) * 6)
            with self.assertRaisesRegex(ValueError, "exactly 20"):
                replace(hand, logical_raw=(255,) * 19)
            with self.assertRaisesRegex(ValueError, "12x6"):
                FingerTactile(matrix_weight_raw=((0,) * 6,) * 11)
        finally:
            bundle.close()

    def test_synergy_mapping_preserves_hardware_and_simulator_counts(self) -> None:
        mapper = MockHandSynergyMapper()
        synergy = HandSynergy(0.5, 0.25, -0.20)
        logical = mapper.to_g20_logical(ArmSide.RIGHT, synergy)
        simulated = mapper.to_sim_independent(ArmSide.RIGHT, synergy)
        self.assertEqual(len(logical), 20)
        self.assertEqual(logical[11:15], (255, 255, 255, 255))
        self.assertEqual(len(simulated), 16)
        restored = mapper.from_g20_state(ArmSide.RIGHT, logical)
        self.assertAlmostEqual(restored.closure, synergy.closure, places=2)

    def test_control_lease_is_exclusive_and_expires(self) -> None:
        clock = ManualClock()
        leases = ControlLeaseManager(clock)
        first = leases.acquire(owner_id="one", side=ArmSide.RIGHT, ttl_s=0.5)
        with self.assertRaises(ControlBusyError):
            leases.acquire(owner_id="two", side=ArmSide.RIGHT, ttl_s=0.5)
        clock.advance_s(0.5)
        second = leases.acquire(owner_id="two", side=ArmSide.RIGHT, ttl_s=0.5)
        self.assertNotEqual(first.lease_id, second.lease_id)
        leases.release(second)
        self.assertIsNone(leases.active(ArmSide.RIGHT))


if __name__ == "__main__":
    unittest.main()
