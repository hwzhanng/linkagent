from __future__ import annotations

import math
import unittest

import numpy as np

from tools.local_arm_can.advanced_controller import AdvancedArmController
from tools.local_arm_can.kinematics import ArmKinematics
from tools.local_arm_can.trajectory import sample_movej, sample_smooth_movej


class KinematicsTests(unittest.TestCase):
    def test_fk_matches_live_pi_right_zero(self) -> None:
        pose = ArmKinematics("right").forward([0.0] * 7)
        np.testing.assert_allclose(
            pose.position,
            [-3.91e-7, -0.222001, -0.6082],
            atol=2.0e-7,
        )
        np.testing.assert_allclose(pose.rpy, [0.0, 0.0, 0.0], atol=1.0e-10)

    def test_fk_matches_live_pi_left_zero(self) -> None:
        pose = ArmKinematics("left").forward([0.0] * 7)
        np.testing.assert_allclose(
            pose.position,
            [-1.582e-6, 0.222, -0.6082],
            atol=2.0e-7,
        )

    def test_fk_matches_live_pi_nontrivial_right(self) -> None:
        # This out-of-limit vector was sent only to the Pi's read-only FK
        # service.  It is useful because it exercises every axis/sign.
        pose = ArmKinematics("right").forward([10, -10, 15, 20, -15, 10, -5])
        np.testing.assert_allclose(
            pose.position,
            [0.0657892105, -0.1544815514, -0.3161363166],
            atol=2.0e-8,
        )
        np.testing.assert_allclose(
            pose.rpy,
            [-2.106149844, -0.291948881, 0.275955114],
            atol=2.0e-8,
        )

    def test_ik_round_trip(self) -> None:
        kinematics = ArmKinematics("right")
        expected = [0.12, -0.35, 0.18, -0.42, 0.15, 0.2, -0.12]
        pose = kinematics.forward(expected)
        result = kinematics.inverse(pose.position, pose.rpy, [0.0] * 7)
        self.assertTrue(result.success)
        recovered_pose = kinematics.forward(result.joints)
        np.testing.assert_allclose(recovered_pose.position, pose.position, atol=2.0e-5)
        self.assertLess(result.rotation_error, 2.0e-4)

    def test_ik_rejects_unreachable_pose(self) -> None:
        result = ArmKinematics("right").inverse((5.0, 5.0, 5.0), (0.0, 0.0, 0.0), [0.0] * 7)
        self.assertFalse(result.success)


class TrajectoryTests(unittest.TestCase):
    def test_movej_uses_official_defaults_and_100_hz(self) -> None:
        samples = sample_movej([0.0] * 7, [0.1] + [0.0] * 6, 0.0, 0.0)
        self.assertEqual(samples[0].time, 0.0)
        self.assertAlmostEqual(samples[1].time, 0.01)
        self.assertAlmostEqual(samples[-1].positions[0], 0.1)
        self.assertTrue(all(math.isclose(sample.positions[1], 0.0) for sample in samples))

    def test_movej_short_axes_hold_at_target(self) -> None:
        samples = sample_movej([0.0] * 7, [0.5, 0.1] + [0.0] * 5, 0.5, 0.5)
        short_at_goal = [sample for sample in samples if math.isclose(sample.positions[1], 0.1)]
        self.assertGreater(len(short_at_goal), 1)
        self.assertAlmostEqual(samples[-1].positions[0], 0.5)

    def test_smooth_profile_has_zero_endpoint_velocity(self) -> None:
        samples = sample_smooth_movej([0.0] * 7, [0.2] * 7, 1.0)
        np.testing.assert_allclose(samples[0].velocities, [0.0] * 7, atol=1.0e-12)
        np.testing.assert_allclose(samples[-1].velocities, [0.0] * 7, atol=1.0e-12)
        np.testing.assert_allclose(samples[-1].positions, [0.2] * 7, atol=1.0e-12)


class _ReadOnlyRuntime:
    arm = "right"

    def __init__(self) -> None:
        self.commands: list[list[float]] = []

    def snapshot(self) -> dict:
        return {
            "tx_enabled": False,
            "software_enabled": False,
            "all_joints_healthy": True,
            "healthy_joint_count": 7,
            "joints": [{"position": 0.0} for _ in range(7)],
        }

    def require_physical_confirmation(self, _value: str) -> None:
        raise AssertionError("read-only preflight must fail before area confirmation")

    def command_positions(self, positions: list[float], *_args: object) -> None:
        self.commands.append(positions)


class _EnabledRuntime(_ReadOnlyRuntime):
    def snapshot(self) -> dict:
        result = super().snapshot()
        result.update(tx_enabled=True, software_enabled=True)
        return result

    def require_physical_confirmation(self, value: str) -> None:
        if value != "RIGHT_ARM_AREA_CLEAR":
            raise PermissionError("missing physical confirmation")


class AdvancedSafetyTests(unittest.TestCase):
    def test_read_only_controller_never_dispatches_motion(self) -> None:
        runtime = _ReadOnlyRuntime()
        controller = AdvancedArmController(runtime)  # type: ignore[arg-type]
        with self.assertRaises(PermissionError):
            controller.movej([0.01] * 7, 0.1, 0.2, True, "RIGHT_ARM_AREA_CLEAR")
        self.assertEqual(runtime.commands, [])

    def test_target_limit_is_checked_before_dispatch(self) -> None:
        runtime = _ReadOnlyRuntime()
        controller = AdvancedArmController(runtime)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "software limits"):
            controller.movej([9.0] * 7, 0.1, 0.2, True, "RIGHT_ARM_AREA_CLEAR")
        self.assertEqual(runtime.commands, [])

    def test_can_dispatch_is_limited_to_50_hz_and_keeps_exact_final_target(self) -> None:
        runtime = _EnabledRuntime()
        controller = AdvancedArmController(runtime)  # type: ignore[arg-type]
        planned = sample_movej([0.0] * 7, [0.01] + [0.0] * 6, 0.5, 0.5)
        self.assertTrue(
            controller.movej(
                [0.01] + [0.0] * 6,
                0.5,
                0.5,
                True,
                "RIGHT_ARM_AREA_CLEAR",
            )
        )
        self.assertGreater(len(runtime.commands), 1)
        self.assertLessEqual(len(runtime.commands), math.ceil(len(planned) / 2) + 1)
        self.assertAlmostEqual(runtime.commands[-1][0], 0.01)


if __name__ == "__main__":
    unittest.main()
