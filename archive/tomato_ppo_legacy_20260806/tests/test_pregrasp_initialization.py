from __future__ import annotations

import unittest

from simulation.tomato_pick.initialization import (
    LEFT_ARM_PARK_JOINTS_RAD,
    PREGRASP_RESET_SPEC,
    RIGHT_ARM_PREGRASP_JOINTS_RAD,
    PreGraspResetSpec,
    minimum_joint_limit_margin_fraction,
)


class PreGraspInitializationTests(unittest.TestCase):
    def test_nominal_arm_poses_are_explicit_bent_seven_joint_states(self) -> None:
        self.assertEqual(len(RIGHT_ARM_PREGRASP_JOINTS_RAD), 7)
        self.assertEqual(len(LEFT_ARM_PARK_JOINTS_RAD), 7)
        self.assertNotEqual(RIGHT_ARM_PREGRASP_JOINTS_RAD, (0.0,) * 7)
        self.assertNotEqual(RIGHT_ARM_PREGRASP_JOINTS_RAD[3], 0.0)
        self.assertNotEqual(RIGHT_ARM_PREGRASP_JOINTS_RAD[5], 0.0)
        self.assertEqual(LEFT_ARM_PARK_JOINTS_RAD[0], -RIGHT_ARM_PREGRASP_JOINTS_RAD[0])
        self.assertEqual(LEFT_ARM_PARK_JOINTS_RAD[1], -RIGHT_ARM_PREGRASP_JOINTS_RAD[1])
        self.assertEqual(LEFT_ARM_PARK_JOINTS_RAD[3:], RIGHT_ARM_PREGRASP_JOINTS_RAD[3:])

    def test_nominal_arm_pose_keeps_ten_percent_hard_limit_margin(self) -> None:
        right_lower = (-2.91, -3.20, -2.69, -2.05, -2.69, -1.59, -1.59)
        right_upper = (2.91, 0.07, 2.69, 2.05, 2.69, 1.59, 1.59)
        left_lower = (-2.91, -0.07, -2.69, -2.05, -2.69, -1.59, -1.59)
        left_upper = (2.91, 3.20, 2.69, 2.05, 2.69, 1.59, 1.59)
        self.assertGreaterEqual(
            minimum_joint_limit_margin_fraction(
                RIGHT_ARM_PREGRASP_JOINTS_RAD, right_lower, right_upper
            ),
            PREGRASP_RESET_SPEC.minimum_arm_joint_limit_margin_fraction,
        )
        self.assertGreaterEqual(
            minimum_joint_limit_margin_fraction(
                LEFT_ARM_PARK_JOINTS_RAD, left_lower, left_upper
            ),
            PREGRASP_RESET_SPEC.minimum_arm_joint_limit_margin_fraction,
        )

    def test_pregrasp_closure_opens_as_fruit_radius_increases(self) -> None:
        spec = PREGRASP_RESET_SPEC
        small = spec.closure_for_lateral_radius(spec.small_fruit_lateral_radius_m)
        middle = spec.closure_for_lateral_radius(
            (spec.small_fruit_lateral_radius_m + spec.large_fruit_lateral_radius_m) / 2.0
        )
        large = spec.closure_for_lateral_radius(spec.large_fruit_lateral_radius_m)
        self.assertGreater(small, middle)
        self.assertGreater(middle, large)
        self.assertAlmostEqual(small, spec.small_fruit_closure)
        self.assertAlmostEqual(large, spec.large_fruit_closure)

    def test_invalid_reset_contract_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PreGraspResetSpec(maximum_reset_steps=4)

if __name__ == "__main__":
    unittest.main()
