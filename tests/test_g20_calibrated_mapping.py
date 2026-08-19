from __future__ import annotations

from pathlib import Path
import unittest

from module.robotics.g20_calibrated import CalibratedG20Mapper
from module.skills.tomato_pick.hand_synergy import (
    PREGRASP_SYNERGY,
    g20_synergy_to_joint_targets,
)
from proto import ArmSide, HandSynergy


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/robotics/g20_motion_calibration_20260726.json"


class CalibratedG20MapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = CalibratedG20Mapper.from_file(PROFILE)

    def test_left_verified_endpoints_and_reserved_channels(self) -> None:
        low_raw = (50, 15, 5, 5, 15, 247, 185, 140, 97, 42, 235, 75, 5, 5, 10, 5)
        closed = (0.675294, 1.562353, 1.627451, 1.627451, 1.562353,
                  -1.001686, -0.117255, -0.02549, 0.062196, 0.174353,
                  0.113725, 0.931765, 1.333333, 1.333333, 1.306667, 1.333333)
        logical = self.mapper.to_g20_joint_positions(ArmSide.LEFT, closed)
        self.assertEqual(low_raw[:11], logical[:11])
        self.assertEqual((255, 255, 255, 255), logical[11:15])
        self.assertEqual(low_raw[11:], logical[15:])

    def test_right_mirrors_only_yaw_and_roll_joint_signs(self) -> None:
        left_open = self.mapper.to_sim_independent(
            ArmSide.LEFT, HandSynergy(0.0, 0.0, 0.0)
        )
        right_open = self.mapper.to_sim_independent(
            ArmSide.RIGHT, HandSynergy(0.0, 0.0, 0.0)
        )
        self.assertEqual(left_open[:5], right_open[:5])
        self.assertEqual(left_open[11:], right_open[11:])
        for left, right in zip(left_open[5:11], right_open[5:11]):
            self.assertAlmostEqual(left, -right)

    def test_out_of_calibrated_command_range_is_rejected(self) -> None:
        open_pose = list(self.mapper.to_sim_independent(
            ArmSide.LEFT, HandSynergy(0.0, 0.0, 0.0)
        ))
        open_pose[0] = 0.8
        with self.assertRaisesRegex(ValueError, "outside verified range"):
            self.mapper.to_g20_joint_positions(ArmSide.LEFT, tuple(open_pose))

    def test_non_neutral_synergy_is_not_invented(self) -> None:
        with self.assertRaisesRegex(ValueError, "use 16D joint positions"):
            self.mapper.to_g20_logical(
                ArmSide.LEFT, HandSynergy(0.5, 0.0, 0.0)
            )

    def test_current_clean_v5_pregrasp_exposes_unverified_yaw_roll_gap(self) -> None:
        target = g20_synergy_to_joint_targets(PREGRASP_SYNERGY)
        with self.assertRaisesRegex(ValueError, "outside verified range"):
            self.mapper.to_g20_joint_positions(ArmSide.RIGHT, target)


if __name__ == "__main__":
    unittest.main()
