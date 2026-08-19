from __future__ import annotations

import copy
import unittest

import torch

from simulation.ppo_tomato_grasp.tactile_dataset_contract import summarize_episode, validate_episode


def valid_fixture(samples: int = 20):
    fields = {
        "timestamp_s": torch.arange(samples).float() * 0.02,
        "joint_position_rad": torch.zeros((samples, 16)),
        "joint_velocity_rad_s": torch.zeros((samples, 16)),
        "tactile_raw": torch.zeros((samples, 5, 12, 6)),
        "tactile_valid": torch.ones((samples, 5), dtype=torch.bool),
        "sample_age_s": torch.zeros((samples, 5)),
        "object_pose_in_palm_xyzw": torch.tensor((0, 0, 0, 0, 0, 0, 1)).float().expand(samples, -1).clone(),
        "object_pose_valid": torch.ones(samples, dtype=torch.bool),
        "reference_force_n": torch.zeros((samples, 6)),
        "reference_force_valid": torch.zeros((samples, 6), dtype=torch.bool),
        "phase_code": torch.arange(samples, dtype=torch.int64) % 6,
    }
    metadata = {
        "schema": "left_g20_tactile_episode_v1",
        "episode_id": "synthetic-001",
        "hand": "left_g20",
        "timestamp_source": "synthetic_monotonic",
        "tactile_unit": "raw_count",
        "joint_position_unit": "rad",
        "joint_velocity_unit": "rad/s",
        "reference_force_unit": "N",
        "object_description": "synthetic fixture",
        "operator_supervised": False,
    }
    return fields, metadata


class TactileDatasetContractTests(unittest.TestCase):
    def test_valid_episode_summarizes_rate_and_quality(self):
        fields, metadata = valid_fixture()
        summary = summarize_episode(fields, metadata)
        self.assertAlmostEqual(summary["median_sample_rate_hz"], 50.0, places=3)
        self.assertEqual(summary["samples"], 20)
        self.assertFalse(summary["hardware_release_allowed"])

    def test_nonmonotonic_time_bad_quaternion_and_wrong_mask_fail_closed(self):
        fields, metadata = valid_fixture()
        bad = copy.deepcopy(fields)
        bad["timestamp_s"][4] = bad["timestamp_s"][3]
        with self.assertRaises(ValueError):
            validate_episode(bad, metadata)
        bad = copy.deepcopy(fields)
        bad["object_pose_in_palm_xyzw"][:, 3:] = 0.0
        with self.assertRaises(ValueError):
            validate_episode(bad, metadata)
        bad = copy.deepcopy(fields)
        bad["tactile_valid"] = bad["tactile_valid"].float()
        with self.assertRaises(ValueError):
            validate_episode(bad, metadata)

    def test_metadata_units_and_extra_fields_fail_closed(self):
        fields, metadata = valid_fixture()
        bad_metadata = dict(metadata)
        bad_metadata["joint_position_unit"] = "degree"
        with self.assertRaises(ValueError):
            validate_episode(fields, bad_metadata)
        bad_fields = dict(fields)
        bad_fields["unexpected"] = torch.zeros(20)
        with self.assertRaises(ValueError):
            validate_episode(bad_fields, metadata)


if __name__ == "__main__":
    unittest.main()
