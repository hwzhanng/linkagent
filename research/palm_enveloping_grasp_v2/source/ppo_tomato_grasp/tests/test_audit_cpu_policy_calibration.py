from __future__ import annotations

import math
import unittest

import torch

from simulation.ppo_tomato_grasp.audit_cpu_policy_calibration import perturb_observation
from simulation.ppo_tomato_grasp.hand_manifold import LEFT_SAFE_LOWER_RAD, LEFT_SAFE_UPPER_RAD
from simulation.ppo_tomato_grasp.hand_ppo_contract import ACTOR_OBSERVATION_BLOCKS


class CalibrationPerturbationTests(unittest.TestCase):
    def test_joint_bias_matches_two_degrees_in_normalized_coordinates(self):
        observation = torch.zeros((1, 110))
        changed = perturb_observation(observation, "joint_zero_bias_positive_2deg")
        section = ACTOR_OBSERVATION_BLOCKS["hand_joint_position"]
        span = torch.tensor(LEFT_SAFE_UPPER_RAD) - torch.tensor(LEFT_SAFE_LOWER_RAD)
        expected = (2.0 * math.radians(2.0) / span).clamp(-1.0, 1.0)
        self.assertTrue(torch.allclose(changed[0, section], expected))

    def test_tactile_gain_changes_load_fields_only(self):
        observation = torch.full((1, 110), 0.2)
        changed = perturb_observation(observation, "tactile_gain_1p30")
        tactile = changed[:, ACTOR_OBSERVATION_BLOCKS["tactile_matrix_features"]]
        for index in range(50):
            expected = 0.26 if index % 10 in (0, 1, 7) else 0.2
            self.assertAlmostEqual(float(tactile[0, index]), expected, places=6)

    def test_unknown_scenario_fails_closed(self):
        with self.assertRaises(ValueError):
            perturb_observation(torch.zeros((1, 110)), "unknown")

    def test_extended_biases_have_expected_relative_magnitudes(self):
        source = torch.zeros((1, 110))
        joint_2 = perturb_observation(source, "joint_zero_bias_positive_2deg")
        joint_5 = perturb_observation(source, "joint_zero_bias_positive_5deg")
        section = ACTOR_OBSERVATION_BLOCKS["hand_joint_position"]
        self.assertTrue(torch.all(joint_5[:, section] >= joint_2[:, section]))
        self.assertTrue(torch.any(joint_5[:, section] > joint_2[:, section]))
        vision_5 = perturb_observation(source, "vision_center_bias_x_5mm")
        vision_20 = perturb_observation(source, "vision_center_bias_x_20mm")
        center = ACTOR_OBSERVATION_BLOCKS["fruit_center_in_palm"]
        self.assertAlmostEqual(
            float(vision_20[:, center.start]),
            4.0 * float(vision_5[:, center.start]),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
