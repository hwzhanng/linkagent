from __future__ import annotations

import unittest

import torch

from simulation.ppo_tomato_grasp.audit_cpu_policy_observations import alter_observation
from simulation.ppo_tomato_grasp.hand_ppo_contract import ACTOR_OBSERVATION_BLOCKS


class ObservationAblationTests(unittest.TestCase):
    def test_nominal_is_equal_but_not_aliased(self):
        observation = torch.randn((3, 110))
        altered = alter_observation(observation, "nominal")
        self.assertTrue(torch.equal(altered, observation))
        self.assertNotEqual(altered.data_ptr(), observation.data_ptr())

    def test_stale_tactile_is_explicitly_flagged(self):
        observation = torch.randn((2, 110))
        altered = alter_observation(observation, "tactile_stale")
        tactile = altered[:, ACTOR_OBSERVATION_BLOCKS["tactile_matrix_features"]]
        expected = torch.zeros_like(tactile)
        expected[:, 9::10] = 1.0
        self.assertTrue(torch.equal(tactile, expected))
        self.assertTrue(torch.equal(altered[:, :50], observation[:, :50]))

    def test_proprioception_only_preserves_exactly_two_blocks(self):
        observation = torch.randn((2, 110))
        altered = alter_observation(observation, "proprioception_only")
        expected = torch.zeros_like(observation)
        for name in ("hand_joint_position", "hand_joint_velocity"):
            section = ACTOR_OBSERVATION_BLOCKS[name]
            expected[:, section] = observation[:, section]
        self.assertTrue(torch.equal(altered, expected))

    def test_unknown_mode_fails_closed(self):
        with self.assertRaises(ValueError):
            alter_observation(torch.zeros((1, 110)), "invented")


if __name__ == "__main__":
    unittest.main()
