from __future__ import annotations

import unittest

import torch

from simulation.ppo_tomato_grasp.stage0_logic import (
    enclosure_reward,
    normalized_contact_loads,
    proxy_tip_positions,
)


class Stage0LogicTests(unittest.TestCase):
    def setUp(self):
        self.object_pos = torch.zeros((1, 3), dtype=torch.float32)
        self.radius = 0.035

    def metrics_for(self, actions: torch.Tensor):
        tips = proxy_tip_positions(self.object_pos, actions, self.radius)
        loads, gaps = normalized_contact_loads(tips, self.object_pos, self.radius)
        reward, metrics = enclosure_reward(loads, gaps, torch.zeros_like(actions))
        return reward, loads, metrics

    def test_gentle_enclosure_beats_open_and_over_squeeze(self):
        open_reward, _, _ = self.metrics_for(torch.full((1, 6), -1.0))
        gentle_reward, _, gentle = self.metrics_for(torch.tensor([[0.5, 0.78, 0.78, 0.78, 0.78, 0.78]]))
        squeezed_reward, _, squeezed = self.metrics_for(torch.ones((1, 6)))

        self.assertGreater(gentle_reward.item(), open_reward.item())
        self.assertGreater(gentle_reward.item(), squeezed_reward.item())
        self.assertGreater(gentle_reward.item() - squeezed_reward.item(), 2.0)
        self.assertTrue(gentle["stable"].item())
        self.assertFalse(squeezed["stable"].item())
        self.assertGreater(squeezed["overload"].item(), 0.0)

    def test_each_non_thumb_finger_has_an_independent_action(self):
        baseline = torch.full((1, 6), -1.0)
        _, base_loads, _ = self.metrics_for(baseline)
        for action_index, load_index in ((2, 1), (3, 2), (4, 3), (5, 4)):
            action = baseline.clone()
            action[:, action_index] = 1.0
            _, loads, _ = self.metrics_for(action)
            self.assertGreater(loads[0, load_index].item(), base_loads[0, load_index].item())
            unchanged = [idx for idx in range(1, 5) if idx != load_index]
            self.assertTrue(torch.allclose(loads[0, unchanged], base_loads[0, unchanged]))

    def test_shape_validation_is_explicit(self):
        with self.assertRaises(ValueError):
            proxy_tip_positions(self.object_pos, torch.zeros((1, 5)), self.radius)


if __name__ == "__main__":
    unittest.main()
