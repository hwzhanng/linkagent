from __future__ import annotations

import unittest

import torch

from simulation.ppo_tomato_grasp.hand_manifold import (
    LEFT_ACTIVE_JOINT_ORDER,
    LEFT_SAFE_LOWER_RAD,
    LEFT_SAFE_UPPER_RAD,
    effective_digit_closure,
    hybrid_action_to_joint_targets,
    rate_limit_joint_targets,
    reorder_joint_targets,
    synergy_to_joint_targets,
)


class HandManifoldTests(unittest.TestCase):
    def test_random_actions_remain_inside_measured_safe_ranges(self):
        generator = torch.Generator().manual_seed(20260811)
        actions = 2.0 * torch.rand((10000, 10), generator=generator) - 1.0
        targets = hybrid_action_to_joint_targets(actions)
        lower = torch.tensor(LEFT_SAFE_LOWER_RAD)
        upper = torch.tensor(LEFT_SAFE_UPPER_RAD)
        self.assertTrue(torch.all(targets >= lower - 1.0e-7))
        self.assertTrue(torch.all(targets <= upper + 1.0e-7))

    def test_five_coarse_coordinates_have_independent_local_directions(self):
        origin = torch.zeros((1, 5))
        base = synergy_to_joint_targets(origin)
        columns = []
        for index in range(5):
            perturbed = origin.clone()
            perturbed[0, index] = 1.0e-3
            columns.append(((synergy_to_joint_targets(perturbed) - base) / 1.0e-3)[0])
        jacobian = torch.stack(columns, dim=1)
        self.assertEqual(int(torch.linalg.matrix_rank(jacobian).item()), 5)

        thumb_changed = (columns[4].abs() > 1.0e-8).nonzero().flatten().tolist()
        spread_changed = (columns[3].abs() > 1.0e-8).nonzero().flatten().tolist()
        # Thumb opposition drives the two opposition axes (joints 5 and 10)
        # plus the MCP (joint 0); the IP joint (11) stays on the digit
        # residual.  Updated to match the current calibrated manifold.
        self.assertEqual(thumb_changed, [0, 5, 10])
        self.assertEqual(spread_changed, [6, 7, 8, 9])

    def test_each_tactile_residual_changes_only_one_digit(self):
        base_action = torch.zeros((1, 10))
        base = hybrid_action_to_joint_targets(base_action)
        expected = ((0, 11), (1, 12), (2, 13), (3, 14), (4, 15))
        for residual_index, joint_indices in enumerate(expected, start=5):
            action = base_action.clone()
            action[0, residual_index] = 0.5
            changed = (
                (hybrid_action_to_joint_targets(action) - base).abs() > 1.0e-8
            ).nonzero()[:, 1].tolist()
            self.assertEqual(changed, list(joint_indices))

    def test_rate_limit_is_per_joint_and_conservative(self):
        lower = torch.tensor(LEFT_SAFE_LOWER_RAD).unsqueeze(0)
        upper = torch.tensor(LEFT_SAFE_UPPER_RAD).unsqueeze(0)
        limited = rate_limit_joint_targets(
            lower, upper, policy_dt_s=0.05, velocity_scale=0.5
        )
        self.assertTrue(torch.all(limited > lower))
        self.assertTrue(torch.all(limited < upper))
        self.assertLessEqual(float((limited - lower).max()), 0.05 * 0.5 * 3.1523 + 1.0e-6)

    def test_reorder_is_name_based_and_round_trips(self):
        values = torch.arange(16.0).unsqueeze(0)
        reverse_order = tuple(reversed(LEFT_ACTIVE_JOINT_ORDER))
        reversed_values = reorder_joint_targets(values, reverse_order)
        self.assertTrue(torch.equal(reversed_values, torch.flip(values, dims=(-1,))))
        with self.assertRaises(ValueError):
            reorder_joint_targets(values, LEFT_ACTIVE_JOINT_ORDER[:-1])

    def test_effective_closure_accepts_coarse_and_hybrid_actions(self):
        coarse = effective_digit_closure(torch.zeros((3, 5)))
        hybrid = effective_digit_closure(torch.zeros((3, 10)))
        self.assertEqual(tuple(coarse.shape), (3, 5))
        self.assertTrue(torch.allclose(coarse, hybrid))
        with self.assertRaises(ValueError):
            effective_digit_closure(torch.zeros((3, 6)))


if __name__ == "__main__":
    unittest.main()
