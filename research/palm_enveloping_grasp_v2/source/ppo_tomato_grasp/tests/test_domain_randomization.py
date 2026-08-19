from __future__ import annotations

import unittest

import torch

from simulation.ppo_tomato_grasp.domain_randomization import (
    ROBUSTNESS_BUCKET_ORDER,
    pearson_correlation,
    robustness_bucket_masks,
    sample_hand_grasp_profiles,
)


class DomainRandomizationTests(unittest.TestCase):
    def sample(self, count=20000, fraction=1.0, seed=42):
        return sample_hand_grasp_profiles(
            count,
            curriculum_fraction=fraction,
            generator=torch.Generator().manual_seed(seed),
        )

    def test_sampling_is_deterministic_and_bounded(self):
        first = self.sample(count=1000, seed=9)
        second = self.sample(count=1000, seed=9)
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertTrue(torch.all((first["axes_m"] >= 0.021) & (first["axes_m"] <= 0.044)))
        self.assertTrue(torch.all((first["static_friction"] >= 0.25) & (first["static_friction"] <= 0.90)))
        self.assertTrue(torch.all(first["dynamic_friction"] < first["static_friction"]))
        self.assertTrue(torch.all((first["tactile_delay_steps"] >= 0) & (first["tactile_delay_steps"] <= 3)))
        self.assertEqual(first["tactile_failure_mask"].dtype, torch.bool)

    def test_zero_curriculum_contracts_physics_to_nominal(self):
        result = self.sample(count=1000, fraction=0.0)
        self.assertTrue(torch.allclose(result["axes_m"], torch.tensor((0.0325, 0.0325, 0.031525)).expand(1000, -1)))
        self.assertTrue(torch.all(result["static_friction"] == 0.55))
        self.assertTrue(torch.all(result["contact_stiffness_n_m"] == 5600.0))
        self.assertTrue(torch.all(result["object_position_offset_m"] == 0.0))
        self.assertTrue(torch.all(result["tactile_gain"] == 1.0))
        self.assertFalse(bool(result["tactile_failure_mask"].any()))
        self.assertTrue(torch.all(result["actuator_response_scale"] == 1.0))

    def test_correlations_follow_physical_intent(self):
        result = self.sample(count=50000, fraction=1.0, seed=2026)
        wet_friction = pearson_correlation(result["wetness"], result["static_friction"])
        ripe_stiffness = pearson_correlation(result["ripeness"], result["contact_stiffness_n_m"])
        size_mass = pearson_correlation(result["size_latent"], result["mass_kg"])
        self.assertLess(wet_friction, -0.6)
        self.assertLess(ripe_stiffness, -0.8)
        self.assertGreater(size_mass, 0.9)

    def test_all_robustness_buckets_are_nonempty_at_full_range(self):
        result = self.sample(count=50000, fraction=1.0, seed=7)
        buckets = robustness_bucket_masks(result)
        self.assertEqual(tuple(buckets), ROBUSTNESS_BUCKET_ORDER)
        for name, mask in buckets.items():
            self.assertGreater(int(mask.sum()), 100, name)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            self.sample(count=0)
        with self.assertRaises(ValueError):
            self.sample(fraction=1.1)
        with self.assertRaises(ValueError):
            sample_hand_grasp_profiles(4, curriculum_fraction=1.0, tail_focus_fraction=1.1)
        with self.assertRaises(ValueError):
            robustness_bucket_masks({})

    def test_tail_focus_increases_joint_hard_cases_without_leaving_envelope(self):
        focused = sample_hand_grasp_profiles(
            20000,
            curriculum_fraction=1.0,
            tail_focus_fraction=0.5,
            generator=torch.Generator().manual_seed(87),
        )
        mask = focused["tail_focus_mask"]
        self.assertGreater(float(mask.float().mean()), 0.48)
        self.assertLess(float(mask.float().mean()), 0.52)
        self.assertTrue(torch.all(focused["size_latent"][mask] <= 0.25))
        self.assertTrue(torch.all(focused["ripeness"][mask] >= 0.75))
        self.assertTrue(torch.all(focused["wetness"][mask] >= 0.65))
        self.assertTrue(torch.all((focused["static_friction"] >= 0.25) & (focused["static_friction"] <= 0.90)))


if __name__ == "__main__":
    unittest.main()
