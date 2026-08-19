from __future__ import annotations

import unittest

import torch

from simulation.ppo_tomato_grasp.tactile_features import (
    extract_tactile_features,
    fit_tactile_baseline,
)


class TactileFeatureTests(unittest.TestCase):
    def extract(self, matrices, **kwargs):
        batch = matrices.shape[0]
        defaults = {
            "previous_total_load": torch.zeros((batch, 5)),
            "dt_s": 0.05,
            "contact_taxel_threshold": 1.0,
            "total_load_full_scale": 100.0,
            "peak_taxel_full_scale": 20.0,
            "load_rate_full_scale_per_s": 1000.0,
        }
        defaults.update(kwargs)
        return extract_tactile_features(matrices, **defaults)

    def test_point_contact_has_expected_finger_and_centroid(self):
        matrices = torch.zeros((2, 5, 12, 6))
        matrices[0, 2, 11, 0] = 10.0
        result = self.extract(matrices)
        per_finger = result.features.reshape(2, 5, 10)
        self.assertTrue(result.contact_mask[0, 2])
        self.assertFalse(result.contact_mask[0, 1])
        self.assertAlmostEqual(float(per_finger[0, 2, 0]), 0.1, places=6)
        self.assertAlmostEqual(float(per_finger[0, 2, 3]), 1.0, places=6)
        self.assertAlmostEqual(float(per_finger[0, 2, 4]), -1.0, places=6)
        self.assertAlmostEqual(float(per_finger[0, 2, 5]), 0.0, places=6)

    def test_translation_changes_centroid_not_load(self):
        matrices = torch.zeros((2, 5, 12, 6))
        matrices[0, 0, 2:4, 1:3] = 4.0
        matrices[1, 0, 7:9, 3:5] = 4.0
        result = self.extract(matrices)
        features = result.features.reshape(2, 5, 10)
        self.assertAlmostEqual(float(features[0, 0, 0]), float(features[1, 0, 0]), places=6)
        self.assertLess(float(features[0, 0, 3]), float(features[1, 0, 3]))
        self.assertLess(float(features[0, 0, 4]), float(features[1, 0, 4]))

    def test_stale_or_invalid_is_not_silently_no_contact(self):
        matrices = torch.ones((1, 5, 12, 6)) * 5.0
        ages = torch.zeros((1, 5))
        ages[0, 1] = 0.5
        valid = torch.ones((1, 5), dtype=torch.bool)
        valid[0, 3] = False
        result = self.extract(matrices, sample_age_s=ages, finger_valid=valid)
        features = result.features.reshape(1, 5, 10)
        for finger in (1, 3):
            self.assertTrue(torch.all(features[0, finger, :8] == 0.0))
            self.assertEqual(float(features[0, finger, 8]), 0.0)
            self.assertEqual(float(features[0, finger, 9]), 1.0)
        self.assertEqual(float(features[0, 0, 8]), 1.0)

    def test_bad_taxel_and_baseline_are_removed(self):
        matrices = torch.full((1, 5, 12, 6), 3.0)
        matrices[0, 0, 0, 0] = 1000.0
        baseline = torch.full((5, 12, 6), 3.0)
        bad = torch.zeros((5, 12, 6), dtype=torch.bool)
        bad[0, 0, 0] = True
        result = self.extract(matrices, baseline=baseline, bad_taxel_mask=bad)
        self.assertTrue(torch.all(result.total_load == 0.0))
        self.assertFalse(result.contact_mask.any())

    def test_load_rate_preserves_sign_and_is_bounded(self):
        matrices = torch.zeros((1, 5, 12, 6))
        matrices[:, :, 0, 0] = 10.0
        previous = torch.tensor([[20.0, 10.0, 0.0, 1000.0, 5.0]])
        result = self.extract(matrices, previous_total_load=previous)
        rate = result.features.reshape(1, 5, 10)[0, :, 7]
        self.assertLess(float(rate[0]), 0.0)
        self.assertAlmostEqual(float(rate[1]), 0.0, places=6)
        self.assertGreater(float(rate[2]), 0.0)
        self.assertEqual(float(rate[3]), -1.0)

    def test_robust_baseline_flags_nonfinite_and_noisy_taxels(self):
        generator = torch.Generator().manual_seed(4)
        samples = 10.0 + 0.1 * torch.randn((21, 5, 12, 6), generator=generator)
        samples[:, 0, 0, 0] = float("nan")
        samples[:, 1, 1, 1] = torch.arange(21.0)
        fitted = fit_tactile_baseline(samples, maximum_noise_mad=1.0)
        self.assertTrue(fitted.bad_taxel_mask[0, 0, 0])
        self.assertTrue(fitted.bad_taxel_mask[1, 1, 1])
        self.assertFalse(fitted.bad_taxel_mask[2, 2, 2])
        self.assertLess(abs(float(fitted.baseline[2, 2, 2]) - 10.0), 0.2)

    def test_shape_and_scale_errors_fail_closed(self):
        with self.assertRaises(ValueError):
            self.extract(torch.zeros((1, 5, 6, 12)))
        with self.assertRaises(ValueError):
            self.extract(torch.zeros((1, 5, 12, 6)), total_load_full_scale=0.0)


if __name__ == "__main__":
    unittest.main()
