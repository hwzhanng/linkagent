from __future__ import annotations

import unittest

from simulation.ppo_tomato_grasp.evaluate_cpu_checkpoint_large_scale import wilson_interval


class LargeScaleEvaluationTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(93, 100)
        self.assertLess(low, 0.93)
        self.assertGreater(high, 0.93)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_invalid_binomial_counts_fail_closed(self):
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)


if __name__ == "__main__":
    unittest.main()
