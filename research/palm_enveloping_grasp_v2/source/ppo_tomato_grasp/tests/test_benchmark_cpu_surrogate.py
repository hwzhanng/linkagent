from __future__ import annotations

import unittest

from simulation.ppo_tomato_grasp.benchmark_cpu_surrogate import benchmark_size


class CpuSurrogateBenchmarkTests(unittest.TestCase):
    def test_small_benchmark_is_finite(self):
        result = benchmark_size(8, steps=2, warmup=1, seed=3)
        self.assertTrue(result["all_finite"])
        self.assertGreater(result["environment_transitions_per_second"], 0.0)

    def test_invalid_settings_fail_closed(self):
        with self.assertRaises(ValueError):
            benchmark_size(0, steps=1, warmup=0, seed=1)


if __name__ == "__main__":
    unittest.main()
