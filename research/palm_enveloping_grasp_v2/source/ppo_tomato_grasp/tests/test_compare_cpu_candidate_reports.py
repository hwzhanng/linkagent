from __future__ import annotations

import copy
import unittest

from simulation.ppo_tomato_grasp.compare_cpu_candidate_reports import compare_reports


def report(name: str, success: float, lower: float, force: float, damage: float, tail: float, tail_lower: float):
    return {
        "checkpoint": name,
        "checkpoint_update": 1,
        "episodes": 100000,
        "overall": {
            "success_rate": success,
            "success_rate_wilson_95": [lower, success + 0.001],
            "drop_rate": 0.001,
            "peak_force_mean_n": force,
            "damage_score_mean": damage,
        },
        "stress_slices": {
            "small_ripe_and_wet_low_friction": {
                "count": 2000,
                "success_rate": tail,
                "success_rate_wilson_95": [tail_lower, tail + 0.01],
            }
        },
    }


class CpuCandidateComparisonTests(unittest.TestCase):
    def setUp(self):
        self.baseline = report("base.pt", 0.99, 0.989, 1.0, 0.0, 0.90, 0.88)

    def test_gentle_improvement_is_promoted_without_releasing_hardware(self):
        candidate = report("better.pt", 0.995, 0.994, 1.01, 0.000001, 0.93, 0.91)
        result = compare_reports(self.baseline, [candidate])
        self.assertFalse(result["baseline_retained"])
        self.assertEqual(result["selected_cpu_default_checkpoint"], "better.pt")
        self.assertFalse(result["hardware_release_allowed"])

    def test_force_tradeoff_retains_baseline(self):
        candidate = report("strong.pt", 0.999, 0.998, 1.03, 0.0, 0.95, 0.94)
        result = compare_reports(self.baseline, [candidate])
        self.assertTrue(result["baseline_retained"])
        self.assertFalse(result["candidates"][0]["promotion_gates"]["mean_force_increase_bounded"])

    def test_missing_tail_slice_fails_closed(self):
        broken = copy.deepcopy(self.baseline)
        broken["stress_slices"] = {}
        with self.assertRaises(ValueError):
            compare_reports(self.baseline, [broken])


if __name__ == "__main__":
    unittest.main()
