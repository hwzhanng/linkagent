from __future__ import annotations

import unittest

from simulation.ppo_tomato_grasp.summarize_calibration_breakpoints import summarize


class CalibrationBreakpointSummaryTests(unittest.TestCase):
    def test_force_warning_is_not_hidden_by_high_success(self):
        scan = {
            "extended_breakpoint_scan": True,
            "checkpoint": "candidate.pt",
            "checkpoint_update": 2,
            "results": [
                {
                    "scenario": "baseline",
                    "drop_rate": 0.001,
                    "success_rate_delta_from_baseline": 0.0,
                    "peak_force_delta_from_baseline_n": 0.0,
                    "action_near_limit_fraction": 0.0,
                    "joint_near_limit_fraction": 0.0,
                },
                {
                    "scenario": "high_force",
                    "drop_rate": 0.001,
                    "success_rate_delta_from_baseline": 0.0,
                    "peak_force_delta_from_baseline_n": 0.20,
                    "action_near_limit_fraction": 0.0,
                    "joint_near_limit_fraction": 0.0,
                },
            ],
        }
        report = summarize(scan)
        self.assertEqual(report["warning_count"], 1)
        self.assertIn("peak_force_increase_n", report["warnings"][0]["failed_metrics"])
        self.assertFalse(report["hardware_release_allowed"])

    def test_nonextended_or_missing_baseline_fails_closed(self):
        with self.assertRaises(ValueError):
            summarize({"extended_breakpoint_scan": False, "results": []})
        with self.assertRaises(ValueError):
            summarize({"extended_breakpoint_scan": True, "results": [{"scenario": "x"}]})


if __name__ == "__main__":
    unittest.main()
