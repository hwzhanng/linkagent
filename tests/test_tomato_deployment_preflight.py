from __future__ import annotations

from pathlib import Path
import unittest

from module.robotics import run_deployment_preflight


ROOT = Path(__file__).resolve().parents[1]
BEST = ROOT / "config/skills/tomato_pick_clean_v5_model1350_best.json"
CANDIDATE = ROOT / "config/skills/tomato_pick_clean_v5_model1600_candidate.json"


class TomatoDeploymentPreflightTests(unittest.TestCase):
    def test_selected_best_passes_simulation_release_gate(self) -> None:
        report = run_deployment_preflight(
            skill_profile=BEST,
            mode="simulation",
            minimum_success_rate=0.90,
        )
        self.assertTrue(report.passed, report.as_dict())
        names = {check.name for check in report.checks if check.passed}
        self.assertIn("skill_manifest_and_hashes", names)
        self.assertIn("evaluation_selection_provenance", names)

    def test_candidate_cannot_be_started_as_a_release(self) -> None:
        report = run_deployment_preflight(
            skill_profile=CANDIDATE,
            mode="simulation",
            minimum_success_rate=0.90,
        )
        self.assertFalse(report.passed)
        failed = {check.name for check in report.checks if not check.passed}
        self.assertIn("fixed_multiseed_evaluation", failed)
        self.assertIn("evaluation_selection_provenance", failed)

    def test_hardware_fails_closed_without_real_validation(self) -> None:
        report = run_deployment_preflight(
            skill_profile=BEST,
            mode="hardware",
            robot_deployment=ROOT / "config/robot_bimanual_offline.yaml",
            verifier_thresholds=ROOT / "config/verifiers/tomato_pick_real_pending.json",
            minimum_success_rate=0.90,
        )
        self.assertFalse(report.passed)
        failed = {check.name for check in report.checks if not check.passed}
        self.assertTrue(
            {
                "skill_hardware_approval",
                "robot_deployment",
                "tactile_calibration",
                "verifier_thresholds",
                "vendor_hardware_validation",
            }.issubset(failed),
            report.as_dict(),
        )


if __name__ == "__main__":
    unittest.main()
