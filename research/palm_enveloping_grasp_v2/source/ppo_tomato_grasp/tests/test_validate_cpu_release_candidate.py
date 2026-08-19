from __future__ import annotations

import copy
import unittest

from simulation.ppo_tomato_grasp.validate_cpu_release_candidate import validate_reports


def valid_reports():
    contract = {
        "surrogate_env_version": "test_v1",
        "reward_version": "reward_v1",
        "randomization_version": "randomization_v1",
    }
    offline = {"passed_cpu_scope": True, "blocking_gates": ["physics"]}
    large = {
        "checkpoint": "/tmp/candidate.pt",
        "checkpoint_update": 7,
        "environment_contract": contract,
        "episodes": 100_000,
        "overall": {
            "success_rate_wilson_95": [0.996, 0.997],
            "drop_rate_wilson_95": [0.0001, 0.0008],
        },
        "buckets": {
            "small": {"count": 10_000, "success_rate_wilson_95": [0.99, 0.995]},
        },
    }
    reproducibility = {
        "status": "cpu_surrogate_reproducibility_only_not_physical_evidence",
        "environment_contract": contract,
        "configuration": {"initialization": "weak_semantic_prior_trainable"},
        "runs": [
            {"seed": 1, "checkpoint": "/tmp/seed1.pt"},
            {"seed": 2, "checkpoint": "/tmp/seed2.pt"},
            {"seed": 3, "checkpoint": "/tmp/seed3.pt"},
        ],
        "aggregate": {
            "all_finite": True,
            "success_rate_minimum": 0.995,
            "success_rate_population_std": 0.001,
        },
    }
    observations = {
        "checkpoint": "/tmp/candidate.pt",
        "checkpoint_update": 7,
        "results": [
            {"mode": "nominal", "action_near_limit_fraction": 0.002, "joint_near_limit_fraction": 0.1},
            {"mode": "no_tactile", "success_rate_delta_from_nominal": -0.1},
        ]
    }
    calibration = {
        "checkpoint": "/tmp/candidate.pt",
        "checkpoint_update": 7,
        "results": [
            {"scenario": "baseline", "success_rate_delta_from_baseline": 0.0},
            {"scenario": "gain", "success_rate_delta_from_baseline": -0.01},
        ]
    }
    export = {
        "source_checkpoint": "/tmp/candidate.pt",
        "source_update": 7,
        "maximum_absolute_error": 0.0,
        "hardware_command_allowed": False,
        "formal_training_allowed": False,
    }
    return offline, large, reproducibility, observations, calibration, export


class CpuCandidateGateTests(unittest.TestCase):
    def test_valid_evidence_passes_but_never_releases_physics_or_hardware(self):
        report = validate_reports(*valid_reports())
        self.assertTrue(report["passed_cpu_candidate_scope"])
        self.assertFalse(report["isaac_physx_release_allowed"])
        self.assertFalse(report["hardware_release_allowed"])
        self.assertEqual(report["failed_checks"], [])

    def test_weak_bucket_fails_closed(self):
        reports = list(valid_reports())
        reports[1] = copy.deepcopy(reports[1])
        reports[1]["buckets"]["small"]["success_rate_wilson_95"][0] = 0.90
        report = validate_reports(*reports)
        self.assertFalse(report["passed_cpu_candidate_scope"])
        self.assertIn("large_scale_bucket_floor", report["failed_checks"])

    def test_missing_tactile_ablation_fails_closed(self):
        reports = list(valid_reports())
        reports[3] = copy.deepcopy(reports[3])
        reports[3]["results"] = [reports[3]["results"][0]]
        report = validate_reports(*reports)
        self.assertFalse(report["passed_cpu_candidate_scope"])
        self.assertIn("policy_uses_tactile_information", report["failed_checks"])

    def test_mixed_checkpoint_evidence_fails_closed(self):
        reports = list(valid_reports())
        reports[4] = copy.deepcopy(reports[4])
        reports[4]["checkpoint_update"] = 8
        report = validate_reports(*reports)
        self.assertFalse(report["passed_cpu_candidate_scope"])
        self.assertIn("candidate_evidence_lineage", report["failed_checks"])

    def test_old_environment_reproducibility_fails_closed(self):
        reports = list(valid_reports())
        reports[2] = copy.deepcopy(reports[2])
        reports[2]["environment_contract"]["surrogate_env_version"] = "old_v0"
        report = validate_reports(*reports)
        self.assertFalse(report["passed_cpu_candidate_scope"])
        self.assertIn("reproducibility_environment_lineage", report["failed_checks"])

    def test_duplicate_seed_or_migration_parent_fails_closed(self):
        reports = list(valid_reports())
        reports[2] = copy.deepcopy(reports[2])
        reports[2]["runs"][1]["seed"] = 1
        report = validate_reports(*reports)
        self.assertIn("independent_seed_count", report["failed_checks"])

        reports = list(valid_reports())
        reports[2] = copy.deepcopy(reports[2])
        reports[2]["configuration"]["initialization"] = "independent_parent_checkpoint_migration"
        for run in reports[2]["runs"]:
            run["parent_checkpoint"] = "/tmp/same-parent.pt"
        report = validate_reports(*reports)
        self.assertIn("independent_seed_count", report["failed_checks"])


if __name__ == "__main__":
    unittest.main()
