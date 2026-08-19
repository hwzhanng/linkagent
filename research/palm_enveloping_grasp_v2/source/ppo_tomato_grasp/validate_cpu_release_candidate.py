#!/usr/bin/env python3
"""Fail-closed release gates for the CPU hand-grasp surrogate candidate.

Passing this module means that the tensor interface and CPU-only surrogate
evidence are internally consistent.  It never authorizes Isaac training or
hardware commands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


THRESHOLDS = {
    "large_scale_min_episodes": 100_000,
    "large_scale_success_wilson_lower": 0.995,
    "large_scale_drop_wilson_upper": 0.001,
    "bucket_min_count": 1_000,
    "bucket_success_wilson_lower": 0.985,
    "reproducibility_min_seeds": 3,
    "reproducibility_success_minimum": 0.99,
    "reproducibility_success_std_maximum": 0.005,
    "nominal_action_near_limit_maximum": 0.01,
    "nominal_joint_near_limit_maximum": 0.15,
    "no_tactile_success_drop_minimum": 0.02,
    "calibration_success_drop_maximum": 0.02,
    "torchscript_parity_error_maximum": 1.0e-6,
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _check(name: str, passed: bool, actual: Any, requirement: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "requirement": requirement,
    }


def validate_reports(
    offline: dict[str, Any],
    large: dict[str, Any],
    reproducibility: dict[str, Any],
    observations: dict[str, Any],
    calibration: dict[str, Any],
    export: dict[str, Any],
) -> dict[str, Any]:
    """Return a machine-readable, fail-closed CPU-scope validation report."""

    checks: list[dict[str, Any]] = []
    candidate_update = large.get("checkpoint_update")
    candidate_path = large.get("checkpoint")
    lineage = {
        "large_scale": [candidate_path, candidate_update],
        "observations": [observations.get("checkpoint"), observations.get("checkpoint_update")],
        "calibration": [calibration.get("checkpoint"), calibration.get("checkpoint_update")],
        "export": [export.get("source_checkpoint"), export.get("source_update")],
    }
    lineage_matches = (
        candidate_path is not None
        and candidate_update is not None
        and all(
            path == candidate_path and update == candidate_update
            for path, update in lineage.values()
        )
    )
    checks.append(_check(
        "candidate_evidence_lineage",
        lineage_matches,
        lineage,
        "large-scale, observation, calibration, and export evidence must name one exact checkpoint",
    ))
    checks.append(_check(
        "offline_structural_audit",
        offline.get("passed_cpu_scope") is True,
        offline.get("passed_cpu_scope"),
        "passed_cpu_scope must be true",
    ))
    candidate_contract = large.get("environment_contract")
    reproducibility_contract = reproducibility.get("environment_contract")
    checks.append(_check(
        "reproducibility_environment_lineage",
        isinstance(candidate_contract, dict)
        and bool(candidate_contract)
        and reproducibility_contract == candidate_contract,
        {
            "candidate": candidate_contract,
            "reproducibility": reproducibility_contract,
        },
        "independent-seed campaign must use the exact candidate environment contract",
    ))

    episodes = int(large.get("episodes", 0))
    overall = large.get("overall", {})
    success_interval = overall.get("success_rate_wilson_95", [0.0, 0.0])
    drop_interval = overall.get("drop_rate_wilson_95", [1.0, 1.0])
    checks.extend([
        _check(
            "large_scale_episode_count",
            episodes >= THRESHOLDS["large_scale_min_episodes"],
            episodes,
            f">= {THRESHOLDS['large_scale_min_episodes']}",
        ),
        _check(
            "large_scale_success_confidence",
            float(success_interval[0]) >= THRESHOLDS["large_scale_success_wilson_lower"],
            success_interval,
            f"95% Wilson lower >= {THRESHOLDS['large_scale_success_wilson_lower']}",
        ),
        _check(
            "large_scale_drop_confidence",
            float(drop_interval[1]) <= THRESHOLDS["large_scale_drop_wilson_upper"],
            drop_interval,
            f"95% Wilson upper <= {THRESHOLDS['large_scale_drop_wilson_upper']}",
        ),
    ])

    bucket_failures: list[dict[str, Any]] = []
    for name, bucket in sorted(large.get("buckets", {}).items()):
        count = int(bucket.get("count", 0))
        lower = float(bucket.get("success_rate_wilson_95", [0.0, 0.0])[0])
        if count < THRESHOLDS["bucket_min_count"] or lower < THRESHOLDS["bucket_success_wilson_lower"]:
            bucket_failures.append({"bucket": name, "count": count, "success_wilson_lower": lower})
    checks.append(_check(
        "large_scale_bucket_floor",
        bool(large.get("buckets")) and not bucket_failures,
        bucket_failures,
        (
            f"every bucket count >= {THRESHOLDS['bucket_min_count']} and "
            f"success Wilson lower >= {THRESHOLDS['bucket_success_wilson_lower']}"
        ),
    ))

    aggregate = reproducibility.get("aggregate", {})
    runs = reproducibility.get("runs", [])
    seeds = [run.get("seed") for run in runs]
    run_checkpoints = [run.get("checkpoint") for run in runs]
    initialization = reproducibility.get("configuration", {}).get("initialization")
    parent_checkpoints = [run.get("parent_checkpoint") for run in runs]
    distinct_training_lineage = (
        len(runs) >= THRESHOLDS["reproducibility_min_seeds"]
        and None not in seeds
        and len(set(seeds)) == len(seeds)
        and None not in run_checkpoints
        and len(set(run_checkpoints)) == len(run_checkpoints)
        and (
            initialization != "independent_parent_checkpoint_migration"
            or (
                None not in parent_checkpoints
                and len(set(parent_checkpoints)) == len(parent_checkpoints)
            )
        )
    )
    checks.extend([
        _check(
            "reproducibility_report_scope",
            reproducibility.get("status")
            == "cpu_surrogate_reproducibility_only_not_physical_evidence"
            and aggregate.get("all_finite") is True,
            {
                "status": reproducibility.get("status"),
                "all_finite": aggregate.get("all_finite"),
            },
            "must be a finite CPU-surrogate reproducibility report",
        ),
        _check(
            "independent_seed_count",
            distinct_training_lineage,
            {
                "run_count": len(runs),
                "seeds": seeds,
                "checkpoints": run_checkpoints,
                "parent_checkpoints": parent_checkpoints,
            },
            (
                f">= {THRESHOLDS['reproducibility_min_seeds']} unique seeds and output "
                "checkpoints; migrated campaigns also require unique parents"
            ),
        ),
        _check(
            "independent_seed_success_floor",
            float(aggregate.get("success_rate_minimum", 0.0))
            >= THRESHOLDS["reproducibility_success_minimum"],
            aggregate.get("success_rate_minimum"),
            f">= {THRESHOLDS['reproducibility_success_minimum']}",
        ),
        _check(
            "independent_seed_dispersion",
            float(aggregate.get("success_rate_population_std", 1.0))
            <= THRESHOLDS["reproducibility_success_std_maximum"],
            aggregate.get("success_rate_population_std"),
            f"<= {THRESHOLDS['reproducibility_success_std_maximum']}",
        ),
    ])

    observation_by_mode = {row["mode"]: row for row in observations.get("results", [])}
    nominal = observation_by_mode.get("nominal", {})
    no_tactile = observation_by_mode.get("no_tactile", {})
    tactile_drop = -float(no_tactile.get("success_rate_delta_from_nominal", 0.0))
    checks.extend([
        _check(
            "nominal_action_boundary_margin",
            float(nominal.get("action_near_limit_fraction", 1.0))
            <= THRESHOLDS["nominal_action_near_limit_maximum"],
            nominal.get("action_near_limit_fraction"),
            f"<= {THRESHOLDS['nominal_action_near_limit_maximum']}",
        ),
        _check(
            "nominal_joint_boundary_margin",
            float(nominal.get("joint_near_limit_fraction", 1.0))
            <= THRESHOLDS["nominal_joint_near_limit_maximum"],
            nominal.get("joint_near_limit_fraction"),
            f"<= {THRESHOLDS['nominal_joint_near_limit_maximum']}",
        ),
        _check(
            "policy_uses_tactile_information",
            "no_tactile" in observation_by_mode
            and tactile_drop >= THRESHOLDS["no_tactile_success_drop_minimum"],
            tactile_drop,
            f"success drop >= {THRESHOLDS['no_tactile_success_drop_minimum']}",
        ),
    ])

    calibration_rows = calibration.get("results", [])
    worst_calibration_drop = max(
        (-float(row.get("success_rate_delta_from_baseline", 0.0)) for row in calibration_rows),
        default=1.0,
    )
    checks.append(_check(
        "bounded_calibration_stress",
        bool(calibration_rows)
        and worst_calibration_drop <= THRESHOLDS["calibration_success_drop_maximum"],
        worst_calibration_drop,
        f"worst success drop <= {THRESHOLDS['calibration_success_drop_maximum']}",
    ))

    checks.extend([
        _check(
            "torchscript_parity",
            float(export.get("maximum_absolute_error", 1.0))
            <= THRESHOLDS["torchscript_parity_error_maximum"],
            export.get("maximum_absolute_error"),
            f"<= {THRESHOLDS['torchscript_parity_error_maximum']}",
        ),
        _check(
            "hardware_release_remains_forbidden",
            export.get("hardware_command_allowed") is False
            and export.get("formal_training_allowed") is False,
            {
                "hardware_command_allowed": export.get("hardware_command_allowed"),
                "formal_training_allowed": export.get("formal_training_allowed"),
            },
            "both flags must be false",
        ),
    ])

    passed = all(item["passed"] for item in checks)
    return {
        "schema_version": 1,
        "status": (
            "cpu_surrogate_candidate_passed_physics_and_hardware_blocked"
            if passed
            else "cpu_surrogate_candidate_failed"
        ),
        "passed_cpu_candidate_scope": passed,
        "isaac_physx_release_allowed": False,
        "hardware_release_allowed": False,
        "thresholds": THRESHOLDS,
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
        "remaining_blocking_gates": offline.get("blocking_gates", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--large-scale", type=Path, required=True)
    parser.add_argument("--reproducibility", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--export-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_reports(
        _load(args.offline),
        _load(args.large_scale),
        _load(args.reproducibility),
        _load(args.observations),
        _load(args.calibration),
        _load(args.export_metadata),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed_cpu_candidate_scope"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
