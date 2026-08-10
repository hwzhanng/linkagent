#!/usr/bin/env python3
"""Validate the fixed-action Physics-v1 throughput benchmark.

This gate intentionally compares only runs with the same environment count,
seed, action stream, warm-up, measured horizon, physics rate, and control
decimation.  It does not treat reduced fidelity or fewer environments as a
performance improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PERF_DIR = ROOT / "logs/qualification/tomato_physics_v1/performance"
ENV_COUNTS = (1024, 4096, 8192)
MATCH_FIELDS = (
    "benchmark",
    "num_envs",
    "seed",
    "action_scale",
    "warmup_steps",
    "measure_steps",
    "block_steps",
    "physics_dt_s",
    "decimation",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-prefix", default="baseline")
    parser.add_argument("--candidate-prefix", default="final_candidate_v4_fair")
    parser.add_argument(
        "--output",
        type=Path,
        default=PERF_DIR / "performance_gate_v4.json",
    )
    parser.add_argument("--min-8192-speedup", type=float, default=2.0)
    parser.add_argument("--max-8192-rollout-seconds", type=float, default=20.0)
    args = parser.parse_args()

    comparisons: dict[str, dict] = {}
    checks: dict[str, bool] = {}
    for env_count in ENV_COUNTS:
        baseline_path = PERF_DIR / f"{args.baseline_prefix}_{env_count}.json"
        candidate_path = PERF_DIR / f"{args.candidate_prefix}_{env_count}.json"
        baseline = _load(baseline_path)
        candidate = _load(candidate_path)

        matched = {field: baseline.get(field) == candidate.get(field) for field in MATCH_FIELDS}
        baseline_seconds = float(baseline["measurement_seconds"])
        candidate_seconds = float(candidate["measurement_seconds"])
        speedup = baseline_seconds / candidate_seconds
        improvement = 1.0 - candidate_seconds / baseline_seconds

        comparisons[str(env_count)] = {
            "baseline": {
                "path": str(baseline_path.relative_to(ROOT)),
                "sha256": _sha256(baseline_path),
                "measurement_seconds": baseline_seconds,
                "transitions_per_second": float(baseline["transitions_per_second"]),
                "terminated_count": int(baseline["terminated_count"]),
            },
            "candidate": {
                "path": str(candidate_path.relative_to(ROOT)),
                "sha256": _sha256(candidate_path),
                "measurement_seconds": candidate_seconds,
                "transitions_per_second": float(candidate["transitions_per_second"]),
                "terminated_count": int(candidate["terminated_count"]),
            },
            "matched_fields": matched,
            "speedup": speedup,
            "improvement_fraction": improvement,
        }
        checks[f"matched_inputs_{env_count}"] = all(matched.values())
        checks[f"finite_{env_count}"] = bool(baseline.get("finite")) and bool(candidate.get("finite"))
        checks[f"faster_{env_count}"] = speedup > 1.0
        checks[f"no_termination_regression_{env_count}"] = int(candidate["terminated_count"]) <= int(
            baseline["terminated_count"]
        )

    candidate_8192 = comparisons["8192"]["candidate"]["measurement_seconds"]
    speedup_8192 = comparisons["8192"]["speedup"]
    checks["physics_rate_120hz"] = comparisons["8192"]["candidate"]["path"] != "" and abs(
        _load(ROOT / comparisons["8192"]["candidate"]["path"])["physics_dt_s"] - 1.0 / 120.0
    ) < 1.0e-12
    checks["control_decimation_6"] = (
        _load(ROOT / comparisons["8192"]["candidate"]["path"])["decimation"] == 6
    )
    checks["8192_rollout_within_budget"] = candidate_8192 <= args.max_8192_rollout_seconds
    checks["8192_speedup_meets_target"] = speedup_8192 >= args.min_8192_speedup

    report = {
        "schema_version": 1,
        "gate": "tomato_physics_v1_fixed_action_performance",
        "passed": all(checks.values()),
        "thresholds": {
            "min_8192_speedup": args.min_8192_speedup,
            "max_8192_rollout_seconds": args.max_8192_rollout_seconds,
        },
        "checks": checks,
        "comparisons": comparisons,
        "constraints": {
            "no_environment_count_reduction": True,
            "physics_rate_hz": 120,
            "control_decimation": 6,
            "fixed_seed": 42,
            "fixed_action_scale": 0.45,
            "fixed_warmup_steps": 16,
            "fixed_measure_steps": 32,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("TOMATO_PHYSICS_V1_PERFORMANCE_GATE=" + json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
