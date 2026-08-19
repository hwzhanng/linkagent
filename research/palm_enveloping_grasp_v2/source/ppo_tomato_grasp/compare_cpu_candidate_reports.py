#!/usr/bin/env python3
"""Compare CPU surrogate candidates without silently trading gentleness for success."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MAXIMUM_FORCE_INCREASE_N = 0.02
MAXIMUM_DAMAGE_SCORE_INCREASE = 1.0e-5
TAIL_SLICE = "small_ripe_and_wet_low_friction"


def _load(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def _extract(report: dict[str, Any]) -> dict[str, Any]:
    overall = report["overall"]
    tail = report.get("stress_slices", {}).get(TAIL_SLICE)
    if not isinstance(tail, dict) or int(tail.get("count", 0)) < 1:
        raise ValueError(f"report lacks nonempty stress slice {TAIL_SLICE}")
    return {
        "checkpoint": report.get("checkpoint"),
        "checkpoint_update": report.get("checkpoint_update"),
        "episodes": int(report.get("episodes", 0)),
        "success_rate": float(overall["success_rate"]),
        "success_wilson_lower": float(overall["success_rate_wilson_95"][0]),
        "drop_rate": float(overall["drop_rate"]),
        "peak_force_mean_n": float(overall["peak_force_mean_n"]),
        "damage_score_mean": float(overall["damage_score_mean"]),
        "tail_count": int(tail["count"]),
        "tail_success_rate": float(tail["success_rate"]),
        "tail_success_wilson_lower": float(tail["success_rate_wilson_95"][0]),
    }


def compare_reports(
    baseline_report: dict[str, Any],
    candidate_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    if not candidate_reports:
        raise ValueError("at least one candidate report is required")
    baseline = _extract(baseline_report)
    candidates = []
    for report in candidate_reports:
        candidate = _extract(report)
        deltas = {
            "success_rate": candidate["success_rate"] - baseline["success_rate"],
            "tail_success_rate": candidate["tail_success_rate"] - baseline["tail_success_rate"],
            "peak_force_mean_n": candidate["peak_force_mean_n"] - baseline["peak_force_mean_n"],
            "damage_score_mean": candidate["damage_score_mean"] - baseline["damage_score_mean"],
        }
        gates = {
            "overall_confidence_not_worse": (
                candidate["success_wilson_lower"] >= baseline["success_wilson_lower"]
            ),
            "tail_confidence_not_worse": (
                candidate["tail_success_wilson_lower"] >= baseline["tail_success_wilson_lower"]
            ),
            "mean_force_increase_bounded": (
                deltas["peak_force_mean_n"] <= MAXIMUM_FORCE_INCREASE_N
            ),
            "damage_increase_bounded": (
                deltas["damage_score_mean"] <= MAXIMUM_DAMAGE_SCORE_INCREASE
            ),
        }
        candidate["deltas_from_baseline"] = deltas
        candidate["promotion_gates"] = gates
        candidate["eligible_for_cpu_default_promotion"] = all(gates.values())
        candidates.append(candidate)
    eligible = [item for item in candidates if item["eligible_for_cpu_default_promotion"]]
    selected = max(eligible, key=lambda item: item["tail_success_wilson_lower"]) if eligible else baseline
    return {
        "schema_version": 1,
        "status": "cpu_surrogate_comparison_only_not_physical_evidence",
        "selection_policy": {
            "overall_and_tail_confidence_must_not_regress": True,
            "maximum_mean_force_increase_n": MAXIMUM_FORCE_INCREASE_N,
            "maximum_damage_score_increase": MAXIMUM_DAMAGE_SCORE_INCREASE,
            "physical_calibration_available": False,
        },
        "baseline": baseline,
        "candidates": candidates,
        "selected_cpu_default_checkpoint": selected["checkpoint"],
        "baseline_retained": selected is baseline,
        "isaac_physx_release_allowed": False,
        "hardware_release_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_reports(_load(args.baseline), [_load(path) for path in args.candidate])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
