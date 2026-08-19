#!/usr/bin/env python3
"""Turn an extended CPU calibration scan into multi-metric warning gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LIMITS = {
    "maximum_success_rate_drop": 0.02,
    "maximum_peak_force_increase_n": 0.15,
    "maximum_drop_rate_increase": 0.002,
    "maximum_action_near_limit_fraction": 0.01,
    "maximum_joint_near_limit_fraction": 0.15,
}


def summarize(scan: dict[str, Any]) -> dict[str, Any]:
    rows = scan.get("results", [])
    if not scan.get("extended_breakpoint_scan") or not rows:
        raise ValueError("an extended, nonempty calibration scan is required")
    baseline = next((row for row in rows if row.get("scenario") == "baseline"), None)
    if baseline is None:
        raise ValueError("calibration scan has no baseline")
    assessed = []
    for row in rows:
        if row is baseline:
            continue
        metrics = {
            "success_rate_drop": max(0.0, -float(row["success_rate_delta_from_baseline"])),
            "peak_force_increase_n": max(0.0, float(row["peak_force_delta_from_baseline_n"])),
            "drop_rate_increase": max(0.0, float(row["drop_rate"]) - float(baseline["drop_rate"])),
            "action_near_limit_fraction": float(row["action_near_limit_fraction"]),
            "joint_near_limit_fraction": float(row["joint_near_limit_fraction"]),
        }
        failed = []
        for name, value in metrics.items():
            limit_name = "maximum_" + name
            if value > LIMITS[limit_name]:
                failed.append(name)
        assessed.append({
            "scenario": row["scenario"],
            "metrics": metrics,
            "within_cpu_warning_envelope": not failed,
            "failed_metrics": failed,
        })
    warnings = [row for row in assessed if not row["within_cpu_warning_envelope"]]
    return {
        "schema_version": 1,
        "status": "cpu_surrogate_calibration_warning_only_not_hardware_tolerance",
        "checkpoint": scan.get("checkpoint"),
        "checkpoint_update": scan.get("checkpoint_update"),
        "limits": LIMITS,
        "scenario_count": len(assessed),
        "warning_count": len(warnings),
        "warnings": warnings,
        "all_scenarios": assessed,
        "recommended_pre_physx_targets": {
            "tactile_multiplicative_gain_error": "keep inside trained 0.82-1.18 envelope",
            "vision_center_bias": "target <= 5 mm; treat 10 mm as investigation threshold",
            "observation_delay": "target <= 3 policy steps (150 ms at 20 Hz)",
            "joint_zero_bias": "target <= 2 degrees after URDF/encoder calibration",
        },
        "physical_tolerance_claimed": False,
        "hardware_release_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scan = json.loads(args.scan.read_text(encoding="utf-8"))
    report = summarize(scan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
