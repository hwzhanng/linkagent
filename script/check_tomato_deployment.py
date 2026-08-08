#!/usr/bin/env python3
"""Run fail-closed tomato-pick deployment checks and emit a JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.robotics.preflight import run_deployment_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-profile", type=Path, required=True)
    parser.add_argument("--mode", choices=("simulation", "shadow", "hardware"), required=True)
    parser.add_argument("--robot-deployment", type=Path)
    parser.add_argument("--tactile-calibration", type=Path)
    parser.add_argument("--verifier-thresholds", type=Path)
    parser.add_argument("--minimum-success-rate", type=float, default=0.80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_deployment_preflight(
        skill_profile=args.skill_profile,
        mode=args.mode,
        robot_deployment=args.robot_deployment,
        tactile_calibration=args.tactile_calibration,
        verifier_thresholds=args.verifier_thresholds,
        minimum_success_rate=args.minimum_success_rate,
    )
    serialized = json.dumps(report.as_dict(), indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if not report.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
