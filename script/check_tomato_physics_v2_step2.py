"""Combine the Physics-v2 static asset audit and independent-layer gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script.audit_tomato_physics_v2_robot_asset import audit
from simulation.tomato_pick.physics_v2 import physics_v2_current_stage_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    static_audit = audit()
    identity = physics_v2_current_stage_report()
    checks = {
        "static_robot_structure_passed": static_audit["passed_structural_gate"],
        "parent_identity_mismatch_is_explicit": (
            not static_audit["physics_v1_identity_gate"]
            and any(
                item["id"] == "physics_v1_declared_asset_hash_mismatch"
                for item in static_audit["findings"]
            )
        ),
        "independent_v2_identity_passed": identity["passed"],
        "training_remains_blocked": identity["training_allowed"] is False,
    }
    report = {
        "schema_version": 1,
        "gate_id": "tomato_pick_physics_v2_step2",
        "passed": all(checks.values()),
        "checks": checks,
        "identity": identity,
        "static_audit_summary": static_audit["summary"],
        "findings": static_audit["findings"],
        "next_stage": "fruit_contact_and_damage_proxy",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
