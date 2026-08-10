"""Check metamorphic success-contract boundaries without Isaac or a GPU."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.tomato_pick.physics_v2_scenarios import physical_task_predicates


def _predicate(
    force: list[float],
    *,
    palm: float = 0.0,
    detached: bool = True,
    distance: float = 0.03,
    speed: float = 0.02,
    hold: int = 20,
) -> dict[str, bool]:
    result = physical_task_predicates(
        torch.tensor([force], dtype=torch.float64),
        physical_palm_force_n=torch.tensor([palm], dtype=torch.float64),
        stem_detached=torch.tensor([detached], dtype=torch.bool),
        center_distance_m=torch.tensor([distance], dtype=torch.float64),
        relative_speed_m_s=torch.tensor([speed], dtype=torch.float64),
        hold_steps=torch.tensor([hold], dtype=torch.long),
        maximum_fingertip_force_n=8.0,
    )
    return {name: bool(value.item()) for name, value in result.items()}


def build_success_surface_audit() -> dict[str, object]:
    checks: dict[str, bool] = {}
    observations: dict[str, object] = {}

    force_levels = [0.0, 0.24, 0.25, 4.0, 8.0, 8.0001]
    force_success = [
        _predicate([level, level, 0.0, 0.0, 0.0])["success"]
        for level in force_levels
    ]
    observations["opposed_fingertip_force_levels_n"] = force_levels
    observations["opposed_fingertip_success"] = force_success
    checks["fingertip_force_boundary_is_exact"] = force_success == [
        False,
        False,
        True,
        True,
        True,
        False,
    ]

    distance_levels = [0.0599, 0.06, 0.0601]
    distance_success = [
        _predicate([4.0, 4.0, 0.0, 0.0, 0.0], distance=level)["success"]
        for level in distance_levels
    ]
    observations["distance_levels_m"] = distance_levels
    observations["distance_success"] = distance_success
    checks["distance_boundary_is_non_strict"] = distance_success == [True, True, False]

    speed_levels = [0.0999, 0.10, 0.1001]
    speed_success = [
        _predicate([4.0, 4.0, 0.0, 0.0, 0.0], speed=level)["success"]
        for level in speed_levels
    ]
    observations["speed_levels_m_s"] = speed_levels
    observations["speed_success"] = speed_success
    checks["speed_boundary_is_non_strict"] = speed_success == [True, True, False]

    hold_levels = [19, 20, 21]
    hold_success = [
        _predicate([4.0, 4.0, 0.0, 0.0, 0.0], hold=level)["success"]
        for level in hold_levels
    ]
    observations["hold_levels_policy_steps"] = hold_levels
    observations["hold_success"] = hold_success
    checks["hold_boundary_is_non_strict"] = hold_success == [False, True, True]

    palm_levels = [0.2499, 0.25, 8.0, 8.0001]
    palm_success = [
        _predicate([0.0, 0.0, 0.0, 0.0, 0.0], palm=level)["success"]
        for level in palm_levels
    ]
    observations["palm_force_levels_n"] = palm_levels
    observations["palm_success"] = palm_success
    checks["palm_boundary_is_exact"] = palm_success == [False, True, True, False]
    checks["palm_contact_before_detach_is_not_support"] = not _predicate(
        [0.0, 0.0, 0.0, 0.0, 0.0], palm=2.0, detached=False
    )["success"]
    checks["same_side_fingertip_contact_cannot_restore_success"] = not _predicate(
        [0.0, 4.0, 4.0, 4.0, 0.0]
    )["success"]
    checks["palm_support_does_not_remove_force_ceiling"] = not _predicate(
        [0.0, 0.0, 0.0, 0.0, 0.0], palm=8.0001
    )["within_force_limit"]

    return {
        "schema_version": 1,
        "audit_id": "tomato_pick_physics_v2_success_surface",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "pure CPU predicate metamorphic checks; no Isaac, PhysX, GPU, PPO, or hardware",
        "observations": observations,
        "checks": checks,
        "passed": all(checks.values()),
        "formal_training_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_success_surface_audit()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["passed"]}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
