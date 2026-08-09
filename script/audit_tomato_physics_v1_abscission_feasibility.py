"""Audit the provisional Physics-v1 abscission envelope without Isaac Sim."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from simulation.tomato_pick.physics_v1 import (
    PHYSICS_V1_PROFILE_PATH,
    load_physics_v1_profile,
    sample_correlated_fruit_profiles,
    update_stem_damage,
)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    levels = torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], dtype=values.dtype)
    result = torch.quantile(values, levels)
    return {
        name: float(value)
        for name, value in zip(("minimum", "p05", "median", "p95", "maximum"), result)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 1000:
        raise ValueError("samples must be at least 1000")

    profile = load_physics_v1_profile()
    values = sample_correlated_fruit_profiles(
        args.samples, generator=torch.Generator().manual_seed(args.seed)
    )
    force = values["stem_tensile_force_n"]
    capacity = values["stem_safe_tangential_capacity_n"]
    required_normal = force / (3.0 * values["static_friction"])
    feasible = force <= capacity

    order = torch.argsort(force)
    selected = order[torch.tensor([0, args.samples // 2, args.samples - 1])]
    critical = values["stem_critical_duration_s"][selected]
    dt = 1.0 / 120.0
    attached = torch.ones(3, dtype=torch.bool)
    damage = torch.zeros(3)
    damage, spike_broken = update_stem_damage(
        damage,
        torch.ones(3),
        dt_s=dt,
        critical_duration_s=critical,
        onset_ratio=profile["stem_prior"]["damage_onset_ratio"],
        recovery_time_s=profile["stem_prior"]["damage_recovery_time_s"],
        attached=attached,
    )
    break_steps = torch.full((3,), -1, dtype=torch.long)
    for step in range(1, 18):
        damage, broken = update_stem_damage(
            damage,
            torch.ones(3),
            dt_s=dt,
            critical_duration_s=critical,
            onset_ratio=profile["stem_prior"]["damage_onset_ratio"],
            recovery_time_s=profile["stem_prior"]["damage_recovery_time_s"],
            attached=attached,
        )
        new = broken & (break_steps < 0)
        break_steps[new] = step + 1

    checks = {
        "all_profiles_within_three_finger_safe_capacity": bool(feasible.all()),
        "all_profiles_below_eight_newtons_per_finger": bool(
            (required_normal < 8.0).all()
        ),
        "single_substep_does_not_break": not bool(spike_broken.any()),
        "weak_median_strong_break_under_sustained_unit_load": bool(
            (break_steps > 1).all()
        ),
        "intact_stem_prior_preserved_as_audit_only": bool(
            (values["stem_intact_tensile_force_n"] > force).all()
        ),
    }
    result = {
        "schema_version": 1,
        "qualification": "provisional_abscission_envelope_feasibility",
        "passed": all(checks.values()),
        "checks": checks,
        "samples": args.samples,
        "seed": args.seed,
        "profile": str(PHYSICS_V1_PROFILE_PATH),
        "profile_sha256": hashlib.sha256(PHYSICS_V1_PROFILE_PATH.read_bytes()).hexdigest(),
        "detachment_force_n": _quantiles(force),
        "intact_stem_tensile_force_n_audit_only": _quantiles(
            values["stem_intact_tensile_force_n"]
        ),
        "three_finger_safe_tangential_capacity_n": _quantiles(capacity),
        "required_normal_force_per_finger_n": _quantiles(required_normal),
        "capacity_margin_n": _quantiles(capacity - force),
        "weak_median_strong": {
            "detachment_force_n": [float(value) for value in force[selected]],
            "critical_duration_s": [float(value) for value in critical],
            "break_substep": [int(value) for value in break_steps],
        },
        "unchanged": [
            "reward",
            "observations",
            "actions",
            "reset_distribution",
            "five_dimensional_hand_manifold",
            "collision_geometry",
            "contact_parameters",
            "ppo",
            "success_predicate",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
