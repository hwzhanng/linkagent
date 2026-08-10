"""Run analytic isolated-physics gates without importing Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from script.audit_tomato_physics_v2_robot_asset import audit
from simulation.tomato_pick.physics_v2 import load_physics_v2_profile
from simulation.tomato_pick.physics_v2_isolated import (
    incline_will_slide,
    restitution_energy_fraction,
    semi_implicit_free_fall,
    simulate_damped_compression,
)
from simulation.tomato_pick.physics_v2_models import (
    fixed_fruit_contact_cases,
    linear_compliant_contact_force,
    sample_correlated_fruit_contact_profiles,
    sample_stem_profiles_v2,
    sample_uncertainty_profiles_v2,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-seed", type=int, default=4096)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples_per_seed < 1 or args.seeds < 2:
        raise ValueError("samples-per-seed must be positive and seeds must be >= 2")

    profile = load_physics_v2_profile()
    cases = fixed_fruit_contact_cases(dtype=torch.float64)
    audit_report = audit()
    free_fall = semi_implicit_free_fall(
        height_m=1.0,
        mass_kg=float(cases["nominal"]["mass_kg"]),
        dt_s=1.0 / profile["learning_contract"]["physics_rate_hz"],
        steps=30,
    )
    free_fall_acceleration = torch.diff(free_fall["velocity_m_s"]) * 120.0
    free_fall_relative_energy_loss = float(
        (free_fall["energy_j"][0] - free_fall["energy_j"][-1])
        / free_fall["energy_j"][0]
    )

    static_friction = torch.stack(
        tuple(case["static_friction"] for case in cases.values())
    )
    compression_energy_checks: dict[str, bool] = {}
    compression_energy_ratios: dict[str, float] = {}
    for name, case in cases.items():
        trace = simulate_damped_compression(
            mass_kg=float(case["mass_kg"]),
            stiffness_n_m=float(case["effective_stiffness_n_m"]),
            damping_n_s_m=float(case["effective_damping_n_s_m"]),
            initial_indentation_m=0.002,
            initial_closing_speed_m_s=0.05,
            dt_s=1.0 / 20000.0,
            steps=1000,
        )
        compression_energy_checks[name] = bool(
            (torch.diff(trace["energy_j"]) <= 1.0e-10).all()
        )
        compression_energy_ratios[name] = float(
            trace["energy_j"][-1] / trace["energy_j"][0]
        )

    indentation = torch.linspace(0.0, 0.010, 101, dtype=torch.float64)
    force = linear_compliant_contact_force(
        indentation,
        torch.zeros_like(indentation),
        torch.full_like(indentation, 9000.0),
        torch.full_like(indentation, 40.0),
    )

    finite_by_seed: dict[str, bool] = {}
    deterministic_by_seed: dict[str, bool] = {}
    for seed in range(args.seeds):
        first = sample_correlated_fruit_contact_profiles(
            args.samples_per_seed,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float64,
        )
        repeat = sample_correlated_fruit_contact_profiles(
            args.samples_per_seed,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float64,
        )
        uncertainty = sample_uncertainty_profiles_v2(
            args.samples_per_seed,
            generator=torch.Generator().manual_seed(seed + 1000),
            dtype=torch.float64,
        )
        stem = sample_stem_profiles_v2(first, mode="learnability")
        all_values = tuple(first.values()) + tuple(uncertainty.values()) + tuple(stem.values())
        finite_by_seed[str(seed)] = all(
            value.dtype is torch.bool or bool(torch.isfinite(value).all())
            for value in all_values
        )
        deterministic_by_seed[str(seed)] = all(
            torch.equal(first[key], repeat[key]) for key in first
        )

    metadata = audit_report["summary"]["physics_v1_layer_metadata"]
    self_collision_static_evidence = (
        audit_report["usd_metadata_checks"]["enabled_self_collision_pairs"]
        and audit_report["usd_metadata_checks"]["filtered_self_collision_pairs"]
        and audit_report["usd_metadata_checks"]["outside_solve_contact_disabled"]
        and metadata["physicsV1EnabledPairsPerRobot"] == 59
        and metadata["physicsV1FilteredPairsPerRobot"] == 2426
        and metadata["physicsV1StoredActiveHandFilteredPairCount"] == 7
    )
    mimic_static_evidence = all(
        hand["independent_joint_count"] == 16
        and hand["passive_mimic_count"] == 5
        and len(hand["mimics"]) == 5
        for hand in audit_report["summary"]["hands"].values()
    )
    restitution_fraction = restitution_energy_fraction(
        float(profile["fruit_contact_v2"]["restitution"])
    )
    checks = {
        "free_fall_acceleration_matches_gravity": bool(
            torch.allclose(
                free_fall_acceleration,
                torch.full_like(free_fall_acceleration, -9.81),
                atol=1.0e-12,
            )
        ),
        "free_fall_has_no_energy_gain": bool(
            (torch.diff(free_fall["energy_j"]) <= 1.0e-12).all()
        ) and free_fall_relative_energy_loss < 0.011,
        "twenty_degree_incline_holds_all_fixed_cases": not bool(
            incline_will_slide(static_friction, 20.0).any()
        ),
        "thirty_five_degree_incline_slides_all_fixed_cases": bool(
            incline_will_slide(static_friction, 35.0).all()
        ),
        "compression_force_is_nonnegative_and_monotonic": bool(
            (force >= 0.0).all() and (torch.diff(force) >= 0.0).all()
        ),
        "damped_compression_has_no_energy_gain": all(compression_energy_checks.values()),
        "rebound_energy_is_strictly_dissipative": 0.0 <= restitution_fraction < 1.0,
        "static_self_collision_policy_is_accounted_for": self_collision_static_evidence,
        "passive_mimic_topology_is_accounted_for": mimic_static_evidence,
        "multi_seed_samples_are_finite": all(finite_by_seed.values()),
        "multi_seed_sampling_is_repeatable": all(deterministic_by_seed.values()),
        "training_remains_blocked": profile["implementation_boundary"]["training_allowed"] is False,
        "integrated_physx_gate_is_not_claimed": True,
    }
    report = {
        "schema_version": 1,
        "gate_id": "tomato_pick_physics_v2_step6_analytic_isolated",
        "scope": "analytic_and_static_only_no_isaac_or_physx_import",
        "passed": all(checks.values()),
        "checks": checks,
        "samples_per_seed": args.samples_per_seed,
        "seeds": args.seeds,
        "metrics": {
            "free_fall_relative_energy_loss": free_fall_relative_energy_loss,
            "fixed_static_friction": static_friction.tolist(),
            "compression_final_to_initial_energy_ratio": compression_energy_ratios,
            "restitution_energy_fraction": restitution_fraction,
            "self_collision_enabled_pairs": metadata["physicsV1EnabledPairsPerRobot"],
            "self_collision_filtered_pairs": metadata["physicsV1FilteredPairsPerRobot"],
            "finite_by_seed": finite_by_seed,
            "deterministic_by_seed": deterministic_by_seed,
        },
        "limitations": [
            "No articulation free-motion, collision response, or contact solver behavior was executed.",
            "The integrated Isaac/PhysX gate requires the exact frozen runtime and remains blocked.",
        ],
        "next_stage": "scripted_wrench_level_feasibility_and_anti_reward_hacking",
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
