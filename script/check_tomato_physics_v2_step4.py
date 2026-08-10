"""Pure feasibility and load-duration gate for Physics-v2 stem modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from simulation.tomato_pick.physics_v1 import update_stem_damage
from simulation.tomato_pick.physics_v2 import load_physics_v2_profile
from simulation.tomato_pick.physics_v2_models import (
    sample_correlated_fruit_contact_profiles,
    sample_stem_profiles_v2,
    stem_load_components_v2,
)


def _fraction(value: torch.Tensor) -> float:
    return float(value.float().mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fruit = sample_correlated_fruit_contact_profiles(
        args.samples, generator=torch.Generator().manual_seed(args.seed)
    )
    modes = {
        name: sample_stem_profiles_v2(fruit, mode=name)
        for name in ("learnability", "literature_axial", "bend_twist_prior")
    }
    mode_metrics = {}
    for name, values in modes.items():
        mode_metrics[name] = {
            "tensile_force_n": {
                "minimum": float(values["stem_tensile_force_n"].min()),
                "median": float(values["stem_tensile_force_n"].median()),
                "maximum": float(values["stem_tensile_force_n"].max()),
            },
            "bending_moment_nm": {
                "minimum": float(values["stem_bending_moment_nm"].min()),
                "median": float(values["stem_bending_moment_nm"].median()),
                "maximum": float(values["stem_bending_moment_nm"].max()),
            },
            "torsional_moment_nm": {
                "minimum": float(values["stem_torsional_moment_nm"].min()),
                "median": float(values["stem_torsional_moment_nm"].median()),
                "maximum": float(values["stem_torsional_moment_nm"].max()),
            },
            "translation_feasible_fraction": _fraction(values["stem_translation_feasible"]),
            "bending_feasible_fraction": _fraction(values["stem_bending_feasible"]),
            "torsion_feasible_fraction": _fraction(values["stem_torsion_feasible"]),
        }

    profile = load_physics_v2_profile()
    common = profile["stem_modes_v2"]["common"]
    quantiles = torch.tensor([0, args.samples // 2, args.samples - 1])
    ordered = torch.argsort(modes["learnability"]["stem_tensile_force_n"])[quantiles]
    critical = modes["learnability"]["stem_critical_duration_s"][ordered]
    damage = torch.zeros(3)
    attached = torch.ones(3, dtype=torch.bool)
    damage, spike_break = update_stem_damage(
        damage,
        torch.ones(3),
        dt_s=1.0 / 120.0,
        critical_duration_s=critical,
        onset_ratio=common["damage_onset_ratio"],
        recovery_time_s=common["damage_recovery_time_s"],
        attached=attached,
    )
    one_substep_damage = damage.clone()
    for _ in range(16):
        damage, sustained_break = update_stem_damage(
            damage,
            torch.ones(3),
            dt_s=1.0 / 120.0,
            critical_duration_s=critical,
            onset_ratio=common["damage_onset_ratio"],
            recovery_time_s=common["damage_recovery_time_s"],
            attached=attached,
        )

    axis = torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1)
    components = stem_load_components_v2(
        torch.tensor([[0.0, 0.0, -10.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.1]]),
        axis,
        torch.full((4,), 10.0),
        torch.full((4,), 5.0),
        torch.full((4,), 0.2),
        torch.full((4,), 0.1),
    )
    checks = {
        "learnability_is_explicitly_safe_capacity_bounded": mode_metrics["learnability"]["translation_feasible_fraction"] == 1.0,
        "literature_axial_is_not_silently_weakened": mode_metrics["literature_axial"]["tensile_force_n"]["minimum"] >= 20.0 and mode_metrics["literature_axial"]["translation_feasible_fraction"] < 0.01,
        "bend_twist_moment_paths_are_conservatively_feasible": mode_metrics["bend_twist_prior"]["bending_feasible_fraction"] > 0.99 and mode_metrics["bend_twist_prior"]["torsion_feasible_fraction"] > 0.99,
        "single_substep_does_not_break": not bool(spike_break.any()),
        "weak_median_strong_sustained_load_breaks": bool(sustained_break.all()),
        "four_unit_load_modes_are_present": bool(torch.allclose(components["combined"], torch.ones(4), atol=1.0e-6)),
        "active_mode_is_recorded": profile["stem_modes_v2"]["active_offline_mode"] == "learnability",
    }
    report = {
        "schema_version": 1,
        "gate_id": "tomato_pick_physics_v2_step4",
        "passed": all(checks.values()),
        "checks": checks,
        "samples": args.samples,
        "seed": args.seed,
        "mode_metrics": mode_metrics,
        "load_duration": {
            "critical_duration_s": [float(value) for value in critical],
            "damage_after_one_substep": [float(value) for value in one_substep_damage],
            "damage_after_sustained_load": [float(value) for value in damage],
        },
        "next_stage": "actuator_and_tactile_uncertainty",
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
