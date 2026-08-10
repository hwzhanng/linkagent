"""Pure CPU gate for Physics-v2 fruit/contact and damage diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from simulation.tomato_pick.physics_v2 import load_physics_v2_profile
from simulation.tomato_pick.physics_v2_models import (
    fixed_fruit_contact_cases,
    initial_fruit_damage_state,
    linear_compliant_contact_force,
    quantize_contact_material_bank,
    sample_correlated_fruit_contact_profiles,
    update_fruit_damage_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1000:
        raise ValueError("samples must be >= 1000")
    profile = load_physics_v2_profile()
    values = sample_correlated_fruit_contact_profiles(
        args.samples, generator=torch.Generator().manual_seed(args.seed)
    )
    expected_mass = 4.0 / 3.0 * math.pi * values["axes_m"].prod(dim=1) * values[
        "density_kg_m3"
    ]
    stiffness_ripeness = float(
        torch.corrcoef(torch.stack((values["contact_stiffness_n_m"], values["ripeness"])))[0, 1]
    )
    friction_wetness = float(
        torch.corrcoef(torch.stack((values["static_friction"], values["wetness"])))[0, 1]
    )
    bank_cfg = profile["fruit_contact_v2"]["material_bank"]
    bank = quantize_contact_material_bank(
        values,
        friction_bins=bank_cfg["friction_bins"],
        compliance_bins=bank_cfg["compliance_bins"],
    )
    cases = fixed_fruit_contact_cases()
    indentation = torch.linspace(-0.001, 0.006, 100)
    force = linear_compliant_contact_force(
        indentation,
        torch.zeros_like(indentation),
        torch.full_like(indentation, 5500.0),
        torch.full_like(indentation, 28.0),
    )
    damage_cfg = profile["fruit_damage_proxy_v2"]
    state = initial_fruit_damage_state(torch.zeros((2, 5)))
    for step in range(120):
        safe_force = min(4.0, step * 0.2)
        state = update_fruit_damage_diagnostics(
            state,
            torch.stack((torch.full((5,), safe_force), torch.full((5,), 7.0))),
            torch.stack((torch.full((5,), safe_force / 5500.0), torch.full((5,), 0.006))),
            dt_s=1.0 / 120.0,
            config=damage_cfg,
        )
    checks = {
        "mass_matches_geometry_and_density": bool(torch.allclose(values["mass_kg"], expected_mass, atol=1.0e-8)),
        "mass_range_is_plausible": float(values["mass_kg"].min()) > 0.045 and float(values["mass_kg"].max()) < 0.22,
        "ripeness_reduces_stiffness": stiffness_ripeness < -0.70,
        "wetness_reduces_friction": friction_wetness < -0.60,
        "dynamic_friction_below_static": bool((values["dynamic_friction"] < values["static_friction"]).all()),
        "all_material_bins_exercised": len(torch.unique(bank["material_bin"])) == 16,
        "fixed_stiffness_order": bool(cases["soft"]["effective_stiffness_n_m"] < cases["nominal"]["effective_stiffness_n_m"] < cases["firm"]["effective_stiffness_n_m"]),
        "contact_is_unilateral_and_monotonic": float(force[0]) == 0.0 and bool((force[1:] >= force[:-1]).all()),
        "safe_slow_contact_has_zero_damage_score": float(state["damage_score"][0].max()) == 0.0,
        "damaging_contact_accumulates_risk": float(state["damage_score"][1].min()) > 0.50,
        "damage_is_diagnostic_only": damage_cfg["affects_reward"] is False and damage_cfg["affects_termination"] is False,
    }
    report = {
        "schema_version": 1,
        "gate_id": "tomato_pick_physics_v2_step3",
        "passed": all(checks.values()),
        "checks": checks,
        "samples": args.samples,
        "seed": args.seed,
        "metrics": {
            "mass_kg_min": float(values["mass_kg"].min()),
            "mass_kg_max": float(values["mass_kg"].max()),
            "stiffness_ripeness_correlation": stiffness_ripeness,
            "friction_wetness_correlation": friction_wetness,
            "material_bin_count": len(torch.unique(bank["material_bin"])),
            "safe_damage_score": float(state["damage_score"][0].max()),
            "damaging_damage_score": float(state["damage_score"][1].min()),
        },
        "next_stage": "stem_mode_separation_and_feasibility",
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
