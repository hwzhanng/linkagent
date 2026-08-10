"""CPU-only distribution and sensitivity audit for Physics-v2 priors.

The reported Pearson coefficients are screening indicators, not calibrated
causal effects.  The script samples only the analytic Physics-v2 prior models;
it does not import Isaac Sim, run PhysX, train a policy, or connect hardware.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from simulation.tomato_pick.physics_v2 import load_physics_v2_profile
from simulation.tomato_pick.physics_v2_models import (
    quantize_contact_material_bank,
    sample_correlated_fruit_contact_profiles,
    sample_stem_profiles_v2,
    sample_uncertainty_profiles_v2,
)


PROFILE_PATH = ROOT / "simulation/config/tomato_pick_physics_v2.yaml"
MODEL_PATH = ROOT / "simulation/tomato_pick/physics_v2_models.py"
MODES = ("learnability", "literature_axial", "bend_twist_prior")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    quantiles = torch.quantile(flat, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64))
    return {
        "minimum": float(flat.min()),
        "p05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "mean": float(flat.mean()),
        "p95": float(quantiles[2]),
        "maximum": float(flat.max()),
    }


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left64 = left.to(dtype=torch.float64).reshape(-1)
    right64 = right.to(dtype=torch.float64).reshape(-1)
    left_centered = left64 - left64.mean()
    right_centered = right64 - right64.mean()
    denominator = torch.sqrt((left_centered.square().sum()) * (right_centered.square().sum()))
    if float(denominator) == 0.0:
        return 0.0
    return float((left_centered * right_centered).sum() / denominator)


def _rank_correlations(
    inputs: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
) -> dict[str, list[dict[str, float | str]]]:
    result: dict[str, list[dict[str, float | str]]] = {}
    for output_name, output in outputs.items():
        values = [
            {"input": input_name, "pearson_r": _pearson(value, output)}
            for input_name, value in inputs.items()
        ]
        result[output_name] = sorted(
            values, key=lambda item: abs(float(item["pearson_r"])), reverse=True
        )
    return result


def build_sensitivity_audit(*, samples: int, seed: int) -> dict[str, object]:
    if samples < 1000:
        raise ValueError("samples must be at least 1000")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fruit = sample_correlated_fruit_contact_profiles(
        samples, generator=generator, device="cpu", dtype=torch.float64
    )
    profile = load_physics_v2_profile()
    bank_cfg = profile["fruit_contact_v2"]["material_bank"]
    bank = quantize_contact_material_bank(
        fruit,
        friction_bins=int(bank_cfg["friction_bins"]),
        compliance_bins=int(bank_cfg["compliance_bins"]),
    )
    stem = {mode: sample_stem_profiles_v2(fruit, mode=mode) for mode in MODES}
    uncertainty = sample_uncertainty_profiles_v2(
        samples,
        generator=torch.Generator(device="cpu").manual_seed(seed + 1),
        device="cpu",
        dtype=torch.float64,
    )

    input_latents = {
        "size": fruit["size_latent"],
        "ripeness": fruit["ripeness"],
        "wetness": fruit["wetness"],
        "surface_texture": fruit["surface_texture"],
        "biological": fruit["biological_latent"],
    }
    selected_outputs = {
        "mass_kg": fruit["mass_kg"],
        "contact_stiffness_n_m": fruit["contact_stiffness_n_m"],
        "static_friction": fruit["static_friction"],
        "learnability_tensile_force_n": stem["learnability"]["stem_tensile_force_n"],
        "bend_twist_bending_moment_nm": stem["bend_twist_prior"]["stem_bending_moment_nm"],
        "critical_duration_s": stem["learnability"]["stem_critical_duration_s"],
    }
    distributions = {
        name: _summary(values)
        for name, values in {
            **selected_outputs,
            "dynamic_friction": fruit["dynamic_friction"],
            "contact_damping_n_s_m": fruit["contact_damping_n_s_m"],
            "hand_response_scale": uncertainty["hand_response_scale"],
            "hand_deadband_rad": uncertainty["hand_deadband_rad"],
            "hand_backlash_rad": uncertainty["hand_backlash_rad"],
            "tactile_gain": uncertainty["tactile_gain"],
            "tactile_bias_n": uncertainty["tactile_bias_n"],
            "tactile_dropout_probability": uncertainty["tactile_dropout_probability"],
        }.items()
    }
    feasibility = {
        mode: {
            "translation": float(values["stem_translation_feasible"].double().mean()),
            "bending": float(values["stem_bending_feasible"].double().mean()),
            "torsion": float(values["stem_torsion_feasible"].double().mean()),
        }
        for mode, values in stem.items()
    }
    devices = {
        str(value.device)
        for group in (fruit, uncertainty)
        for value in group.values()
        if isinstance(value, torch.Tensor)
    }
    finite = all(
        bool(torch.isfinite(value).all())
        for group in (fruit, uncertainty, *stem.values())
        for value in group.values()
        if isinstance(value, torch.Tensor) and value.dtype != torch.bool
    )
    unique_bins, bin_counts = torch.unique(bank["material_bin"], return_counts=True)
    material_coverage = {
        "observed_bin_count": int(unique_bins.numel()),
        "expected_bin_count": int(bank_cfg["friction_bins"] * bank_cfg["compliance_bins"]),
        "minimum_samples_per_observed_bin": int(bin_counts.min()),
        "maximum_samples_per_observed_bin": int(bin_counts.max()),
    }
    correlations = _rank_correlations(input_latents, selected_outputs)
    checks = {
        "all_tensors_on_cpu": devices == {"cpu"},
        "all_values_finite": finite,
        # Correlated priors make the corner bins legitimately sparse.  The
        # fixed seed-42 stage-3 gate separately proves all 16 bins are
        # reachable; this multi-seed screening audit requires broad coverage
        # without pretending every finite sample must occupy every corner.
        "broad_material_bank_coverage": material_coverage["observed_bin_count"]
        >= 12,
        "mass_positive_and_bounded": distributions["mass_kg"]["minimum"] > 0.045
        and distributions["mass_kg"]["maximum"] < 0.22,
        "dynamic_friction_below_static": bool(
            (fruit["dynamic_friction"] < fruit["static_friction"]).all()
        ),
        "ripeness_reduces_stiffness": _pearson(
            fruit["ripeness"], fruit["contact_stiffness_n_m"]
        )
        < -0.70,
        "wetness_reduces_static_friction": _pearson(
            fruit["wetness"], fruit["static_friction"]
        )
        < -0.60,
        "learnability_translation_feasible": feasibility["learnability"]["translation"]
        == 1.0,
        "literature_axial_not_silently_weakened": feasibility["literature_axial"][
            "translation"
        ]
        < 0.01,
        "damage_proxy_remains_diagnostic_only": profile["fruit_damage_proxy_v2"][
            "affects_reward"
        ]
        is False
        and profile["fruit_damage_proxy_v2"]["affects_termination"] is False,
    }
    return {
        "schema_version": 1,
        "audit_id": "tomato_pick_physics_v2_cpu_parameter_sensitivity",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": (
            "analytic prior screening only; no Isaac, PhysX, PPO, GPU, hardware, "
            "or cultivar-calibration claim"
        ),
        "samples": samples,
        "seed": seed,
        "source_identity": {
            "profile": str(PROFILE_PATH.relative_to(ROOT)),
            "profile_sha256": _sha256(PROFILE_PATH),
            "model": str(MODEL_PATH.relative_to(ROOT)),
            "model_sha256": _sha256(MODEL_PATH),
        },
        "devices": sorted(devices),
        "distributions": distributions,
        "screening_correlations": correlations,
        "correlation_interpretation": (
            "Absolute Pearson correlation ranks first-order screening influence only; "
            "correlated inputs prevent a causal or Sobol interpretation."
        ),
        "stem_mode_feasibility": feasibility,
        "material_bank_coverage": material_coverage,
        "checks": checks,
        "passed": all(checks.values()),
        "calibration_status": "hardware_and_cultivar_measurements_pending",
    }


def render_markdown(report: dict[str, object]) -> str:
    distributions = report["distributions"]
    correlations = report["screening_correlations"]
    feasibility = report["stem_mode_feasibility"]
    checks = report["checks"]
    assert isinstance(distributions, dict)
    assert isinstance(correlations, dict)
    assert isinstance(feasibility, dict)
    assert isinstance(checks, dict)
    lines = [
        "# Physics-v2 CPU parameter sensitivity audit",
        "",
        f"Samples: `{report['samples']}` · Seed: `{report['seed']}` · Passed: `{report['passed']}`",
        "",
        "This report screens analytic priors only. It is not tomato calibration and does not use Isaac Sim or a GPU.",
        "",
        "## Key distributions",
        "",
        "| Quantity | P05 | Median | P95 |",
        "|---|---:|---:|---:|",
    ]
    for name, values in distributions.items():
        assert isinstance(values, dict)
        lines.append(
            f"| {name} | {values['p05']:.6g} | {values['median']:.6g} | {values['p95']:.6g} |"
        )
    lines.extend(("", "## Strongest screening relationship", ""))
    for name, ranking in correlations.items():
        assert isinstance(ranking, list) and ranking
        strongest = ranking[0]
        lines.append(
            f"- `{name}`: `{strongest['input']}` (Pearson r = `{strongest['pearson_r']:.4f}`)"
        )
    lines.extend(("", "## Stem feasibility fractions", "", "| Mode | Translation | Bending | Torsion |", "|---|---:|---:|---:|"))
    for name, values in feasibility.items():
        assert isinstance(values, dict)
        lines.append(
            f"| {name} | {values['translation']:.4f} | {values['bending']:.4f} | {values['torsion']:.4f} |"
        )
    lines.extend(("", "## Gate checks", ""))
    for name, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(
        (
            "",
            "> These coefficients are not causal sensitivity indices because several sampled parameters are intentionally correlated.",
            "",
        )
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    report = build_sensitivity_audit(samples=args.samples, seed=args.seed)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    markdown = args.markdown or output.with_suffix(".md")
    if not markdown.is_absolute():
        markdown = ROOT / markdown
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["passed"]}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
