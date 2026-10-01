"""Run an immutable candidate123 baseline replay through the legacy evaluator.

This is the P0 compatibility replay only.  P1 and later use the v3 strict
telemetry runner so current physical five-digit contact, rather than an
episode-level measured-contact latch, controls the strict window.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from simulation.ppo_tomato_grasp_v3.provenance import atomic_json_write, sha256_file
from simulation.ppo_tomato_grasp_v3.scenario import derive_run_scenario, load_scenario


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO = Path(__file__).resolve().parent / "scenarios" / "v3_large90_nominal.json"
DEFAULT_ISAAC_PYTHON = Path("/home/miobarnacle/anaconda3/envs/isaacsim/bin/python")
ISAACLAB_SOURCE = Path("/home/miobarnacle/linkbot/IsaacLab-2.3.2/source/isaaclab")


def _csv(values: list[float]) -> str:
    return ",".join(str(float(value)) for value in values)


def build_command(
    scenario: dict, *, python: Path, output: Path, seed: int, num_envs: int, steps: int
) -> list[str]:
    obj = scenario["object"]
    geometry = scenario["contact_geometry"]
    strict = scenario["strict"]
    safety = scenario["safety"]
    controller = scenario["controller"]
    checkpoint = PROJECT / scenario["evaluation"]["checkpoint"]
    evaluator = PROJECT / "simulation/ppo_tomato_grasp_v2/evaluate_v2.py"
    command = [
        str(python), str(evaluator),
        "--checkpoint", str(checkpoint),
        "--output", str(output),
        "--profile", scenario["evaluation"]["profile"],
        "--num-envs", str(num_envs),
        "--steps", str(steps),
        "--seed", str(seed),
        "--stage", "1",
        "--strict-coverage-deg", str(strict["angular_coverage_min_deg"]),
        "--strict-hold-steps", str(strict["diagnostic_hold_steps"]),
        "--physical-fingertip-proxy",
        "--proxy-only-collision",
        "--fingertip-proxy-radius", str(geometry["fingertip_proxy_radius_m"]),
        "--strict-require-measured-distal-contact",
        "--tomato-contact-offset", str(obj["contact_offset_m"]),
        "--tomato-rest-offset", str(obj["rest_offset_m"]),
        "--proxy-contact-offset", str(geometry["proxy_contact_offset_m"]),
        "--proxy-rest-offset", str(geometry["proxy_rest_offset_m"]),
        "--reset-clearance", str(obj["reset_clearance_m"]),
        "--cup-half-extent", _csv(geometry["palm_cup_inner_half_extent_m"]),
        "--rim-height", str(geometry["palm_cup_rim_height_m"]),
        "--command-rate-rad-s", str(safety["command_rate_limit_rad_s"]),
        "--safe-force-n", str(safety["safe_force_n"]),
        "--force-ceiling", _csv(safety["force_ceiling_n"]),
        "--mass-min", str(obj["accepted_mass_kg"][0]),
        "--mass-max", str(obj["accepted_mass_kg"][1]),
        "--density-min", str(obj["density_kg_m3"]),
        "--density-max", str(obj["density_kg_m3"]),
        "--axis-range-m", str(obj["semi_axes_m"][0]),
        "--contact-stiffness-n-m", str(obj["contact_stiffness_n_m"]),
        "--distal-filter-alpha", str(controller["distal_filter_alpha"]),
        "--residual-base-controller",
        "--skip-actor-forward",
        "--residual-scale", str(controller["residual_scale"]),
        "--residual-mask", controller["residual_mask"],
        # The candidate begins with a negative value.  Keep option and value
        # in one argv token so argparse cannot mistake the CSV for a flag.
        "--residual-target-action=" + _csv(controller["target_action"]),
        "--residual-approach-steps", str(controller["approach_steps"]),
        "--residual-per-digit-steps", str(controller["per_digit_steps"]),
        "--residual-hold-steps", str(controller["hold_steps"]),
        "--residual-approach-smoothing", str(controller["approach_smoothing"]),
        "--residual-thumb-delay-steps", str(controller["thumb_delay_steps"]),
        "--residual-thumb-ramp-steps", str(controller["thumb_ramp_steps"]),
        "--residual-thumb-first-steps", str(controller["thumb_first_steps"]),
        "--residual-thumb-first-global-scale", str(controller["thumb_first_global_scale"]),
        "--residual-lower-force-per-digit", _csv(controller["straight_digit_force_lower_n"]),
        "--residual-upper-force-per-digit", _csv(controller["straight_digit_force_upper_n"]),
        "--residual-insertion-step-per-digit", _csv(controller["insertion_step_per_digit"]),
        "--residual-retreat-per-digit", _csv(controller["retreat_step_per_digit"]),
        "--residual-thumb-hold-offset", _csv(controller["thumb_hold_target_offset"]),
        "--residual-thumb-lower-force", str(controller["thumb_force_band_n"][0]),
        "--residual-thumb-upper-force", str(controller["thumb_force_band_n"][1]),
        "--residual-thumb-search-amplitude", _csv(controller["thumb_search_amplitude"]),
        "--residual-thumb-search-period", str(controller["thumb_search_period_steps"]),
        "--residual-thumb-contact-gate-steps", str(controller["thumb_contact_gate_steps"]),
        "--residual-thumb-contact-latch-force", str(controller["thumb_contact_latch_force_n"]),
        "--residual-thumb-contact-latch-steps", str(controller["thumb_contact_latch_steps"]),
        "--headless",
        "--device", "cuda:0",
    ]
    if controller["thumb_contact_latch_freeze_coarse"]:
        command.append("--residual-thumb-contact-latch-freeze-coarse")
    if not geometry["virtual_distal_contact_for_shaping"]:
        command.append("--disable-virtual-distal-contact")
    return command


def _assert_resolved_match(report: dict, scenario: dict) -> None:
    resolved = report["resolved_contact_geometry"]
    obj = scenario["object"]
    geometry = scenario["contact_geometry"]
    expected = {
        "tomato_contact_offset_m": obj["contact_offset_m"],
        "tomato_rest_offset_m": obj["rest_offset_m"],
        "fingertip_proxy_radius_m": geometry["fingertip_proxy_radius_m"],
        "fingertip_proxy_contact_offset_m": geometry["proxy_contact_offset_m"],
        "fingertip_proxy_rest_offset_m": geometry["proxy_rest_offset_m"],
        "contact_stiffness_n_m": obj["contact_stiffness_n_m"],
        "command_rate_limit_rad_s": scenario["safety"]["command_rate_limit_rad_s"],
    }
    mismatches = {
        key: {"expected": value, "actual": resolved.get(key)}
        for key, value in expected.items()
        if abs(float(resolved.get(key, float("nan"))) - float(value)) > 1.0e-9
    }
    if mismatches:
        raise RuntimeError(f"resolved evaluator configuration drifted: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--python", type=Path, default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--purpose", default="p0_matched_baseline")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    scenario, scenario_hash = load_scenario(args.scenario)
    defaults = scenario["evaluation"]
    seed = defaults["seed"] if args.seed is None else args.seed
    num_envs = defaults["num_envs"] if args.num_envs is None else args.num_envs
    steps = defaults["steps"] if args.steps is None else args.steps
    run_scenario, run_hash = derive_run_scenario(
        scenario, seed=seed, num_envs=num_envs, steps=steps, purpose=args.purpose
    )
    output = args.output.expanduser().resolve()
    command = build_command(
        scenario, python=args.python.expanduser().resolve(), output=output,
        seed=seed, num_envs=num_envs, steps=steps,
    )
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = {
        "version": "v3_p0_run_manifest_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": run_scenario,
        "scenario_sha256": scenario_hash,
        "run_sha256": run_hash,
        "checkpoint_sha256": sha256_file(PROJECT / scenario["evaluation"]["checkpoint"]),
        "legacy_evaluator_sha256": sha256_file(PROJECT / "simulation/ppo_tomato_grasp_v2/evaluate_v2.py"),
        "isaaclab_source": str(ISAACLAB_SOURCE),
        "command": command,
        "hardware_command_allowed": False,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(shlex.join(command))
        return
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite P0 artifact: {output} or {manifest_path}")
    if not (ISAACLAB_SOURCE / "isaaclab/__init__.py").is_file():
        raise FileNotFoundError(f"Isaac Lab 2.3.2 source is missing: {ISAACLAB_SOURCE}")
    atomic_json_write(manifest_path, manifest)
    child_env = os.environ.copy()
    inherited = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in (PROJECT, ISAACLAB_SOURCE) if str(path)
    ) + (os.pathsep + inherited if inherited else "")
    completed = subprocess.run(command, cwd=PROJECT, env=child_env, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    report = json.loads(output.read_text(encoding="utf-8"))
    _assert_resolved_match(report, scenario)
    report["v3_provenance"] = {
        "scenario_sha256": scenario_hash,
        "run_sha256": run_hash,
        "manifest": str(manifest_path),
        "legacy_compatibility_replay": True,
        "strict_current_physical_five_digit_contact_audited": False,
    }
    atomic_json_write(output, report)
    print(output)


if __name__ == "__main__":
    main()
