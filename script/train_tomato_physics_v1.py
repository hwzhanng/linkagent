"""Validate, plan, or explicitly start Physics-v1 formal PPO from scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_isaac_root_value = os.environ.get("LINKAGENT_ISAACLAB_ROOT")
if not _isaac_root_value:
    raise RuntimeError("set LINKAGENT_ISAACLAB_ROOT to the Isaac Lab checkout")
ISAAC_ROOT = Path(_isaac_root_value).expanduser().resolve()
ISAAC_ENV = ISAAC_ROOT / "env_isaaclab"
PROFILE_PATH = PROJECT_ROOT / "simulation/config/tomato_pick_physics_v1.yaml"
QUALIFICATION_ROOT = PROJECT_ROOT / "logs/qualification/tomato_physics_v1"
DEFAULT_MANIFEST = QUALIFICATION_ROOT / "formal_training_manifest.json"
CONFIRMATION = "START_PHYSICS_V1_FORMAL_FROM_SCRATCH"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_passed(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("passed") is not True:
        raise RuntimeError(f"qualification report did not pass: {path}")
    return value


def build_plan(*, num_envs: int, iterations: int, seed: int) -> dict:
    if num_envs != 8192 or iterations != 1800 or seed != 42:
        raise ValueError("formal Physics-v1 plan is frozen at 8192 envs, 1800 updates, seed 42")
    profile = json.loads(PROFILE_PATH.read_text())
    if profile.get("schema_version") != 1:
        raise RuntimeError("unexpected Physics-v1 profile schema")
    if profile.get("profile_id") != "tomato_pick_physics_v1":
        raise RuntimeError("unexpected Physics-v1 profile id")
    gate = profile["training_gate"]
    if gate["ready"] is not True or gate["formal_training_started"] is not False:
        raise RuntimeError("Physics-v1 formal training gate is not ready-and-stopped")
    if gate["blockers"]:
        raise RuntimeError(f"Physics-v1 still has blockers: {gate['blockers']}")

    asset = profile["robot_asset"]
    source_usd = PROJECT_ROOT / asset["source_usd"]
    derived_usd = PROJECT_ROOT / asset["derived_usd"]
    if _sha256(source_usd) != asset["source_usd_sha256"]:
        raise RuntimeError("Physics-v1 source robot USD hash changed")
    if _sha256(derived_usd) != asset["derived_usd_sha256"]:
        raise RuntimeError("Physics-v1 derived robot USD hash changed")
    if asset["collision_matrix_baked"] is not True:
        raise RuntimeError("Physics-v1 collision matrix is not baked")
    if asset["runtime_authored_collision_pairs"] != 0:
        raise RuntimeError("Physics-v1 still authors collision pairs at runtime")

    feasibility_path = QUALIFICATION_ROOT / "abscission_envelope_feasibility.json"
    isolated_path = QUALIFICATION_ROOT / "isolated_physics_gate_abscission_v1.json"
    preflight_path = (
        QUALIFICATION_ROOT / "ppo_256x4_secure_kinematic_credit_gate.json"
    )
    reward_math_path = QUALIFICATION_ROOT / "reward_math_audit.json"
    comparison_path = QUALIFICATION_ROOT / "clean_v5_training_reward_comparison.json"
    guidance_path = QUALIFICATION_ROOT / "historical_checkpoint_guidance/summary.json"
    first_success_path = (
        QUALIFICATION_ROOT
        / "first_success_8192x50_secure_kinematic_credit_gate.json"
    )
    feasibility = _read_passed(feasibility_path)
    isolated = _read_passed(isolated_path)
    preflight = _read_passed(preflight_path)
    reward_math = _read_passed(reward_math_path)
    comparison = _read_passed(comparison_path)
    guidance = _read_passed(guidance_path)
    first_success = _read_passed(first_success_path)
    if feasibility.get("qualification") != "provisional_abscission_envelope_feasibility":
        raise RuntimeError("abscission feasibility gate lost its provisional qualification")
    if isolated.get("qualification") != "literature_prior_provisional":
        raise RuntimeError("isolated gate lost its provisional qualification")
    if preflight.get("num_envs") != 256 or len(preflight.get("iterations", [])) != 4:
        raise RuntimeError("PPO preflight is not the fixed 256x4 gate")
    preflight_credit = preflight.get("credit_gate", {})
    if not (
        preflight_credit.get("break_substep_secure_support_required") is True
        and preflight_credit.get("break_substep_kinematic_retained_required") is True
    ):
        raise RuntimeError("PPO preflight did not qualify secure+kinematic break credit")
    if reward_math.get("checks", {}).get("discovery_first_success_is_best") is not True:
        raise RuntimeError("reward discovery ordering gate is no longer valid")
    if comparison.get("checks", {}).get("ppo_hyperparameters_exactly_reused") is not True:
        raise RuntimeError("Physics-v1 no longer exactly reuses the Clean-v5 PPO contract")
    if guidance.get("checks", {}).get("released_best_reproduces_over_90_percent") is not True:
        raise RuntimeError("historical checkpoint guidance lost its Clean-v5 reference")
    if first_success.get("num_envs") != 8192 or first_success.get("iterations") != 50:
        raise RuntimeError("first-success report is not the formal-scale 8192x50 gate")
    first_success_iteration = first_success.get("first_success_iteration")
    if first_success_iteration is None or int(first_success_iteration) > 49:
        raise RuntimeError("Physics-v1 did not produce success within the 50-update gate")
    required_tail_checks = {
        "tail_window_complete",
        "tail_success_repeated",
        "tail_retention_persistent",
        "tail_retention_not_regressed",
        "tail_drop_detach_ratio_not_worse",
    }
    first_success_checks = first_success.get("checks", {})
    if any(first_success_checks.get(name) is not True for name in required_tail_checks):
        raise RuntimeError("Physics-v1 first-success gate did not pass tail trends")

    parent_checkpoint = PROJECT_ROOT / profile["parent_evidence"]["checkpoint"]
    if _sha256(parent_checkpoint) != profile["parent_evidence"]["checkpoint_sha256"]:
        raise RuntimeError("frozen Clean-v5 evidence checkpoint hash changed")

    tracked_inputs = [
        PROFILE_PATH,
        PROJECT_ROOT / "module/skills/tomato_pick/hand_synergy_calibrated_v2.py",
        PROJECT_ROOT / "simulation/tomato_pick/physics_v1.py",
        PROJECT_ROOT / "simulation/tomato_pick/reset_v1.py",
        PROJECT_ROOT / "simulation/tomato_pick/reward.py",
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v1/env_cfg.py",
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v1/env.py",
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v1/assets.py",
        PROJECT_ROOT
        / "simulation/tomato_pick/isaac/physics_v1/agents/rsl_rl_ppo_cfg.py",
        PROJECT_ROOT / "script/audit_tomato_checkpoint_guidance.py",
        PROJECT_ROOT / "script/audit_tomato_physics_v1_clean_v5_contract.py",
        PROJECT_ROOT / "script/gate_tomato_physics_v1_first_success.py",
        derived_usd,
        feasibility_path,
        isolated_path,
        preflight_path,
        reward_math_path,
        comparison_path,
        guidance_path,
        first_success_path,
    ]
    input_hashes = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in tracked_inputs
    }
    command = [
        str(ISAAC_ENV / "bin" / "python"),
        str(ISAAC_ROOT / "scripts/reinforcement_learning/train.py"),
        "--rl_library",
        "rsl_rl",
        "--external_callback",
        "simulation.tomato_pick.isaac.physics_v1.register_task",
        "--task",
        "LinkAgent-TomatoPick-Physics-v1",
        "--num_envs",
        str(num_envs),
        "--seed",
        str(seed),
        "--max_iterations",
        str(iterations),
        "--logger",
        "wandb",
        "--experiment_name",
        "tomato_pick_physics_v1",
        "--run_name",
        "tomato_pick_physics_v1_from_scratch_8192",
        "--viz",
        "none",
    ]
    return {
        "schema_version": 1,
        "task_id": "LinkAgent-TomatoPick-Physics-v1",
        "status": "formal_training_ready_not_started",
        "qualification": "literature_prior_provisional",
        "formal_training_started": False,
        "from_scratch": True,
        "checkpoint_loaded": None,
        "num_envs": num_envs,
        "rollout_steps": 32,
        "iterations": iterations,
        "seed": seed,
        "total_transitions_if_completed": num_envs * 32 * iterations,
        "logger": "wandb",
        "wandb_project": "PPO",
        "reward_iteration_offset": 0,
        "command": command,
        "environment": {
            "LINKAGENT_ISAACLAB_ROOT": str(ISAAC_ROOT),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONPATH": str(PROJECT_ROOT),
            "TOMATO_REWARD_ITERATION_OFFSET": "0",
        },
        "qualification_reports": {
            "abscission_feasibility": str(feasibility_path.relative_to(PROJECT_ROOT)),
            "isolated_physics": str(isolated_path.relative_to(PROJECT_ROOT)),
            "ppo_256x4": str(preflight_path.relative_to(PROJECT_ROOT)),
            "reward_math": str(reward_math_path.relative_to(PROJECT_ROOT)),
            "clean_v5_contract": str(comparison_path.relative_to(PROJECT_ROOT)),
            "historical_checkpoint_guidance": str(guidance_path.relative_to(PROJECT_ROOT)),
            "first_success_8192x50": str(first_success_path.relative_to(PROJECT_ROOT)),
        },
        "input_sha256": input_hashes,
        "hardware_boundary": "no SDK, CAN, or real-hardware connection",
        "hardware_transfer_status": "blocked_pending_user_measurements",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation")
    args = parser.parse_args(argv)
    plan = build_plan(
        num_envs=args.num_envs, iterations=args.iterations, seed=args.seed
    )
    manifest = args.write_manifest or DEFAULT_MANIFEST
    if args.write_manifest is not None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    if args.confirmation != CONFIRMATION:
        raise RuntimeError(
            "formal training requires --confirmation " + CONFIRMATION
        )
    environment = dict(os.environ)
    environment.update(plan["environment"])
    environment["VIRTUAL_ENV"] = str(ISAAC_ENV)
    environment.pop("CONDA_PREFIX", None)
    environment["PATH"] = os.pathsep.join(
        (str(ISAAC_ENV / "bin"), environment.get("PATH", ""))
    )
    return subprocess.run(
        plan["command"], cwd=PROJECT_ROOT, env=environment, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
