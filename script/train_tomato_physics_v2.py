"""Plan or explicitly start Physics-v2 PPO from scratch after Q1-Q6 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

from script.physics_v2_audit_inputs import (
    AUDITED_RUNTIME_INPUTS,
    require_current_input_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_isaac_root_value = os.environ.get("LINKAGENT_ISAACLAB_ROOT")
if not _isaac_root_value:
    raise RuntimeError("set LINKAGENT_ISAACLAB_ROOT to the Isaac Lab checkout")
ISAAC_ROOT = Path(_isaac_root_value).expanduser().resolve()
ISAAC_ENV = ISAAC_ROOT / "env_isaaclab"
PROFILE_PATH = (
    PROJECT_ROOT
    / "simulation/config/tomato_pick_physics_v2_public_prior_v1.json"
)
QUALIFICATION_ROOT = PROJECT_ROOT / "logs/qualification/tomato_physics_v2"
CPU_GATE = QUALIFICATION_ROOT / "public_prior_cpu_gate.json"
ISAAC_GATE = QUALIFICATION_ROOT / "isaac_smoke_gate_64x128.json"
DEFAULT_MANIFEST = QUALIFICATION_ROOT / "training_manifest.json"
CONFIRMATION = "START_PHYSICS_V2_PUBLIC_PRIOR_FROM_SCRATCH"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_passed(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("passed") is not True:
        raise RuntimeError(f"qualification report did not pass: {path}")
    return value


def build_plan(*, num_envs: int, seed: int) -> dict:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if profile.get("profile_id") != "tomato_pick_physics_v2_public_prior_v1":
        raise RuntimeError("unexpected Physics-v2 profile")
    if profile.get("status") != (
        "qualified_ready_for_external_audit_training_not_started"
    ):
        raise RuntimeError("Physics-v2 profile is not audit-ready")
    if profile.get("qualification", {}).get("passed") is not True:
        raise RuntimeError("Physics-v2 profile qualification is not marked passed")
    training = profile["training"]
    gate_spec = training["first_success_gate"]
    if num_envs != int(gate_spec["num_envs"]) or seed != int(gate_spec["seed"]):
        raise ValueError(
            "Physics-v2 initial plan must match the audited first-success "
            "environment count and seed"
        )
    first_success_gate = PROJECT_ROOT / str(gate_spec["output"])
    if not (
        training["from_scratch_required"] is True
        and training["physics_v1_checkpoint_resume_allowed"] is False
    ):
        raise RuntimeError("Physics-v2 from-scratch boundary changed")
    cpu = _read_passed(CPU_GATE)
    isaac = _read_passed(ISAAC_GATE)
    first_success = _read_passed(first_success_gate)
    if cpu.get("qualification") != "physics_v2_public_prior_cpu_q1_q6":
        raise RuntimeError("unexpected Physics-v2 CPU qualification identity")
    if set(cpu.get("gates", {})) != {
        "Q1_morphology",
        "Q2_pure_axial",
        "Q3_bending",
        "Q4_torsion",
        "Q5_twist_then_pull",
        "Q6_strategy_sanity",
    }:
        raise RuntimeError("Physics-v2 CPU qualification is missing Q1-Q6")
    if cpu.get("engineering_gap_prior_audit", {}).get("passed") is not True:
        raise RuntimeError("Physics-v2 engineering gap priors did not pass")
    q6_checks = cpu["gates"]["Q6_strategy_sanity"].get("checks", {})
    required_q6_checks = {
        "moderate_pull_does_not_instantly_break_intact_az",
        "pure_axial_overload_breaks",
        "pure_shear_overload_breaks",
        "pure_bending_overload_breaks",
        "half_axial_half_torsion_is_subcritical",
        "360deg_twist_reduces_integrity_but_keeps_attachment",
        "staged_moderate_pull_breaks_in_finite_time",
        "continued_twist_breaks_by_public_550deg_bound",
    }
    if not required_q6_checks.issubset(q6_checks) or not all(
        q6_checks[name] for name in required_q6_checks
    ):
        raise RuntimeError("Physics-v2 Q6 counterfactual strategy matrix is incomplete")
    if isaac.get("num_envs") != 64 or isaac.get("steps") != 128:
        raise RuntimeError("Physics-v2 Isaac gate is not the frozen 64x128 smoke gate")
    if isaac.get("training_started") is not False:
        raise RuntimeError("Physics-v2 Isaac gate unexpectedly started training")
    if len(isaac.get("checks", {})) < 21 or not all(isaac["checks"].values()):
        raise RuntimeError("Physics-v2 Isaac gate is incomplete or failed")
    if (
        first_success.get("qualification")
        != "physics_v2_formal_scale_first_success_gate"
        or first_success.get("num_envs") != int(gate_spec["num_envs"])
        or first_success.get("iterations") != int(gate_spec["updates"])
        or first_success.get("seed") != int(gate_spec["seed"])
        or first_success.get("configured_gate_spec", {}).get("candidate_id")
        != gate_spec["candidate_id"]
    ):
        raise RuntimeError("Physics-v2 first-success gate identity changed")
    require_current_input_sha256(
        first_success,
        PROJECT_ROOT,
        (*AUDITED_RUNTIME_INPUTS, "script/gate_tomato_physics_v2_first_success.py"),
        report_name=str(first_success_gate.relative_to(PROJECT_ROOT)),
    )

    tracked = [
        PROFILE_PATH,
        PROJECT_ROOT / profile["public_prior"]["path"],
        PROJECT_ROOT / profile["engineering_prior"]["path"],
        PROJECT_ROOT / "simulation/tomato_pick/physics_v2.py",
        PROJECT_ROOT / "simulation/tomato_pick/isaac/env.py",
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v2/__init__.py",
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v2/assets.py",
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v2/env_cfg.py",
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v2/env.py",
        PROJECT_ROOT
        / "simulation/tomato_pick/isaac/physics_v2/agents/rsl_rl_ppo_cfg.py",
        PROJECT_ROOT / "script/qualify_tomato_physics_v2_public_priors.py",
        PROJECT_ROOT / "script/physics_v2_audit_inputs.py",
        PROJECT_ROOT / "script/check_tomato_physics_v2.py",
        PROJECT_ROOT / "script/evaluate_tomato_checkpoint.py",
        PROJECT_ROOT / "script/run_tomato_physics_v2_curve_controller.sh",
        PROJECT_ROOT / "script/train_tomato_physics_v2.py",
        CPU_GATE,
        ISAAC_GATE,
        first_success_gate,
    ]
    input_sha256 = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in tracked
    }
    controller = PROJECT_ROOT / training["curve_controller"]
    command = [
        str(controller),
        "--execute",
        "START_PHYSICS_V2_CURVE_CONTROLLER",
    ]
    initial_segment_command = [
        str(ISAAC_ENV / "bin/python"),
        str(ISAAC_ROOT / "scripts/reinforcement_learning/train.py"),
        "--rl_library",
        "rsl_rl",
        "--external_callback",
        "simulation.tomato_pick.isaac.physics_v2.register_task",
        "--task",
        profile["task_id"],
        "--num_envs",
        str(num_envs),
        "--seed",
        str(seed),
        "--max_iterations",
        str(training["segment_updates"]),
        "--logger",
        "wandb",
        "--experiment_name",
        "tomato_pick_physics_v2_public_prior_v1",
        "--run_name",
        "tomato_pick_physics_v2_from0_to400_8192",
        "--viz",
        "none",
    ]
    return {
        "schema_version": 1,
        "task_id": profile["task_id"],
        "profile_id": profile["profile_id"],
        "candidate_id": gate_spec["candidate_id"],
        "status": "ready_for_user_authorized_from_scratch_training",
        "from_scratch": True,
        "physics_v1_checkpoint_loaded": None,
        "checkpoint_loaded": None,
        "num_envs": num_envs,
        "rollout_steps": 32,
        "seed": seed,
        "fixed_total_iterations": None,
        "segment_updates": training["segment_updates"],
        "stop_policy": "curve-based convergence; no fixed update cap",
        "command": command,
        "initial_segment_command": initial_segment_command,
        "environment": {
            "LINKAGENT_ISAACLAB_ROOT": str(ISAAC_ROOT),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONPATH": str(PROJECT_ROOT),
            "TOMATO_REWARD_ITERATION_OFFSET": "0",
            "TOMATO_V2_DISCOVERY_CAPACITY_CURRICULUM": "1",
        },
        "qualification_reports": {
            "Q1_Q6_cpu": str(CPU_GATE.relative_to(PROJECT_ROOT)),
            "isaac_64x128": str(ISAAC_GATE.relative_to(PROJECT_ROOT)),
            "first_success_configured_gate": str(
                first_success_gate.relative_to(PROJECT_ROOT)
            ),
        },
        "input_sha256": input_sha256,
        "hardware_boundary": "no SDK, CAN, or real-hardware connection",
        "hardware_transfer_status": "blocked_pending_hardware_calibration",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation")
    args = parser.parse_args(argv)
    plan = build_plan(num_envs=args.num_envs, seed=args.seed)
    manifest = args.write_manifest or DEFAULT_MANIFEST
    if args.write_manifest is not None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return 0
    if args.confirmation != CONFIRMATION:
        raise RuntimeError("training requires --confirmation " + CONFIRMATION)
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
