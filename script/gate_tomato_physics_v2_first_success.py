"""Run the frozen formal-scale first-success gate for Physics-v2.

The gate is a bounded from-scratch diagnostic, not formal training.  Its
environment count, update budget and seed come from the audited Physics-v2
profile.  The released default remains 8192 environments for 50 updates.  A
longer gate therefore requires an explicit, reviewable profile change rather
than an ad-hoc command-line override.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Sequence

from script.physics_v2_audit_inputs import (
    AUDITED_RUNTIME_INPUTS,
    input_sha256,
    require_current_input_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_isaac_root_value = os.environ.get("LINKAGENT_ISAACLAB_ROOT")
if not _isaac_root_value:
    raise RuntimeError("set LINKAGENT_ISAACLAB_ROOT to the Isaac Lab checkout")
ISAAC_ROOT = Path(_isaac_root_value).expanduser().resolve()
ISAAC_ENV = ISAAC_ROOT / "env_isaaclab"
QUALIFICATION_ROOT = PROJECT_ROOT / "logs/qualification/tomato_physics_v2"
PROFILE_PATH = (
    PROJECT_ROOT
    / "simulation/config/tomato_pick_physics_v2_public_prior_v1.json"
)
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _metric(line: str, name: str) -> float | None:
    match = re.search(rf"{re.escape(name)}:\s*([-+0-9.eE]+)", line)
    return float(match.group(1)) if match else None


def _parse_training_log(text: str) -> list[dict[str, float | int]]:
    clean = ANSI.sub("", text)
    records: list[dict[str, float | int]] = []
    current: dict[str, float | int] | None = None
    names = {
        "Metrics/success_rate": "success_rate",
        "Metrics/drop_rate": "drop_rate",
        "Metrics/detach_rate": "detach_rate",
        "Metrics/retained_rate": "retained_rate",
        "Metrics/mean_hold_steps": "mean_hold_steps",
        "Metrics/ever_retained_rate": "ever_retained_rate",
        "Metrics/mean_max_hold_steps": "mean_max_hold_steps",
        "PhysicsV2/mean_break_distance_m": "mean_break_distance_m",
        "PhysicsV2/mean_break_relative_speed_m_s": (
            "mean_break_relative_speed_m_s"
        ),
        "PhysicsV2/break_secure_support_rate": "break_secure_support_rate",
        "PhysicsV2/break_distance_ok_rate": "break_distance_ok_rate",
        "PhysicsV2/break_speed_ok_rate": "break_speed_ok_rate",
        "PhysicsV2/break_kinematic_retained_rate": (
            "break_kinematic_retained_rate"
        ),
        "PhysicsV2/break_joint_ready_rate": "break_joint_ready_rate",
        "PhysicsV2/discovery_capacity_scale": "discovery_capacity_scale",
        "PhysicsV2/tensile_shear_capacity_scale": (
            "tensile_shear_capacity_scale"
        ),
        "PhysicsV2/bending_torsion_capacity_scale": (
            "bending_torsion_capacity_scale"
        ),
        "PhysicsV2/twist_assisted_break_rate": "twist_assisted_break_rate",
        "PhysicsV2/bend_assisted_break_rate": "bend_assisted_break_rate",
        "PhysicsV2/moderate_pull_break_rate": "moderate_pull_break_rate",
        "PhysicsV2/paper_strategy_candidate_rate": (
            "paper_strategy_candidate_rate"
        ),
        "PhysicsV2/paper_strategy_success_rate": (
            "paper_strategy_success_rate"
        ),
        "PhysicsV2/bending_torsion_curriculum_assisted_break_rate": (
            "bending_torsion_curriculum_assisted_break_rate"
        ),
        "PhysicsV2/mean_break_thumb_force_n": "mean_break_thumb_force_n",
        "PhysicsV2/mean_break_other_force_n": "mean_break_other_force_n",
        "PhysicsV2/mean_break_opposition_target_n": (
            "mean_break_opposition_target_n"
        ),
        "PhysicsV2/mean_break_load_bearing_opposition_score": (
            "mean_break_load_bearing_opposition_score"
        ),
        "Episode_Reward/break_grasped": "break_grasped_reward",
        "Episode_Reward/stem_load_progress": "stem_load_progress_reward",
        "Episode_Reward/retention_progress": "retention_progress_reward",
        "Episode_Reward/stable_detach": "stable_detach_reward",
        "Episode_Reward/time_cost": "time_cost_reward",
        "Episode_Reward/drop_failure": "drop_failure_reward",
        "Episode_Reward/excess_force_failure": "excess_force_failure_reward",
        "Episode_Reward/out_of_bounds_failure": (
            "out_of_bounds_failure_reward"
        ),
        "Episode_Reward/timeout_failure": "timeout_failure_reward",
        "Mean action std": "mean_action_std",
        "Mean value loss": "mean_value_loss",
        "Mean surrogate loss": "mean_surrogate_loss",
        "Mean reward": "mean_reward",
        "Steps per second": "steps_per_second",
        "Iteration time": "iteration_time_seconds",
    }
    for line in clean.splitlines():
        match = re.search(r"Learning iteration\s+(\d+)/(\d+)", line)
        if match:
            if current is not None:
                records.append(current)
            current = {
                "iteration": int(match.group(1)),
                "configured_max_iterations": int(match.group(2)),
            }
            continue
        if current is None:
            continue
        for displayed, key in names.items():
            value = _metric(line, displayed)
            if value is not None:
                current[key] = value
                break
    if current is not None:
        records.append(current)
    return records


def _tail_summary(
    records: list[dict[str, float | int]], *, window: int = 10
) -> tuple[dict[str, float | int], dict[str, bool]]:
    tail = records[-window:]
    previous = records[-2 * window : -window]

    def mean(rows: list[dict[str, float | int]], key: str) -> float:
        return (
            sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
            if rows
            else 0.0
        )

    def positive(rows: list[dict[str, float | int]], key: str) -> int:
        return sum(float(row.get(key, 0.0)) > 0.0 for row in rows)

    def drop_detach_ratio(rows: list[dict[str, float | int]]) -> float:
        detach = sum(float(row.get("detach_rate", 0.0)) for row in rows)
        drop = sum(float(row.get("drop_rate", 0.0)) for row in rows)
        return drop / max(detach, 1.0e-12)

    summary: dict[str, float | int] = {
        "window": window,
        "tail_start_iteration": int(tail[0]["iteration"]) if tail else -1,
        "tail_end_iteration": int(tail[-1]["iteration"]) if tail else -1,
        "tail_success_positive_iterations": positive(tail, "success_rate"),
        "tail_retained_positive_iterations": positive(tail, "retained_rate"),
        "tail_ever_retained_positive_iterations": positive(
            tail, "ever_retained_rate"
        ),
        "tail_joint_ready_positive_iterations": positive(
            tail, "break_joint_ready_rate"
        ),
        "tail_mean_success_rate": mean(tail, "success_rate"),
        "tail_mean_retained_rate": mean(tail, "retained_rate"),
        "previous_mean_retained_rate": mean(previous, "retained_rate"),
        "tail_mean_ever_retained_rate": mean(tail, "ever_retained_rate"),
        "previous_mean_ever_retained_rate": mean(
            previous, "ever_retained_rate"
        ),
        "tail_drop_detach_ratio": drop_detach_ratio(tail),
        "previous_drop_detach_ratio": drop_detach_ratio(previous),
        "drop_detach_ratio_improvement": (
            drop_detach_ratio(previous) - drop_detach_ratio(tail)
        ),
        "tail_min_discovery_capacity_scale": min(
            (float(row.get("discovery_capacity_scale", 0.0)) for row in tail),
            default=0.0,
        ),
        "tail_min_tensile_shear_capacity_scale": min(
            (
                float(row.get("tensile_shear_capacity_scale", 0.0))
                for row in tail
            ),
            default=0.0,
        ),
        "tail_min_bending_torsion_capacity_scale": min(
            (
                float(row.get("bending_torsion_capacity_scale", 0.0))
                for row in tail
            ),
            default=0.0,
        ),
    }
    checks = {
        "tail_window_complete": len(tail) == window and len(previous) == window,
        "tail_success_repeated": positive(tail, "success_rate") >= 2,
        "tail_retention_persistent": positive(tail, "ever_retained_rate") >= 3,
        "tail_retention_not_regressed": mean(tail, "ever_retained_rate")
        >= mean(previous, "ever_retained_rate"),
        "tail_drop_detach_ratio_not_worse": drop_detach_ratio(tail)
        <= drop_detach_ratio(previous) + 0.01,
        "tail_full_public_prior_capacity": bool(tail)
        and all(
            float(row.get("discovery_capacity_scale", 0.0)) >= 0.999
            and float(row.get("tensile_shear_capacity_scale", 0.0)) >= 0.999
            and float(row.get("bending_torsion_capacity_scale", 0.0)) >= 0.999
            for row in tail
        ),
    }
    return summary, checks


def main(argv: Sequence[str] | None = None) -> int:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    gate_spec = profile["training"]["first_success_gate"]
    configured_envs = int(gate_spec["num_envs"])
    configured_updates = int(gate_spec["updates"])
    configured_seed = int(gate_spec["seed"])
    default_output = PROJECT_ROOT / str(gate_spec["output"])
    default_log = Path(
        f"/tmp/tomato_physics_v2_first_success_"
        f"{configured_envs}x{configured_updates}.log"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=configured_envs)
    parser.add_argument("--iterations", type=int, default=configured_updates)
    parser.add_argument("--seed", type=int, default=configured_seed)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--log", type=Path, default=default_log)
    parser.add_argument(
        "--run-name",
        default=(
            f"physics_v2_gate_{configured_envs}x{configured_updates}_not_formal"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "validate the frozen gate and fresh prerequisite reports, print the "
            "resolved command, and exit without starting Isaac or training"
        ),
    )
    args = parser.parse_args(argv)
    if (args.num_envs, args.iterations, args.seed) != (
        configured_envs,
        configured_updates,
        configured_seed,
    ):
        raise ValueError(
            "Physics-v2 first-success gate must match the audited profile; "
            "change the profile only after review and approval"
        )
    if (
        configured_envs <= 0
        or configured_updates < 20
        or configured_seed < 0
    ):
        raise RuntimeError("profile first-success gate boundary is invalid")
    prerequisite_inputs = {
        "public_prior_cpu_gate.json": (
            *AUDITED_RUNTIME_INPUTS,
            "script/qualify_tomato_physics_v2_public_priors.py",
        ),
        "isaac_smoke_gate_64x128.json": (
            *AUDITED_RUNTIME_INPUTS,
            "script/check_tomato_physics_v2.py",
        ),
    }
    for report_name, required_inputs in prerequisite_inputs.items():
        report = json.loads(
            (QUALIFICATION_ROOT / report_name).read_text(encoding="utf-8")
        )
        if report.get("passed") is not True:
            raise RuntimeError(f"prerequisite qualification did not pass: {report_name}")
        if report.get("candidate_id") != gate_spec["candidate_id"]:
            raise RuntimeError(
                f"prerequisite qualification candidate changed: {report_name}"
            )
        require_current_input_sha256(
            report,
            PROJECT_ROOT,
            required_inputs,
            report_name=report_name,
        )

    environment = dict(os.environ)
    environment.update(
        {
            "VIRTUAL_ENV": str(ISAAC_ENV),
            "LINKAGENT_ISAACLAB_ROOT": str(ISAAC_ROOT),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONPATH": os.pathsep.join(
                (str(PROJECT_ROOT), environment.get("PYTHONPATH", ""))
            ),
            "TOMATO_REWARD_ITERATION_OFFSET": "0",
            "TOMATO_V2_DISCOVERY_CAPACITY_CURRICULUM": "1",
        }
    )
    environment.pop("CONDA_PREFIX", None)
    environment["PATH"] = os.pathsep.join(
        (str(ISAAC_ENV / "bin"), environment.get("PATH", ""))
    )
    command = [
        str(ISAAC_ENV / "bin/python"),
        str(ISAAC_ROOT / "scripts/reinforcement_learning/train.py"),
        "--rl_library",
        "rsl_rl",
        "--external_callback",
        "simulation.tomato_pick.isaac.physics_v2.register_task",
        "--task",
        profile["task_id"],
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--max_iterations",
        str(args.iterations),
        "--logger",
        "tensorboard",
        "--experiment_name",
        "tomato_pick_physics_v2_first_success_gate",
        "--run_name",
        args.run_name,
        "--viz",
        "none",
    ]
    tracked_inputs = (
        *AUDITED_RUNTIME_INPUTS,
        "script/gate_tomato_physics_v2_first_success.py",
    )
    if args.preflight_only:
        preflight = {
            "schema_version": 1,
            "passed": True,
            "qualification": "physics_v2_first_success_gate_preflight",
            "candidate_id": gate_spec["candidate_id"],
            "configured_gate_spec": gate_spec,
            "command": command,
            "input_sha256": input_sha256(PROJECT_ROOT, tracked_inputs),
            "training_started": False,
            "hardware_connected": False,
        }
        print(
            "PHYSICS_V2_FIRST_SUCCESS_GATE_PREFLIGHT="
            + json.dumps(preflight, sort_keys=True)
        )
        return 0
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    text = args.log.read_text(encoding="utf-8", errors="replace")
    records = _parse_training_log(text)
    first_success = next(
        (
            int(record["iteration"])
            for record in records
            if float(record.get("success_rate", 0.0)) > 0.0
        ),
        None,
    )
    finite = bool(records) and all(
        math.isfinite(float(value))
        for record in records
        for key, value in record.items()
        if key not in {"iteration", "configured_max_iterations"}
    )
    error_patterns = {
        "traceback": r"Traceback \(most recent call last\)",
        "nan_or_inf": r"(?i)(?<![A-Za-z])(nan|inf)(?![A-Za-z])",
        "physx_overflow": r"(?i)PhysX[^\n]*overflow",
        "out_of_memory": r"(?i)(out of memory|CUDA error)",
    }
    errors = {
        name: bool(re.search(pattern, text))
        for name, pattern in error_patterns.items()
    }
    last = records[-1] if records else {}
    tail_trend, tail_checks = _tail_summary(records)
    checks = {
        "process_completed": completed.returncode == 0,
        "configured_updates_recorded": bool(
            len(records) >= configured_updates
            and int(last.get("iteration", -1)) >= configured_updates - 1
        ),
        "metrics_finite": finite,
        "no_runtime_error_signature": not any(errors.values()),
        "fracture_event_reached": any(
            float(record.get("detach_rate", 0.0)) > 0.0
            or float(record.get("break_grasped_reward", 0.0)) > 0.0
            for record in records
        ),
        "first_success_within_configured_gate": first_success is not None
        and first_success <= configured_updates - 1,
        "retention_chain_reached": any(
            float(record.get("ever_retained_rate", 0.0)) > 0.0
            for record in records
        ),
        **tail_checks,
    }
    healthy_and_full = (
        checks["process_completed"]
        and checks["configured_updates_recorded"]
        and checks["metrics_finite"]
        and checks["no_runtime_error_signature"]
        and checks["tail_full_public_prior_capacity"]
    )
    isolated_outcome = first_success is not None or any(
        float(record.get("ever_retained_rate", 0.0)) > 0.0
        for record in records
    )
    repeated_joint_ready_with_material_drop_improvement = (
        int(tail_trend["tail_joint_ready_positive_iterations"]) >= 3
        and float(tail_trend["drop_detach_ratio_improvement"]) >= 0.05
    )
    extension_assessment = {
        "automatic_extension": False,
        "requires_user_approval": True,
        "eligible_to_request_extension": bool(
            not all(checks.values())
            and healthy_and_full
            and (
                isolated_outcome
                or repeated_joint_ready_with_material_drop_improvement
            )
        ),
        "evidence_policy": (
            "request a longer gate only when the configured run is healthy, "
            "the tail uses full public-prior mechanics, and either an isolated "
            "retained/success outcome already exists or joint-ready fractures "
            "repeat while drop/detach improves by at least five percentage points"
        ),
        "isolated_success_or_retention_observed": isolated_outcome,
        "repeated_joint_ready_with_material_drop_improvement": (
            repeated_joint_ready_with_material_drop_improvement
        ),
    }
    iteration_times = [
        float(record["iteration_time_seconds"])
        for record in records
        if "iteration_time_seconds" in record
    ]
    report = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "qualification": "physics_v2_formal_scale_first_success_gate",
        "from_scratch": True,
        "checkpoint_loaded": None,
        "num_envs": args.num_envs,
        "rollout_steps": 32,
        "iterations": args.iterations,
        "seed": args.seed,
        "first_success_iteration": first_success,
        "checks": checks,
        "error_signatures": errors,
        "tail_trend": tail_trend,
        "extension_assessment": extension_assessment,
        "last_iteration": last,
        "records": records,
        "log": str(args.log),
        "command": command,
        "run_name": args.run_name,
        "configured_gate_spec": gate_spec,
        "training_curriculum": {
            "enabled": True,
            "initial_tensile_shear_capacity_scale": profile[
                "training_curriculum"
            ]["initial_tensile_shear_capacity_scale"],
            "initial_bending_torsion_capacity_scale": profile[
                "training_curriculum"
            ]["initial_bending_torsion_capacity_scale"],
            "hold_initial_scale_iterations": profile["training_curriculum"][
                "hold_initial_scale_iterations"
            ],
            "full_public_prior_iteration": profile["training_curriculum"][
                "full_public_prior_iteration"
            ],
            "bending_torsion_assistance_requires_secure_support": profile[
                "training_curriculum"
            ]["bending_torsion_assistance_requires_secure_support"],
            "bending_torsion_assistance_requires_kinematic_retention": profile[
                "training_curriculum"
            ]["bending_torsion_assistance_requires_kinematic_retention"],
            "stem_load_progress_uses_public_capacity": profile[
                "training_curriculum"
            ]["stem_load_progress_uses_public_capacity"],
            "stem_load_progress_requires_kinematic_retention": profile[
                "training_curriculum"
            ]["stem_load_progress_requires_kinematic_retention"],
            "stem_load_progress_uses_retention_margin": profile[
                "training_curriculum"
            ]["stem_load_progress_uses_retention_margin"],
            "retention_margin_score": profile["training_curriculum"][
                "retention_margin_score"
            ],
            "tail_requires_full_public_prior": True,
        },
        "training_observation": profile["training_observation"],
        "training_policy": profile["training_policy"],
        "training_objective": profile["training_objective"],
        "training_shaping": profile["training_shaping"],
        "performance": {
            "iteration_time_seconds_mean": (
                sum(iteration_times) / len(iteration_times)
                if iteration_times
                else None
            ),
            "iteration_time_seconds_min": (
                min(iteration_times) if iteration_times else None
            ),
            "iteration_time_seconds_max": (
                max(iteration_times) if iteration_times else None
            ),
        },
        "input_sha256": input_sha256(PROJECT_ROOT, tracked_inputs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PHYSICS_V2_FIRST_SUCCESS_GATE=" + json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
