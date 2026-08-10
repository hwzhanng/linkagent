"""Run the formal-scale, bounded first-success gate for Physics-v1.

This is deliberately not a formal W&B run and never loads a checkpoint.  It
uses the exact formal environment count and PPO settings for at most 50
updates, then emits a machine-readable qualification report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_isaac_root_value = os.environ.get("LINKAGENT_ISAACLAB_ROOT")
if not _isaac_root_value:
    raise RuntimeError("set LINKAGENT_ISAACLAB_ROOT to the Isaac Lab checkout")
ISAAC_ROOT = Path(_isaac_root_value).expanduser().resolve()
ISAAC_ENV = ISAAC_ROOT / "env_isaaclab"
QUALIFICATION_ROOT = PROJECT_ROOT / "logs/qualification/tomato_physics_v1"
PROFILE_PATH = PROJECT_ROOT / "simulation/config/tomato_pick_physics_v1.yaml"
DEFAULT_PERFORMANCE_GATE = QUALIFICATION_ROOT / "performance/performance_gate_v4.json"
DEFAULT_OUTPUT = QUALIFICATION_ROOT / "first_success_8192x50_gate.json"
DEFAULT_LOG = Path("/tmp/tomato_physics_v1_first_success_8192x50.log")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "Mean action std": "mean_action_std",
        "Mean value loss": "mean_value_loss",
        "Mean surrogate loss": "mean_surrogate_loss",
        "Mean reward": "mean_reward",
        "Steps per second": "steps_per_second",
        "Iteration time": "iteration_time_seconds",
        "Episode_Reward/break_grasped": "break_grasped_reward",
        "Episode_Reward/retention_progress": "retention_progress_reward",
        "Episode_Reward/stable_detach": "stable_detach_reward",
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


def _tail_trend_summary(
    records: list[dict[str, float | int]], *, window: int = 10
) -> tuple[dict[str, float | int], dict[str, bool]]:
    """Summarize whether success is repeatable near the end of the gate."""

    tail = records[-window:]
    previous = records[-2 * window : -window]

    def mean(rows: list[dict[str, float | int]], key: str) -> float:
        if not rows:
            return 0.0
        return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)

    def positive_count(rows: list[dict[str, float | int]], key: str) -> int:
        return sum(float(row.get(key, 0.0)) > 0.0 for row in rows)

    def drop_detach_ratio(rows: list[dict[str, float | int]]) -> float:
        detach = sum(float(row.get("detach_rate", 0.0)) for row in rows)
        drop = sum(float(row.get("drop_rate", 0.0)) for row in rows)
        return drop / max(detach, 1.0e-12)

    summary: dict[str, float | int] = {
        "window": window,
        "tail_start_iteration": int(tail[0]["iteration"]) if tail else -1,
        "tail_end_iteration": int(tail[-1]["iteration"]) if tail else -1,
        "tail_success_positive_iterations": positive_count(tail, "success_rate"),
        "tail_retained_positive_iterations": positive_count(tail, "retained_rate"),
        "tail_mean_success_rate": mean(tail, "success_rate"),
        "tail_mean_retained_rate": mean(tail, "retained_rate"),
        "previous_mean_retained_rate": mean(previous, "retained_rate"),
        "tail_drop_detach_ratio": drop_detach_ratio(tail),
        "previous_drop_detach_ratio": drop_detach_ratio(previous),
    }
    checks = {
        "tail_window_complete": len(tail) == window and len(previous) == window,
        # A single success can be a one-environment accident. Require success
        # on at least two distinct PPO updates and retained signal on three.
        "tail_success_repeated": positive_count(tail, "success_rate") >= 2,
        "tail_retention_persistent": positive_count(tail, "retained_rate") >= 3,
        "tail_retention_not_regressed": mean(tail, "retained_rate")
        >= mean(previous, "retained_rate"),
        "tail_drop_detach_ratio_not_worse": drop_detach_ratio(tail)
        <= drop_detach_ratio(previous) + 0.01,
    }
    return summary, checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--run-name", default="from_scratch_8192x50_not_formal")
    parser.add_argument("--performance-gate", type=Path, default=DEFAULT_PERFORMANCE_GATE)
    args = parser.parse_args(argv)
    if (args.num_envs, args.iterations, args.seed) != (8192, 50, 42):
        raise ValueError("first-success gate is frozen at 8192 envs, 50 updates, seed 42")
    performance_gate_path = args.performance_gate.resolve()
    performance_gate = json.loads(performance_gate_path.read_text(encoding="utf-8"))
    if performance_gate.get("passed") is not True:
        raise RuntimeError("selected Physics-v1 performance gate has not passed")

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
        "simulation.tomato_pick.isaac.physics_v1.register_task",
        "--task",
        "LinkAgent-TomatoPick-Physics-v1",
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--max_iterations",
        str(args.iterations),
        "--logger",
        "tensorboard",
        "--experiment_name",
        "tomato_pick_physics_v1_first_success_gate",
        "--run_name",
        args.run_name,
        "--viz",
        "none",
    ]
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
    finite = all(
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
        name: bool(re.search(pattern, text)) for name, pattern in error_patterns.items()
    }
    last = records[-1] if records else {}
    tail_trend, tail_checks = _tail_trend_summary(records)
    checks = {
        "process_completed": completed.returncode == 0,
        "fifty_updates_recorded": bool(records and int(last["iteration"]) >= 49),
        "metrics_finite": finite and bool(records),
        "no_runtime_error_signature": not any(errors.values()),
        "first_success_by_update_50": first_success is not None and first_success <= 49,
        "retention_chain_reached": any(
            float(record.get("retained_rate", 0.0)) > 0.0 for record in records
        ),
        **tail_checks,
    }
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    tracked_inputs = [
        PROFILE_PATH,
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v1/env_cfg.py",
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v1/env.py",
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v1/assets.py",
        PROJECT_ROOT / "simulation/tomato_pick/physics_v1.py",
        PROJECT_ROOT / "simulation/tomato_pick/reset_v1.py",
        PROJECT_ROOT / "simulation/tomato_pick/reward.py",
        performance_gate_path,
    ]
    iteration_times = [
        float(record["iteration_time_seconds"])
        for record in records
        if "iteration_time_seconds" in record
    ]
    report = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "qualification": "formal_scale_first_success_gate_not_formal_training",
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
        "last_iteration": last,
        "records": records,
        "log": str(args.log),
        "command": command,
        "run_name": args.run_name,
        "physics_representation": {
            "self_collision_representation": profile["robot_asset"][
                "self_collision_representation"
            ],
            **profile["simulation"],
        },
        "performance": {
            "iteration_time_seconds_mean": (
                sum(iteration_times) / len(iteration_times) if iteration_times else None
            ),
            "iteration_time_seconds_min": min(iteration_times) if iteration_times else None,
            "iteration_time_seconds_max": max(iteration_times) if iteration_times else None,
        },
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in tracked_inputs
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("PHYSICS_V1_FIRST_SUCCESS_GATE=" + json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
