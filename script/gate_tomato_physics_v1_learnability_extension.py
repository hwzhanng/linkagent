"""Extend a healthy Physics-v1 8192-env gate to a bounded target update.

The frozen 50-update result remains immutable evidence.  This bounded extension
resumes its final checkpoint with the complete RSL-RL state and changes no MDP,
reward, reset, physics, or PPO setting.  It exists only to distinguish a slower
exploration curriculum from an unreachable task after the reset expansion.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Sequence

from gate_tomato_physics_v1_first_success import _parse_training_log


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_isaac_root_value = os.environ.get("LINKAGENT_ISAACLAB_ROOT")
if not _isaac_root_value:
    raise RuntimeError("set LINKAGENT_ISAACLAB_ROOT to the Isaac Lab checkout")
ISAAC_ROOT = Path(_isaac_root_value).expanduser().resolve()
ISAAC_ENV = ISAAC_ROOT / "env_isaaclab"
QUALIFICATION_ROOT = PROJECT_ROOT / "logs/qualification/tomato_physics_v1"
SOURCE_REPORT = QUALIFICATION_ROOT / "first_success_8192x50_gate.json"
SOURCE_RUN = (
    PROJECT_ROOT
    / "logs/rsl_rl/tomato_pick_physics_v1_first_success_gate"
    / "2026-08-09_00-23-09_from_scratch_8192x50_not_formal"
)
SOURCE_CHECKPOINT = SOURCE_RUN / "model_49.pt"
DEFAULT_OUTPUT = QUALIFICATION_ROOT / "first_success_8192x100_extended_gate.json"
DEFAULT_LOG = Path("/tmp/tomato_physics_v1_first_success_8192x100_extended.log")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(records: list[dict[str, float | int]]) -> bool:
    return bool(records) and all(
        math.isfinite(float(value))
        for record in records
        for key, value in record.items()
        if key not in {"iteration", "configured_max_iterations"}
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--source-report", type=Path, default=SOURCE_REPORT)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument("--source-iterations", type=int, default=50)
    parser.add_argument("--target-iterations", type=int, default=100)
    parser.add_argument(
        "--run-name", default="continue_model49_to100_not_formal"
    )
    args = parser.parse_args(argv)

    source_report = args.source_report.expanduser().resolve()
    source_run = args.source_run.expanduser().resolve()
    source_checkpoint = args.source_checkpoint.expanduser().resolve()
    if args.source_iterations <= 0:
        raise ValueError("source-iterations must be positive")
    if args.target_iterations <= args.source_iterations:
        raise ValueError("target-iterations must exceed source-iterations")
    if not source_report.is_file():
        raise FileNotFoundError(source_report)
    source = json.loads(source_report.read_text())
    source_checks = source.get("checks", {})
    if args.source_iterations == 50:
        source_iterations_recorded = source_checks.get("fifty_updates_recorded") is True
    else:
        source_iterations_recorded = (
            int(source.get("cumulative_iterations", -1)) == args.source_iterations
            and int(source.get("last_iteration", {}).get("iteration", -1))
            >= args.source_iterations - 1
        )
    required_source_checks = (
        source_checks.get("process_completed") is True
        and source_iterations_recorded
        and source_checks.get("metrics_finite") is True
        and source_checks.get("no_runtime_error_signature") is True
        and source.get("num_envs") == 8192
        and source.get("seed") == 42
        and source.get("from_scratch_lineage", True) is True
    )
    if not required_source_checks:
        raise RuntimeError("the source gate is not a healthy resumable lineage")
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    expected_checkpoint = f"model_{args.source_iterations - 1}.pt"
    if source_checkpoint.name != expected_checkpoint:
        raise RuntimeError(
            f"source checkpoint must be {expected_checkpoint}, got {source_checkpoint.name}"
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
            "TOMATO_REWARD_ITERATION_OFFSET": str(args.source_iterations - 1),
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
        "8192",
        "--seed",
        "42",
        # RSL-RL logs the loaded checkpoint iteration once, so the bounded
        # inclusive segment length is target - source + 1.
        "--max_iterations",
        str(args.target_iterations - args.source_iterations + 1),
        "--logger",
        "tensorboard",
        "--experiment_name",
        "tomato_pick_physics_v1_first_success_gate",
        "--run_name",
        args.run_name,
        "--resume",
        "--load_run",
        f"^{re.escape(source_run.name)}$",
        "--checkpoint",
        f"^{re.escape(source_checkpoint.name)}$",
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
    extension_records = _parse_training_log(text)
    source_records = list(source.get("records", []))
    combined_by_iteration = {
        int(record["iteration"]): record for record in source_records + extension_records
    }
    records = [combined_by_iteration[key] for key in sorted(combined_by_iteration)]
    first_success = next(
        (
            int(record["iteration"])
            for record in records
            if float(record.get("success_rate", 0.0)) > 0.0
        ),
        None,
    )
    first_retained = next(
        (
            int(record["iteration"])
            for record in records
            if float(record.get("retained_rate", 0.0)) > 0.0
        ),
        None,
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
    checks = {
        "healthy_source_verified": required_source_checks,
        "source_checkpoint_nonempty": source_checkpoint.stat().st_size > 1_000_000,
        "process_completed": completed.returncode == 0,
        "target_updates_recorded": bool(
            records
            and int(last.get("iteration", -1)) >= args.target_iterations - 1
        ),
        "metrics_finite": _finite(records),
        "no_runtime_error_signature": not any(errors.values()),
        "first_success_by_target_update": (
            first_success is not None
            and first_success <= args.target_iterations - 1
        ),
        "retention_chain_reached": first_retained is not None,
    }
    report = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "qualification": "bounded_learnability_extension_not_formal_training",
        "from_scratch_lineage": True,
        "resumed_complete_training_state": True,
        "num_envs": 8192,
        "rollout_steps": 32,
        "cumulative_iterations": args.target_iterations,
        "seed": 42,
        "first_success_iteration": first_success,
        "first_retained_iteration": first_retained,
        "source_report": str(source_report),
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "checks": checks,
        "error_signatures": errors,
        "last_iteration": last,
        "records": records,
        "log": str(args.log),
        "command": command,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("PHYSICS_V1_EXTENDED_GATE=" + json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
