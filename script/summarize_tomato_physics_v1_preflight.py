"""Validate and summarize the fixed 256-env Physics-v1 PPO preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = PROJECT_ROOT / "simulation/config/tomato_pick_physics_v1.yaml"
DEFAULT_LOG = Path("/tmp/tomato_physics_v1_ppo_preflight.log")
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "logs/qualification/tomato_physics_v1/ppo_256x4_gate.json"
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(block: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s+([-+0-9.eE]+)", block)
    if match is None:
        raise RuntimeError(f"missing preflight metric: {label}")
    return float(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    text = ANSI.sub("", args.log.read_text())
    if "Number of environments: 256" not in text:
        raise RuntimeError("preflight did not use exactly 256 environments")
    if "Training time:" not in text or "SimulationContext cleared" not in text:
        raise RuntimeError("preflight did not finish cleanly")
    forbidden = re.search(
        r"Traceback|CUDA out of memory|PhysX[^\n]*overflow|\b(?:nan|inf)\b",
        text,
        flags=re.IGNORECASE,
    )
    if forbidden is not None:
        raise RuntimeError(f"preflight log contains error token: {forbidden.group(0)}")

    matches = list(re.finditer(r"Learning iteration (\d+)/4", text))
    if [int(match.group(1)) for match in matches] != [0, 1, 2, 3]:
        raise RuntimeError("preflight does not contain exactly iterations 0..3")
    iterations = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start():stop]
        values = {
            "iteration": index,
            "steps_per_second": _metric(block, "Steps per second"),
            "value_loss": _metric(block, "Mean value loss"),
            "surrogate_loss": _metric(block, "Mean surrogate loss"),
            "entropy_loss": _metric(block, "Mean entropy loss"),
            "mean_reward": _metric(block, "Mean reward"),
            "mean_action_std": _metric(block, "Mean action std"),
        }
        if not all(math.isfinite(value) for key, value in values.items() if key != "iteration"):
            raise RuntimeError(f"non-finite iteration metrics: {values}")
        iterations.append(values)

    run_root = PROJECT_ROOT / "logs/rsl_rl/tomato_pick_physics_v1_preflight"
    runs = sorted(path for path in run_root.iterdir() if path.is_dir())
    if not runs:
        raise RuntimeError("preflight run directory was not created")
    run = runs[-1]
    final_checkpoint = run / "model_3.pt"
    if not final_checkpoint.is_file() or final_checkpoint.stat().st_size < 1_000_000:
        raise RuntimeError("final preflight checkpoint is missing or truncated")
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    secure_success = profile["secure_success"]
    tracked_inputs = [
        PROFILE_PATH,
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v1/env_cfg.py",
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v1/env.py",
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v1/assets.py",
        PROJECT_ROOT / "simulation/tomato_pick/physics_v1.py",
        PROJECT_ROOT / "tests/test_tomato_physics_v1.py",
    ]
    result = {
        "schema_version": 1,
        "task_id": "LinkAgent-TomatoPick-Physics-v1",
        "qualification": "numerical_preflight_not_formal_training",
        "num_envs": 256,
        "seed": 42,
        "iterations": iterations,
        "total_transitions": 256 * 32 * 4,
        "run_directory": str(run.resolve()),
        "final_checkpoint": str(final_checkpoint.resolve()),
        "final_checkpoint_sha256": _sha256(final_checkpoint),
        "log": str(args.log.resolve()),
        "credit_gate": {
            "break_substep_secure_support_required": True,
            "break_substep_kinematic_retained_required": True,
            "maximum_center_distance_m": secure_success["maximum_center_distance_m"],
            "maximum_relative_speed_m_s": secure_success["maximum_relative_speed_m_s"],
        },
        "physics_representation": {
            "self_collision_representation": profile["robot_asset"][
                "self_collision_representation"
            ],
            "solver_position_iteration_count": profile["simulation"][
                "solver_position_iteration_count"
            ],
            "solver_velocity_iteration_count": profile["simulation"][
                "solver_velocity_iteration_count"
            ],
        },
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in tracked_inputs
        },
        "checks": {
            "completed_four_updates": True,
            "losses_and_std_are_finite": True,
            "no_traceback_nan_inf_overflow_or_oom": True,
            "final_checkpoint_is_nonempty": True,
        },
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("TOMATO_PHYSICS_V1_PREFLIGHT=" + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
