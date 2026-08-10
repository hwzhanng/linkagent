"""Start the bounded eight-hour Physics-v2 PPO learning validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "logs/qualification/tomato_physics_v2/eight_hour_training_manifest.json"
PROBE_REPORT = PROJECT_ROOT / "logs/qualification/tomato_physics_v2/ppo_256x4_probe.json"
MIN_FREE_BYTES = 8 * 1024**3


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args(argv)
    if args.num_envs != 256 or not 100 <= args.iterations <= 500:
        raise ValueError("learning validation is bounded to 256 envs and 100..500 updates")
    probe = json.loads(PROBE_REPORT.read_text(encoding="utf-8"))
    if probe.get("passed") is not True or probe.get("update_count") != 4:
        raise RuntimeError("the fixed Physics-v2 numerical probe has not passed")
    if shutil.disk_usage(PROJECT_ROOT).free < MIN_FREE_BYTES:
        raise RuntimeError("less than 8 GiB free; refusing to start bounded training")
    root_value = os.environ.get("LINKAGENT_ISAACLAB_ROOT")
    if not root_value:
        raise RuntimeError("set LINKAGENT_ISAACLAB_ROOT to the frozen Isaac Lab checkout")
    isaac_root = Path(root_value).expanduser().resolve()
    isaac_python = isaac_root / "env_isaaclab/bin/python"
    if not isaac_python.is_file():
        raise RuntimeError(f"managed Isaac Lab Python is missing: {isaac_python}")

    environment = dict(os.environ)
    environment.update(
        {
            "VIRTUAL_ENV": str(isaac_root / "env_isaaclab"),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "TOMATO_REWARD_ITERATION_OFFSET": "0",
            "WANDB_MODE": "online",
            "PYTHONUNBUFFERED": "1",
        }
    )
    environment.pop("CONDA_PREFIX", None)
    environment["PATH"] = os.pathsep.join(
        (str(isaac_root / "env_isaaclab/bin"), environment.get("PATH", ""))
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(PROJECT_ROOT), environment.get("PYTHONPATH", ""))
    )
    run_name = "eight_hour_learning_seed43_from_scratch"
    command = [
        str(isaac_python),
        str(isaac_root / "scripts/reinforcement_learning/train.py"),
        "--rl_library",
        "rsl_rl",
        "--external_callback",
        "simulation.tomato_pick.isaac.physics_v2.register_task",
        "--task",
        "LinkAgent-TomatoPick-Physics-v2",
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--max_iterations",
        str(args.iterations),
        "--logger",
        "wandb",
        "--log_project_name",
        "linkagent-tomato-physics-v2",
        "--experiment_name",
        "tomato_pick_physics_v2_learning",
        "--run_name",
        run_name,
        "--viz",
        "none",
    ]
    manifest = {
        "schema_version": 1,
        "task_id": "LinkAgent-TomatoPick-Physics-v2",
        "qualification": "bounded_eight_hour_learning_validation",
        "from_scratch": True,
        "num_envs": args.num_envs,
        "num_steps_per_env": 32,
        "max_updates": args.iterations,
        "maximum_transitions": args.num_envs * 32 * args.iterations,
        "seed": args.seed,
        "logger": "wandb",
        "wandb_project": "linkagent-tomato-physics-v2",
        "source_probe_report": str(PROBE_REPORT),
        "hardware_boundary": "simulation_only_no_robot_connections",
        "automatic_stop_policy": "external monitor: error, thermal, disk, or progress stall",
        "command": command,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BOUNDARY=simulation_only_bounded_500_updates", flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
