"""Run the bounded Physics-v2 PPO numerical probe (never formal training)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_BYTES = 8 * 1024**3


def _runtime() -> tuple[Path, Path]:
    value = os.environ.get("LINKAGENT_ISAACLAB_ROOT")
    if not value:
        raise RuntimeError("set LINKAGENT_ISAACLAB_ROOT to the frozen Isaac Lab checkout")
    root = Path(value).expanduser().resolve()
    python = root / "env_isaaclab/bin/python"
    if not python.is_file():
        raise RuntimeError(f"managed Isaac Lab Python is missing: {python}")
    return root, python


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.num_envs != 256 or args.iterations != 4:
        raise ValueError("the Physics-v2 numerical probe is fixed to exactly 256 envs x 4 updates")
    if shutil.disk_usage(PROJECT_ROOT).free < MIN_FREE_BYTES:
        raise RuntimeError("less than 8 GiB free; refusing to start even the bounded probe")

    isaac_root, isaac_python = _runtime()
    environment = dict(os.environ)
    environment.update(
        {
            "VIRTUAL_ENV": str(isaac_root / "env_isaaclab"),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "TOMATO_REWARD_ITERATION_OFFSET": "0",
            "WANDB_MODE": "online",
        }
    )
    environment.pop("CONDA_PREFIX", None)
    environment["PATH"] = os.pathsep.join(
        (str(isaac_root / "env_isaaclab/bin"), environment.get("PATH", ""))
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(PROJECT_ROOT), environment.get("PYTHONPATH", ""))
    )
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
        "tomato_pick_physics_v2_numerical_probe",
        "--run_name",
        "256x4_seed42_not_formal_training",
        "--viz",
        "none",
    ]
    print("BOUNDARY=no_hardware_no_formal_training_256x4_only", flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
