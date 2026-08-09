"""Run the bounded Physics-v1 PPO numerical preflight (never a formal run)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_isaac_root_value = os.environ.get("LINKAGENT_ISAACLAB_ROOT")
if not _isaac_root_value:
    raise RuntimeError("set LINKAGENT_ISAACLAB_ROOT to the Isaac Lab checkout")
ISAAC_ROOT = Path(_isaac_root_value).expanduser().resolve()
ISAAC_ENV = ISAAC_ROOT / "env_isaaclab"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.num_envs != 256:
        raise ValueError("the release preflight is fixed to exactly 256 environments")
    if not 1 <= args.iterations <= 8:
        raise ValueError("preflight iterations must be in 1..8")
    environment = dict(os.environ)
    environment["VIRTUAL_ENV"] = str(ISAAC_ENV)
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    environment["TOMATO_REWARD_ITERATION_OFFSET"] = "0"
    environment.pop("CONDA_PREFIX", None)
    environment["PATH"] = os.pathsep.join(
        (str(ISAAC_ENV / "bin"), environment.get("PATH", ""))
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(PROJECT_ROOT), environment.get("PYTHONPATH", ""))
    )
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
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--max_iterations",
        str(args.iterations),
        "--logger",
        "tensorboard",
        "--experiment_name",
        "tomato_pick_physics_v1_preflight",
        "--run_name",
        "numeric_gate_not_formal_training",
        "--viz",
        "none",
    ]
    return subprocess.run(
        command, cwd=PROJECT_ROOT, env=environment, check=False
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
