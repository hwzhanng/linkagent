"""Run a deliberately short TensorBoard-only numerical PPO probe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAAC_ROOT = Path("/home/descfly/isaaclab/IsaacLab-3.0.0-beta2.patch1")
ISAAC_ENV = ISAAC_ROOT / "env_isaaclab"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.num_envs <= 0 or not 1 <= args.iterations <= 32:
        raise ValueError("probe requires num-envs > 0 and 1..32 iterations")
    environment = dict(os.environ)
    environment["VIRTUAL_ENV"] = str(ISAAC_ENV)
    environment.pop("CONDA_PREFIX", None)
    environment["PATH"] = os.pathsep.join((str(ISAAC_ENV / "bin"), environment.get("PATH", "")))
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(PROJECT_ROOT), environment.get("PYTHONPATH", ""))
    )
    command = [
        str(ISAAC_ENV / "bin" / "python"),
        str(ISAAC_ROOT / "scripts/reinforcement_learning/train.py"),
        "--rl_library",
        "rsl_rl",
        "--external_callback",
        "simulation.tomato_pick.isaac.register_task",
        "--task",
        "LinkAgent-TomatoPick-Direct-v0",
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed),
        "--max_iterations",
        str(args.iterations),
        "--logger",
        "tensorboard",
        "--experiment_name",
        "tomato_pick_positive_grasp_v5_probe",
        "--run_name",
        "numeric_gate",
        "--viz",
        "none",
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
