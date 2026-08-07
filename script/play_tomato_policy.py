"""Play one trained minimal tomato checkpoint through Isaac Lab's runner."""

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
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-length", type=int, default=400)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    environment = dict(os.environ)
    environment["VIRTUAL_ENV"] = str(ISAAC_ENV)
    environment.pop("CONDA_PREFIX", None)
    environment["PATH"] = os.pathsep.join((str(ISAAC_ENV / "bin"), environment.get("PATH", "")))
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(PROJECT_ROOT), environment.get("PYTHONPATH", ""))
    )
    command = [
        str(ISAAC_ENV / "bin" / "python"),
        str(ISAAC_ROOT / "scripts/reinforcement_learning/play.py"),
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
        "--checkpoint",
        str(checkpoint),
        "--viz",
        "kit" if args.render else "none",
    ]
    if args.video:
        command.extend(("--video", "--video_length", str(args.video_length)))
    return subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
