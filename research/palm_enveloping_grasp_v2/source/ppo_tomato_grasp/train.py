"""Local PPO entry point and headless smoke test.

Run with Isaac Lab's Python environment, for example:
`python train.py --headless --num_envs 16 --steps 256`.
The first milestone intentionally uses random actions unless `--ppo` is
requested; this keeps installation validation independent from long training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_ROOT = Path(__file__).resolve().parents[3] / "isaaclab-v2.3.2"
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Minimal tomato grasp PPO smoke test")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=256)
parser.add_argument("--ppo", action="store_true", help="reserved for the next milestone")
parser.add_argument("--linkbot", action="store_true", help="use the Linkbot USD articulation instead of the proxy")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

from simulation.ppo_tomato_grasp.tomato_grasp_env import TomatoGraspEnv, TomatoGraspEnvCfg


def main() -> None:
    print("[tomato-grasp] constructing config", flush=True)
    cfg = TomatoGraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.use_linkbot_asset = args_cli.linkbot
    print("[tomato-grasp] constructing environment", flush=True)
    env = TomatoGraspEnv(cfg=cfg, render_mode=None)
    print("[tomato-grasp] resetting environment", flush=True)
    obs, _ = env.reset()
    print(f"[tomato-grasp] obs={obs['policy'].shape} action={env.action_space}", flush=True)
    for step in range(args_cli.steps):
        actions = 2.0 * torch.rand((args_cli.num_envs, cfg.action_space), device=env.device) - 1.0
        obs, reward, terminated, truncated, _ = env.step(actions)
        if step in (0, args_cli.steps - 1):
            print(f"[tomato-grasp] step={step + 1} reward_mean={reward.mean().item():.4f}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
