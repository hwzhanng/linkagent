#!/usr/bin/env python3
"""验证受控 Isaac Lab 3.0 运行时的当前物理环境可构造。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
from simulation.tomato_pick.isaac import TASK_ID
from simulation.tomato_pick.isaac.env_cfg import TomatoPickEnvCfg


def main() -> None:
    cfg = TomatoPickEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = gym.make(TASK_ID, cfg=cfg)
    try:
        obs, _ = env.reset(seed=42)
        print(f"current_runtime_probe policy={tuple(obs['policy'].shape)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
