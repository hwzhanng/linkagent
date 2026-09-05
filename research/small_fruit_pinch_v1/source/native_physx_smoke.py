#!/usr/bin/env python3
"""原生左手小果捏取环境的最小 PhysX 冒烟测试。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym

from research.small_fruit_pinch_v1.source.native_env import (
    NativeSmallFruitPinchCfg,
    NativeSmallFruitPinchEnv,
)


def main() -> None:
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        observations, _ = env.reset(seed=42)
        action = observations["policy"].new_zeros((args.num_envs, cfg.action_space))
        for _ in range(5):
            observations, _, _, _, info = env.step(action)
        print(
            "native_physx_smoke "
            f"policy={tuple(observations['policy'].shape)} "
            f"critic={tuple(observations['critic'].shape)} "
            f"info_keys={sorted(info.keys())}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
