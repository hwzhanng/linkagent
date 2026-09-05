#!/usr/bin/env python3
"""在同一个 Isaac/PhysX 进程中比较结构化先验的独立回合结果。

该探针只运行零 PPO 残差，不保存权重，也不改变历史训练目录。目的在于
把“按指几何偏置”与训练噪声分离，避免用一个批次的 guided_probe 事件数
误判为 episode 成功率。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=256)
parser.add_argument("--seed", type=int, default=20260961)
parser.add_argument("--offset-x-mm", type=float, default=13.0)
parser.add_argument("--offset-y-mm", type=float, default=30.0)
parser.add_argument("--offset-z-mm", type=float, default=24.0)
parser.add_argument("--geometry-scale", type=float, default=0.30)
parser.add_argument("--variants", default="p1,thumb_index,thumb_index_middle_soft,thumb_index_middle_tiny,all_three,thumb_only,index_only")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.utils.seed import configure_seed

from research.small_fruit_pinch_v1.source.guided_residual_env import (
    GuidedResidualSmallFruitPinchEnv,
)
from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg


VARIANTS = {
    "p1": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "thumb_index": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "thumb_index_middle_soft": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "thumb_index_middle_tiny": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "all_three": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "thumb_only": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "index_only": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}


def main() -> None:
    if args.num_envs <= 0 or args.episodes <= 0:
        raise ValueError("num-envs 与 episodes 必须为正数")
    configure_seed(args.seed, True)
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.pinch_working_offset_m = (
        args.offset_x_mm / 1000.0,
        args.offset_y_mm / 1000.0,
        args.offset_z_mm / 1000.0,
    )
    GuidedResidualSmallFruitPinchEnv.geometry_prior_enabled = True
    GuidedResidualSmallFruitPinchEnv.geometry_prior_scale = args.geometry_scale
    GuidedResidualSmallFruitPinchEnv.residual_scale = 0.0
    GuidedResidualSmallFruitPinchEnv.precontact_residual_fraction = 0.0
    env = GuidedResidualSmallFruitPinchEnv(cfg=cfg)
    try:
        requested = [name.strip() for name in args.variants.split(",") if name.strip()]
        unknown = [name for name in requested if name not in VARIANTS]
        if unknown:
            raise ValueError(f"未知先验变体：{unknown}")
        for name in requested:
            values = VARIANTS[name]
            GuidedResidualSmallFruitPinchEnv.geometry_prior_residual = values
            env.reset(seed=args.seed)
            action = torch.zeros((env.num_envs, 14), device=env.device)
            episodes = success = drop = overload = collision = 0
            while episodes < args.episodes:
                _, _, terminated, truncated, info = env.step(action)
                step = info["step"]
                success += int(step.get("success_count", 0))
                drop += int(step.get("drop_count", 0))
                overload += int(step.get("overload_count", 0))
                collision += int(step.get("self_collision_count", 0))
                episodes += int((terminated | truncated).sum().item())
            total = max(episodes, 1)
            print(
                "prior_probe "
                f"variant={name} episodes={episodes} "
                f"strict_success_rate={success / total:.4f} "
                f"drop_rate={drop / total:.4f} "
                f"overload_rate={overload / total:.4f} "
                f"self_collision_rate={collision / total:.4f}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
