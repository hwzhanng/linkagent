#!/usr/bin/env python3
"""测量三根手指局部 residual 对真实 PhysX 接触力的方向。

每个方向在同一个 reset seed 上重复，先用 P1 建立接触，再只改变一个
residual 维度。该探针不训练、不保存权重，用于给结构性受力通道提供符号
证据，而不是凭经验把“闭合”和“回撤”混为一谈。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=20260965)
parser.add_argument("--contact-steps", type=int, default=70)
parser.add_argument("--probe-steps", type=int, default=20)
parser.add_argument("--probe-amplitude", type=float, default=0.30)
parser.add_argument("--offset-x-mm", type=float, default=13.0)
parser.add_argument("--offset-y-mm", type=float, default=30.0)
parser.add_argument("--offset-z-mm", type=float, default=24.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.utils.seed import configure_seed

from research.small_fruit_pinch_v1.source.guided_residual_env import (
    GuidedResidualSmallFruitPinchEnv,
    P1_GUIDE_ACTION,
)
from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg


def run(action: torch.Tensor, *, env: GuidedResidualSmallFruitPinchEnv) -> tuple[torch.Tensor, torch.Tensor]:
    env.reset(seed=args.seed)
    peak = torch.zeros((env.num_envs, 5), device=env.device)
    for step in range(args.contact_steps + args.probe_steps):
        env.step(action if step < args.contact_steps else action)
        peak = torch.maximum(peak, env._last_physical_force_n)
    return env._last_physical_force_n[:, :3].mean(dim=0), peak[:, :3].mean(dim=0)


def main() -> None:
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
    GuidedResidualSmallFruitPinchEnv.geometry_prior_enabled = False
    GuidedResidualSmallFruitPinchEnv.residual_scale = 1.0
    GuidedResidualSmallFruitPinchEnv.precontact_residual_fraction = 1.0
    env = GuidedResidualSmallFruitPinchEnv(cfg=cfg)
    try:
        base = torch.tensor(P1_GUIDE_ACTION, device=env.device).expand(env.num_envs, -1).clone()
        for index, name in ((5, "thumb_root"), (6, "index_root"), (7, "middle_root"), (10, "index_distal"), (11, "middle_distal")):
            rows = []
            for sign in (-1.0, 1.0):
                action = torch.zeros_like(base)
                action[:, index] = sign * args.probe_amplitude
                # Run P1 through the guide and add a one-dimensional probe only
                # after the acquisition window; the guide path itself remains
                # identical in every comparison.
                env.reset(seed=args.seed)
                peak = torch.zeros((env.num_envs, 3), device=env.device)
                force_at_switch = torch.zeros((env.num_envs, 3), device=env.device)
                zero = torch.zeros_like(action)
                for step in range(args.contact_steps + args.probe_steps):
                    current = zero if step < args.contact_steps else action
                    env.step(current)
                    force = env._last_physical_force_n[:, :3]
                    peak = torch.maximum(peak, force)
                    if step == args.contact_steps - 1:
                        force_at_switch.copy_(force)
                rows.append(
                    {
                        "sign": sign,
                        "force_at_switch": force_at_switch.mean(dim=0).detach().cpu().tolist(),
                        "force_after": env._last_physical_force_n[:, :3].mean(dim=0).detach().cpu().tolist(),
                        "peak": peak.mean(dim=0).detach().cpu().tolist(),
                    }
                )
            print(f"direction_probe finger={name} rows={rows}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
