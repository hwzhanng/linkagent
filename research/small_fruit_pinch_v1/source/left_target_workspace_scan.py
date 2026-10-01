#!/usr/bin/env python3
"""在孤立左手的静态 FK 上搜索给定水果中心的三指接触动作。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=20260907)
parser.add_argument("--target-center-mm", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
parser.add_argument("--radius-mm", type=float, default=33.5)
parser.add_argument("--top-k", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv
from research.small_fruit_pinch_v1.source.pinch_action_manifold import pinch_action_to_joint_targets


def main() -> None:
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        generator = torch.Generator(device=env.device)
        generator.manual_seed(args.seed)
        action = torch.empty((args.num_envs, 14), device=env.device).uniform_(-1.0, 1.0, generator=generator)
        action[:, 0] = torch.empty(args.num_envs, device=env.device).uniform_(0.55, 1.0, generator=generator)
        action[:, 4] = torch.empty(args.num_envs, device=env.device).uniform_(0.20, 1.0, generator=generator)
        action[:, 5:8] = torch.empty((args.num_envs, 3), device=env.device).uniform_(-1.0, 1.0, generator=generator)
        action[:, 10:14] = torch.empty((args.num_envs, 4), device=env.device).uniform_(-1.0, 1.0, generator=generator)
        targets = pinch_action_to_joint_targets(action)
        env._joint_targets[:, env._hand_joint_ids] = targets
        env._update_mimic_targets(torch.arange(env.num_envs, device=env.device))
        env._robot.write_joint_position_to_sim_index(position=env._joint_targets, joint_ids=env._actuated_joint_ids,
                                                     env_ids=torch.arange(env.num_envs, device=env.device))
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)
        contacts = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids]
        origins = env.scene.env_origins
        center = torch.tensor(args.target_center_mm, dtype=torch.float32, device=env.device).div(1000.0)
        center_w = origins + center
        radius = float(args.radius_mm) / 1000.0
        distances = torch.linalg.vector_norm(contacts[:, :3] - center_w[:, None, :], dim=2)
        surface = (distances[:, :3] - radius).abs()
        # 小的穿透/间隙都可作为接触候选，但惩罚明显穿透，避免把
        # 一个已经压进球体的姿态误判为好动作。
        loss = surface.mean(dim=1) + 0.25 * torch.relu(radius * 0.75 - distances[:, :3]).mean(dim=1)
        order = torch.argsort(loss)[: args.top_k]
        for rank, index in enumerate(order.tolist(), start=1):
            print({
                "rank": rank,
                "loss_m": round(float(loss[index]), 6),
                "distance_m": [round(float(v), 5) for v in distances[index, :3].tolist()],
                "surface_error_m": [round(float(v), 5) for v in surface[index].tolist()],
                "action": [round(float(v), 4) for v in action[index].tolist()],
            }, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
