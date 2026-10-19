#!/usr/bin/env python3
"""用批量静态 FK 搜索孤立左手三指接触候选，不推进 PhysX 时间。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=20260907)
parser.add_argument("--top-k", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv
from research.small_fruit_pinch_v1.source.pinch_action_manifold import pinch_action_to_joint_targets


def main() -> None:
    if args.num_envs < args.top_k:
        raise ValueError("num-envs 必须不小于 top-k")
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
        # 保证候选都进入闭合工作区；其余维度保留独立远端自由度。
        action[:, 0] = torch.empty(args.num_envs, device=env.device).uniform_(0.55, 1.0, generator=generator)
        action[:, 4] = torch.empty(args.num_envs, device=env.device).uniform_(0.35, 1.0, generator=generator)
        action[:, 5:8] = torch.empty((args.num_envs, 3), device=env.device).uniform_(-1.0, 1.0, generator=generator)
        action[:, 10:12] = torch.empty((args.num_envs, 2), device=env.device).uniform_(-1.0, 1.0, generator=generator)
        targets = pinch_action_to_joint_targets(action)
        env._joint_targets[:, env._hand_joint_ids] = targets
        env._update_mimic_targets(torch.arange(env.num_envs, device=env.device))
        env._robot.write_joint_position_to_sim_index(
            position=env._joint_targets,
            joint_ids=env._actuated_joint_ids,
            env_ids=torch.arange(env.num_envs, device=env.device),
        )
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)
        contacts = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids]
        fruit = env._tomato.data.root_pos_w.torch
        origins = env.scene.env_origins
        radius = env._fruit_axes_m.mean(dim=1)
        distances = torch.linalg.vector_norm(contacts[:, :3] - fruit[:, None, :], dim=2)
        # 表面误差 + 三指接触点在果实周向分离的鼓励。距离落在球半径
        # 附近并不等于成功，只用于选出值得做动态 PhysX 验证的候选。
        surface = (distances[:, :3] - radius[:, None]).abs()
        contact_score = surface.mean(dim=1) + 0.30 * torch.relu(0.028 - distances[:, :3]).mean(dim=1)
        order = torch.argsort(contact_score)[: args.top_k]
        for rank, index in enumerate(order.tolist(), start=1):
            print(
                {
                    "rank": rank,
                    "env": index,
                    "loss_m": round(float(contact_score[index]), 6),
                    "distance_m": [round(float(v), 5) for v in distances[index, :3].tolist()],
                    "surface_error_m": [round(float(v), 5) for v in surface[index].tolist()],
                    "fruit_m": [round(float(v), 5) for v in fruit[index].tolist()],
                    "fruit_local_m": [round(float(v), 5) for v in (fruit[index] - origins[index]).tolist()],
                    "action": [round(float(v), 4) for v in action[index].tolist()],
                },
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
