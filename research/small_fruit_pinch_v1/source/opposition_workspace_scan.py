#!/usr/bin/env python3
"""扫描孤立左手的动作--果心组合，寻找真实拇指对向几何。

该脚本不推进物理时间，只把安全关节目标写入 PhysX 做 FK。它同时随机化
球心和 14D 动作，按三指表面距离、拇指对向角和食中指夹持弦偏心排序，
避免只找“所有指尖都在球同一侧”的假接触。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--batches", type=int, default=8)
parser.add_argument("--top-k", type=int, default=12)
parser.add_argument("--radius-mm", type=float, default=33.5)
parser.add_argument("--seed", type=int, default=20260908)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from research.small_fruit_pinch_v1.source.native_env import (
    NativeSmallFruitPinchCfg,
    NativeSmallFruitPinchEnv,
)
from research.small_fruit_pinch_v1.source.pinch_action_manifold import (
    pinch_action_to_joint_targets,
)


def main() -> None:
    if args.num_envs <= 0 or args.batches <= 0 or args.top_k <= 0:
        raise ValueError("num-envs、batches 和 top-k 必须为正数")
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        generator = torch.Generator(device=env.device)
        generator.manual_seed(args.seed)
        origins = env.scene.env_origins
        radius = float(args.radius_mm) / 1000.0
        candidates: list[tuple[float, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for batch in range(args.batches):
            # 64--70 mm 直径对应 32--35 mm 半轴；这里取等效球半径
            # 33.5 mm，并在实际工作区内扫描球心。动作采样偏向闭合段，
            # 但拇指横摆、食中指 spread 和四个新增远端维仍覆盖全范围。
            action = torch.empty(
                (args.num_envs, 14), device=env.device
            ).uniform_(-1.0, 1.0, generator=generator)
            action[:, 0] = torch.empty(args.num_envs, device=env.device).uniform_(
                0.55, 1.0, generator=generator
            )
            action[:, 1] = torch.empty(args.num_envs, device=env.device).uniform_(
                -1.0, 0.65, generator=generator
            )
            action[:, 4] = torch.empty(args.num_envs, device=env.device).uniform_(
                -1.0, 1.0, generator=generator
            )
            action[:, 5:8] = torch.empty(
                (args.num_envs, 3), device=env.device
            ).uniform_(-1.0, 1.0, generator=generator)
            action[:, 10:14] = torch.empty(
                (args.num_envs, 4), device=env.device
            ).uniform_(-1.0, 1.0, generator=generator)
            center = torch.empty(
                (args.num_envs, 3), device=env.device
            ).uniform_(-1.0, 1.0, generator=generator)
            center = center * center.new_tensor((0.045, 0.090, 0.060))
            center += center.new_tensor((0.055, 0.0, 0.165))
            targets = pinch_action_to_joint_targets(action)
            env._joint_targets[:, env._hand_joint_ids] = targets
            env._update_mimic_targets(torch.arange(env.num_envs, device=env.device))
            env._robot.write_joint_position_to_sim_index(
                position=env._joint_targets,
                joint_ids=env._actuated_joint_ids,
                env_ids=torch.arange(env.num_envs, device=env.device),
            )
            fruit_pose = env._tomato.data.default_root_pose.torch.clone()
            fruit_pose[:, :3] = origins + center
            fruit_pose[:, 3:7] = fruit_pose.new_tensor((0.0, 0.0, 0.0, 1.0))
            env._tomato.write_root_pose_to_sim_index(
                root_pose=fruit_pose,
                env_ids=torch.arange(env.num_envs, device=env.device),
            )
            env._tomato.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((env.num_envs, 6), device=env.device),
                env_ids=torch.arange(env.num_envs, device=env.device),
            )
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(dt=0.0)
            points = env._robot.data.body_pos_w.torch[
                :, env._fingertip_contact_body_ids
            ]
            relative = points - fruit_pose[:, None, :3]
            distance = torch.linalg.vector_norm(relative, dim=2)
            surface = (distance[:, :3] - radius).abs()
            unit = torch.nn.functional.normalize(relative[:, :3], dim=2)
            finger_mean = torch.nn.functional.normalize(
                relative[:, 1:3].mean(dim=1), dim=1
            )
            theta = torch.rad2deg(
                torch.acos(
                    torch.clamp((unit[:, 0] * -finger_mean).sum(dim=1), -1.0, 1.0)
                )
            )
            # ``d_chord`` 是拇指接触点到食指/中指中点的夹持作用线
            # （不是食指--中指短连线）。这对应“拇指对准两指中间”，
            # 也避免把相邻两指间距误当作整条夹持作用线。
            finger_mid = relative[:, 1:3].mean(dim=1)
            line = finger_mid - relative[:, 0]
            line_sq = torch.sum(line * line, dim=1).clamp_min(1.0e-12)
            line_t = torch.clamp(
                torch.sum(-relative[:, 0] * line, dim=1) / line_sq, 0.0, 1.0
            )
            closest = relative[:, 0] + line_t.unsqueeze(1) * line
            chord_ratio = torch.linalg.vector_norm(closest, dim=1) / radius
            loss = (
                surface.mean(dim=1) / 0.005
                + torch.relu(theta - 30.0) / 30.0
                + torch.relu(chord_ratio - 0.40) * 2.0
            )
            for idx in torch.argsort(loss)[: args.top_k].tolist():
                candidates.append(
                    (
                        float(loss[idx]),
                        batch,
                        action[idx].detach().cpu(),
                        center[idx].detach().cpu(),
                        distance[idx, :3].detach().cpu(),
                        theta[idx].detach().cpu(),
                        chord_ratio[idx].detach().cpu(),
                    )
                )
        candidates.sort(key=lambda item: item[0])
        print(
            f"opposition_scan num_envs={args.num_envs} batches={args.batches} "
            f"radius_mm={args.radius_mm} candidates={len(candidates)}",
            flush=True,
        )
        for rank, item in enumerate(candidates[: args.top_k], start=1):
            loss, batch, action, center, distance, theta, chord_ratio = item
            print(
                {
                    "rank": rank,
                    "loss": round(loss, 6),
                    "batch": batch,
                    "center_mm": [round(float(v * 1000.0), 3) for v in center],
                    "distance_mm": [round(float(v * 1000.0), 3) for v in distance],
                    "theta_opp_deg": round(float(theta), 3),
                    "chord_ratio": round(float(chord_ratio), 4),
                    "action": [round(float(v), 4) for v in action],
                },
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
