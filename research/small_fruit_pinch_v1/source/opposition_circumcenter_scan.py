#!/usr/bin/env python3
"""用三指接触点的外接球反求果心，审计拇指是否能真正对向。

随机动作先通过孤立左手 PhysX FK 得到拇指、食指和中指接触点。对每个
三点，解析计算半径 33.5 mm 的两个可能球心，再按工作区、拇指对向角
和食中指夹持弦偏心排序。这样不会把“球心随机不合适”误认为动作空间
没有对向解，也不会只奖励三根手指同侧擦边。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--batches", type=int, default=24)
parser.add_argument("--top-k", type=int, default=12)
parser.add_argument("--radius-mm", type=float, default=33.5)
parser.add_argument("--min-center-z-mm", type=float, default=90.0)
parser.add_argument("--max-center-x-mm", type=float, default=130.0)
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


def _sample_actions(count: int, device: str, generator: torch.Generator) -> torch.Tensor:
    action = torch.empty((count, 14), device=device).uniform_(-1.0, 1.0, generator=generator)
    action[:, 0] = torch.empty(count, device=device).uniform_(0.55, 1.0, generator=generator)
    action[:, 1] = torch.empty(count, device=device).uniform_(-1.0, 0.65, generator=generator)
    action[:, 4] = torch.empty(count, device=device).uniform_(-1.0, 1.0, generator=generator)
    action[:, 5:8] = torch.empty((count, 3), device=device).uniform_(-1.0, 1.0, generator=generator)
    action[:, 10:14] = torch.empty((count, 4), device=device).uniform_(-1.0, 1.0, generator=generator)
    return action


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
        env_ids = torch.arange(env.num_envs, device=env.device)
        origins = env.scene.env_origins
        radius = float(args.radius_mm) / 1000.0
        records: list[tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _ in range(args.batches):
            action = _sample_actions(args.num_envs, env.device, generator)
            targets = pinch_action_to_joint_targets(action)
            env._joint_targets[:, env._hand_joint_ids] = targets
            env._update_mimic_targets(env_ids)
            env._robot.write_joint_position_to_sim_index(
                position=env._joint_targets, joint_ids=env._actuated_joint_ids, env_ids=env_ids
            )
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(dt=0.0)
            points = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids][:, :3]
            # 工作区间在每个环境的本地基座坐标中表达。
            points = points - origins[:, None, :]
            a = points[:, 1] - points[:, 0]
            b = points[:, 2] - points[:, 0]
            normal = torch.linalg.cross(a, b, dim=1)
            normal_sq = torch.sum(normal * normal, dim=1)
            a_sq = torch.sum(a * a, dim=1)
            b_sq = torch.sum(b * b, dim=1)
            denom = (2.0 * normal_sq).clamp_min(1.0e-12)
            circum_offset = (
                a_sq.unsqueeze(1) * torch.linalg.cross(b, normal, dim=1)
                + b_sq.unsqueeze(1) * torch.linalg.cross(normal, a, dim=1)
            ) / denom.unsqueeze(1)
            circumcenter = points[:, 0] + circum_offset
            circumradius_sq = torch.sum(circum_offset * circum_offset, dim=1)
            valid = (normal_sq > 1.0e-8) & (circumradius_sq <= radius * radius)
            height = torch.sqrt(torch.relu(radius * radius - circumradius_sq))
            normal_unit = normal / torch.sqrt(normal_sq).clamp_min(1.0e-8).unsqueeze(1)
            centers = torch.stack((circumcenter + height.unsqueeze(1) * normal_unit,
                                   circumcenter - height.unsqueeze(1) * normal_unit), dim=1)
            for side in range(2):
                center = centers[:, side]
                relative = points - center[:, None, :]
                unit = torch.nn.functional.normalize(relative, dim=2)
                finger_mean = torch.nn.functional.normalize(relative[:, 1:3].mean(dim=1), dim=1)
                theta = torch.rad2deg(torch.acos(torch.clamp((unit[:, 0] * -finger_mean).sum(dim=1), -1.0, 1.0)))
                # 夹持弦定义为拇指接触点到食指/中指中点的线段。
                # 这样指标直接衡量拇指是否对准两指中间，而不受相邻
                # 食指--中指间距这一几何常数的误导。
                finger_mid = relative[:, 1:3].mean(dim=1)
                line = finger_mid - relative[:, 0]
                line_sq = torch.sum(line * line, dim=1).clamp_min(1.0e-12)
                line_t = torch.clamp(
                    torch.sum(-relative[:, 0] * line, dim=1) / line_sq, 0.0, 1.0
                )
                closest = relative[:, 0] + line_t.unsqueeze(1) * line
                chord_ratio = torch.linalg.vector_norm(closest, dim=1) / radius
                # 这些球心还必须能从当前 USD 的开位安全启动；否则 reset
                # 的保守 AABB 修正会把球心推离闭合工作区，扫描结果就不
                # 能直接用于动态验证。
                clearance = env._hand_collision_clearance_for(
                    origins + center, env._fruit_axes_m, env_ids
                )
                inside = (
                    valid
                    & (center[:, 0] >= 0.0) & (center[:, 0] <= args.max_center_x_mm / 1000.0)
                    & (center[:, 1] >= -0.12) & (center[:, 1] <= 0.12)
                    & (center[:, 2] >= args.min_center_z_mm / 1000.0) & (center[:, 2] <= 0.24)
                    & (clearance >= cfg.minimum_initial_clearance_m)
                )
                loss = theta / 30.0 + torch.relu(chord_ratio - 0.40) * 2.0
                loss = torch.where(inside, loss, torch.full_like(loss, 1.0e4))
                for idx in torch.argsort(loss)[: args.top_k].tolist():
                    records.append((
                        float(loss[idx]),
                        center[idx].detach().cpu(),
                        action[idx].detach().cpu(),
                        theta[idx].detach().cpu(),
                        chord_ratio[idx].detach().cpu(),
                        torch.linalg.vector_norm(relative[idx], dim=1)[:3].detach().cpu(),
                        clearance[idx].detach().cpu(),
                    ))
        records.sort(key=lambda record: record[0])
        print(
            f"circumcenter_scan num_envs={args.num_envs} batches={args.batches} "
            f"radius_mm={args.radius_mm} records={len(records)}",
            flush=True,
        )
        for rank, (loss, center, action, theta, chord_ratio, distance, clearance) in enumerate(records[: args.top_k], start=1):
            print({
                "rank": rank,
                "loss": round(loss, 6),
                "center_mm": [round(float(v * 1000.0), 3) for v in center],
                "distance_mm": [round(float(v * 1000.0), 3) for v in distance],
                "open_clearance_mm": round(float(clearance * 1000.0), 3),
                "theta_opp_deg": round(float(theta), 3),
                "chord_ratio": round(float(chord_ratio), 4),
                "action": [round(float(v), 4) for v in action],
            }, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
