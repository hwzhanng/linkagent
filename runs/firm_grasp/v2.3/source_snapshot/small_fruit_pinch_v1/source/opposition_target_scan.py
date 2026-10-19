#!/usr/bin/env python3
"""搜索无掌托、可从开位启动的 64--70 mm 三指对向候选。

``opposition_circumcenter_scan`` 解决的是“三个点能否落在同一个球面”问题，
但它的外接球心可能已经穿进孤立手掌。这个扫描直接同时随机化果心和 14D
动作，用实际接触刚体位置计算表面误差、拇指对向角、夹持弦偏心及开位
clearance，避免把求解器第一步的排斥冲量误认为抓取动作。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--batches", type=int, default=60)
parser.add_argument("--top-k", type=int, default=20)
parser.add_argument("--radius-mm", type=float, default=33.5)
parser.add_argument("--seed", type=int, default=20260912)
parser.add_argument("--center-x-min-mm", type=float, default=68.0)
parser.add_argument("--center-x-max-mm", type=float, default=140.0)
parser.add_argument("--center-y-min-mm", type=float, default=-75.0)
parser.add_argument("--center-y-max-mm", type=float, default=75.0)
parser.add_argument("--center-z-min-mm", type=float, default=100.0)
parser.add_argument("--center-z-max-mm", type=float, default=245.0)
parser.add_argument("--clearance-mm", type=float, default=1.0)
parser.add_argument("--maximum-opposition-angle-deg", type=float, default=28.0)
parser.add_argument("--maximum-chord-ratio", type=float, default=0.35)
parser.add_argument("--minimum-support-span-mm", type=float, default=20.0)
parser.add_argument("--maximum-tip-gap-mm", type=float, default=6.0)
parser.add_argument("--base-action", nargs=14, type=float, default=None)
parser.add_argument("--base-center-mm", nargs=3, type=float, default=None)
parser.add_argument("--action-noise", type=float, default=0.20)
parser.add_argument("--center-noise-mm", type=float, default=12.0)
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
    # bias the random proposal toward a closed but non-limit hand while keeping
    # all 14 dimensions independent enough to expose distal solutions.
    action[:, 0] = torch.empty(count, device=device).uniform_(0.45, 1.0, generator=generator)
    action[:, 1] = torch.empty(count, device=device).uniform_(-0.80, 0.80, generator=generator)
    action[:, 3] = torch.empty(count, device=device).uniform_(-0.80, 0.80, generator=generator)
    action[:, 4] = torch.empty(count, device=device).uniform_(-0.65, 0.90, generator=generator)
    action[:, 5:8] = torch.empty((count, 3), device=device).uniform_(-1.0, 1.0, generator=generator)
    action[:, 10:12] = torch.empty((count, 2), device=device).uniform_(-1.0, 1.0, generator=generator)
    return action


def main() -> None:
    if args.num_envs <= 0 or args.batches <= 0 or args.top_k <= 0:
        raise ValueError("num-envs、batches 和 top-k 必须为正数")
    if args.radius_mm <= 0.0 or args.clearance_mm < 0.0:
        raise ValueError("radius-mm 必须为正数，clearance-mm 不能为负数")
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
        clearance_threshold = float(args.clearance_mm) / 1000.0
        minimum_support_span = float(args.minimum_support_span_mm) / 1000.0
        maximum_tip_gap = float(args.maximum_tip_gap_mm) / 1000.0
        body_names = tuple(
            env._robot.body_names[index] for index in env._hand_collision_body_ids
        )
        # collision body 与接触传感器绑定在倒数第二节，frame marker
        # 在末端固定刚体；两者都属于允许的远端区域，不能把 link3
        # 误当成近端禁触体，否则所有真实指垫候选都会被判非法。
        tip_names = ("thumb_link4", "index_link3", "middle_link3")
        allowed_distal_names = set(tip_names) | {
            "thumb_link5", "index_link4", "middle_link4"
        }
        tip_indices = torch.tensor(
            [body_names.index(name) for name in tip_names],
            dtype=torch.long,
            device=env.device,
        )
        # `closest_points_w` 是按碰撞 AABB 列表排列的，不能用它推断真实
        # 指尖刚体中心。动态教师/传感器使用 articulation body id；静态
        # 候选也必须使用同一组 id，避免扫描通过而动态执行偏离数厘米。
        tip_body_ids = torch.tensor(
            [env._robot.find_bodies(name)[0][0] for name in tip_names],
            dtype=torch.long,
            device=env.device,
        )
        forbidden_mask = torch.tensor(
            [
                name.startswith(("thumb_", "index_", "middle_"))
                and name not in allowed_distal_names
                for name in body_names
            ],
            dtype=torch.bool,
            device=env.device,
        )
        forbidden_names = [
            name for name, selected in zip(body_names, forbidden_mask.cpu().tolist()) if selected
        ]
        records: list[tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        valid_records = 0
        for _ in range(args.batches):
            if args.base_action is None:
                action = _sample_actions(args.num_envs, env.device, generator)
            else:
                action = torch.tensor(
                    args.base_action, dtype=torch.float32, device=env.device
                ).expand(args.num_envs, -1).clone()
                action += torch.randn(
                    action.shape, device=env.device, generator=generator
                ) * float(args.action_noise)
                action.clamp_(-1.0, 1.0)
            if args.base_center_mm is None:
                center = torch.empty((args.num_envs, 3), device=env.device)
                center[:, 0].uniform_(args.center_x_min_mm / 1000.0, args.center_x_max_mm / 1000.0, generator=generator)
                center[:, 1].uniform_(args.center_y_min_mm / 1000.0, args.center_y_max_mm / 1000.0, generator=generator)
                center[:, 2].uniform_(args.center_z_min_mm / 1000.0, args.center_z_max_mm / 1000.0, generator=generator)
            else:
                center = torch.tensor(
                    args.base_center_mm, dtype=torch.float32, device=env.device
                ).div(1000.0).expand(args.num_envs, -1).clone()
                center += torch.randn(
                    center.shape, device=env.device, generator=generator
                ) * (float(args.center_noise_mm) / 1000.0)
            targets = pinch_action_to_joint_targets(action)
            env._joint_targets[:, env._hand_joint_ids] = targets
            env._update_mimic_targets(env_ids)
            env._robot.write_joint_position_to_sim_index(
                position=env._joint_targets, joint_ids=env._actuated_joint_ids, env_ids=env_ids
            )
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(dt=0.0)
            center_w = origins + center
            clearance_matrix = env._hand_collision_clearance_matrix_for(
                center_w, env._fruit_axes_m, env_ids
            )
            closest_points_w = env._hand_collision_closest_points_for(center_w, env_ids)
            points = env._robot.data.body_pos_w.torch[:, tip_body_ids]
            relative = points - center[:, None, :]
            distances = torch.linalg.vector_norm(relative, dim=2)
            tip_clearance = clearance_matrix[:, tip_indices]
            surface = tip_clearance.abs()
            unit = relative / distances.clamp_min(1.0e-8).unsqueeze(-1)
            finger_mean = torch.nn.functional.normalize(unit[:, 1:3].mean(dim=1), dim=1)
            theta = torch.rad2deg(torch.acos(torch.clamp((unit[:, 0] * -finger_mean).sum(dim=1), -1.0, 1.0)))
            finger_mid = relative[:, 1:3].mean(dim=1)
            line = finger_mid - relative[:, 0]
            line_sq = torch.sum(line * line, dim=1).clamp_min(1.0e-12)
            line_t = torch.clamp(torch.sum(-relative[:, 0] * line, dim=1) / line_sq, 0.0, 1.0)
            closest = relative[:, 0] + line_t.unsqueeze(1) * line
            chord_ratio = torch.linalg.vector_norm(closest, dim=1) / radius
            support_span = torch.linalg.vector_norm(points[:, 1] - points[:, 2], dim=1)
            forbidden_values = clearance_matrix[:, forbidden_mask]
            forbidden_clearance, forbidden_worst = forbidden_values.min(dim=1)
            clearance = clearance_matrix.amin(dim=1)
            # Surface error is the primary objective; geometry is only accepted
            # when all required fingers are near the sphere and the thumb is on
            # the opposite side of the index/middle midpoint.
            loss = (
                surface[:, :3].mean(dim=1) / radius
                + theta / 35.0
                + 2.0 * torch.relu(chord_ratio - 0.45)
                + 2.0 * torch.relu(
                    (minimum_support_span - support_span) / minimum_support_span
                )
                + 4.0 * torch.relu(
                    (clearance_threshold - forbidden_clearance) / radius
                )
                + 0.4 * torch.relu(0.70 - distances[:, :3].min(dim=1).values / radius)
            )
            valid = (
                (forbidden_clearance >= clearance_threshold)
                & (surface[:, :3].amax(dim=1) <= maximum_tip_gap)
                & (theta <= float(args.maximum_opposition_angle_deg))
                & (chord_ratio <= float(args.maximum_chord_ratio))
                & (support_span >= minimum_support_span)
            )
            valid_records += int(valid.sum().item())
            for idx in torch.argsort(loss)[: args.top_k].tolist():
                records.append((
                    float(loss[idx]), center[idx].detach().cpu(), action[idx].detach().cpu(),
                    theta[idx].detach().cpu(), chord_ratio[idx].detach().cpu(),
                    support_span[idx].detach().cpu(), forbidden_clearance[idx].detach().cpu(),
                    forbidden_names[int(forbidden_worst[idx])], clearance[idx].detach().cpu(),
                    tip_clearance[idx].detach().cpu(), bool(valid[idx]),
                ))
        records.sort(key=lambda record: record[0])
        print(
            f"opposition_target_scan num_envs={args.num_envs} batches={args.batches} "
            f"radius_mm={args.radius_mm} records={len(records)} valid={valid_records}",
            flush=True,
        )
        for rank, (loss, center, action, theta, chord_ratio, support_span, forbidden_clearance, forbidden_body, clearance, tip_clearance, valid) in enumerate(records[: args.top_k], start=1):
            print({
                "rank": rank,
                "loss": round(loss, 6),
                "center_mm": [round(float(v * 1000.0), 3) for v in center],
                "minimum_clearance_mm": round(float(clearance * 1000.0), 3),
                "tip_clearance_mm": [round(float(v * 1000.0), 3) for v in tip_clearance],
                "forbidden_clearance_mm": round(float(forbidden_clearance * 1000.0), 3),
                "forbidden_worst_body": forbidden_body,
                "theta_opp_deg": round(float(theta), 3),
                "chord_ratio": round(float(chord_ratio), 4),
                "index_middle_span_mm": round(float(support_span * 1000.0), 3),
                "valid": valid,
                "action": [round(float(v), 4) for v in action],
            }, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
