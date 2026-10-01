#!/usr/bin/env python3
"""用 true-tip 碰撞面搜索 64--70 mm 三指对向构型。

该脚本只写仿真关节状态，不推进物理时间、不训练 PPO，也不连接实机。
它联合优化 14D 手型和基座坐标下的果心，拒绝把刚体原点或中段指节当作
指尖接触点。输出仍必须经过动态 PhysX 探针和人工视觉确认。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--iterations", type=int, default=50)
parser.add_argument("--elite-fraction", type=float, default=0.08)
parser.add_argument("--seed", type=int, default=20260923)
parser.add_argument("--radius-mm", type=float, default=33.5)
parser.add_argument("--top-k", type=int, default=10)
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
    if args.num_envs < 32 or args.iterations <= 0:
        raise ValueError("num-envs 至少为 32，iterations 必须为正数")
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        device = env.device
        count = env.num_envs
        ids = torch.arange(count, device=device)
        origins = env.scene.env_origins
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

        body_names = tuple(
            env._robot.body_names[index] for index in env._hand_collision_body_ids
        )
        # 真实受力壳在倒数第二刚体；末端固定小刚体只作为指垫中心标记。
        tip_names = ("thumb_link4", "index_link3", "middle_link3")
        tip_marker_names = ("thumb_link5", "index_link4", "middle_link4")
        tip_indices = torch.tensor(
            [body_names.index(name) for name in tip_names], device=device
        )
        marker_body_ids = torch.tensor(
            [env._robot.body_names.index(name) for name in tip_marker_names],
            device=device,
        )
        contact_body_ids = torch.tensor(
            [env._robot.body_names.index(name) for name in tip_names],
            device=device,
        )
        forbidden_mask = torch.tensor(
            [
                name.startswith(("thumb_", "index_", "middle_"))
                and name not in tip_names
                and name not in tip_marker_names
                for name in body_names
            ],
            dtype=torch.bool,
            device=device,
        )
        forbidden_names = [
            name
            for name, selected in zip(body_names, forbidden_mask.cpu().tolist())
            if selected
        ]

        # 14D action + xyz centre (metres).  The initial mean is deliberately
        # broad and does not encode a terminal grasp pose.
        mean = torch.tensor(
            [
                0.70, 0.00, 0.00, 0.55, 0.75,
                0.00, 0.00, 0.00, 0.00, 0.00,
                0.00, 0.00, 0.00, 0.00,
                0.065, -0.025, 0.170,
            ],
            dtype=torch.float32,
            device=device,
        )
        std = torch.tensor(
            [
                0.35, 0.55, 0.55, 0.55, 0.45,
                0.55, 0.55, 0.55, 0.05, 0.05,
                0.55, 0.55, 0.05, 0.05,
                0.035, 0.040, 0.055,
            ],
            dtype=torch.float32,
            device=device,
        )
        lower = torch.tensor(
            [-1.0] * 14 + [0.020, -0.085, 0.090], device=device
        )
        upper = torch.tensor(
            [1.0] * 14 + [0.125, 0.040, 0.245], device=device
        )
        # 环指/小指局部残差在三指任务中固定为零，避免浪费搜索维度。
        fixed_dims = (8, 9, 12, 13)
        elite_count = max(8, int(count * args.elite_fraction))
        radius = float(args.radius_mm) / 1000.0
        nominal_axes = torch.full((count, 3), radius, device=device)
        best_records: list[tuple[float, torch.Tensor, dict[str, object]]] = []

        for iteration in range(args.iterations):
            samples = mean + std * torch.randn(
                (count, 17), device=device, generator=generator
            )
            samples = torch.maximum(lower, torch.minimum(upper, samples))
            samples[:, list(fixed_dims)] = 0.0
            action = samples[:, :14]
            center = samples[:, 14:17]
            targets = pinch_action_to_joint_targets(action)
            env._joint_targets[:, env._hand_joint_ids] = targets
            env._update_mimic_targets(ids)
            env._robot.write_joint_position_to_sim_index(
                position=env._joint_targets,
                joint_ids=env._actuated_joint_ids,
                env_ids=ids,
            )
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(dt=0.0)

            center_w = origins + center
            clearances = env._hand_collision_clearance_matrix_for(
                center_w, nominal_axes, ids
            )
            closest = env._hand_collision_closest_points_for(center_w, ids)
            points = closest[:, tip_indices]
            marker_points = env._robot.data.body_pos_w.torch[:, marker_body_ids]
            pad_distance = torch.linalg.vector_norm(points - marker_points, dim=2)
            contact_origins = env._robot.data.body_pos_w.torch[:, contact_body_ids]
            tip_axes = marker_points - contact_origins
            tip_axis_sq = (tip_axes * tip_axes).sum(dim=2).clamp_min(1.0e-10)
            pad_progress = (
                ((points - contact_origins) * tip_axes).sum(dim=2) / tip_axis_sq
            )
            relative = points - center_w.unsqueeze(1)
            distance = torch.linalg.vector_norm(relative, dim=2).clamp_min(1.0e-8)
            unit = relative / distance.unsqueeze(2)
            opposing_mean = torch.nn.functional.normalize(
                unit[:, 1:3].mean(dim=1), dim=1
            )
            theta = torch.rad2deg(
                torch.acos(
                    torch.clamp((unit[:, 0] * -opposing_mean).sum(dim=1), -1.0, 1.0)
                )
            )
            finger_mid = relative[:, 1:3].mean(dim=1)
            line = finger_mid - relative[:, 0]
            line_sq = (line * line).sum(dim=1).clamp_min(1.0e-12)
            line_t = torch.clamp(
                (-relative[:, 0] * line).sum(dim=1) / line_sq, 0.0, 1.0
            )
            chord = torch.linalg.vector_norm(
                relative[:, 0] + line_t.unsqueeze(1) * line, dim=1
            ) / radius
            support_span = torch.linalg.vector_norm(points[:, 1] - points[:, 2], dim=1)
            tip_clearance = clearances[:, tip_indices]
            forbidden_values = clearances[:, forbidden_mask]
            forbidden_clearance, forbidden_worst = forbidden_values.min(dim=1)
            thumb_index_margin = env._hand_obb_pair_clearance_for(
                ("thumb_link2", "thumb_link3", "thumb_link4"),
                ("index_link1", "index_link2", "index_link3"),
                ids,
            )

            # 先求可达，再求严格对向。非末端穿透的代价高于角度和间距，
            # 防止 CEM 用指节挤进果实换取表面上的三指几何分数。
            loss = (
                tip_clearance.abs().mean(dim=1) / radius
                + 6.0 * torch.relu((-forbidden_clearance) / radius)
                + 4.0 * torch.relu(0.68 - pad_progress).mean(dim=1)
                + theta / 25.0
                + 3.0 * torch.relu(chord - 0.30)
                + 3.0 * torch.relu((0.020 - support_span) / 0.020)
                + 8.0 * torch.relu((0.003 - thumb_index_margin) / 0.003)
            )
            elite = torch.topk(loss, elite_count, largest=False).indices
            elite_samples = samples[elite]
            mean = 0.25 * mean + 0.75 * elite_samples.mean(dim=0)
            std = (0.25 * std + 0.75 * elite_samples.std(dim=0)).clamp_min(
                torch.tensor([0.015] * 14 + [0.0015] * 3, device=device)
            )
            mean[list(fixed_dims)] = 0.0

            best = int(torch.argmin(loss))
            metrics = {
                "iteration": iteration,
                "tip_clearance_mm": [
                    round(float(value * 1000.0), 3)
                    for value in tip_clearance[best]
                ],
                "tip_pad_distance_mm": [
                    round(float(value * 1000.0), 3)
                    for value in pad_distance[best]
                ],
                "tip_axial_progress": [
                    round(float(value), 4) for value in pad_progress[best]
                ],
                "forbidden_clearance_mm": round(
                    float(forbidden_clearance[best] * 1000.0), 3
                ),
                "forbidden_worst_body": forbidden_names[
                    int(forbidden_worst[best])
                ],
                "theta_opp_deg": round(float(theta[best]), 3),
                "chord_ratio": round(float(chord[best]), 4),
                "index_middle_span_mm": round(float(support_span[best] * 1000.0), 3),
                "thumb_index_margin_mm": round(
                    float(thumb_index_margin[best] * 1000.0), 3
                ),
            }
            best_records.append(
                (float(loss[best]), samples[best].detach().cpu(), metrics)
            )
            if iteration % 5 == 0 or iteration == args.iterations - 1:
                print(
                    f"cem iteration={iteration:03d} loss={float(loss[best]):.6f} "
                    f"metrics={metrics}",
                    flush=True,
                )

        best_records.sort(key=lambda item: item[0])
        print("opposition_surface_cem_best", flush=True)
        for rank, (loss, sample, metrics) in enumerate(
            best_records[: args.top_k], start=1
        ):
            print(
                {
                    "rank": rank,
                    "loss": round(loss, 6),
                    "action": [round(float(value), 5) for value in sample[:14]],
                    "center_base_mm": [
                        round(float(value * 1000.0), 3) for value in sample[14:17]
                    ],
                    **metrics,
                },
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
