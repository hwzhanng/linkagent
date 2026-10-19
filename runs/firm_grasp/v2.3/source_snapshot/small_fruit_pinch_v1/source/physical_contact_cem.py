#!/usr/bin/env python3
"""用真实 PhysX 接触搜索三指末端共同接触构型。

本探针把候选关节状态瞬时写入并只推进少量物理子步，以低成本排除
碰撞 AABB 的假阳性。它不训练 PPO、不连接实机；输出仍需连续轨迹探针
验证首接触顺序，不能直接作为成功权重。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--iterations", type=int, default=40)
parser.add_argument("--settle-steps", type=int, default=3)
parser.add_argument("--elite-fraction", type=float, default=0.08)
parser.add_argument("--seed", type=int, default=20260927)
parser.add_argument("--top-k", type=int, default=8)
parser.add_argument("--diameter-mm", type=float, default=70.0)
parser.add_argument("--initial-control", nargs="+", type=float, default=None)
parser.add_argument("--initial-center-mm", nargs=3, type=float, default=None)
parser.add_argument("--fk-center", action="store_true", help="用当前真实指尖 FK 几何中心覆盖候选果心")
parser.add_argument("--control-std", type=float, default=None)
parser.add_argument(
    "--direct-joints",
    action="store_true",
    help="搜索拇指/食指/中指十个 canonical 关节，而不是受耦合的 14D 流形。",
)
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
from research.small_fruit_pinch_v1.source.legacy_bridge import load_legacy_hand_manifold


def main() -> None:
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.episode_length_s = 60.0
    if not 64.0 <= args.diameter_mm <= 70.0:
        raise ValueError("diameter-mm 必须位于当前 B0 的 64–70 mm 范围")
    radius = float(args.diameter_mm) / 2000.0
    cfg.events.fruit_geometry.params["axis_ranges_m"] = {
        axis: (radius, radius) for axis in ("x", "y", "z")
    }
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        device = env.device
        count = env.num_envs
        ids = torch.arange(count, device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        manifold_mean = torch.tensor(
            [
                0.526, 0.72, -0.20, 0.32, 0.93,
                0.10, -0.20, 0.20, 0.0, 0.0,
                0.18, 0.40, 0.0, 0.0,
                0.050, -0.041, 0.156,
            ], device=device
        )
        manifold_std = torch.tensor(
            [
                0.20, 0.25, 0.25, 0.25, 0.12,
                0.25, 0.25, 0.25, 0.02, 0.02,
                0.25, 0.25, 0.02, 0.02,
                0.025, 0.025, 0.035,
            ], device=device
        )
        # 资产实际 canonical 顺序下，拇指为 0/5/10/11，食指为 1/6/12，
        # 中指为 2/7/13；仅搜索这十个必需轴。
        direct_joint_indices = (0, 1, 2, 5, 6, 7, 10, 11, 12, 13)
        legacy = load_legacy_hand_manifold()
        safe_lower = torch.tensor(legacy.LEFT_SAFE_LOWER_RAD, device=device)
        safe_upper = torch.tensor(legacy.LEFT_SAFE_UPPER_RAD, device=device)
        template_action = manifold_mean[:14].unsqueeze(0)
        template_targets = pinch_action_to_joint_targets(template_action)[0]
        if args.direct_joints:
            normalized = 2.0 * (
                (template_targets - safe_lower) / (safe_upper - safe_lower)
            ) - 1.0
            mean = torch.cat(
                (normalized[list(direct_joint_indices)], manifold_mean[14:17])
            )
            std = torch.tensor([0.40] * 10 + [0.025, 0.025, 0.035], device=device)
            lower = torch.tensor([-1.0] * 10 + [0.015, -0.095, 0.080], device=device)
            upper = torch.tensor([1.0] * 10 + [0.105, 0.020, 0.235], device=device)
            control_dim = 10
            fixed_dims: tuple[int, ...] = ()
        else:
            mean = manifold_mean
            std = manifold_std
            lower = torch.tensor([-1.0] * 14 + [0.015, -0.095, 0.080], device=device)
            upper = torch.tensor([1.0] * 14 + [0.105, 0.020, 0.235], device=device)
            control_dim = 14
            # 全 14D 动作都开放；此前固定 12/13 维会锁死中指/环指远端
            # 残差，可能把可行的三指接触错误排除。正式发布仍由左手
            # 合同和近端禁触门筛选，不会因此放宽安全边界。
            fixed_dims = ()
        if args.initial_control is not None:
            if len(args.initial_control) != control_dim:
                raise ValueError(f"initial-control 必须包含 {control_dim} 个值")
            mean[:control_dim] = torch.tensor(args.initial_control, device=device)
        if args.initial_center_mm is not None:
            mean[control_dim:control_dim + 3] = torch.tensor(
                args.initial_center_mm, device=device
            ) / 1000.0
        if args.control_std is not None:
            if args.control_std <= 0.0:
                raise ValueError("control-std 必须为正数")
            std[:control_dim] = float(args.control_std)
        elite_count = max(8, int(count * args.elite_fraction))
        records: list[tuple[float, torch.Tensor, dict[str, object]]] = []
        default_joint_pos = env._robot.data.default_joint_pos.torch
        collision_names = tuple(
            env._robot.body_names[index] for index in env._hand_collision_body_ids
        )
        tip_indices = torch.tensor(
            [collision_names.index(name) for name in ("thumb_link4", "index_link3", "middle_link3")],
            device=device,
        )
        required_forbidden_mask = torch.tensor(
            [
                name.startswith(("thumb_", "index_", "middle_"))
                and name not in ("thumb_link4", "thumb_link5", "index_link3", "index_link4", "middle_link3", "middle_link4")
                for name in collision_names
            ], dtype=torch.bool, device=device,
        )

        for iteration in range(args.iterations):
            samples = mean + std * torch.randn(
                (count, control_dim + 3), generator=generator, device=device
            )
            samples = torch.maximum(lower, torch.minimum(upper, samples))
            samples[:, list(fixed_dims)] = 0.0
            controls = samples[:, :control_dim]
            center = samples[:, control_dim:control_dim + 3]
            if args.direct_joints:
                targets = template_targets.expand(count, -1).clone()
                unit = 0.5 * (controls + 1.0)
                selected = torch.tensor(direct_joint_indices, device=device)
                targets[:, selected] = safe_lower[selected] + unit * (
                    safe_upper[selected] - safe_lower[selected]
                )
            else:
                targets = pinch_action_to_joint_targets(controls)
            env._joint_targets[:, env._hand_joint_ids] = targets
            env._update_mimic_targets(ids)
            joint_pos = default_joint_pos.clone()
            joint_pos[:, env._actuated_joint_ids] = env._joint_targets
            env._robot.write_joint_position_to_sim_index(
                position=joint_pos,
                joint_ids=None,
                env_ids=ids,
            )
            env._robot.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(joint_pos),
                joint_ids=None,
                env_ids=ids,
            )
            fruit_pose = env._tomato.data.default_root_pose.torch.clone()
            fruit_pose[:, :3] = env.scene.env_origins + center
            fruit_pose[:, 3:7] = fruit_pose.new_tensor((0.0, 0.0, 0.0, 1.0))
            env._tomato.write_root_pose_to_sim_index(root_pose=fruit_pose, env_ids=ids)
            env._tomato.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((count, 6), device=device), env_ids=ids
            )
            env.scene.write_data_to_sim()
            env.sim.forward()
            fk_tips = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids[:3]]
            pair01 = torch.linalg.vector_norm(fk_tips[:, 0] - fk_tips[:, 1], dim=1)
            pair02 = torch.linalg.vector_norm(fk_tips[:, 0] - fk_tips[:, 2], dim=1)
            pair12 = torch.linalg.vector_norm(fk_tips[:, 1] - fk_tips[:, 2], dim=1)
            fk_max_pair = torch.maximum(pair01, torch.maximum(pair02, pair12))
            if args.fk_center:
                # 统一使用 articulation body FK，避免碰撞 AABB 局部索引与
                # 动态传感器坐标系不一致。该中心只用于候选接触验证，不是
                # 训练时硬编码的终态目标。
                fk_tips = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids[:3]]
                fk_center = fk_tips.mean(dim=1)
                fruit_pose[:, :3] = fk_center
                env._tomato.write_root_pose_to_sim_index(root_pose=fruit_pose, env_ids=ids)
                env._tomato.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros((count, 6), device=device), env_ids=ids
                )
                env.scene.write_data_to_sim()
                env.sim.forward()
            for _ in range(max(1, args.settle_steps)):
                env._robot.set_joint_position_target_index(
                    target=env._joint_targets, joint_ids=env._actuated_joint_ids
                )
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=cfg.sim.dt)

            effective_all = env._physical_finger_force()
            effective = effective_all[:, :3]
            proximal_force = env._forbidden_proximal_force()
            raw = env._raw_finger_force_n[:, :3]
            valid = env._contact_region_axial_progress[:, :3] >= 0.68
            contact = effective >= 0.02
            # 候选搜索先允许五指中的任意三指组合，避免把“中指几何不可达”
            # 错误地当成整个三指任务不可达；发布阶段仍由合同固定拇指/食指/中指。
            contact_count = contact.sum(dim=1)
            raw_count = (raw >= 0.02).sum(dim=1)
            missing = torch.relu(0.02 - effective).sum(dim=1)
            excessive = torch.relu(effective - 2.0).sum(dim=1)
            forbidden = env._forbidden_contact_force_n.sum(dim=1)
            margin = env._hand_obb_pair_clearance_for(
                ("thumb_link2", "thumb_link3", "thumb_link4"),
                ("index_link1", "index_link2", "index_link3"),
                ids,
            )
            axes = torch.full((count, 3), radius, device=device)
            geom_clearance = env._hand_collision_clearance_matrix_for(
                env._tomato.data.root_pos_w.torch, axes, ids
            )
            tip_geom = geom_clearance[:, tip_indices]
            proximal_geom = geom_clearance[:, required_forbidden_mask].amin(dim=1)
            loss = (
                4.0 * torch.relu(3 - contact_count).float()
                + 0.5 * (3 - raw_count).float()
                + 20.0 * missing
                + 1.5 * forbidden
                + 1.5 * proximal_force.sum(dim=1)
                + 0.25 * excessive
                + 8.0 * torch.relu((0.003 - margin) / 0.003)
                + 0.8 * tip_geom.abs().mean(dim=1) / radius
                + 3.0 * torch.relu(-proximal_geom / radius)
                + 80.0 * torch.relu(fk_max_pair - 0.068)
            )
            elite = torch.topk(loss, elite_count, largest=False).indices
            elite_samples = samples[elite]
            mean = 0.25 * mean + 0.75 * elite_samples.mean(dim=0)
            std = (0.25 * std + 0.75 * elite_samples.std(dim=0)).clamp_min(
                torch.tensor([0.015] * control_dim + [0.0015] * 3, device=device)
            )
            mean[list(fixed_dims)] = 0.0
            best = int(torch.argmin(loss))
            metrics = {
                "iteration": iteration,
                "effective_force_n": [round(float(v), 4) for v in effective_all[best]],
                "raw_force_n": [round(float(v), 4) for v in env._raw_finger_force_n[best]],
                "tip_axial_progress": [round(float(v), 4) for v in env._contact_region_axial_progress[best, :3]],
                "valid_tip_region": [bool(v) for v in valid[best]],
                "thumb_index_margin_mm": round(float(margin[best] * 1000.0), 3),
                "tip_aabb_clearance_mm": [round(float(v * 1000.0), 3) for v in tip_geom[best]],
                "proximal_aabb_clearance_mm": round(float(proximal_geom[best] * 1000.0), 3),
                "proximal_force_n": [round(float(v), 4) for v in proximal_force[best]],
            }
            records.append((float(loss[best]), samples[best].detach().cpu(), metrics))
            if iteration % 5 == 0 or iteration == args.iterations - 1:
                print(f"physical_cem iteration={iteration:03d} loss={float(loss[best]):.5f} metrics={metrics}", flush=True)

        records.sort(key=lambda item: item[0])
        print("physical_contact_cem_best", flush=True)
        for rank, (loss, sample, metrics) in enumerate(records[: args.top_k], 1):
            print({
                "rank": rank,
                "loss": round(loss, 6),
                "contract": "direct_required_joints_v0" if args.direct_joints else "pinch_14d",
                "control": [round(float(v), 5) for v in sample[:control_dim]],
                "canonical_joint_indices": list(direct_joint_indices) if args.direct_joints else None,
                "center_base_mm": [round(float(v * 1000.0), 3) for v in sample[control_dim:control_dim + 3]],
                **metrics,
            }, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
