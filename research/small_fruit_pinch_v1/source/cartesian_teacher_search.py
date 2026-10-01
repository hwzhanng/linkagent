#!/usr/bin/env python3
"""左手三指笛卡尔指尖教师搜索（只读/离线，不训练、不连接实机）。"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--diameter-mm", type=float, default=68.0)
parser.add_argument("--iterations", type=int, default=80)
parser.add_argument("--center-mm", nargs=3, type=float, default=(61.0, -16.0, 168.0))
parser.add_argument("--step-size", type=float, default=0.025)
parser.add_argument("--debug-jacobian", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); app = AppLauncher(args).app

import torch
from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv
from research.small_fruit_pinch_v1.source.pinch_action_manifold import pinch_action_to_joint_targets
from research.small_fruit_pinch_v1.source.legacy_bridge import load_legacy_hand_manifold


def main() -> None:
    cfg = NativeSmallFruitPinchCfg(); cfg.scene.num_envs = args.num_envs; cfg.sim.device = args.device
    cfg.episode_length_s = 30.0
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=20260950); d = env.device; n = env.num_envs
        ids = torch.arange(n, device=d)
        q = pinch_action_to_joint_targets(torch.zeros((n, 14), device=d))
        env._joint_targets[:, env._hand_joint_ids] = q; env._update_mimic_targets(ids)
        full = env._robot.data.default_joint_pos.torch.clone(); full[:, env._hand_joint_ids] = q
        env._robot.write_joint_position_to_sim_index(position=full, joint_ids=None, env_ids=ids)
        env._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(full), joint_ids=None, env_ids=ids)
        env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(dt=cfg.sim.dt)
        center = torch.tensor(args.center_mm, device=d).view(1, 3) / 1000.0 + env.scene.env_origins
        radius = args.diameter_mm / 2000.0
        tip_ids = env._fingertip_contact_body_ids[:3]
        # 以当前 FK 指尖到果心的方向构造球面目标，避免假定 USD 世界轴与
        # 手掌坐标完全一致；这仍保持三点分离，拇指与食/中指位于相对侧。
        initial_pos = env._robot.data.body_pos_w.torch[:, tip_ids]
        radial = initial_pos - center[:, None, :]
        radial = radial / torch.linalg.vector_norm(radial, dim=2, keepdim=True).clamp_min(1e-6)
        desired = center[:, None, :] + radius * radial
        jac_ids = torch.tensor([i + env._robot.num_base_dofs for i in env._hand_joint_ids], device=d)
        legacy = load_legacy_hand_manifold(); lo = torch.tensor(legacy.LEFT_SAFE_LOWER_RAD, device=d); hi = torch.tensor(legacy.LEFT_SAFE_UPPER_RAD, device=d)
        # 每根手指只允许自己的轴运动，避免 Jacobian 伪逆把动作泄漏到其他手指。
        groups = ((0, 5, 10, 11), (1, 6, 12), (2, 7, 13))
        if args.debug_jacobian:
            print("joint_names", [str(env._robot.joint_names[i]) for i in env._hand_joint_ids], flush=True)
            for digit, group in enumerate(groups):
                cols = torch.tensor(group, device=d) + env._robot.num_base_dofs
                J = env._robot.data.body_link_jacobian_w[:, tip_ids[digit], :3, cols]
                print("jacobian", digit, "cols", group, "norm", torch.linalg.vector_norm(J, dim=1)[0].detach().cpu().tolist(), flush=True)
        for it in range(args.iterations):
            pos = env._robot.data.body_pos_w.torch[:, tip_ids]
            err = desired - pos
            for digit, group in enumerate(groups):
                cols = torch.tensor(group, device=d) + env._robot.num_base_dofs
                J = env._robot.data.body_link_jacobian_w[:, tip_ids[digit], :3, cols]
                JJ = J @ J.transpose(1, 2) + 1e-4 * torch.eye(3, device=d).unsqueeze(0)
                dq = J.transpose(1, 2) @ torch.linalg.solve(JJ, err[:, digit].unsqueeze(2))
                # 保守步长，避免一次更新扫过碰撞壳。
                q[:, group] = q[:, group] + args.step_size * dq.squeeze(2)
            q = torch.maximum(lo, torch.minimum(hi, q))
            env._joint_targets[:, env._hand_joint_ids] = q; env._update_mimic_targets(ids)
            full[:, env._hand_joint_ids] = q
            env._robot.write_joint_position_to_sim_index(position=full, joint_ids=None, env_ids=ids)
            env._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(full), joint_ids=None, env_ids=ids)
            env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(dt=cfg.sim.dt)
        pos = env._robot.data.body_pos_w.torch[:, tip_ids]
        print("cartesian_teacher", {
            "tip_distance_mm": torch.linalg.vector_norm(pos - center[:, None, :], dim=2).mean(0).mul(1000).detach().cpu().tolist(),
            "target_error_mm": torch.linalg.vector_norm(pos - desired, dim=2).mean(0).mul(1000).detach().cpu().tolist(),
            "joint_targets": q[0].detach().cpu().tolist(),
            "proximal_force_n": env._forbidden_proximal_force_n[0].detach().cpu().tolist(),
            "tip_force_n": env._last_physical_force_n[0, :3].detach().cpu().tolist(),
        }, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
