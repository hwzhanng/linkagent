#!/usr/bin/env python3
"""扫描无初始穿透条件下的左手三指共同可达区。

每个并行环境从同一个有效重置位姿出发，仅在抓取局部基中平移果实。
候选先通过原生碰撞 AABB 间隙检查，再以同一平滑闭合动作在 PhysX 中验证。
这把“果实可达性”与“PPO 是否学会了动作”明确分离。
"""

from __future__ import annotations

import argparse
import itertools

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=160)
parser.add_argument("--seed", type=int, default=44)
parser.add_argument("--top-k", type=int, default=12)
parser.add_argument("--actions-per-pose", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from research.small_fruit_pinch_v1.source.native_env import (
    NativeSmallFruitPinchCfg,
    NativeSmallFruitPinchEnv,
)


# 相对当前已验证重置位置的 (接近、横向、下垂) 位移，单位 m。
OFFSETS_M = tuple(itertools.product(
    (-0.018, -0.006, 0.006, 0.018),
    (-0.030, -0.010, 0.010, 0.030),
    (-0.024, -0.008, 0.008, 0.024),
))
CLOSE_ACTION = (1.0, 1.0, 0.0, 0.0, 1.0, 0.45, 0.45, 0.65, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0)


def _candidate_actions(count: int, device: str, seed: int) -> torch.Tensor:
    """围绕全局闭合的 Sobol 搜索，保留根部/远端的独立可达性。"""

    engine = torch.quasirandom.SobolEngine(dimension=9, scramble=True, seed=seed)
    unit = engine.draw(count).to(device=device)
    result = torch.tensor(CLOSE_ACTION, device=device).expand(count, -1).clone()
    result[:, 0] = 0.55 + 0.45 * unit[:, 0]
    result[:, 1] = -1.0 + 2.0 * unit[:, 1]
    result[:, 4] = 0.15 + 0.85 * unit[:, 2]
    result[:, 5:8] = -1.0 + 2.0 * unit[:, 3:6]
    result[:, 10:12] = -1.0 + 2.0 * unit[:, 6:8]
    # 第一条是已知可产生拇指/食指载荷的诊断基准，防止覆盖采样误判执行链路。
    result[0] = torch.tensor(CLOSE_ACTION, device=device)
    return result


def main() -> None:
    if args.actions_per_pose <= 0:
        raise ValueError("actions-per-pose 必须为正数")
    offsets_per_pose = args.actions_per_pose
    count = len(OFFSETS_M) * offsets_per_pose
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = count
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=args.seed)
        base_pos = env._tomato.data.root_pos_w.torch.clone()
        offsets = torch.tensor(OFFSETS_M, dtype=torch.float32, device=env.device)
        offsets = offsets.repeat_interleave(offsets_per_pose, dim=0)
        offset_w = torch.bmm(env._grasp_basis_w, offsets.unsqueeze(2)).squeeze(2)
        candidate_pos = base_pos + offset_w
        env_ids = torch.arange(count, device=env.device)
        clearance = env._hand_collision_clearance_for(
            candidate_pos, env._fruit_axes_m, env_ids
        )
        valid = clearance >= cfg.minimum_initial_clearance_m
        pose = env._tomato.data.root_pose_w.torch.clone()
        pose[valid, :3] = candidate_pos[valid]
        velocity = torch.zeros((count, 6), device=env.device)
        env._tomato.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        env._tomato.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
        # 茎连接仅随果实一起平移；不引入捕获、锁定或位置重置力。
        env._stem_rest_attachment_pos_w[valid] += offset_w[valid]
        env._stem_plant_anchor_pos_w[valid] += offset_w[valid]
        env._pinch_reference_center_w.copy_(pose[:, :3])

        action = _candidate_actions(count, env.device, args.seed)
        peak = torch.zeros((count, 3), device=env.device)
        tri_frames = torch.zeros(count, dtype=torch.long, device=env.device)
        for _ in range(args.steps):
            env.step(action)
            force = env._last_physical_force_n[:, :3]
            peak = torch.maximum(peak, force)
            tri_frames += (force >= 0.08).all(dim=1).long()

        score = torch.amin(peak, dim=1)
        score = torch.where(valid, score, torch.full_like(score, -1.0))
        order = torch.argsort(score, descending=True)[: args.top_k]
        rows = [
            {
                "offset_grasp_m": [round(float(v), 4) for v in offsets[index].tolist()],
                "initial_clearance_m": round(float(clearance[index].item()), 5),
                "peak_force_n": [round(float(v), 4) for v in peak[index].tolist()],
                "min_peak_force_n": round(float(score[index].item()), 4),
                "tri_contact_frames": int(tri_frames[index].item()),
                "action": [round(float(v), 4) for v in action[index].tolist()],
            }
            for index in order
        ]
        print(
            "geometry_scan "
            f"valid_rate={valid.float().mean().item():.4f} "
            f"triple_candidate_rate={(tri_frames > 0).float().mean().item():.4f} "
            f"top={rows}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
