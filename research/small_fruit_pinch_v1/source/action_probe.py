#!/usr/bin/env python3
"""用确定性 14D 手型扫描验证左手—果实接触可达性。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=320)
parser.add_argument("--pattern", default="all")
parser.add_argument("--fruit-axis-min-mm", type=float, default=None)
parser.add_argument("--fruit-axis-max-mm", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.utils.math import quat_apply_inverse

from research.small_fruit_pinch_v1.source.native_env import (
    NativeSmallFruitPinchCfg,
    NativeSmallFruitPinchEnv,
)


PATTERNS = {
    "close_opposed": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "close_distal": (1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0),
    "close_residual": (0.8, 0.7, 0.0, 0.0, 0.9, 0.5, 0.5, 0.5, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0),
    "p1_candidate": (0.9562, -0.5214, 0.0, 0.0, 0.4017, -0.7592, 0.2345, -0.9126, 0.0, 0.0, -0.3269, 0.0349, 0.0, 0.0),
    # P1 邻域的单指方向诊断；只用于确定动作坐标的物理正负，不参与训练。
    "p1_thumb_more": (0.9562, -0.5214, 0.0, 0.0, 0.4017, -1.0, 0.2345, -0.9126, 0.0, 0.0, -0.3269, 0.0349, 0.0, 0.0),
    "p1_thumb_less": (0.9562, -0.5214, 0.0, 0.0, 0.4017, -0.50, 0.2345, -0.9126, 0.0, 0.0, -0.3269, 0.0349, 0.0, 0.0),
    "p1_index_more": (0.9562, -0.5214, 0.0, 0.0, 0.4017, -0.7592, 0.55, -0.9126, 0.0, 0.0, -0.10, 0.0349, 0.0, 0.0),
    "p1_middle_more": (0.9562, -0.5214, 0.0, 0.0, 0.4017, -0.7592, 0.2345, -0.65, 0.0, 0.0, -0.3269, 0.20, 0.0, 0.0),
    # 静态左手 FK 批量搜索得到的三指候选，仅作为动态 PhysX 的待验证
    # 动作，不是成功姿态或训练终点。
    "fk_candidate": (0.9797, -0.7249, 0.0786, 0.0633, 0.5832, -0.1982, 0.4831, 0.8889, -0.6976, 0.8117, -0.3553, -0.5200, -0.7945, -0.5582),
    "workspace_candidate": (0.8239, -0.8590, 0.0811, 0.6406, 0.4097, 0.2718, -0.7128, 0.5314, -0.9198, 0.1658, -0.7514, -0.8352, -0.5911, 0.7236),
}


def main() -> None:
    if args.pattern != "all" and args.pattern not in PATTERNS:
        raise ValueError(f"未知 pattern：{args.pattern}")
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    if (args.fruit_axis_min_mm is None) != (args.fruit_axis_max_mm is None):
        raise ValueError("fruit-axis-min-mm 与 fruit-axis-max-mm 必须同时提供")
    if args.fruit_axis_min_mm is not None:
        if not (0.0 < args.fruit_axis_min_mm <= args.fruit_axis_max_mm):
            raise ValueError("果径半轴范围必须为正且 min <= max")
        axis_min_m = args.fruit_axis_min_mm / 1000.0
        axis_max_m = args.fruit_axis_max_mm / 1000.0
        cfg.events.fruit_geometry.params["axis_ranges_m"] = {
            "x": (axis_min_m, axis_max_m),
            "y": (axis_min_m, axis_max_m),
            "z": (axis_min_m, axis_max_m),
        }
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    report: dict[str, dict[str, list[float]]] = {}
    try:
        patterns = PATTERNS.items() if args.pattern == "all" else ((args.pattern, PATTERNS[args.pattern]),)
        for offset, (name, values) in enumerate(patterns):
            env.reset(seed=42 + offset)
            ee_pose = env._robot.data.body_pose_w.torch[:, env._ee_body_id]
            fruit_pose = env._tomato.data.root_pose_w.torch
            fruit_in_palm = quat_apply_inverse(
                ee_pose[:, 3:7], fruit_pose[:, :3] - ee_pose[:, :3]
            )
            action = torch.tensor(values, device=env.device).expand(args.num_envs, -1).clone()
            peak = torch.zeros((args.num_envs, 5), device=env.device)
            nearest_surface = torch.full((args.num_envs, 5), float("inf"), device=env.device)
            terminals = {"success": 0, "drop": 0, "overload": 0, "self_collision": 0}
            for _ in range(args.steps):
                _, _, terminated, _, info = env.step(action)
                terminals["success"] += int(info["step"]["success_count"])
                terminals["drop"] += int(info["step"]["drop_count"])
                terminals["overload"] += int(info["step"]["overload_count"])
                terminals["self_collision"] += int(info["step"]["self_collision_count"])
                peak = torch.maximum(peak, env._last_physical_force_n)
                tips = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids]
                center = env._tomato.data.root_pos_w.torch
                radial = torch.linalg.vector_norm(tips - center.unsqueeze(1), dim=2)
                surface = (radial - env._fruit_axes_m.mean(dim=1, keepdim=True)).abs()
                nearest_surface = torch.minimum(nearest_surface, surface)
            report[name] = {
                "mean_peak_force_n": peak.mean(dim=0).tolist(),
                "nearest_surface_distance_m": nearest_surface.mean(dim=0).tolist(),
                "fruit_in_left_palm_m_env0": fruit_in_palm[0].tolist(),
                "closed_tip_in_left_palm_m_env0": quat_apply_inverse(
                    env._robot.data.body_pose_w.torch[0, env._ee_body_id, 3:7].expand(5, -1),
                    env._robot.data.body_pos_w.torch[0, env._fingertip_contact_body_ids]
                    - env._robot.data.body_pos_w.torch[0, env._ee_body_id],
                ).tolist(),
                "terminal_events": terminals,
            }
        print(f"action_probe mean_peak_force_n={report}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
