#!/usr/bin/env python3
"""打印孤立左手基座、指尖和复位水果的世界/手掌坐标，用于动作基准校准。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--center-local-mm", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"), help="覆盖父抓取基下的水果中心（mm）")
parser.add_argument("--pregrasp-distance-mm", type=float, default=None,
                    help="沿抓取基接近轴把初始水果移开指定距离（mm）")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from isaaclab.utils.math import quat_apply_inverse

from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv


def main() -> None:
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 20260907
    if args.center_local_mm is not None:
        cfg.closed_grasp_fruit_center_m = tuple(value / 1000.0 for value in args.center_local_mm)
    if args.pregrasp_distance_mm is not None:
        if args.pregrasp_distance_mm < 0.0:
            raise ValueError("pregrasp-distance-mm 不能为负数")
        distance_m = args.pregrasp_distance_mm / 1000.0
        cfg.pregrasp_distance_range_m = (distance_m, distance_m)
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=cfg.seed)
        base = env._robot.data.body_pose_w.torch[:, env._ee_body_id]
        tips = env._robot.data.body_pos_w.torch[:, env._fingertip_frame_body_ids]
        contacts = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids]
        fruit = env._tomato.data.root_pose_w.torch
        print("base_world", base[0].detach().cpu().tolist(), flush=True)
        print("tip_world", tips[0].detach().cpu().tolist(), flush=True)
        print("contact_world", contacts[0].detach().cpu().tolist(), flush=True)
        print("fruit_world", fruit[0].detach().cpu().tolist(), flush=True)
        local_tips = quat_apply_inverse(base[0, 3:7].expand(5, -1), tips[0] - base[0, :3])
        local_contacts = quat_apply_inverse(base[0, 3:7].expand(5, -1), contacts[0] - base[0, :3])
        local_fruit = quat_apply_inverse(base[0, 3:7], fruit[0, :3] - base[0, :3])
        print("tip_in_base", local_tips.detach().cpu().tolist(), flush=True)
        print("contact_in_base", local_contacts.detach().cpu().tolist(), flush=True)
        print("fruit_in_base", local_fruit.detach().cpu().tolist(), flush=True)
        print("grasp_basis_w", env._grasp_basis_w[0].detach().cpu().tolist(), flush=True)
        print("pinch_working_offset_m", list(env.cfg.pinch_working_offset_m), flush=True)
        print("fruit_axes_m", env._fruit_axes_m[0].detach().cpu().tolist(), flush=True)
        collision_names = [
            env._robot.body_names[index] for index in env._hand_collision_body_ids
        ]
        for name in (
            "thumb_link4", "thumb_link5",
            "index_link3", "index_link4",
            "middle_link3", "middle_link4",
        ):
            index = collision_names.index(name)
            print(
                "collision_aabb",
                name,
                "center_m",
                env._hand_collision_aabb_centers[index].detach().cpu().tolist(),
                "half_extent_m",
                env._hand_collision_aabb_half_extents[index].detach().cpu().tolist(),
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
