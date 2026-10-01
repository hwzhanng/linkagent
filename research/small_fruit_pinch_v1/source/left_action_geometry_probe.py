#!/usr/bin/env python3
"""在不推进物理时间的情况下检查 14D 动作对应的左手指尖 FK。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--pattern", default="all")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv
from research.small_fruit_pinch_v1.source.pinch_action_manifold import pinch_action_to_joint_targets


PATTERNS = {
    "open": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "p1": (0.9562, -0.5214, 0.0, 0.0, 0.4017, -0.7592, 0.2345, -0.9126, 0.0, 0.0, -0.3269, 0.0349, 0.0, 0.0),
    "p1_thumb_more": (0.9562, -0.5214, 0.0, 0.0, 0.85, -1.0, 0.2345, -0.9126, 0.0, 0.0, -0.85, 0.60, 0.0, 0.0),
    "close": (1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
    "close_soft": (0.75, 0.0, 0.0, 0.0, 0.90, 0.70, 0.70, 0.70, 0.0, 0.0, 0.8, 0.8, 0.7, 0.7),
    "thumb_only": (0.15, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "thumb_opp_negative": (0.75, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "thumb_axes_low": (0.75, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0),
    "thumb_axes_high": (0.75, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
    "three_finger_spread": (0.85, 0.0, 0.0, 0.65, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
    "fk_candidate": (0.9797, -0.7249, 0.0786, 0.0633, 0.5832, -0.1982, 0.4831, 0.8889, -0.6976, 0.8117, -0.3553, -0.5200, -0.7945, -0.5582),
    "workspace_candidate": (0.8239, -0.8590, 0.0811, 0.6406, 0.4097, 0.2718, -0.7128, 0.5314, -0.9198, 0.1658, -0.7514, -0.8352, -0.5911, 0.7236),
    # 外接球扫描得到的首个拇指对向候选；它的球心在基座坐标约
    # (49.45, -10.49, 177.48) mm，专用于后续动态验证。
    "opposition_candidate": (0.5563, 0.3046, -0.4787, 0.1579, 0.3133, 0.6795, 0.1310, -0.1900, 0.2070, 0.5084, -0.2590, -0.6060, -0.1727, 0.0229),
}


def write_static(env: NativeSmallFruitPinchEnv, action: tuple[float, ...]) -> None:
    values = torch.tensor(action, dtype=torch.float32, device=env.device).expand(env.num_envs, -1)
    targets = pinch_action_to_joint_targets(values)
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
    print("q_after_write", env._robot.data.joint_pos.torch[0].detach().cpu().tolist(), flush=True)


def main() -> None:
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 20260907
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=cfg.seed)
        patterns = PATTERNS.items() if args.pattern == "all" else ((args.pattern, PATTERNS[args.pattern]),)
        for name, action in patterns:
            write_static(env, action)
            tip = env._robot.data.body_pos_w.torch[0, env._fingertip_frame_body_ids]
            contact = env._robot.data.body_pos_w.torch[0, env._fingertip_contact_body_ids]
            print(f"{name} frame_tip={tip.detach().cpu().tolist()}", flush=True)
            print(f"{name} contact_body={contact.detach().cpu().tolist()}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
