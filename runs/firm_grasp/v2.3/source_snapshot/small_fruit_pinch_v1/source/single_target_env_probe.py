#!/usr/bin/env python3
"""在完整 Native 左手执行路径中测单关节目标响应。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--joint", default="thumb_joint1")
parser.add_argument("--target", type=float, default=0.8)
parser.add_argument("--steps", type=int, default=80)
parser.add_argument("--print-every", type=int, default=10)
parser.add_argument("--direct", action="store_true", help="绕过环境 _apply_action，仅写原始目标用于对比")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import warp as wp

from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv


def main() -> None:
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=20260907)
        joint_id = env._robot.joint_names.index(args.joint)
        targets = env._joint_targets.clone()
        targets[:, joint_id] = float(args.target)
        ids = [env._robot.joint_names.index(name) for name in ("thumb_joint0", "thumb_joint1", "thumb_joint2", "thumb_joint3", "thumb_joint4")]
        if args.direct:
            env._robot.set_joint_position_target_index(
                target=targets,
                joint_ids=env._actuated_joint_ids,
            )
            env._robot.set_joint_effort_target_index(
                target=torch.zeros_like(targets),
                joint_ids=env._actuated_joint_ids,
            )
        for step in range(args.steps):
            if args.direct:
                pass
            else:
                env._joint_targets.copy_(targets)
                env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            if step % max(1, args.print_every) == 0 or step == args.steps - 1:
                q = env._robot.data.joint_pos.torch[0]
                v = env._robot.data.joint_vel.torch[0]
                sim_effort = wp.to_torch(env._robot._joint_effort_target_sim)[0]
                target = env._robot.data.joint_pos_target.torch[0]
                print(
                    f"step={step:03d} target={args.target:+.3f} "
                    f"q={[round(float(q[i]),6) for i in ids]} "
                    f"v={[round(float(v[i]),6) for i in ids]} "
                    f"target={float(target[joint_id]):+.5f} effort={float(sim_effort[joint_id]):+.5f}", flush=True,
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
