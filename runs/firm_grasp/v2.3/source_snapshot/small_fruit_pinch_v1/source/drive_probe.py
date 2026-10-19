#!/usr/bin/env python3
"""检查孤立左手 PhysX 关节位置驱动是否真的响应目标。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--joint", type=str, default="thumb_joint2")
parser.add_argument("--effort", type=float, default=0.005)
parser.add_argument("--target", type=float, default=0.0)
parser.add_argument("--effort-limit", type=float, default=None)
parser.add_argument("--stiffness", type=float, default=None)
parser.add_argument("--damping", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv


def main() -> None:
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    for actuator in cfg.robot.actuators.values():
        if args.effort_limit is not None:
            actuator.effort_limit_sim = args.effort_limit
            actuator.effort_limit = args.effort_limit
        if args.stiffness is not None:
            actuator.stiffness = args.stiffness
        if args.damping is not None:
            actuator.damping = args.damping
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=20260907)
        targets = env._joint_targets.clone()
        joint_id = env._robot.joint_names.index(args.joint)
        targets[:, joint_id] = args.target
        env._robot.set_joint_position_target_index(target=targets, joint_ids=env._actuated_joint_ids)
        effort = torch.zeros_like(targets)
        effort[:, joint_id] = args.effort
        env._robot.set_joint_effort_target_index(target=effort, joint_ids=env._actuated_joint_ids)
        for step in range(args.steps):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt * env.cfg.decimation)
            q = env._robot.data.joint_pos.torch[0]
            v = env._robot.data.joint_vel.torch[0]
            names = ("thumb_joint0", "thumb_joint1", "thumb_joint2", "index_joint1", "middle_joint1", "index_joint2", "middle_joint2")
            print(step, {name: (float(q[env._robot.joint_names.index(name)]), float(v[env._robot.joint_names.index(name)])) for name in names}, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
