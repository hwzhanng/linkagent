#!/usr/bin/env python3
"""批量测量隔离左手 USD 的 position-target 方向（仅仿真诊断）。

每次只给一个关节很小的目标，其余关节保持零位；不推进任务奖励，也不
连接任何真实设备。输出 ``target_sign -> q_sign``，用于确认 canonical
动作流形和导入 USD 的关节轴向是否一致。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--magnitude", type=float, default=0.20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from research.small_fruit_pinch_v1.source.native_env import (
    NativeSmallFruitPinchCfg,
    NativeSmallFruitPinchEnv,
)


def settle_zero(env: NativeSmallFruitPinchEnv) -> None:
    env._joint_targets.zero_()
    env._update_mimic_targets(torch.arange(env.num_envs, device=env.device))
    zeros = torch.zeros_like(env._joint_targets)
    ids = torch.arange(env.num_envs, device=env.device)
    env._robot.write_joint_position_to_sim_index(position=zeros, joint_ids=env._actuated_joint_ids, env_ids=ids)
    env._robot.write_joint_velocity_to_sim_index(velocity=zeros, joint_ids=env._actuated_joint_ids, env_ids=ids)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)


def probe(env: NativeSmallFruitPinchEnv, name: str, target_value: float) -> float:
    settle_zero(env)
    target = torch.zeros_like(env._joint_targets)
    target[:, env._robot.joint_names.index(name)] = target_value
    env._joint_targets.copy_(target)
    for _ in range(args.steps):
        # Explicit actuators compute their PD effort in Articulation.write_data
        # after ``set_*_target`` is called.  Calling the environment's narrow
        # action hook here avoids the common false-negative probe that only
        # writes a target buffer and therefore never runs the actuator model.
        env._apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    return float(env._robot.data.joint_pos.torch[0, env._robot.joint_names.index(name)])


def main() -> None:
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=20260907)
        names = tuple(env._robot.joint_names)
        for name in names:
            positive = probe(env, name, args.magnitude)
            negative = probe(env, name, -args.magnitude)
            print(
                f"{name:14s} target=+{args.magnitude:.3f} -> q={positive:+.6f}; "
                f"target=-{args.magnitude:.3f} -> q={negative:+.6f}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
