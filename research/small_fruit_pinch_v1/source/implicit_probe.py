#!/usr/bin/env python3
"""诊断孤立左手的 PhysX 隐式位置驱动（不训练、不连接实机）。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--pattern", choices=("p1", "workspace"), default="workspace")
parser.add_argument("--kp", type=float, default=5.0)
parser.add_argument("--kd", type=float, default=1.5)
parser.add_argument("--effort-limit", type=float, default=0.50)
parser.add_argument("--center-local-mm", nargs=3, type=float, default=(-90.0, -30.0, -150.0))
parser.add_argument("--print-every", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import warp as wp
from isaaclab.actuators import ImplicitActuatorCfg

from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv


P1 = (0.9562, -0.5214, 0.0, 0.0, 0.4017, -0.7592, 0.2345, -0.9126, 0.0, 0.0, -0.3269, 0.0349, 0.0, 0.0)
WORKSPACE = (0.8239, -0.8590, 0.0811, 0.6406, 0.4097, 0.2718, -0.7128, 0.5314, -0.9198, 0.1658, -0.7514, -0.8352, -0.5911, 0.7236)


def main() -> None:
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.closed_grasp_fruit_center_m = tuple(value / 1000.0 for value in args.center_local_mm)
    cfg.robot.actuators = {
        "g20_independent": ImplicitActuatorCfg(
            joint_names_expr=[r"thumb_joint[0-3]", r"(index|middle|ring|little)_joint[0-2]"],
            effort_limit_sim=float(args.effort_limit),
            velocity_limit_sim=1.0,
            stiffness=float(args.kp),
            damping=float(args.kd),
            friction=0.01,
        ),
        "g20_followers": ImplicitActuatorCfg(
            joint_names_expr=[r"thumb_joint4", r"(index|middle|ring|little)_joint3"],
            effort_limit_sim=float(args.effort_limit),
            velocity_limit_sim=1.0,
            stiffness=float(args.kp),
            damping=float(args.kd),
            friction=0.01,
        ),
    }
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=20260907)
        print(f"has_implicit={getattr(env._robot, '_has_implicit_actuators', None)} actuator_types="
              f"{ {name: type(act).__name__ for name, act in env._robot.actuators.items()} }", flush=True)
        action_values = P1 if args.pattern == "p1" else WORKSPACE
        action = torch.tensor(action_values, device=env.device, dtype=torch.float32).view(1, -1)
        thumb_ids = [env._robot.joint_names.index(name) for name in ("thumb_joint0", "thumb_joint1", "thumb_joint2", "thumb_joint3", "thumb_joint4")]
        finger_ids = [env._robot.joint_names.index(name) for name in ("index_joint1", "middle_joint1", "ring_joint1", "little_joint1")]
        for step in range(args.steps):
            _, _, term, trunc, _ = env.step(action)
            if step % max(1, args.print_every) == 0 or bool(term.any() | trunc.any()):
                q = env._robot.data.joint_pos.torch[0]
                target = env._joint_targets[0]
                sim_target = wp.to_torch(env._robot._joint_pos_target_sim)[0]
                tips = env._robot.data.body_pos_w.torch[0, env._fingertip_contact_body_ids]
                print(
                    f"step={step:03d} q_thumb={[round(float(v),4) for v in q[thumb_ids]]} "
                    f"q_finger={[round(float(v),4) for v in q[finger_ids]]} "
                    f"target_thumb={[round(float(v),4) for v in target[thumb_ids]]} "
                    f"sim_thumb={[round(float(v),4) for v in sim_target[thumb_ids]]} "
                    f"tips={[tuple(round(float(v),4) for v in p) for p in tips]} "
                    f"force={[round(float(v),4) for v in env._last_physical_force_n[0]]} "
                    f"term={bool(term[0])} trunc={bool(trunc[0])}",
                    flush=True,
                )
            if bool(term.any() | trunc.any()):
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
