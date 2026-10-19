#!/usr/bin/env python3
"""只读验证当前原生 PhysX 栈可解析左手关节与触觉路径。"""

from __future__ import annotations

import argparse
import re

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
from simulation.tomato_pick.isaac import env as native_env
from simulation.tomato_pick.isaac import TASK_ID
from simulation.tomato_pick.isaac.env_cfg import TomatoPickEnvCfg


def _left_path(path: str) -> str:
    path = path.replace("right_hand", "left_hand")
    return re.sub(r"/R([1-8])_Link", r"/L\1_Link", path)


def configure_native_module_for_left() -> None:
    native_env.RIGHT_HAND_CONTROL_JOINT_NAMES = tuple(
        name.replace("right_hand_", "left_hand_", 1)
        for name in native_env.RIGHT_HAND_CONTROL_JOINT_NAMES
    )
    native_env.RIGHT_FINGERTIP_FRAME_BODY_NAMES = {
        name: value.replace("right_hand_", "left_hand_")
        for name, value in native_env.RIGHT_FINGERTIP_FRAME_BODY_NAMES.items()
    }
    native_env.RIGHT_FINGERTIP_CONTACT_BODY_NAMES = {
        name: value.replace("right_hand_", "left_hand_")
        for name, value in native_env.RIGHT_FINGERTIP_CONTACT_BODY_NAMES.items()
    }
    native_env.RIGHT_HAND_BODY_NAMES = tuple(
        value.replace("right_hand_", "left_hand_")
        for value in native_env.RIGHT_HAND_BODY_NAMES
    )


def main() -> None:
    configure_native_module_for_left()
    cfg = TomatoPickEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    for sensor in cfg.fingertip_contacts.values():
        sensor.prim_path = _left_path(sensor.prim_path)
    env = gym.make(TASK_ID, cfg=cfg)
    try:
        obs, _ = env.reset(seed=42)
        impl = env.unwrapped
        print(
            "native_left_runtime_probe "
            f"policy={tuple(obs['policy'].shape)} "
            f"hand_joints={len(impl._hand_joint_ids)} "
            f"tip_bodies={len(impl._fingertip_contact_body_ids)}",
            flush=True,
        )
        print("hand_joint_names", list(impl._hand_joint_names), flush=True)
        print("hand_joint_ids", list(impl._hand_joint_ids), flush=True)
        print("tip_contact_names", [impl._robot.body_names[i] for i in impl._fingertip_contact_body_ids], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
