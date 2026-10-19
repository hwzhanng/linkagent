#!/usr/bin/env python3
"""只读打印孤立左手触觉传感器与 articulation 刚体绑定关系。"""
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser); args = parser.parse_args(); app = AppLauncher(args).app
from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv

cfg = NativeSmallFruitPinchCfg(); cfg.scene.num_envs = 1; cfg.sim.device = args.device
env = NativeSmallFruitPinchEnv(cfg=cfg)
try:
    env.reset(seed=20260912)
    print("robot_bodies", list(enumerate(env._robot.body_names)), flush=True)
    print("tip_frame_ids", list(env._fingertip_frame_body_ids), flush=True)
    print("tip_contact_ids", list(env._fingertip_contact_body_ids), flush=True)
    print("tip_frame_names", [env._robot.body_names[i] for i in env._fingertip_frame_body_ids], flush=True)
    print("tip_contact_names", [env._robot.body_names[i] for i in env._fingertip_contact_body_ids], flush=True)
    for name, sensor in env._contact_sensors.items():
        print("sensor", name, "prim_path", sensor.cfg.prim_path, "body_names", getattr(sensor.cfg, "body_names", None), flush=True)
        print("sensor_contact_pos_shape", name, None if sensor.data.contact_pos_w is None else tuple(sensor.data.contact_pos_w.shape), flush=True)
        if sensor.data.contact_pos_w is not None:
            print("sensor_contact_pos_value", name, sensor.data.contact_pos_w.torch.detach().cpu().tolist(), flush=True)
finally:
    env.close()
