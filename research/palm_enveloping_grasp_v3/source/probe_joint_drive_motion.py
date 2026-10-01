"""Collision-free ramp tracking audit for the v3 implicit hand drive."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--effort", type=float, required=True)
parser.add_argument("--kp", type=float, required=True)
parser.add_argument("--kd", type=float, required=True)
parser.add_argument("--ramp-steps", type=int, default=1560)
parser.add_argument("--hold-steps", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v3.env import V3EnvelopingPhysxTomatoGraspEnv, V3EnvelopingPhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp_v3.provenance import atomic_json_write, sha256_file


TARGET = (
    0.5874857903, 0.9678264260, 0.4663121402, 0.4773970544,
    0.7955846786, -0.0478790924, -0.1276895702, -0.0374275073,
    0.0457147248, 0.1716733575, 0.0312314257, 0.5707497001,
    0.3816780448, 0.1510673463, 0.0607470423, 0.4736952484,
)


@torch.inference_mode()
def main() -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    cfg = V3EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg = apply_profile(cfg, "v1_nominal_reliable", num_envs=1, seed=20260828)
    cfg.fingertip_collision_proxy_physical = False
    cfg.proxy_only_collision = True
    cfg.v3_urdf_terminal_mimic_enabled = False
    cfg.v3_joint_drive_effort_limit_nm = float(args.effort)
    cfg.v3_joint_drive_stiffness_nm_rad = float(args.kp)
    cfg.v3_joint_drive_damping_nm_s_rad = float(args.kd)
    actuator = cfg.robot_cfg.actuators["left_hand"]
    actuator.effort_limit_sim = float(args.effort)
    actuator.stiffness = float(args.kp)
    actuator.damping = float(args.kd)
    env = V3EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=20260828)
    transforms = env.tomato.root_physx_view.get_transforms().clone()
    transforms[:, 2] += 1.0
    env.tomato.root_physx_view.set_transforms(transforms, env.tomato._ALL_INDICES)
    start = env.robot.data.joint_pos[:, env.active_joint_ids].clone()
    target = start.new_tensor(TARGET).view(1, 16)
    max_error = torch.zeros_like(start)
    max_speed = torch.zeros_like(start)
    for step in range(args.ramp_steps + args.hold_steps):
        fraction = min(1.0, float(step + 1) / float(args.ramp_steps))
        command = start + fraction * (target - start)
        env.robot.set_joint_position_target(command, joint_ids=env.active_joint_ids)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        q = env.robot.data.joint_pos[:, env.active_joint_ids]
        qd = env.robot.data.joint_vel[:, env.active_joint_ids]
        max_error = torch.maximum(max_error, (q - command).abs())
        max_speed = torch.maximum(max_speed, qd.abs())
    final = env.robot.data.joint_pos[:, env.active_joint_ids]
    final_error = final - target
    report = {
        "version": "v3_joint_drive_motion_probe_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_command_allowed": False,
        "effort_limit_nm": args.effort,
        "stiffness_nm_rad": args.kp,
        "damping_nm_s_rad": args.kd,
        "ramp_steps": args.ramp_steps,
        "hold_steps": args.hold_steps,
        "joint_names": list(env.active_joint_names),
        "target_rad": [float(v) for v in target[0].cpu()],
        "final_rad": [float(v) for v in final[0].cpu()],
        "final_error_rad": [float(v) for v in final_error[0].cpu()],
        "final_rmse_rad": float(torch.sqrt(final_error.square().mean()).cpu()),
        "worst_tracking_error_rad": float(max_error.max().cpu()),
        "worst_speed_rad_s": float(max_speed.max().cpu()),
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json_write(output, report)
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
