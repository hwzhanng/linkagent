"""Audit whether the v3 hand can hold its reset pose without object contact."""

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
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--zero-gravity", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v3.env import V3EnvelopingPhysxTomatoGraspEnv, V3EnvelopingPhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp_v3.provenance import atomic_json_write, sha256_file


@torch.inference_mode()
def main() -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    cfg = V3EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    if args.zero_gravity:
        cfg.sim.gravity = (0.0, 0.0, 0.0)
    cfg = apply_profile(cfg, "v1_nominal_reliable", num_envs=1, seed=20260827)
    cfg.fingertip_collision_proxy_physical = False
    cfg.proxy_only_collision = True
    cfg.v3_urdf_terminal_mimic_enabled = False
    cfg.v3_joint_drive_effort_limit_nm = float(args.effort)
    cfg.v3_joint_drive_stiffness_nm_rad = float(args.kp)
    cfg.v3_joint_drive_damping_nm_s_rad = float(args.kd)
    actuator_cfg = cfg.robot_cfg.actuators["left_hand"]
    actuator_cfg.effort_limit_sim = float(args.effort)
    actuator_cfg.stiffness = float(args.kp)
    actuator_cfg.damping = float(args.kd)
    env = V3EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=20260827)
    # Remove every external contact: this probe is only gravity/static hold.
    transforms = env.tomato.root_physx_view.get_transforms().clone()
    transforms[:, 2] += 1.0
    env.tomato.root_physx_view.set_transforms(transforms, env.tomato._ALL_INDICES)
    initial = env.robot.data.joint_pos[:, env.active_joint_ids].clone()
    env.robot.set_joint_position_target(initial, joint_ids=env.active_joint_ids)
    target_after_set = env.robot.data.joint_pos_target[:, env.active_joint_ids].clone()
    stiffness_after_set = env.robot.root_physx_view.get_dof_stiffnesses()[:, env.active_joint_ids].clone()
    damping_after_set = env.robot.root_physx_view.get_dof_dampings()[:, env.active_joint_ids].clone()
    max_force_after_set = env.robot.root_physx_view.get_dof_max_forces()[:, env.active_joint_ids].clone()
    body_masses = env.robot.root_physx_view.get_masses().clone()
    body_coms = env.robot.root_physx_view.get_coms().clone()
    body_inertias = env.robot.root_physx_view.get_inertias().clone()
    max_error = torch.zeros_like(initial)
    max_speed = torch.zeros_like(initial)
    for _ in range(args.steps):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        q = env.robot.data.joint_pos[:, env.active_joint_ids]
        qd = env.robot.data.joint_vel[:, env.active_joint_ids]
        max_error = torch.maximum(max_error, (q - initial).abs())
        max_speed = torch.maximum(max_speed, qd.abs())
    final = env.robot.data.joint_pos[:, env.active_joint_ids]
    physx_target = env.robot.root_physx_view.get_dof_position_targets()[:, env.active_joint_ids]
    report = {
        "version": "v3_joint_drive_hold_probe_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_command_allowed": False,
        "effort_limit_nm": args.effort,
        "stiffness_nm_rad": args.kp,
        "damping_nm_s_rad": args.kd,
        "steps": args.steps,
        "zero_gravity": bool(args.zero_gravity),
        "hand_disable_gravity": bool(cfg.v3_hand_disable_gravity),
        "joint_names": list(env.active_joint_names),
        "initial_rad": [float(v) for v in initial[0].cpu()],
        "target_after_set_rad": [float(v) for v in target_after_set[0].cpu()],
        "stiffness_after_set": [float(v) for v in stiffness_after_set[0].cpu()],
        "damping_after_set": [float(v) for v in damping_after_set[0].cpu()],
        "max_force_after_set": [float(v) for v in max_force_after_set[0].cpu()],
        "actuator_type": type(env.robot.actuators["left_hand"]).__name__,
        "body_masses_kg": {
            name: float(mass) for name, mass in zip(env.robot.body_names, body_masses[0].cpu())
        },
        "total_robot_mass_kg": float(body_masses[0].sum().cpu()),
        "body_com_local_m": {
            name: [float(v) for v in com[:3]]
            for name, com in zip(env.robot.body_names, body_coms[0].cpu())
        },
        "body_inertia_diagonal_kg_m2": {
            name: [float(inertia[0]), float(inertia[4]), float(inertia[8])]
            for name, inertia in zip(env.robot.body_names, body_inertias[0].cpu())
        },
        "final_rad": [float(v) for v in final[0].cpu()],
        "physx_target_rad": [float(v) for v in physx_target[0].cpu()],
        "max_abs_error_rad": [float(v) for v in max_error[0].cpu()],
        "max_abs_speed_rad_s": [float(v) for v in max_speed[0].cpu()],
        "worst_error_rad": float(max_error.max().cpu()),
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
