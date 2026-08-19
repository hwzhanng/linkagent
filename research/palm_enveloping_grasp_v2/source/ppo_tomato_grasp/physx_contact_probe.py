"""Deterministic contact smoke test for the converted hand-only PhysX asset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--overlap-index", action="store_true", help="Place the sphere at the index fingertip for a sensor-only contact check")
parser.add_argument("--action-value", type=float, default=0.0, help="Constant action value applied to all 10 policy dimensions")
parser.add_argument("--action-values", type=float, nargs=10, default=None, metavar=("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"), help="Explicit 10D action vector; overrides --action-value")
parser.add_argument("--ramp-to", type=float, default=None, help="Linearly ramp all 10 policy dimensions from zero to this value")
parser.add_argument("--ramp-steps", type=int, default=40, help="Number of policy steps used by --ramp-to")
parser.add_argument("--direct-position-targets", action="store_true", help="Bypass IsaacLab actuator write and hold the mapped target directly in the PhysX view")
parser.add_argument("--direct-positions", action="store_true", help="Teleport to the mapped joint pose and inspect kinematic fingertip placement")
parser.add_argument("--sphere-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Override sphere initial position for a geometry probe")
parser.add_argument("--compact", action="store_true", help="Print only the geometry/contact summary")
parser.add_argument("--trace", action="store_true", help="Print a compact state line after every policy step")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from simulation.ppo_tomato_grasp.physx_sphere_env import PhysxSphereGraspEnv, PhysxSphereGraspEnvCfg


try:
    cfg = PhysxSphereGraspEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    if args.sphere_pos is not None:
        cfg.sphere_cfg.init_state.pos = tuple(args.sphere_pos)
    env = PhysxSphereGraspEnv(cfg)
    obs, _ = env.reset()
    if args.trace:
        actuator = env.robot.actuators.get("left_hand")
        print({
            "active_joint_ids": env.active_joint_ids,
            "actuator_joint_indices": getattr(actuator, "joint_indices", None),
            "actuator_joint_names": getattr(actuator, "joint_names", None),
            "actuator_stiffness": getattr(actuator, "stiffness", None),
            "actuator_damping": getattr(actuator, "damping", None),
            "initial_active_q": env.robot.data.joint_pos[:, env.active_joint_ids].detach().cpu().tolist(),
            "initial_active_qd": env.robot.data.joint_vel[:, env.active_joint_ids].detach().cpu().tolist(),
            "initial_prev_targets": env.prev_targets.detach().cpu().tolist(),
        }, flush=True)
    if args.overlap_index:
        transforms = env.sphere.root_physx_view.get_transforms().clone()
        transforms[:, :3] = env._tip_positions()[:, 1].to(device=transforms.device)
        transforms[:, 3:7] = transforms.new_tensor((0.0, 0.0, 0.0, 1.0))
        env.sphere.root_physx_view.set_transforms(
            transforms, indices=torch.arange(args.num_envs, device=transforms.device)
        )
    if args.action_values is None:
        action = torch.full((args.num_envs, cfg.action_space), args.action_value, device=env.device)
    else:
        action = torch.tensor(args.action_values, device=env.device, dtype=torch.float32).expand(args.num_envs, -1).clone()
    peak = torch.zeros((args.num_envs, 5), device=env.device)
    if args.direct_positions:
        from simulation.ppo_tomato_grasp.hand_manifold import hybrid_action_to_joint_targets
        target = hybrid_action_to_joint_targets(action)
        full_pos = env.robot.root_physx_view.get_dof_positions().clone()
        full_vel = env.robot.root_physx_view.get_dof_velocities().clone()
        full_pos[:, env.active_joint_ids] = target.to(device=full_pos.device)
        full_vel.zero_()
        env.robot.root_physx_view.set_dof_positions(full_pos, env.robot._ALL_INDICES)
        env.robot.root_physx_view.set_dof_velocities(full_vel, env.robot._ALL_INDICES)
        env.sim.forward()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        env._read_contact_forces()
        if args.overlap_index:
            transforms = env.sphere.root_physx_view.get_transforms().clone()
            transforms[:, :3] = env._tip_positions()[:, 1].to(device=transforms.device)
            transforms[:, 3:7] = transforms.new_tensor((0.0, 0.0, 0.0, 1.0))
            env.sphere.root_physx_view.set_transforms(
                transforms, indices=torch.arange(args.num_envs, device=transforms.device)
            )
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            env._read_contact_forces()
            peak = torch.maximum(peak, env.contact_forces)
    elif args.direct_position_targets:
        from simulation.ppo_tomato_grasp.hand_manifold import hybrid_action_to_joint_targets
        target = hybrid_action_to_joint_targets(action)
        full_target = env.robot.root_physx_view.get_dof_position_targets().clone()
        full_target[:, env.active_joint_ids] = target.to(device=full_target.device)
        for _ in range(args.steps * cfg.decimation):
            env.robot.root_physx_view.set_dof_position_targets(full_target, env.robot._ALL_INDICES)
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            env._read_contact_forces()
            peak = torch.maximum(peak, env.contact_forces)
    else:
        for step in range(args.steps):
            if args.ramp_to is not None:
                value = args.ramp_to * min(1.0, float(step + 1) / max(1, args.ramp_steps))
                action.mul_(0.0).add_(value)
            obs, reward, terminated, truncated, _ = env.step(action)
            peak = torch.maximum(peak, env.contact_forces)
            if args.trace:
                print({
                    "step": step + 1,
                    "action0": float(action[0, 0]),
                    "target0": float(env.prev_targets[0, 0]),
                    "q0": float(env.robot.data.joint_pos[0, env.active_joint_ids[0]]),
                    "physx_q0": float(env.robot.root_physx_view.get_dof_positions()[0, env.active_joint_ids[0]]),
                    "physx_target0": float(env.robot.root_physx_view.get_dof_position_targets()[0, env.active_joint_ids[0]]),
                    "data_target0": float(env.robot.data.joint_pos_target[0, env.active_joint_ids[0]]),
                    "sim_target0": float(env.robot._joint_pos_target_sim[0, env.active_joint_ids[0]]),
                    "actuation0": float(env.robot.root_physx_view.get_dof_actuation_forces()[0, env.active_joint_ids[0]]),
                    "max_force0": float(env.robot.root_physx_view.get_dof_max_forces()[0, env.active_joint_ids[0]]),
                    "max_velocity0": float(env.robot.root_physx_view.get_dof_max_velocities()[0, env.active_joint_ids[0]]),
                    "projected_force0": float(env.robot.root_physx_view.get_dof_projected_joint_forces()[0, env.active_joint_ids[0]]),
                    "computed_torque0": float(env.robot.data.computed_torque[0, env.active_joint_ids[0]]),
                    "applied_torque0": float(env.robot.data.applied_torque[0, env.active_joint_ids[0]]),
                    "computed_torque1": float(env.robot.data.computed_torque[0, env.active_joint_ids[1]]),
                    "q1": float(env.robot.data.joint_pos[0, env.active_joint_ids[1]]),
                    "q_thumb0": float(env.robot.data.joint_pos[0, env.active_joint_ids[5]]),
                    "q_thumb1": float(env.robot.data.joint_pos[0, env.active_joint_ids[10]]),
                    "q_thumb3": float(env.robot.data.joint_pos[0, env.active_joint_ids[11]]),
                    "target_thumb0": float(env.robot.root_physx_view.get_dof_position_targets()[0, env.active_joint_ids[5]]),
                    "target_thumb1": float(env.robot.root_physx_view.get_dof_position_targets()[0, env.active_joint_ids[10]]),
                    "target_thumb3": float(env.robot.root_physx_view.get_dof_position_targets()[0, env.active_joint_ids[11]]),
                    "sphere": env.sphere.data.root_pos_w[0].detach().cpu().tolist(),
                    "force": env.contact_forces[0].detach().cpu().tolist(),
                    "stable_steps": int(env.stable_steps[0]),
                    "success": bool(env.success[0]),
                    "terminated": bool(terminated[0]),
                    "truncated": bool(truncated[0]),
                }, flush=True)
    report = {
            "steps": args.steps,
            "tip_names": env.robot.body_names,
            "active_joint_names": env.active_joint_names,
            "joint_names": env.robot.joint_names,
            "physx_view_methods": [name for name in dir(env.robot.root_physx_view) if "dof" in name.lower() or "name" in name.lower()],
            "physx_dof_paths": getattr(env.robot.root_physx_view, "dof_paths", None),
            "physx_dof_names": (
                env.robot.root_physx_view.get_dof_names()
                if hasattr(env.robot.root_physx_view, "get_dof_names")
                else getattr(env.robot.root_physx_view, "dof_names", None)
            ),
            "active_joint_pos": env.robot.data.joint_pos[:, env.active_joint_ids].detach().cpu().tolist(),
            "active_joint_vel": env.robot.data.joint_vel[:, env.active_joint_ids].detach().cpu().tolist(),
            "active_joint_target": env.prev_targets.detach().cpu().tolist(),
            "active_joint_pos_target_buffer": env.robot.data.joint_pos_target[:, env.active_joint_ids].detach().cpu().tolist(),
            "active_physx_position_target": env.robot.root_physx_view.get_dof_position_targets()[:, env.active_joint_ids].detach().cpu().tolist(),
            "data_joint_pos_full": env.robot.data.joint_pos.detach().cpu().tolist(),
            "physx_joint_pos_full": env.robot.root_physx_view.get_dof_positions().detach().cpu().tolist(),
            "physx_joint_vel_full": env.robot.root_physx_view.get_dof_velocities().detach().cpu().tolist(),
            "physx_joint_limits_full": env.robot.root_physx_view.get_dof_limits().detach().cpu().tolist(),
            "physx_damping_full": env.robot.root_physx_view.get_dof_dampings().detach().cpu().tolist(),
            "physx_stiffness_full": env.robot.root_physx_view.get_dof_stiffnesses().detach().cpu().tolist(),
            "physx_actuation_forces_full": env.robot.root_physx_view.get_dof_actuation_forces().detach().cpu().tolist(),
            "physx_projected_joint_forces_full": env.robot.root_physx_view.get_dof_projected_joint_forces().detach().cpu().tolist(),
            "physx_body_masses": env.robot.root_physx_view.get_masses().detach().cpu().tolist(),
            "active_joint_limits": env.robot.data.joint_limits[:, env.active_joint_ids].detach().cpu().tolist(),
            "sphere_pos": env.sphere.data.root_pos_w.detach().cpu().tolist(),
            "tip_pos": env._tip_positions().detach().cpu().tolist(),
            "peak_force_n": peak.detach().cpu().tolist(),
            "contact_sensor_names": list(env.scene.sensors.keys()),
        }
    if args.compact:
        print({
            "steps": args.steps,
            "active_joint_pos": report["active_joint_pos"],
            "active_joint_target": report["active_joint_target"],
            "sphere_pos": report["sphere_pos"],
            "tip_pos": report["tip_pos"],
            "peak_force_n": report["peak_force_n"],
            "success": env.success.detach().cpu().tolist(),
            "stable_steps": env.stable_steps.detach().cpu().tolist(),
        }, flush=True)
    else:
        print(report, flush=True)
finally:
    try:
        env.close()
    except Exception:
        pass
    app.close()
