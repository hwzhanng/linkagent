#!/usr/bin/env python3
"""输出少量环境的每步力/动作轨迹，定位过载发生在接触前还是接触后。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--guided", action="store_true")
parser.add_argument("--pattern", choices=("p1", "workspace"), default="workspace")
parser.add_argument("--center-local-mm", nargs=3, type=float, metavar=("X", "Y", "Z"), default=None)
parser.add_argument("--pregrasp-distance-mm", type=float, default=None)
parser.add_argument("--effort-limit", type=float, default=None)
parser.add_argument("--stiffness", type=float, default=None)
parser.add_argument("--damping", type=float, default=None)
parser.add_argument("--armature", type=float, default=None)
parser.add_argument("--velocity-limit", type=float, default=None)
parser.add_argument("--interpolation-steps", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import warp as wp

from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv


P1 = (0.9562, -0.5214, 0.0, 0.0, 0.4017, -0.7592, 0.2345, -0.9126, 0.0, 0.0, -0.3269, 0.0349, 0.0, 0.0)
WORKSPACE = (0.8239, -0.8590, 0.0811, 0.6406, 0.4097, 0.2718, -0.7128, 0.5314, -0.9198, 0.1658, -0.7514, -0.8352, -0.5911, 0.7236)


def main() -> None:
    env_cls = NativeSmallFruitPinchEnv
    if args.guided:
        from research.small_fruit_pinch_v1.source.guided_residual_env import GuidedResidualSmallFruitPinchEnv

        env_cls = GuidedResidualSmallFruitPinchEnv
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    if args.center_local_mm is not None:
        cfg.closed_grasp_fruit_center_m = tuple(value / 1000.0 for value in args.center_local_mm)
    if args.pregrasp_distance_mm is not None:
        if args.pregrasp_distance_mm < 0.0:
            raise ValueError("pregrasp-distance-mm 不能为负数")
        distance_m = args.pregrasp_distance_mm / 1000.0
        cfg.pregrasp_distance_range_m = (distance_m, distance_m)
    for actuator in cfg.robot.actuators.values():
        if args.effort_limit is not None:
            actuator.effort_limit_sim = args.effort_limit
            actuator.effort_limit = args.effort_limit
        if args.stiffness is not None:
            actuator.stiffness = args.stiffness
        if args.damping is not None:
            actuator.damping = args.damping
        if args.armature is not None:
            actuator.armature = args.armature
        if args.velocity_limit is not None:
            actuator.velocity_limit_sim = args.velocity_limit
            actuator.velocity_limit = args.velocity_limit
    if args.interpolation_steps is not None:
        env_cls.position_target_interpolation_steps = max(1, args.interpolation_steps)
    env = env_cls(cfg=cfg)
    try:
        obs, _ = env.reset(seed=20260924)
        print(f"joint_names={env._robot.joint_names}", flush=True)
        print(f"hand_joint_ids={env._hand_joint_ids} hand_joint_names={env._hand_joint_names}", flush=True)
        print(f"joint_pos_initial={env._robot.data.joint_pos.torch[0].detach().cpu().tolist()}", flush=True)
        print(f"joint_limits={env._robot.data.joint_pos_limits.torch[0].detach().cpu().tolist()}", flush=True)
        print(f"body_names={env._robot.body_names}", flush=True)
        print(f"body_mass={env._robot.data.body_mass.torch[0].detach().cpu().tolist()}", flush=True)
        print(f"fruit_world_initial={env._tomato.data.root_pos_w.torch[0].detach().cpu().tolist()}", flush=True)
        print(
            "actuator_joint_indices="
            + str({name: {"ids": actuator.joint_indices.tolist(), "names": list(actuator.joint_names), "limits": getattr(actuator, "_joint_lower_limit", None).detach().cpu().tolist() if hasattr(actuator, "_joint_lower_limit") else None, "effort_sim": getattr(actuator, "effort_limit_sim", None).detach().cpu().tolist() if hasattr(getattr(actuator, "effort_limit_sim", None), "detach") else getattr(actuator, "effort_limit_sim", None), "effort": getattr(actuator, "effort_limit", None).detach().cpu().tolist() if hasattr(getattr(actuator, "effort_limit", None), "detach") else getattr(actuator, "effort_limit", None)} for name, actuator in env._robot.actuators.items()}),
            flush=True,
        )
        action = torch.tensor(WORKSPACE if args.pattern == "workspace" else P1, device=env.device).expand(args.num_envs, -1)
        best = torch.zeros((args.num_envs, 5), device=env.device)
        for step in range(args.steps):
            obs, _, term, trunc, info = env.step(action)
            force = env._last_physical_force_n.detach().cpu()
            best = torch.maximum(best, env._last_physical_force_n)
            if step < 20 or step % 5 == 0 or bool((force > torch.tensor((2.0, 2.0, 2.0, 2.0, 2.0))).any()):
                tips = env._robot.data.body_pos_w.torch[0, env._fingertip_contact_body_ids].detach().cpu().tolist()
                thumb_ids = [env._robot.joint_names.index(name) for name in ("thumb_joint0", "thumb_joint1", "thumb_joint2", "thumb_joint3", "thumb_joint4")]
                thumb_q = env._robot.data.joint_pos.torch[0, thumb_ids].detach().cpu().tolist()
                thumb_v = env._robot.data.joint_vel.torch[0, thumb_ids].detach().cpu().tolist()
                active_q = env._robot.data.joint_pos.torch[0, env._hand_joint_ids].detach().cpu().tolist()
                active_v = env._robot.data.joint_vel.torch[0, env._hand_joint_ids].detach().cpu().tolist()
                thumb_target = env._joint_targets[0, thumb_ids].detach().cpu().tolist()
                target_buffer = env._robot.data.joint_pos_target.torch[0, thumb_ids].detach().cpu().tolist()
                sim_target = wp.to_torch(env._robot._joint_pos_target_sim)[0, thumb_ids].detach().cpu().tolist()
                computed_torque = env._robot.data.computed_torque.torch[0, thumb_ids].detach().cpu().tolist()
                applied_torque = env._robot.data.applied_torque.torch[0, thumb_ids].detach().cpu().tolist()
                effort_target_sim = wp.to_torch(env._robot._joint_effort_target_sim)[0, thumb_ids].detach().cpu().tolist()
                active_target = env._joint_targets[0, env._hand_joint_ids].detach().cpu().tolist()
                active_applied = env._robot.data.applied_torque.torch[0, env._hand_joint_ids].detach().cpu().tolist()
                print(
                    f"step={step:03d} max_force={force.amax(dim=1).tolist()} "
                    f"mean_force={force.mean(dim=0).tolist()} "
                    f"term={int(term.sum())} max_a={env._executed_actions.detach().cpu().abs().amax(dim=1).tolist()} "
                    f"a0-4={env._executed_actions[0,:5].detach().cpu().tolist()} "
                    f"tips0={tips} thumb_q={thumb_q} thumb_v={thumb_v} thumb_target={thumb_target} "
                    f"target_buffer={target_buffer} sim_target={sim_target} computed_torque={computed_torque} applied_torque={applied_torque} "
                    f"effort_target_sim={effort_target_sim} "
                    f"active_q={active_q} active_v={active_v} active_target={active_target} active_applied={active_applied}",
                    flush=True,
                )
        print(f"best={best.detach().cpu().tolist()}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
