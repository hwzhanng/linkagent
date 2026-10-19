#!/usr/bin/env python3
"""以固定候选动作推进单左手 PhysX，输出可读的动作/接触摘要。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=220)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--pattern", choices=("p1", "workspace"), default="p1")
parser.add_argument("--center-local-mm", nargs=3, type=float, default=(-90.0, -4.0, -165.0))
parser.add_argument(
    "--center-base-mm",
    nargs=3,
    type=float,
    default=None,
    metavar="X",
    help="直接覆盖 reset 后的基座坐标球心（mm），用于验证保守 AABB 之外的候选",
)
parser.add_argument(
    "--action-values",
    nargs=14,
    type=float,
    default=None,
    metavar="A",
    help="覆盖 pattern 的 14D 动作（用于 FK 扫描得到的候选动作）",
)
parser.add_argument("--print-every", type=int, default=10)
parser.add_argument(
    "--continue-after-done",
    action="store_true",
    help="诊断自动 reset；默认首个终止即停止，避免把下一回合零状态混进轨迹。",
)
parser.add_argument(
    "--episode-seconds",
    type=float,
    default=None,
    help="仅探针延长单回合时长（秒），避免候选动作尚未到位就被默认时限截断。",
)
parser.add_argument(
    "--stem-frequency",
    type=float,
    default=None,
    help="仅探针覆盖茎约束线性固有频率（rad/s），用于隔离果实漂移影响。",
)
parser.add_argument(
    "--stem-damping",
    type=float,
    default=None,
    help="仅探针覆盖茎约束阻尼比。",
)
parser.add_argument(
    "--palm-fruit-collision-filter",
    action="store_true",
    help="诊断模式：过滤仅 base_link↔Tomato 的碰撞，以验证掌部粗糙 AABB 是否掩盖对向动作；不用于最终发布。",
)
parser.add_argument(
    "--explicit-hand-pd",
    action="store_true",
    help="将两组手部执行器切换为显式 IdealPD，便于隔离 USD 的轨迹响应诊断。",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg, NativeSmallFruitPinchEnv


P1 = (0.9562, -0.5214, 0.0, 0.0, 0.4017, -0.7592, 0.2345, -0.9126, 0.0, 0.0, -0.3269, 0.0349, 0.0, 0.0)
WORKSPACE = (0.8239, -0.8590, 0.0811, 0.6406, 0.4097, 0.2718, -0.7128, 0.5314, -0.9198, 0.1658, -0.7514, -0.8352, -0.5911, 0.7236)


def main() -> None:
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    if args.episode_seconds is not None:
        if args.episode_seconds <= 0.0:
            raise ValueError("episode-seconds 必须为正数")
        cfg.episode_length_s = float(args.episode_seconds)
    if args.stem_frequency is not None:
        if args.stem_frequency <= 0.0:
            raise ValueError("stem-frequency 必须为正数")
        cfg.stem_linear_natural_frequency_rad_s = float(args.stem_frequency)
    if args.stem_damping is not None:
        if args.stem_damping <= 0.0:
            raise ValueError("stem-damping 必须为正数")
        cfg.stem_linear_damping_ratio = float(args.stem_damping)
    cfg.closed_grasp_fruit_center_m = tuple(value / 1000.0 for value in args.center_local_mm)
    cfg.palm_fruit_collision_filter_enabled = bool(args.palm_fruit_collision_filter)
    if args.explicit_hand_pd:
        from isaaclab.actuators import IdealPDActuatorCfg

        for name in ("g20_independent", "g20_followers"):
            source = cfg.robot.actuators[name]
            cfg.robot.actuators[name] = IdealPDActuatorCfg(
                joint_names_expr=list(source.joint_names_expr),
                effort_limit=source.effort_limit,
                velocity_limit=source.velocity_limit,
                effort_limit_sim=source.effort_limit_sim,
                velocity_limit_sim=source.velocity_limit_sim,
                stiffness=source.stiffness,
                damping=source.damping,
                armature=source.armature,
                friction=source.friction,
                dynamic_friction=source.dynamic_friction,
                viscous_friction=source.viscous_friction,
            )
    env = NativeSmallFruitPinchEnv(cfg=cfg)
    try:
        env.reset(seed=20260907)
        if args.center_base_mm is not None:
            center = torch.tensor(args.center_base_mm, dtype=torch.float32, device=env.device) / 1000.0
            previous_center = env._tomato.data.root_pos_w.torch.clone()
            fruit_pose = env._tomato.data.default_root_pose.torch.clone()
            fruit_pose[:, :3] = env.scene.env_origins + center
            fruit_pose[:, 3:7] = fruit_pose.new_tensor((0.0, 0.0, 0.0, 1.0))
            env._tomato.write_root_pose_to_sim_index(
                root_pose=fruit_pose,
                env_ids=torch.arange(env.num_envs, device=env.device),
            )
            env._tomato.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((env.num_envs, 6), device=env.device),
                env_ids=torch.arange(env.num_envs, device=env.device),
            )
            center_delta = fruit_pose[:, :3] - previous_center
            env._stem_rest_attachment_pos_w += center_delta
            env._stem_plant_anchor_pos_w += center_delta
            env._pinch_reference_center_w.copy_(fruit_pose[:, :3])
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(dt=0.0)
        action_values = (
            tuple(args.action_values)
            if args.action_values is not None
            else (P1 if args.pattern == "p1" else WORKSPACE)
        )
        action = torch.tensor(action_values, dtype=torch.float32, device=env.device).expand(args.num_envs, -1)
        thumb_names = ("thumb_joint0", "thumb_joint1", "thumb_joint2", "thumb_joint3", "thumb_joint4")
        thumb_ids = [env._robot.joint_names.index(name) for name in thumb_names]
        finger_ids = [env._robot.joint_names.index(name) for name in ("index_joint1", "middle_joint1", "ring_joint1", "little_joint1")]
        best = torch.zeros((args.num_envs, 5), device=env.device)
        for step in range(args.steps):
            _, _, terminated, truncated, _ = env.step(action)
            force = env._last_physical_force_n
            forbidden_force = env._forbidden_contact_force_n
            best = torch.maximum(best, force)
            if step % max(1, args.print_every) == 0 or bool(terminated.any() | truncated.any()):
                q = env._robot.data.joint_pos.torch
                tips = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids]
                center = env._tomato.data.root_pos_w.torch
                contact_points = []
                for sensor in env._contact_sensors.values():
                    raw = sensor.data.contact_pos_w
                    point = None
                    if raw is not None:
                        values = raw.torch[0, 0]
                        finite = torch.isfinite(values).all(dim=-1)
                        if bool(finite.any()):
                            point = tuple(
                                round(float(value), 5)
                                for value in values[finite][0].detach().cpu()
                            )
                    contact_points.append(point)
                print(
                    f"step={step:03d} q_thumb={[round(float(v),4) for v in q[0, thumb_ids]]} "
                    f"q_finger={[round(float(v),4) for v in q[0, finger_ids]]} "
                    f"q_all={[round(float(v),4) for v in q[0]]} "
                    f"tip_xyz={[tuple(round(float(v),4) for v in p) for p in tips[0]]} "
                    f"fruit={tuple(round(float(v),4) for v in center[0])} "
                    f"force={[round(float(v),4) for v in force[0]]} "
                    f"forbidden_force={[round(float(v),4) for v in forbidden_force[0]]} "
                    f"forbidden_proximal={[round(float(v),4) for v in env._forbidden_proximal_force_n[0]]} "
                    f"forbidden_first={bool(env._forbidden_first_contact[0])} "
                    f"tip_axial={[round(float(v),4) for v in env._contact_region_axial_progress[0]]} "
                    f"thumb_index_margin_mm={round(float(env._thumb_index_surface_margin_m[0])*1000.0,3)} "
                    f"contact_points={contact_points} "
                    f"term={bool(terminated[0])} trunc={bool(truncated[0])}",
                    flush=True,
                )
            if bool(terminated.any() | truncated.any()):
                if not args.continue_after_done:
                    break
        print("best_force_n", [[round(float(v), 5) for v in row] for row in best.detach().cpu()], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
