#!/usr/bin/env python3
"""输出单左手 PPO 权重的一条可复核物理轨迹。

该探针只用于首版物理动作验收：果径固定在 64--70 mm（内部半轴
32--35 mm），场景来自 ``NativeSmallFruitPinchEnv``，不包含双臂或右手。
它打印策略的实际 14D 输出、关节状态和五个指尖的 PhysX 接触力，避免
把固定引导姿态误认为 PPO 已经学会了动作。
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=220)
parser.add_argument("--print-every", type=int, default=10)
parser.add_argument("--seed", type=int, default=20260908)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument(
    "--zero-policy",
    action="store_true",
    help="消融：将 Actor 输出置零，仅保留环境中的弱先验/安全层；不用于训练或发布。",
)
parser.add_argument("--guide-action-values", default=None)
parser.add_argument("--residual-scale", type=float, default=0.15)
parser.add_argument("--distal-residual-scale", type=float, default=0.20)
parser.add_argument("--precontact-residual-fraction", type=float, default=0.10)
parser.add_argument("--explicit-hand-pd", action="store_true")
parser.add_argument(
    "--center-local-mm",
    nargs=3,
    type=float,
    default=None,
    metavar=("X", "Y", "Z"),
    help="覆盖左手抓取基坐标中的固定果心（mm）。",
)
parser.add_argument(
    "--palm-fruit-collision-filter",
    action="store_true",
    help="诊断模式：过滤仅 base_link↔Tomato 的碰撞。",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.seed import configure_seed
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from research.small_fruit_pinch_v1.source.agents import SmallFruitPinchPPORunnerCfg
from research.small_fruit_pinch_v1.source.guided_residual_env import (
    GuidedResidualSmallFruitPinchEnv,
)
from research.small_fruit_pinch_v1.source.native_env import NativeSmallFruitPinchCfg


GUIDE_DEFAULT = (
    0.8239,
    -0.8590,
    0.0811,
    0.6406,
    0.4097,
    0.2718,
    -0.7128,
    0.5314,
    -0.9198,
    0.1658,
    -0.7514,
    -0.8352,
    -0.5911,
    0.7236,
)


def _parse_guide(raw: str | None) -> tuple[float, ...]:
    if raw is None:
        return GUIDE_DEFAULT
    try:
        values = tuple(float(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError("guide-action-values 必须是 14 个浮点数") from exc
    if len(values) != 14 or any(abs(value) > 1.0 for value in values):
        raise ValueError("guide-action-values 必须包含 14 个 [-1, 1] 值")
    return values


def main() -> None:
    if args.steps <= 0 or args.print_every <= 0 or args.num_envs <= 0:
        raise ValueError("steps、print-every 和 num-envs 必须为正数")
    configure_seed(args.seed, True)
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    if args.center_local_mm is not None:
        cfg.closed_grasp_fruit_center_m = tuple(
            value / 1000.0 for value in args.center_local_mm
        )
    cfg.palm_fruit_collision_filter_enabled = bool(
        args.palm_fruit_collision_filter
    )
    # 配置语义是果径；环境事件接收椭球半轴，因此 64--70 mm 直径
    # 对应这里的 32--35 mm 半轴。不要传入 64--70 以免扩大一倍。
    cfg.events.fruit_geometry.params["axis_ranges_m"] = {
        "x": (0.032, 0.035),
        "y": (0.032, 0.035),
        "z": (0.032, 0.035),
    }
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
    GuidedResidualSmallFruitPinchEnv.guide_action_values = _parse_guide(
        args.guide_action_values
    )
    GuidedResidualSmallFruitPinchEnv.residual_scale = float(args.residual_scale)
    GuidedResidualSmallFruitPinchEnv.distal_residual_scale = float(
        args.distal_residual_scale
    )
    GuidedResidualSmallFruitPinchEnv.precontact_residual_fraction = float(
        args.precontact_residual_fraction
    )
    env = GuidedResidualSmallFruitPinchEnv(cfg=cfg)
    agent_cfg = SmallFruitPinchPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, metadata.version("rsl-rl-lib")
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(args.checkpoint, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        obs = wrapped.get_observations()
        required_contact_steps = torch.zeros(args.num_envs, device=env.device)
        first_contact_step = torch.full(
            (args.num_envs, 5), -1, dtype=torch.long, device=env.device
        )
        contact_order_history: list[list[int]] = [[] for _ in range(args.num_envs)]
        peak_force = torch.zeros((args.num_envs, 5), device=env.device)
        latest_contact_points = torch.full(
            (args.num_envs, 5, 3), float("nan"), device=env.device
        )
        best_contact_points = latest_contact_points.clone()
        best_contact_force = torch.zeros((args.num_envs, 5), device=env.device)
        best_geometry_points = latest_contact_points.clone()
        best_geometry_fruit = torch.full(
            (args.num_envs, 3), float("nan"), device=env.device
        )
        best_geometry_angle = torch.full(
            (args.num_envs,), float("inf"), device=env.device
        )
        best_geometry_chord = torch.full(
            (args.num_envs,), float("inf"), device=env.device
        )
        best_geometry_step = torch.full(
            (args.num_envs,), -1, dtype=torch.long, device=env.device
        )
        policy_action_history: list[torch.Tensor] = []
        print(
            "policy_trace_start "
            f"checkpoint={args.checkpoint} num_envs={args.num_envs} "
            "scene=isolated_left_hand fruit_diameter_mm=64..70 "
            f"zero_policy={args.zero_policy}",
            flush=True,
        )
        for step in range(args.steps):
            with torch.inference_mode():
                action = policy(obs)
                if args.zero_policy:
                    action = torch.zeros_like(action)
            policy_action_history.append(action.detach().cpu())
            obs, _, dones, extras = wrapped.step(action)
            force = env._last_physical_force_n.detach()
            peak_force = torch.maximum(peak_force, force)
            required = force[:, :3] >= 0.08
            required_contact_steps += required.all(dim=1).float()
            contact_now = force >= 0.08
            newly_contacted = contact_now & (first_contact_step < 0)
            if bool(newly_contacted.any()):
                step_tensor = torch.full_like(first_contact_step, step)
                first_contact_step = torch.where(
                    newly_contacted, step_tensor, first_contact_step
                )
                for env_index in range(args.num_envs):
                    for digit_index in torch.where(newly_contacted[env_index])[0].tolist():
                        contact_order_history[env_index].append(int(digit_index))
            for digit_index, sensor in enumerate(env._contact_sensors.values()):
                raw = sensor.data.contact_pos_w
                if raw is None:
                    continue
                values = raw.torch[:, 0]
                finite = torch.isfinite(values).all(dim=-1)
                for env_index in range(args.num_envs):
                    if bool(finite[env_index].any()):
                        latest_contact_points[env_index, digit_index] = values[
                            env_index, finite[env_index]
                        ].mean(dim=0)
            contact_quality = required.float().sum(dim=1)
            replace = contact_quality > best_contact_force[:, :3].gt(0.08).sum(dim=1)
            if bool(replace.any()):
                best_contact_points[replace] = latest_contact_points[replace]
                best_contact_force[replace] = force[replace]
            fruit_now = env._tomato.data.root_pos_w.torch
            valid_geometry = required[:, :3].all(dim=1) & torch.isfinite(
                latest_contact_points[:, :3]
            ).all(dim=(1, 2))
            if bool(valid_geometry.any()):
                relative_now = latest_contact_points[:, :3] - fruit_now.unsqueeze(1)
                units_now = torch.nn.functional.normalize(relative_now, dim=2)
                finger_now = torch.nn.functional.normalize(
                    relative_now[:, 1:3].mean(dim=1), dim=1
                )
                angle_now = torch.rad2deg(
                    torch.acos(
                        torch.clamp(
                            (units_now[:, 0] * -finger_now).sum(dim=1), -1.0, 1.0
                        )
                    )
                )
                line_now = relative_now[:, 1:3].mean(dim=1) - relative_now[:, 0]
                line_sq_now = (line_now * line_now).sum(dim=1).clamp_min(1.0e-12)
                line_t_now = torch.clamp(
                    (-relative_now[:, 0] * line_now).sum(dim=1) / line_sq_now,
                    0.0,
                    1.0,
                )
                chord_now = torch.linalg.vector_norm(
                    relative_now[:, 0] + line_t_now.unsqueeze(1) * line_now,
                    dim=1,
                )
                better = valid_geometry & (
                    angle_now < best_geometry_angle
                )
                if bool(better.any()):
                    best_geometry_points[better] = latest_contact_points[better]
                    best_geometry_fruit[better] = fruit_now[better]
                    best_geometry_angle[better] = angle_now[better]
                    best_geometry_chord[better] = chord_now[better]
                    best_geometry_step[better] = step
            if step % args.print_every == 0 or bool(dones.any()):
                q = env._robot.data.joint_pos.torch[0, env._hand_joint_ids]
                target = env._joint_targets[0, env._hand_joint_ids]
                print(
                    f"step={step:03d} "
                    f"action={[round(float(v),4) for v in action[0].detach().cpu()]} "
                    f"q={[round(float(v),4) for v in q.detach().cpu()]} "
                    f"target={[round(float(v),4) for v in target.detach().cpu()]} "
                    f"force={[round(float(v),4) for v in force[0].cpu()]} "
                    f"all_required_contact={bool(required[0].all())} "
                    f"done={bool(dones[0])}",
                    flush=True,
                )
            if bool(dones.any()):
                env.reset()
        # 用实际 ContactSensor 接触点做与力值相互独立的方向审计：夹持侧
        # 的拇指向量应与食指/中指平均向量相反；食指--中指连线到球心
        # 的偏移越小，越接近“拇指对准两指中间”的目标。没有有效接触点
        # 时才回退到接触体位置，并在输出中明确标记，避免把 frame tip
        # 的姿态误报为真实接触几何。
        measured_points = best_geometry_points[0].clone()
        tips = env._robot.data.body_pos_w.torch[0, env._fingertip_frame_body_ids]
        fruit = best_geometry_fruit[0].clone()
        if not bool(torch.isfinite(fruit).all()):
            fruit = env._tomato.data.root_pos_w.torch[0]
        body_points = env._robot.data.body_pos_w.torch[0, env._fingertip_contact_body_ids]
        measured_valid = torch.isfinite(measured_points).all(dim=1)
        geometry_source = "contact_points"
        if not bool(measured_valid[:3].all()):
            measured_points = body_points
            geometry_source = "contact_body_fallback"
        relative = measured_points - fruit
        unit = torch.nn.functional.normalize(relative, dim=1)
        finger_mean = torch.nn.functional.normalize(relative[1:3].mean(dim=0), dim=0)
        opposition_angle = torch.rad2deg(
            torch.acos(torch.clamp(torch.dot(unit[0], -finger_mean), -1.0, 1.0))
        )
        finger_mid = relative[1:3].mean(dim=0)
        line = finger_mid - relative[0]
        line_sq = torch.dot(line, line).clamp_min(1.0e-12)
        line_t = torch.clamp(torch.dot(-relative[0], line) / line_sq, 0.0, 1.0)
        chord_offset_m = torch.linalg.vector_norm(relative[0] + line_t * line)
        print(
            "policy_geometry_summary "
            f"source={geometry_source} "
            f"best_step={int(best_geometry_step[0])} "
            f"tip_distance_mm={[round(float(v*1000.0),2) for v in torch.linalg.vector_norm(relative, dim=1).cpu()]} "
            f"opposition_angle_deg={float(opposition_angle):.2f} "
            f"thumb_to_index_middle_chord_offset_mm={float(chord_offset_m*1000.0):.2f} "
            f"contact_xyz={[tuple(round(float(v),5) for v in row) for row in measured_points.cpu()]} "
            f"frame_tip_xyz={[tuple(round(float(v),5) for v in row) for row in tips.cpu()]} "
            f"fruit_xyz={[round(float(v),5) for v in fruit.cpu()]}",
            flush=True,
        )
        print(
            "policy_trace_summary "
            f"peak_force_n={[[round(float(v),5) for v in row] for row in peak_force.cpu()]} "
            f"required_contact_steps={[int(v) for v in required_contact_steps.cpu()]} "
            f"first_contact_step={first_contact_step.cpu().tolist()} "
            f"contact_order={contact_order_history} "
            f"terminated={int(dones.sum().item())}",
            flush=True,
        )
        geometry_valid = torch.isfinite(best_geometry_angle)
        geometry_angles = [
            round(float(value), 2) if bool(valid) else None
            for value, valid in zip(
                best_geometry_angle.detach().cpu(), geometry_valid.detach().cpu()
            )
        ]
        geometry_chords_mm = [
            round(float(value) * 1000.0, 2) if bool(valid) else None
            for value, valid in zip(
                best_geometry_chord.detach().cpu(), geometry_valid.detach().cpu()
            )
        ]
        print(
            "policy_population_summary "
            f"all_required_env_count={int((required_contact_steps > 0).sum().item())}/{args.num_envs} "
            f"required_contact_steps_total={int(required_contact_steps.sum().item())} "
            f"geometry_valid_env_count={int(geometry_valid.sum().item())}/{args.num_envs} "
            f"best_opposition_angle_deg={geometry_angles} "
            f"best_chord_offset_mm={geometry_chords_mm}",
            flush=True,
        )
        if policy_action_history:
            action_history = torch.stack(policy_action_history)
            action_delta = action_history[1:] - action_history[:-1]
            print(
                "policy_action_stats "
                f"mean_abs={float(action_history.abs().mean()):.6f} "
                f"std={float(action_history.std(unbiased=False)):.6f} "
                f"delta_p95={float(torch.quantile(action_delta.abs(), 0.95)) if action_delta.numel() else 0.0:.6f} "
                f"max_abs={float(action_history.abs().max()):.6f}",
                flush=True,
            )
    finally:
        wrapped.close()


if __name__ == "__main__":
    main()
