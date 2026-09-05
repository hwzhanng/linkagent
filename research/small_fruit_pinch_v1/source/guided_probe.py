#!/usr/bin/env python3
"""验证 compliant prior 在零 PPO residual 下的物理基线。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=320)
parser.add_argument("--episodes", type=int, default=0)
parser.add_argument("--seed", type=int, default=20260923)
parser.add_argument(
    "--guide-feedback",
    action="store_true",
    help="启用低层先验的压力反馈偏置；仅用于结构化基线探针，不改变训练默认值。",
)
parser.add_argument(
    "--no-adaptive-closure",
    action="store_true",
    help="关闭按指接触的自适应闭合，仅用于与固定 P1 先验做物理回归对照。",
)
parser.add_argument(
    "--adaptive-closure",
    action="store_true",
    help="启用按上一帧真实接触力渐进闭合的低层探针。",
)
parser.add_argument(
    "--geometry-prior",
    action="store_true",
    help="启用扫描得到的拇指/食指/中指几何弱先验。",
)
parser.add_argument(
    "--geometry-prior-scale",
    type=float,
    default=None,
    help="覆盖几何弱先验幅度（仅用于容量探针）。",
)
parser.add_argument(
    "--geometry-prior-values",
    default=None,
    help="覆盖 14D 几何先验向量；用于按指消融，格式为 14 个逗号分隔值。",
)
parser.add_argument(
    "--geometry-prior-after-contact",
    action="store_true",
    help="每根手指建立真实接触后才施加该指的几何弱先验。",
)
parser.add_argument(
    "--force-balance",
    action="store_true",
    help="启用带死区和限幅的三指压力平衡闭环。",
)
parser.add_argument(
    "--force-servo",
    action="store_true",
    help="启用每指受力误差双通道探针。",
)
parser.add_argument("--force-servo-seek-gain", type=float, default=None)
parser.add_argument("--force-servo-band-seek-gain", type=float, default=None)
parser.add_argument("--force-servo-target-gain", type=float, default=None)
parser.add_argument("--force-servo-retract-gain", type=float, default=None)
parser.add_argument("--force-servo-alpha", type=float, default=None)
parser.add_argument("--force-servo-max-offset", type=float, default=None)
parser.add_argument("--force-servo-warning-fraction", type=float, default=None)
parser.add_argument("--force-servo-seek-direction", default=None, help="3 个逗号分隔的方向符号")
parser.add_argument("--force-servo-retract-direction", default=None, help="3 个逗号分隔的方向符号")
parser.add_argument(
    "--force-gate-start-fraction",
    type=float,
    default=None,
    help="从软压力带的该比例开始收缩 residual 权限；仅用于探针。",
)
parser.add_argument(
    "--approach-speed",
    type=float,
    default=None,
    help="temporal filter 的接近速度上限。",
)
parser.add_argument("--warning-fraction", type=float, default=None)
parser.add_argument("--proportional-gain", type=float, default=None)
parser.add_argument("--max-guide-offset", type=float, default=None)
parser.add_argument("--guide-action-alpha", type=float, default=None)
parser.add_argument("--projection-guard-fraction", type=float, default=None)
parser.add_argument("--phase-controller", action="store_true")
parser.add_argument("--phase-search-step", type=float, default=None)
parser.add_argument("--phase-close-step", type=float, default=None)
parser.add_argument("--phase-retract-step", type=float, default=None)
parser.add_argument("--phase-offset-alpha", type=float, default=None)
parser.add_argument("--phase-max-offset", type=float, default=None)
parser.add_argument("--phase-root-fraction", type=float, default=None)
parser.add_argument("--phase-low-delay-steps", type=int, default=None)
parser.add_argument("--phase-low-ramp-steps", type=int, default=None)
parser.add_argument("--phase-hysteresis-n", type=float, default=None)
parser.add_argument("--hand-stiffness", type=float, default=None)
parser.add_argument("--direct-force-projection", action="store_true")
parser.add_argument("--direct-force-warning-fraction", type=float, default=None)
parser.add_argument("--direct-force-high-gain", type=float, default=None)
parser.add_argument("--direct-force-low-gain", type=float, default=None)
parser.add_argument("--direct-force-alpha", type=float, default=None)
parser.add_argument("--direct-force-max-open", type=float, default=None)
parser.add_argument("--direct-force-root-fraction", type=float, default=None)
parser.add_argument("--geometry-approach", action="store_true")
parser.add_argument("--geometry-approach-target-gap-mm", type=float, default=None)
parser.add_argument("--geometry-approach-scale-mm", type=float, default=None)
parser.add_argument("--geometry-approach-gain", type=float, default=None)
parser.add_argument("--geometry-approach-max-offset", type=float, default=None)
parser.add_argument("--geometry-approach-alpha", type=float, default=None)
parser.add_argument("--external-forces-every-iteration", action="store_true")
parser.add_argument("--enhanced-determinism", action="store_true")
parser.add_argument("--robot-max-depenetration-velocity", type=float, default=None)
parser.add_argument("--fruit-max-depenetration-velocity", type=float, default=None)
parser.add_argument("--fruit-contact-offset", type=float, default=None)
parser.add_argument("--fruit-rest-offset", type=float, default=None)
parser.add_argument(
    "--residual-scale",
    type=float,
    default=0.0,
    help="PPO residual 的结构化权限；仅用于固定动作容量探针，不改变训练默认值。",
)
parser.add_argument(
    "--precontact-residual-fraction",
    type=float,
    default=0.0,
    help="接触前 residual 权限比例；仅用于固定动作容量探针。",
)
parser.add_argument(
    "--residual-values",
    default=None,
    help="可选的 14 个逗号分隔残差动作；省略时使用全零残差。",
)
parser.add_argument("--trace-guide", action="store_true", help="打印前若干步先验动作/复位计数，定位闭环状态。")
parser.add_argument("--trace-geometry", action="store_true", help="在轨迹探针中打印三根指尖到椭球表面的间隙/径向速度。")
parser.add_argument("--trace-failure", action="store_true", help="首次过载/掉落时打印该环境的力与几何状态。")
parser.add_argument("--trace-axis", action="store_true", help="打印本进程固定随机化果实半轴的分布摘要。")
parser.add_argument("--trace-env", type=int, default=None, help="逐步打印指定环境的力/动作，配合接触瞬态审计。")
parser.add_argument("--fruit-axis-min-mm", type=float, default=None)
parser.add_argument("--fruit-axis-max-mm", type=float, default=None)
parser.add_argument("--offset-x-mm", type=float, default=None)
parser.add_argument("--offset-y-mm", type=float, default=None)
parser.add_argument("--offset-z-mm", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab.utils.seed import configure_seed

from research.small_fruit_pinch_v1.source.guided_residual_env import (
    GuidedResidualSmallFruitPinchEnv,
    NativeSmallFruitPinchCfg,
)


def main() -> None:
    configure_seed(args.seed, True)
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    if args.external_forces_every_iteration:
        cfg.sim.physics.enable_external_forces_every_iteration = True
    if args.enhanced_determinism:
        cfg.sim.physics.enable_enhanced_determinism = True
    if args.robot_max_depenetration_velocity is not None:
        if args.robot_max_depenetration_velocity <= 0.0:
            raise ValueError("robot-max-depenetration-velocity 必须为正数")
        cfg.robot.spawn.rigid_props.max_depenetration_velocity = args.robot_max_depenetration_velocity
    if args.fruit_max_depenetration_velocity is not None:
        if args.fruit_max_depenetration_velocity <= 0.0:
            raise ValueError("fruit-max-depenetration-velocity 必须为正数")
        cfg.tomato.spawn.rigid_props.max_depenetration_velocity = args.fruit_max_depenetration_velocity
    if args.fruit_contact_offset is not None:
        if args.fruit_contact_offset <= 0.0:
            raise ValueError("fruit-contact-offset 必须为正数")
        cfg.tomato.spawn.collision_props.contact_offset = args.fruit_contact_offset
    if args.fruit_rest_offset is not None:
        if args.fruit_rest_offset < 0.0:
            raise ValueError("fruit-rest-offset 不能为负数")
        cfg.tomato.spawn.collision_props.rest_offset = args.fruit_rest_offset
    if (args.fruit_axis_min_mm is None) != (args.fruit_axis_max_mm is None):
        raise ValueError("fruit-axis-min-mm 与 fruit-axis-max-mm 必须同时提供")
    if args.fruit_axis_min_mm is not None:
        if not (0.0 < args.fruit_axis_min_mm <= args.fruit_axis_max_mm):
            raise ValueError("果径半轴范围必须为正且 min <= max")
        axis_min_m = args.fruit_axis_min_mm / 1000.0
        axis_max_m = args.fruit_axis_max_mm / 1000.0
        cfg.events.fruit_geometry.params["axis_ranges_m"] = {
            "x": (axis_min_m, axis_max_m),
            "y": (axis_min_m, axis_max_m),
            "z": (axis_min_m, axis_max_m),
        }
    offset_args = (args.offset_x_mm, args.offset_y_mm, args.offset_z_mm)
    if any(value is not None for value in offset_args):
        if not all(value is not None for value in offset_args):
            raise ValueError("offset-x/y/z-mm 必须同时提供")
        cfg.pinch_working_offset_m = tuple(value / 1000.0 for value in offset_args)
    if args.hand_stiffness is not None:
        if args.hand_stiffness <= 0.0:
            raise ValueError("hand-stiffness 必须为正数")
        for actuator_name in ("g20_independent", "g20_followers"):
            cfg.robot.actuators[actuator_name].stiffness = args.hand_stiffness
    if args.guide_feedback:
        # 探针显式标注，训练入口仍保持默认关闭，避免把一次诊断性的力反馈
        # 修改误写入正式合同。
        GuidedResidualSmallFruitPinchEnv.guide_feedback_enabled = True
    if args.no_adaptive_closure:
        GuidedResidualSmallFruitPinchEnv.adaptive_closure_enabled = False
    elif args.adaptive_closure:
        GuidedResidualSmallFruitPinchEnv.adaptive_closure_enabled = True
    GuidedResidualSmallFruitPinchEnv.geometry_prior_enabled = args.geometry_prior
    GuidedResidualSmallFruitPinchEnv.geometry_prior_after_contact = args.geometry_prior_after_contact
    GuidedResidualSmallFruitPinchEnv.geometry_approach_enabled = args.geometry_approach
    if args.geometry_approach_target_gap_mm is not None:
        if args.geometry_approach_target_gap_mm < 0.0:
            raise ValueError("geometry-approach-target-gap-mm 不能为负数")
        GuidedResidualSmallFruitPinchEnv.geometry_approach_target_gap_m = args.geometry_approach_target_gap_mm / 1000.0
    if args.geometry_approach_scale_mm is not None:
        if args.geometry_approach_scale_mm <= 0.0:
            raise ValueError("geometry-approach-scale-mm 必须为正数")
        GuidedResidualSmallFruitPinchEnv.geometry_approach_scale_m = args.geometry_approach_scale_mm / 1000.0
    for name, value in (
        ("geometry_approach_gain", args.geometry_approach_gain),
        ("geometry_approach_max_offset", args.geometry_approach_max_offset),
        ("geometry_approach_alpha", args.geometry_approach_alpha),
    ):
        if value is not None:
            if value < 0.0:
                raise ValueError(f"{name} 不能为负数")
            setattr(GuidedResidualSmallFruitPinchEnv, name, value)
    if args.geometry_prior_scale is not None:
        if not 0.0 <= args.geometry_prior_scale <= 1.0:
            raise ValueError("geometry-prior-scale 必须在 [0, 1] 内")
        GuidedResidualSmallFruitPinchEnv.geometry_prior_scale = args.geometry_prior_scale
    if args.geometry_prior_values is not None:
        try:
            values = tuple(float(value.strip()) for value in args.geometry_prior_values.split(","))
        except ValueError as exc:
            raise ValueError("geometry-prior-values 必须是 14 个浮点数") from exc
        if len(values) != 14 or any(abs(value) > 1.0 for value in values):
            raise ValueError("geometry-prior-values 必须包含 14 个 [-1, 1] 值")
        GuidedResidualSmallFruitPinchEnv.geometry_prior_residual = values
    if args.approach_speed is not None:
        if args.approach_speed <= 0.0:
            raise ValueError("approach-speed 必须为正数")
        GuidedResidualSmallFruitPinchEnv.approach_speed = args.approach_speed
    for name, value in (
        ("warning_fraction", args.warning_fraction),
        ("proportional_gain", args.proportional_gain),
        ("max_guide_offset", args.max_guide_offset),
        ("guide_action_alpha", args.guide_action_alpha),
        ("projection_guard_fraction", args.projection_guard_fraction),
    ):
        if value is not None:
            if value < 0.0:
                raise ValueError(f"{name} 不能为负数")
            setattr(GuidedResidualSmallFruitPinchEnv, name, value)
    GuidedResidualSmallFruitPinchEnv.force_balance_enabled = args.force_balance
    GuidedResidualSmallFruitPinchEnv.force_servo_enabled = args.force_servo
    GuidedResidualSmallFruitPinchEnv.direct_force_projection_enabled = args.direct_force_projection
    for name, value in (
        ("direct_force_projection_warning_fraction", args.direct_force_warning_fraction),
        ("direct_force_projection_high_gain", args.direct_force_high_gain),
        ("direct_force_projection_low_gain", args.direct_force_low_gain),
        ("direct_force_projection_alpha", args.direct_force_alpha),
        ("direct_force_projection_max_open", args.direct_force_max_open),
        ("direct_force_projection_root_fraction", args.direct_force_root_fraction),
    ):
        if value is not None:
            if value < 0.0:
                raise ValueError(f"{name} 不能为负数")
            setattr(GuidedResidualSmallFruitPinchEnv, name, value)
    GuidedResidualSmallFruitPinchEnv.phase_controller_enabled = args.phase_controller
    for name, value in (
        ("force_servo_seek_gain", args.force_servo_seek_gain),
        ("force_servo_band_seek_gain", args.force_servo_band_seek_gain),
        ("force_servo_target_gain", args.force_servo_target_gain),
        ("force_servo_retract_gain", args.force_servo_retract_gain),
        ("force_servo_alpha", args.force_servo_alpha),
        ("force_servo_max_offset", args.force_servo_max_offset),
        ("force_servo_warning_fraction", args.force_servo_warning_fraction),
    ):
        if value is not None:
            if value < 0.0:
                raise ValueError(f"{name} 不能为负数")
            setattr(GuidedResidualSmallFruitPinchEnv, name, value)
    for name, value in (
        ("phase_search_step", args.phase_search_step),
        ("phase_close_step", args.phase_close_step),
        ("phase_retract_step", args.phase_retract_step),
        ("phase_offset_alpha", args.phase_offset_alpha),
        ("phase_max_offset", args.phase_max_offset),
        ("phase_root_fraction", args.phase_root_fraction),
        ("phase_low_delay_steps", args.phase_low_delay_steps),
        ("phase_low_ramp_steps", args.phase_low_ramp_steps),
        ("phase_hysteresis_n", args.phase_hysteresis_n),
    ):
        if value is not None:
            if value < 0.0:
                raise ValueError(f"{name} 不能为负数")
            setattr(GuidedResidualSmallFruitPinchEnv, name, value)
    for attr, raw in (
        ("force_servo_seek_direction", args.force_servo_seek_direction),
        ("force_servo_retract_direction", args.force_servo_retract_direction),
    ):
        if raw is not None:
            try:
                direction = tuple(float(value.strip()) for value in raw.split(","))
            except ValueError as exc:
                raise ValueError(f"{attr} 必须是 3 个浮点数") from exc
            if len(direction) != 3 or any(abs(value) > 1.0 for value in direction):
                raise ValueError(f"{attr} 必须包含 3 个 [-1, 1] 值")
            setattr(GuidedResidualSmallFruitPinchEnv, attr, direction)
    if args.force_gate_start_fraction is not None:
        if not 0.0 < args.force_gate_start_fraction <= 1.0:
            raise ValueError("force-gate-start-fraction 必须在 (0, 1] 内")
        GuidedResidualSmallFruitPinchEnv.force_gate_start_fraction = args.force_gate_start_fraction
    if not 0.0 <= args.residual_scale <= 1.0:
        raise ValueError("residual-scale 必须在 [0, 1] 内")
    if not 0.0 <= args.precontact_residual_fraction <= 1.0:
        raise ValueError("precontact-residual-fraction 必须在 [0, 1] 内")
    GuidedResidualSmallFruitPinchEnv.residual_scale = args.residual_scale
    GuidedResidualSmallFruitPinchEnv.precontact_residual_fraction = args.precontact_residual_fraction
    if args.residual_values is None:
        residual_values = [0.0] * 14
    else:
        try:
            residual_values = [float(value.strip()) for value in args.residual_values.split(",")]
        except ValueError as exc:
            raise ValueError("residual-values 必须是 14 个浮点数") from exc
        if len(residual_values) != 14:
            raise ValueError("residual-values 必须恰好包含 14 个浮点数")
        if any(abs(value) > 1.0 for value in residual_values):
            raise ValueError("residual-values 每个值必须在 [-1, 1] 内")
    env = GuidedResidualSmallFruitPinchEnv(cfg=cfg)
    try:
        obs, _ = env.reset(seed=args.seed)
        action = obs["policy"].new_tensor(residual_values).expand(args.num_envs, -1).clone()
        if args.trace_axis:
            print(
                f"axis_trace min_mm={(env._fruit_axes_m.amin(dim=0) * 1000.0).detach().cpu().tolist()} "
                f"max_mm={(env._fruit_axes_m.amax(dim=0) * 1000.0).detach().cpu().tolist()} "
                f"mean_mm={(env._fruit_axes_m.mean(dim=0) * 1000.0).detach().cpu().tolist()} "
                f"mass_minmax={(env._fruit_mass_kg.amin().item(), env._fruit_mass_kg.amax().item())} "
                f"mass_mean={env._fruit_mass_kg.mean().item():.6f} "
                f"mass_std={env._fruit_mass_kg.std().item():.6f}",
                flush=True,
            )
        peak = torch.zeros((args.num_envs, 5), device=env.device)
        hard_hits = torch.zeros(5, device=env.device)
        success = drop = overload = collision = 0
        candidate_steps = torch.zeros(args.num_envs, device=env.device)
        failure_reported = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        if args.trace_env is not None and not 0 <= args.trace_env < args.num_envs:
            raise ValueError("trace-env 必须位于 [0, num-envs) 内")
        for _ in range(args.steps):
            step_index = _
            obs, _, terminated, truncated, info = env.step(action)
            if args.trace_guide and (step_index < 5 or step_index % 20 == 0):
                message = (
                    f"guide_trace step={step_index} guide={env._guide_action[0].detach().cpu().tolist()} "
                    f"force={env._last_physical_force_n[0].detach().cpu().tolist()} "
                    f"term={int(terminated.sum())} trunc={int(truncated.sum())}"
                )
                if args.trace_geometry:
                    tips = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids[:3]]
                    center = env._tomato.data.root_pos_w.torch
                    relative = tips - center.unsqueeze(1)
                    radial = torch.linalg.vector_norm(relative, dim=2).clamp_min(1.0e-6)
                    unit = relative / radial.unsqueeze(-1)
                    axes = env._fruit_axes_m[:, None, :].clamp_min(1.0e-6)
                    radius = 1.0 / torch.sqrt(
                        torch.sum(torch.square(unit / axes), dim=2).clamp_min(1.0e-8)
                    )
                    gap_mm = (radial - radius)[0] * 1000.0
                    tip_vel = env._robot.data.body_lin_vel_w.torch[:, env._fingertip_contact_body_ids[:3]]
                    fruit_vel = env._tomato.data.root_lin_vel_w.torch
                    radial_vel = ((tip_vel - fruit_vel.unsqueeze(1)) * unit).sum(dim=2)[0]
                    message += (
                        f" gap_mm={gap_mm.detach().cpu().tolist()}"
                        f" radial_vel={radial_vel.detach().cpu().tolist()}"
                    )
                print(message, flush=True)
            peak = torch.maximum(peak, env._last_physical_force_n)
            if args.trace_env is not None:
                trace_index = args.trace_env
                trace_done = bool((terminated | truncated)[trace_index].item())
                trace_force = (
                    env._last_terminal_force_n[trace_index]
                    if trace_done
                    else env._last_physical_force_n[trace_index]
                )
                trace_reason = (
                    int(env._last_terminal_reason[trace_index].item())
                    if trace_done
                    else 0
                )
                print(
                    f"env_trace step={step_index} force={trace_force.detach().cpu().tolist()} "
                    f"guide={env._guide_action[trace_index].detach().cpu().tolist()} "
                    f"done={int(trace_done)} reason={trace_reason}",
                    flush=True,
                )
            if args.trace_failure:
                done = terminated | truncated
                failure = done & (env._last_terminal_reason >= 2)
                for env_index in torch.where(failure & ~failure_reported)[0].tolist():
                    print(
                        f"failure_trace env={env_index} step={step_index} "
                        f"reason={int(env._last_terminal_reason[env_index].item())} "
                        f"force={env._last_terminal_force_n[env_index].detach().cpu().tolist()} "
                        f"axes_mm={(env._last_terminal_axes_m[env_index] * 1000.0).detach().cpu().tolist()} "
                        f"hold={int(env._last_terminal_hold_age[env_index].item())} "
                        f"contact_ratio={float(env._last_terminal_contact_ratio[env_index].item()):.3f} "
                        f"band_ratio={float(env._last_terminal_band_ratio[env_index].item()):.3f}",
                        flush=True,
                    )
                    failure_reported[env_index] = True
            hard_hits += env._last_physical_force_n.new_tensor(
                info["step"].get("overload_by_finger", [0, 0, 0, 0, 0])
            )
            candidate_steps += info["step"].get("success_count", 0)
            success += int(info["step"].get("success_count", 0))
            drop += int(info["step"].get("drop_count", 0))
            overload += int(info["step"].get("overload_count", 0))
            collision += int(info["step"].get("self_collision_count", 0))
        print(
            "guided_probe "
            f"peak_force_n={peak.mean(dim=0).tolist()} "
            f"hard_hits_by_finger={hard_hits.detach().cpu().tolist()} "
            f"success_events={success} drop_events={drop} overload_events={overload} collision_events={collision} "
            f"mean_hold_steps={env._verifier.hold_age.float().mean().item():.2f} "
            f"mean_guide_offset={env._guide_offset.abs().mean().item():.4f} "
            f"guide_action_mean={env._guide_action.mean(dim=0).detach().cpu().tolist()}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
