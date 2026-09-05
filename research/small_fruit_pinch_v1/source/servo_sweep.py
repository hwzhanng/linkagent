#!/usr/bin/env python3
"""在同一个 Isaac 进程中比较结构化受力伺服的少量保守参数组合。

该脚本只运行零 PPO residual 的诊断，不生成训练权重，也不改变环境或
历史分支。它把每个组合放在独立 reset/seed 下，避免多个 Kit 进程争用
同一张 GPU。
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--seed", type=int, default=20260980)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--variant", default="all")
parser.add_argument("--approach-speed", type=float, default=None)
parser.add_argument("--projection-guard-fraction", type=float, default=None)
parser.add_argument("--direct-force-projection", action="store_true")
parser.add_argument("--direct-force-warning-fraction", type=float, default=None)
parser.add_argument("--direct-force-high-gain", type=float, default=None)
parser.add_argument("--direct-force-high-gain-by-digit", default=None)
parser.add_argument("--direct-force-low-gain", type=float, default=None)
parser.add_argument("--direct-force-alpha", type=float, default=None)
parser.add_argument("--direct-force-max-open", type=float, default=None)
parser.add_argument("--direct-force-root-fraction", type=float, default=None)
parser.add_argument("--hand-damping", type=float, default=None)
parser.add_argument("--hand-stiffness", type=float, default=None)
parser.add_argument("--geometry-approach", action="store_true")
parser.add_argument("--geometry-approach-target-gap-mm", type=float, default=None)
parser.add_argument("--geometry-approach-scale-mm", type=float, default=None)
parser.add_argument("--geometry-approach-gain", type=float, default=None)
parser.add_argument("--geometry-approach-max-offset", type=float, default=None)
parser.add_argument("--geometry-approach-alpha", type=float, default=None)
parser.add_argument("--hold-joint-target-speed", type=float, default=None)
parser.add_argument("--external-forces-every-iteration", action="store_true")
parser.add_argument("--enhanced-determinism", action="store_true")
parser.add_argument("--robot-max-depenetration-velocity", type=float, default=None)
parser.add_argument("--fruit-max-depenetration-velocity", type=float, default=None)
parser.add_argument("--fruit-contact-offset", type=float, default=None)
parser.add_argument("--fruit-rest-offset", type=float, default=None)
parser.add_argument("--geometry-prior-values", default=None)
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


VARIANTS = {
    "current": dict(geometry_scale=0.15, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0),
    # 只提高未接触指的 seek 增益；contact_gap 在建立接触后归零，因而
    # 不会在持稳阶段继续推压。用于区分“时限不足”与“压力过高”。
    "seek_progressive": dict(geometry_scale=0.15, warning=0.78, seek=0.12, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0),
    "geometry_020": dict(geometry_scale=0.20, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0),
    "geometry_025": dict(geometry_scale=0.25, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0),
    "geometry_030": dict(geometry_scale=0.30, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0),
    "late_retract_fast": dict(geometry_scale=0.15, warning=0.78, seek=0.04, target=0.0, retract=0.50, alpha=0.45, max_offset=0.40, rate=0.0),
    "early_moderate": dict(geometry_scale=0.15, warning=0.70, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0),
    "early_fast": dict(geometry_scale=0.15, warning=0.70, seek=0.03, target=0.0, retract=0.50, alpha=0.45, max_offset=0.32, rate=0.0),
    "very_early": dict(geometry_scale=0.15, warning=0.65, seek=0.03, target=0.0, retract=0.45, alpha=0.40, max_offset=0.30, rate=0.0),
    "late_safe": dict(geometry_scale=0.15, warning=0.82, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0),
    "smaller_geometry": dict(geometry_scale=0.10, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0),
    # 只增加一个小的正向力变化率提前回撤项，作为独立消融；其余
    # 几何/压力参数与 current 完全一致，便于判断是否真的抑制冲量。
    "rate_conservative": dict(geometry_scale=0.15, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.10),
    "target_conservative": dict(geometry_scale=0.15, warning=0.78, seek=0.04, target=0.04, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0),
    "target_very_conservative": dict(geometry_scale=0.15, warning=0.78, seek=0.04, target=0.01, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0),
    # 只在某指已接触、但仍低于该指 soft_low 时补极小的闭合量；与
    # contact_gap 和目标力 gap 分离，专门验证“接触充分但带内比例不足”的
    # 超时诊断，不改变高力回撤通道。
    "band_seek_conservative": dict(geometry_scale=0.15, warning=0.78, seek=0.04, band_seek=0.05, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0),
    # 逐指 SEARCH/CLOSE/HOLD/RETRACT 结构控制器；其偏置在 HOLD 中保持，
    # 近端/远端按 root_fraction 分摊，避免 level-only servo 的“补压后
    # 自动衰减回零”。参数均是保守默认，先做静态验证再训练。
    "phase_default": dict(geometry_scale=0.15, warning=0.78, seek=0.0, band_seek=0.0, target=0.0, retract=0.0, alpha=0.30, max_offset=0.18, rate=0.0, phase=True, phase_search=0.0030, phase_close=0.0015, phase_retract=0.0030, phase_alpha=0.30, phase_root=0.55, phase_delay=3, phase_ramp=5, phase_hysteresis=0.03),
    "phase_fast_retract": dict(geometry_scale=0.15, warning=0.78, seek=0.0, band_seek=0.0, target=0.0, retract=0.0, alpha=0.30, max_offset=0.32, rate=0.0, phase=True, phase_search=0.0020, phase_close=0.0010, phase_retract=0.0200, phase_alpha=0.50, phase_root=0.55, phase_delay=4, phase_ramp=8, phase_hysteresis=0.04),
    "phase_slow_contact": dict(geometry_scale=0.15, warning=0.78, seek=0.0, band_seek=0.0, target=0.0, retract=0.0, alpha=0.30, max_offset=0.28, rate=0.0, phase=True, phase_search=0.0015, phase_close=0.0010, phase_retract=0.0160, phase_alpha=0.50, phase_root=0.55, phase_delay=4, phase_ramp=8, phase_hysteresis=0.04, phase_contact_speed=0.22, phase_high_speed=0.08),
    "phase_geometry": dict(geometry_scale=0.15, warning=0.78, seek=0.0, band_seek=0.0, target=0.0, retract=0.0, alpha=0.30, max_offset=0.25, rate=0.0, phase=True, phase_search=0.0015, phase_close=0.0010, phase_retract=0.0120, phase_alpha=0.45, phase_root=0.55, phase_delay=4, phase_ramp=8, phase_hysteresis=0.04, phase_contact_speed=0.32, phase_high_speed=0.14, phase_geometry=True, phase_geometry_step=0.0015, phase_geometry_gap_scale=0.010),
    "phase_target": dict(geometry_scale=0.15, warning=0.78, seek=0.0, band_seek=0.0, target=0.0, retract=0.0, alpha=0.30, max_offset=0.22, rate=0.0, phase=True, phase_search=0.0020, phase_close=0.0010, phase_retract=0.0080, phase_alpha=0.40, phase_root=0.55, phase_delay=4, phase_ramp=8, phase_hysteresis=0.04, phase_contact_speed=0.35, phase_high_speed=0.16, phase_target=True, phase_target_force=(0.70, 0.55, 0.30), phase_target_step=0.006, phase_target_alpha=0.50, phase_target_deadband=0.04, phase_target_scale=(0.70, 0.55, 0.35), phase_target_raw_mix=0.35),
    "geometry_size_adaptive": dict(geometry_scale=0.15, geometry_max_scale=0.30, geometry_size_adaptive=True, warning=0.78, seek=0.04, band_seek=0.0, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0),
    # 将一部分食/中指闭合先验从近端迁移到新增远端自由度。
    "distal_prior": dict(geometry_scale=0.15, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0, geometry_values=(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.5, -0.5, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0)),
    # 16D 目标层逐指受力保护：只在真实力接近/超过软线时平滑回撤，
    # 低压接触时给极小补进。它与 action-domain servo 独立，便于消融。
    "direct_guard": dict(geometry_scale=0.15, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0, direct=True, direct_warning=0.60, direct_high=0.30, direct_low=0.06, direct_alpha=0.30, direct_max=0.28, direct_root=0.45),
    "direct_guard_strong": dict(geometry_scale=0.15, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0, direct=True, direct_warning=0.52, direct_high=0.42, direct_low=0.05, direct_alpha=0.35, direct_max=0.36, direct_root=0.45),
    "direct_guard_distal": dict(geometry_scale=0.15, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0, direct=True, direct_warning=0.60, direct_high=0.30, direct_low=0.06, direct_alpha=0.30, direct_max=0.28, direct_root=0.30),
    # 仅提高中指高力回撤增益；拇指/食指保持 direct_guard，验证是否能
    # 消除中指冲量而不把三指整体提前打开。
    "direct_guard_middle_safe": dict(geometry_scale=0.15, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0, direct=True, direct_warning=0.60, direct_high=0.30, direct_low=0.06, direct_alpha=0.30, direct_max=0.28, direct_root=0.45, direct_high_by_digit=(0.30, 0.30, 0.52)),
    "direct_guard_middle_070": dict(geometry_scale=0.15, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0, direct=True, direct_warning=0.55, direct_high=0.30, direct_low=0.06, direct_alpha=0.40, direct_max=0.34, direct_root=0.45, direct_high_by_digit=(0.52, 0.30, 0.70)),
    "direct_guard_impulse_safe": dict(geometry_scale=0.15, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0, direct=True, direct_warning=0.50, direct_high=0.30, direct_low=0.06, direct_alpha=0.50, direct_max=0.50, direct_root=0.45, direct_high_by_digit=(0.80, 0.40, 1.00)),
    # 保持 direct_guard 的力线和幅度，只提高 canonical 目标层的 EMA
    # 响应速度，专门检验“上一帧冲量来不及回撤”的假设。
    "direct_guard_fast_retract": dict(geometry_scale=0.15, warning=0.78, seek=0.04, target=0.0, retract=0.35, alpha=0.30, max_offset=0.25, rate=0.0, direct=True, direct_warning=0.60, direct_high=0.30, direct_low=0.06, direct_alpha=0.50, direct_max=0.28, direct_root=0.45),
}


def _mean_rows(rows: list[list[float]]) -> list[float] | None:
    if not rows:
        return None
    count = len(rows)
    width = len(rows[0])
    return [sum(row[index] for row in rows) / count for index in range(width)]


def main() -> None:
    if args.repeats <= 0:
        raise ValueError("repeats 必须为正数")
    if args.variant != "all" and args.variant not in VARIANTS:
        raise ValueError(f"未知 variant: {args.variant}")
    configure_seed(args.seed, True)
    cfg = NativeSmallFruitPinchCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.pinch_working_offset_m = (0.013, 0.030, 0.024)
    offset_args = (args.offset_x_mm, args.offset_y_mm, args.offset_z_mm)
    if any(value is not None for value in offset_args):
        if not all(value is not None for value in offset_args):
            raise ValueError("offset-x/y/z-mm 必须同时提供")
        cfg.pinch_working_offset_m = tuple(value / 1000.0 for value in offset_args)
    cfg.events.fruit_geometry.params["axis_ranges_m"] = {
        "x": (0.032, 0.035),
        "y": (0.032, 0.035),
        "z": (0.032, 0.035),
    }
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
    if args.hand_damping is not None:
        if args.hand_damping <= 0.0:
            raise ValueError("hand-damping 必须为正数")
        for actuator_name in ("g20_independent", "g20_followers"):
            cfg.robot.actuators[actuator_name].damping = args.hand_damping
    if args.hand_stiffness is not None:
        if args.hand_stiffness <= 0.0:
            raise ValueError("hand-stiffness 必须为正数")
        for actuator_name in ("g20_independent", "g20_followers"):
            cfg.robot.actuators[actuator_name].stiffness = args.hand_stiffness
    GuidedResidualSmallFruitPinchEnv.residual_scale = 0.0
    GuidedResidualSmallFruitPinchEnv.precontact_residual_fraction = 0.0
    GuidedResidualSmallFruitPinchEnv.geometry_prior_enabled = True
    GuidedResidualSmallFruitPinchEnv.geometry_prior_size_adaptive = False
    GuidedResidualSmallFruitPinchEnv.geometry_prior_after_contact = False
    GuidedResidualSmallFruitPinchEnv.force_servo_enabled = True
    GuidedResidualSmallFruitPinchEnv.direct_force_projection_enabled = bool(args.direct_force_projection)
    GuidedResidualSmallFruitPinchEnv.phase_controller_enabled = False
    GuidedResidualSmallFruitPinchEnv.geometry_approach_enabled = args.geometry_approach
    if args.hold_joint_target_speed is not None:
        if args.hold_joint_target_speed <= 0.0:
            raise ValueError("hold-joint-target-speed 必须为正数")
        GuidedResidualSmallFruitPinchEnv.hold_joint_target_speed_rad_s = args.hold_joint_target_speed
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
    if args.direct_force_high_gain_by_digit is not None:
        values = tuple(float(value.strip()) for value in args.direct_force_high_gain_by_digit.split(","))
        if len(values) != 3 or any(value < 0.0 for value in values):
            raise ValueError("direct-force-high-gain-by-digit 必须包含 3 个非负值")
        GuidedResidualSmallFruitPinchEnv.direct_force_projection_high_gain_by_digit = values
    if args.geometry_prior_values is not None:
        values = tuple(float(value.strip()) for value in args.geometry_prior_values.split(","))
        if len(values) != 14 or any(abs(value) > 1.0 for value in values):
            raise ValueError("geometry-prior-values 必须包含 14 个 [-1, 1] 值")
        GuidedResidualSmallFruitPinchEnv.geometry_prior_residual = values
    GuidedResidualSmallFruitPinchEnv.force_gate_start_fraction = 0.60
    if args.approach_speed is not None:
        if args.approach_speed <= 0.0:
            raise ValueError("approach-speed 必须为正数")
        GuidedResidualSmallFruitPinchEnv.approach_speed = args.approach_speed
    if args.projection_guard_fraction is not None:
        if not 0.0 < args.projection_guard_fraction <= 1.0:
            raise ValueError("projection-guard-fraction 必须在 (0, 1] 内")
        GuidedResidualSmallFruitPinchEnv.projection_guard_fraction = args.projection_guard_fraction
    GuidedResidualSmallFruitPinchEnv.guide_feedback_enabled = False
    GuidedResidualSmallFruitPinchEnv.adaptive_closure_enabled = False
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
    if args.approach_speed is not None:
        if args.approach_speed <= 0.0:
            raise ValueError("approach-speed 必须为正数")
        GuidedResidualSmallFruitPinchEnv.approach_speed = args.approach_speed
    env = GuidedResidualSmallFruitPinchEnv(cfg=cfg)
    zero = torch.zeros((args.num_envs, 14), device=env.device)
    try:
        variants = VARIANTS.items() if args.variant == "all" else ((args.variant, VARIANTS[args.variant]),)
        for variant_index, (name, params) in enumerate(variants):
            GuidedResidualSmallFruitPinchEnv.geometry_prior_scale = params["geometry_scale"]
            GuidedResidualSmallFruitPinchEnv.geometry_prior_size_adaptive = params.get("geometry_size_adaptive", False)
            GuidedResidualSmallFruitPinchEnv.geometry_prior_max_scale = params.get("geometry_max_scale", 0.30)
            if "geometry_values" in params:
                GuidedResidualSmallFruitPinchEnv.geometry_prior_residual = params["geometry_values"]
            GuidedResidualSmallFruitPinchEnv.force_servo_warning_fraction = params["warning"]
            GuidedResidualSmallFruitPinchEnv.force_servo_seek_gain = params["seek"]
            GuidedResidualSmallFruitPinchEnv.force_servo_band_seek_gain = params.get("band_seek", 0.0)
            GuidedResidualSmallFruitPinchEnv.force_servo_target_gain = params["target"]
            GuidedResidualSmallFruitPinchEnv.force_servo_retract_gain = params["retract"]
            GuidedResidualSmallFruitPinchEnv.force_servo_alpha = params["alpha"]
            GuidedResidualSmallFruitPinchEnv.force_servo_max_offset = params["max_offset"]
            GuidedResidualSmallFruitPinchEnv.force_servo_rate_gain = params["rate"]
            GuidedResidualSmallFruitPinchEnv.direct_force_projection_enabled = params.get("direct", bool(args.direct_force_projection))
            GuidedResidualSmallFruitPinchEnv.direct_force_projection_warning_fraction = params.get("direct_warning", 0.60)
            GuidedResidualSmallFruitPinchEnv.direct_force_projection_high_gain = params.get("direct_high", 0.30)
            GuidedResidualSmallFruitPinchEnv.direct_force_projection_high_gain_by_digit = params.get("direct_high_by_digit")
            GuidedResidualSmallFruitPinchEnv.direct_force_projection_low_gain = params.get("direct_low", 0.06)
            GuidedResidualSmallFruitPinchEnv.direct_force_projection_alpha = params.get("direct_alpha", 0.30)
            GuidedResidualSmallFruitPinchEnv.direct_force_projection_max_open = params.get("direct_max", 0.28)
            GuidedResidualSmallFruitPinchEnv.direct_force_projection_root_fraction = params.get("direct_root", 0.45)
            GuidedResidualSmallFruitPinchEnv.force_servo_band_seek_gain = params.get("band_seek", 0.0)
            GuidedResidualSmallFruitPinchEnv.phase_controller_enabled = params.get("phase", False)
            GuidedResidualSmallFruitPinchEnv.phase_search_step = params.get("phase_search", 0.0030)
            GuidedResidualSmallFruitPinchEnv.phase_close_step = params.get("phase_close", 0.0015)
            GuidedResidualSmallFruitPinchEnv.phase_retract_step = params.get("phase_retract", 0.0030)
            GuidedResidualSmallFruitPinchEnv.phase_offset_alpha = params.get("phase_alpha", 0.30)
            GuidedResidualSmallFruitPinchEnv.phase_max_offset = params.get("max_offset", 0.18)
            GuidedResidualSmallFruitPinchEnv.phase_root_fraction = params.get("phase_root", 0.55)
            GuidedResidualSmallFruitPinchEnv.phase_low_delay_steps = params.get("phase_delay", 3)
            GuidedResidualSmallFruitPinchEnv.phase_low_ramp_steps = params.get("phase_ramp", 5)
            GuidedResidualSmallFruitPinchEnv.phase_hysteresis_n = params.get("phase_hysteresis", 0.03)
            GuidedResidualSmallFruitPinchEnv.phase_contact_speed_scale = params.get("phase_contact_speed", 0.42)
            GuidedResidualSmallFruitPinchEnv.phase_high_speed_scale = params.get("phase_high_speed", 0.24)
            GuidedResidualSmallFruitPinchEnv.phase_geometry_enabled = params.get("phase_geometry", False)
            GuidedResidualSmallFruitPinchEnv.phase_geometry_step = params.get("phase_geometry_step", 0.0015)
            GuidedResidualSmallFruitPinchEnv.phase_geometry_gap_scale_m = params.get("phase_geometry_gap_scale", 0.010)
            GuidedResidualSmallFruitPinchEnv.phase_target_controller_enabled = params.get("phase_target", False)
            GuidedResidualSmallFruitPinchEnv.phase_target_force_n = params.get("phase_target_force", (0.70, 0.55, 0.30))
            GuidedResidualSmallFruitPinchEnv.phase_target_step = params.get("phase_target_step", 0.006)
            GuidedResidualSmallFruitPinchEnv.phase_target_alpha = params.get("phase_target_alpha", 0.50)
            GuidedResidualSmallFruitPinchEnv.phase_target_deadband_n = params.get("phase_target_deadband", 0.04)
            GuidedResidualSmallFruitPinchEnv.phase_target_error_scale_n = params.get("phase_target_scale", (0.70, 0.55, 0.35))
            GuidedResidualSmallFruitPinchEnv.phase_target_raw_mix = params.get("phase_target_raw_mix", 0.35)
            for repeat in range(args.repeats):
                env.reset(seed=args.seed + 1000 * repeat + variant_index)
                events = {"episodes": 0, "success": 0, "drop": 0, "overload": 0, "self_collision": 0}
                peak = torch.zeros((args.num_envs, 5), device=env.device)
                # 0 表示超时（Verifier 没有硬故障，但未在时限内形成最终成功）。
                terminal_axes: dict[int, list[list[float]]] = {0: [], 1: [], 2: [], 3: [], 4: []}
                terminal_force: dict[int, list[list[float]]] = {0: [], 1: [], 2: [], 3: [], 4: []}
                terminal_mass: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: [], 4: []}
                terminal_hold: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: [], 4: []}
                terminal_contact_ratio: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: [], 4: []}
                terminal_band_ratio: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: [], 4: []}
                while events["episodes"] < args.episodes:
                    _, _, terminated, truncated, info = env.step(zero)
                    step = info.get("step", {})
                    events["success"] += int(step.get("success_count", 0))
                    events["drop"] += int(step.get("drop_count", 0))
                    events["overload"] += int(step.get("overload_count", 0))
                    events["self_collision"] += int(step.get("self_collision_count", 0))
                    done = terminated | truncated
                    if hasattr(env, "_last_terminal_reason"):
                        for reason in (0, 1, 2, 3, 4):
                            selected = done & (env._last_terminal_reason == reason)
                            if bool(selected.any()):
                                terminal_axes[reason].extend(
                                    env._last_terminal_axes_m[selected].detach().cpu().tolist()
                                )
                                terminal_force[reason].extend(
                                    env._last_terminal_force_n[selected].detach().cpu().tolist()
                                )
                                terminal_hold[reason].extend(
                                    env._last_terminal_hold_age[selected].detach().cpu().tolist()
                                )
                                terminal_contact_ratio[reason].extend(
                                    env._last_terminal_contact_ratio[selected].detach().cpu().tolist()
                                )
                                terminal_band_ratio[reason].extend(
                                    env._last_terminal_band_ratio[selected].detach().cpu().tolist()
                                )
                                terminal_mass[reason].extend(
                                    env._fruit_mass_kg[selected].detach().cpu().tolist()
                                )
                    events["episodes"] += int(done.sum().item())
                    peak = torch.maximum(peak, env._last_physical_force_n)
                total = max(events["episodes"], 1)
                print(
                    "servo_sweep "
                    f"variant={name} repeat={repeat} episodes={total} "
                    f"strict={events['success'] / total:.4f} "
                    f"drop={events['drop'] / total:.4f} "
                    f"overload={events['overload'] / total:.4f} "
                    f"self={events['self_collision'] / total:.4f} "
                    f"peak={peak.mean(dim=0).detach().cpu().tolist()} "
                    f"success_axes_mean={_mean_rows(terminal_axes[1])} "
                    f"success_mass_mean={(sum(terminal_mass[1]) / len(terminal_mass[1])) if terminal_mass[1] else None} "
                    f"timeout_axes_mean={_mean_rows(terminal_axes[0])} "
                    f"timeout_mass_mean={(sum(terminal_mass[0]) / len(terminal_mass[0])) if terminal_mass[0] else None} "
                    f"timeout_hold_mean={(sum(terminal_hold[0]) / len(terminal_hold[0])) if terminal_hold[0] else None} "
                    f"timeout_contact_ratio_mean={(sum(terminal_contact_ratio[0]) / len(terminal_contact_ratio[0])) if terminal_contact_ratio[0] else None} "
                    f"timeout_band_ratio_mean={(sum(terminal_band_ratio[0]) / len(terminal_band_ratio[0])) if terminal_band_ratio[0] else None} "
                    f"timeout_force_mean={_mean_rows(terminal_force[0])} "
                    f"overload_axes_mean={_mean_rows(terminal_axes[3])} "
                    f"overload_mass_mean={(sum(terminal_mass[3]) / len(terminal_mass[3])) if terminal_mass[3] else None} "
                    f"overload_force_mean={_mean_rows(terminal_force[3])}",
                    flush=True,
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
