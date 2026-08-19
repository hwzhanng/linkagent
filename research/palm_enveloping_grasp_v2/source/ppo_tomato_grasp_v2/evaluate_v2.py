"""Independent reset-safe evaluator for v2 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--profile", choices=("v1_training_small", "v1_nominal_reliable", "v2_band1_match", "v2_band2_shape", "v2_band3_sensor_actuator", "v2_band4_stress"), default="v2_band1_match")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=20260820)
parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
parser.add_argument("--ablation", choices=("full14", "freeze_distal", "roots_only"), default="full14",
                    help="counterfactual action mask for P2 distal contribution audit")
parser.add_argument("--strict-coverage-deg", type=float, default=None)
parser.add_argument("--strict-hold-steps", type=int, default=None)
parser.add_argument("--pregrasp-drop-grace-steps", type=int, default=None)
parser.add_argument("--pregrasp-settle-steps", type=int, default=None,
                    help="simulation-only zero-velocity settling steps before releasing the fruit")
parser.add_argument("--physical-fingertip-proxy", action="store_true")
parser.add_argument(
    "--proxy-only-collision",
    action="store_true",
    help="simulation-only: disable uncalibrated vendor hand colliders and keep only v2 fingertip proxies plus palm support",
)
parser.add_argument("--disable-virtual-distal-contact", action="store_true")
parser.add_argument("--fingertip-proxy-radius", type=float, default=None)
parser.add_argument("--strict-require-measured-distal-contact", action="store_true")
parser.add_argument("--palm-first-gate", action="store_true")
parser.add_argument("--palm-first-coarse-scale", type=float, default=1.0)
parser.add_argument("--palm-first-coarse-max-scale", type=float, default=1.0)
parser.add_argument("--palm-first-ramp-steps", type=int, default=None)
parser.add_argument("--palm-first-motion-hold", action="store_true",
                    help="damp fresh coarse commands when supported fruit is moving outward")
parser.add_argument("--palm-first-motion-hold-scale", type=float, default=0.25)
parser.add_argument("--palm-first-motion-hold-speed", type=float, default=0.08)
parser.add_argument("--palm-first-motion-hold-offset", type=float, default=0.55)
parser.add_argument("--smoothing-alpha", type=float, default=None)
parser.add_argument("--distal-filter-alpha", type=float, default=None,
                    help="v2-only distal residual command filter alpha")
parser.add_argument("--distal-filter-alpha-per-digit", type=str, default=None,
                    help="optional four index,middle,ring,little distal filter alphas")
parser.add_argument(
    "--thumb-opposition-override", type=float, default=None,
    help="diagnostic-only override for action channel 4; does not modify the checkpoint",
)
parser.add_argument(
    "--thumb-opposition-ramp-steps", type=int, default=40,
    help="steps used to ramp the diagnostic thumb channel from the open value -1",
)
parser.add_argument(
    "--thumb-residual-override", type=float, default=None,
    help="diagnostic-only override for action channel 5 after the thumb ramp",
)
parser.add_argument(
    "--action-prefix-override", type=str, default=None,
    help="diagnostic-only comma-separated prefix (up to 14 channels), ramped from open; checkpoint is unchanged",
)
parser.add_argument("--action-prefix-ramp-steps", type=int, default=60)
parser.add_argument(
    "--residual-base-controller", action="store_true",
    help="evaluate with the observable staged base controller plus residual_scale*actor output",
)
parser.add_argument(
    "--residual-simultaneous-reach", action="store_true",
    help="ablation: ramp distal channels during the coarse approach; default is palm/root/thumb first",
)
parser.add_argument("--residual-scale", type=float, default=0.0)
parser.add_argument("--residual-teacher-coverage-blend", type=float, default=0.0,
                    help="bounded simulation-only thumb-first teacher delta on channels 2:14 after base approach")
parser.add_argument("--teacher-thumb-first-steps", type=int, default=0)
parser.add_argument("--teacher-thumb-first-global-scale", type=float, default=0.35)
parser.add_argument("--teacher-coverage-delta", type=str, default="0.0,0.0,0.0,0.0")
parser.add_argument("--teacher-motion-smoothing-alpha", type=float, default=1.0)
parser.add_argument(
    "--skip-actor-forward", action="store_true",
    help="diagnostic: do not execute the checkpoint actor when residual_scale is zero",
)
parser.add_argument(
    "--residual-mask", choices=("all", "thumb_distal", "distal", "coverage"), default="thumb_distal",
    help="residual channels exposed to the actor; default protects coarse palm/root actions",
)
parser.add_argument(
    "--residual-no-phase-gate", action="store_true",
    help="ablation: allow residuals during coarse approach; default enables them after the base approach",
)
parser.add_argument(
    "--residual-target-action", type=str,
    default="1.0,-1.0,0.5817804336547852,-0.009242807142138481,1.0,-0.01167266070842743,0.58189457654953,0.9941583275794983,0.11048947274684906,0.05933154374361038,-0.6171775460243225,0.6542128324508667,-0.24184483289718628,0.5135747194290161",
)
parser.add_argument("--residual-approach-steps", type=int, default=240)
parser.add_argument("--residual-per-digit-steps", type=int, default=60)
parser.add_argument("--residual-hold-steps", type=int, default=80)
parser.add_argument("--residual-approach-smoothing", type=float, default=1.0,
                    help="fraction of the staged coarse ramp applied per policy step")
parser.add_argument("--residual-thumb-delay-steps", type=int, default=0)
parser.add_argument("--residual-thumb-ramp-steps", type=int, default=60)
parser.add_argument("--residual-thumb-first-steps", type=int, default=0,
                    help="thumb-only coarse sub-phase before straight-finger roots ramp")
parser.add_argument("--residual-thumb-first-global-scale", type=float, default=0.0,
                    help="fraction of global flexion allowed during the thumb-first sub-phase")
parser.add_argument("--residual-lower-force-per-digit", type=str, default=None)
parser.add_argument("--residual-upper-force-per-digit", type=str, default=None)
parser.add_argument("--residual-insertion-step-fraction", type=float, default=0.08)
parser.add_argument("--residual-retreat-fraction", type=float, default=0.12)
parser.add_argument(
    "--residual-insertion-step-per-digit", type=str, default=None,
    help="optional four index,middle,ring,little insertion fractions",
)
parser.add_argument(
    "--residual-retreat-per-digit", type=str, default=None,
    help="optional four index,middle,ring,little retreat fractions",
)
parser.add_argument("--residual-thumb-hold-offset", type=str, default="0.0,0.0",
                    help="two bounded thumb action offsets used only in residual hold feedback")
parser.add_argument("--residual-thumb-lower-force", type=float, default=0.10)
parser.add_argument("--residual-thumb-upper-force", type=float, default=0.80)
parser.add_argument("--residual-thumb-search-amplitude", type=str, default="0.0,0.0",
                    help="optional bounded two-sided thumb search amplitude during low-force hold")
parser.add_argument("--residual-thumb-search-period", type=int, default=24,
                    help="low-force frames before the signed thumb search offset flips")
parser.add_argument("--residual-thumb-contact-gate-steps", type=int, default=0,
                    help="hold straight-finger insertion until thumb force is in-band for N frames")
parser.add_argument("--residual-thumb-contact-latch-force", type=float, default=0.0,
                    help="measured thumb force that starts a bounded action latch")
parser.add_argument("--residual-thumb-contact-latch-steps", type=int, default=0,
                    help="policy frames to hold the thumb action after contact; zero disables")
parser.add_argument("--residual-thumb-contact-latch-freeze-coarse", action="store_true",
                    help="freeze coarse palm/root channels during the thumb contact latch")
parser.add_argument("--assist-gain", type=float, default=None,
                    help="override the bounded distal-to-proximal reach assist gain for an inference audit")
parser.add_argument("--assist-per-digit", type=str, default=None,
                    help="override four per-digit distal-to-proximal reach gains: index,middle,ring,little")
parser.add_argument(
    "--trace-action-steps", type=str, default=None,
    help="diagnostic comma-separated episode steps to record (action and measured force)",
)
parser.add_argument("--per-digit-root-span", type=float, default=None,
                    help="optional v2-only root residual span for an inference audit")
parser.add_argument("--per-digit-distal-gate", action="store_true")
parser.add_argument("--per-digit-link3-threshold", type=float, default=None)
parser.add_argument("--per-digit-link3-stable-steps", type=int, default=None)
parser.add_argument("--per-digit-distal-min-episode-step", type=int, default=None)
parser.add_argument("--per-digit-near-surface-gate", action="store_true",
                    help="release each distal residual after palm dwell and that fingertip is near the fruit")
parser.add_argument("--per-digit-near-surface-distance", type=float, default=0.043)
parser.add_argument(
    "--distal-near-surface-distance", type=float, default=None,
    help="override the basic per-finger near-surface gate distance in metres",
)
parser.add_argument("--per-digit-near-surface-min-episode-step", type=int, default=20)
parser.add_argument("--per-digit-distal-probe", action="store_true",
                    help="enable bounded post-palm distal contact bootstrap")
parser.add_argument("--per-digit-distal-probe-distance", type=float, default=0.064)
parser.add_argument("--per-digit-distal-probe-action", type=float, default=0.14)
parser.add_argument("--per-digit-distal-probe-action-per-digit", type=str, default=None,
                    help="optional four-value index,middle,ring,little probe action vector")
parser.add_argument("--per-digit-distal-probe-start-step", type=int, default=20)
parser.add_argument("--per-digit-distal-probe-ramp-steps", type=int, default=80)
parser.add_argument("--per-digit-distal-probe-staggered", action="store_true",
                    help="stagger the simulation-only distal bootstrap across the four straight fingers")
parser.add_argument("--per-digit-distal-probe-digit-interval", type=int, default=20)
parser.add_argument("--tomato-contact-offset", type=float, default=None)
parser.add_argument("--tomato-rest-offset", type=float, default=None)
parser.add_argument("--proxy-contact-offset", type=float, default=None)
parser.add_argument("--proxy-rest-offset", type=float, default=None)
parser.add_argument("--palm-support-contact-offset", type=float, default=None)
parser.add_argument("--palm-support-rest-offset", type=float, default=None)
parser.add_argument("--reset-clearance", type=float, default=None)
parser.add_argument("--cup-half-extent", type=str, default=None)
parser.add_argument("--rim-height", type=float, default=None)
parser.add_argument("--debug-collision-digit", choices=("", "palm", "all", "index", "middle", "ring", "little"), default="")
parser.add_argument("--command-rate-rad-s", type=float, default=None)
parser.add_argument("--force-ceiling", type=str, default=None,
                    help="five comma-separated simulation safety ceilings: thumb,index,middle,ring,little")
parser.add_argument("--safe-force-n", type=float, default=None)
parser.add_argument("--mass-min", type=float, default=None)
parser.add_argument("--mass-max", type=float, default=None)
parser.add_argument("--density-min", type=float, default=None)
parser.add_argument("--density-max", type=float, default=None)
parser.add_argument("--axis-range-m", type=str, default=None,
                    help="override isotropic tomato semi-axis range as one or two metres")
parser.add_argument("--contact-stiffness-n-m", type=float, default=None,
                    help="override authored tomato compliant-contact stiffness for this simulation run")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch
from torch import nn

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile, profile_definition
from simulation.ppo_tomato_grasp_v2.contract import V2_ACTOR_OBSERVATION_DIM, V2_ACTION_DIM
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv, V2EnvelopingPhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp_v2.residual_controller import ResidualBaseController, ResidualBaseControllerCfg
from simulation.ppo_tomato_grasp.enveloping_teacher import EnvelopingTeacher, EnvelopingTeacherCfg


def parse_pair(value: str, name: str) -> tuple[float, float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 2 or any(item <= 0.0 for item in values):
        raise ValueError(f"{name} requires two positive comma-separated metres")
    return float(values[0]), float(values[1])


def parse_axis_range(value: str, name: str = "--axis-range-m") -> tuple[float, float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) == 1:
        values = [values[0], values[0]]
    if len(values) != 2 or any(item <= 0.0 for item in values) or values[0] > values[1]:
        raise ValueError(f"{name} requires one positive value or min,max with 0 < min <= max")
    return float(values[0]), float(values[1])


def parse_four(value: str, name: str) -> tuple[float, float, float, float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 4 or any(item < 0.0 for item in values):
        raise ValueError(f"{name} requires four non-negative comma-separated values")
    return tuple(float(item) for item in values)  # type: ignore[return-value]


def parse_signed_four(value: str, name: str) -> tuple[float, float, float, float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 4 or any(abs(item) > 0.30 for item in values):
        raise ValueError(f"{name} requires four values in [-0.30,0.30]")
    return tuple(values)  # type: ignore[return-value]


def parse_signed_pair(value: str, name: str) -> tuple[float, float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 2 or any(abs(item) > 1.0 for item in values):
        raise ValueError(f"{name} requires two values in [-1,1]")
    return float(values[0]), float(values[1])


def parse_force_ceiling(value: str, name: str = "--force-ceiling") -> tuple[float, float, float, float, float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 5 or any(item <= 0.0 for item in values):
        raise ValueError(f"{name} requires five positive comma-separated values")
    return tuple(float(item) for item in values)  # type: ignore[return-value]


def parse_prefix(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not 1 <= len(values) <= 14 or any(abs(item) > 1.000001 for item in values):
        raise ValueError("--action-prefix-override requires 1..14 values in [-1,1]")
    return values


def parse_trace_steps(value: str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if any(step < 0 for step in steps):
        raise ValueError("--trace-action-steps requires non-negative integer steps")
    return steps


def residual_action_mask(name: str, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(V2_ACTION_DIM, device=device)
    if name == "all":
        mask[:] = 1.0
    elif name == "thumb_distal":
        mask[5] = 1.0
        mask[10:] = 1.0
    elif name == "distal":
        mask[10:] = 1.0
    elif name == "coverage":
        # Coverage residuals may adjust radial/ulnar balance, spread, thumb
        # opposition/residual, straight-finger roots and distal joints, while
        # protecting global flexion/curl channels that establish palm support.
        mask[2:] = 1.0
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown residual mask {name!r}")
    return mask


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(V2_ACTOR_OBSERVATION_DIM, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh())
        self.mean = nn.Linear(128, V2_ACTION_DIM)

    def forward(self, obs):
        return torch.tanh(self.mean(self.body(obs)))


def make_cfg():
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args_cli.device
    cfg.training_stage = args_cli.stage
    cfg.strict_enveloping_enabled = True
    cfg.strict_enveloping_require_stem_break = args_cli.stage >= 2
    cfg.episode_length_s = 12.0
    cfg.safe_force_n = 12.0
    cfg.force_ceiling_n = (12.0, 9.75, 9.75, 8.25, 8.25)
    if args_cli.safe_force_n is not None:
        if args_cli.safe_force_n <= 0.0:
            raise ValueError("--safe-force-n must be positive")
        cfg.safe_force_n = float(args_cli.safe_force_n)
    if args_cli.force_ceiling is not None:
        cfg.force_ceiling_n = parse_force_ceiling(args_cli.force_ceiling)
    if args_cli.strict_coverage_deg is not None:
        cfg.strict_angular_coverage_min_deg = float(args_cli.strict_coverage_deg)
    if args_cli.strict_hold_steps is not None:
        cfg.strict_hold_steps = int(args_cli.strict_hold_steps)
    if args_cli.pregrasp_drop_grace_steps is not None:
        cfg.pregrasp_drop_grace_steps = int(args_cli.pregrasp_drop_grace_steps)
    if args_cli.pregrasp_settle_steps is not None:
        if args_cli.pregrasp_settle_steps < 0:
            raise ValueError("--pregrasp-settle-steps must be non-negative")
        cfg.pregrasp_settle_steps = int(args_cli.pregrasp_settle_steps)
    # ``proxy-only-collision`` is a physical contact audit: disabling the
    # vendor colliders while leaving the proxies non-physical would silently
    # create a no-fingertip-contact scene.  Make the safe implication
    # explicit; users can still run a virtual shell audit without
    # ``--proxy-only-collision``.
    cfg.fingertip_collision_proxy_physical = bool(
        args_cli.physical_fingertip_proxy or args_cli.proxy_only_collision
    )
    cfg.proxy_only_collision = bool(args_cli.proxy_only_collision)
    if args_cli.fingertip_proxy_radius is not None:
        cfg.fingertip_collision_proxy_radius_m = float(args_cli.fingertip_proxy_radius)
    if args_cli.disable_virtual_distal_contact:
        cfg.virtual_distal_contact_enabled = False
    cfg.strict_require_measured_distal_contact = bool(args_cli.strict_require_measured_distal_contact)
    cfg.palm_first_distal_gate = bool(args_cli.palm_first_gate)
    cfg.palm_first_coarse_scale = float(args_cli.palm_first_coarse_scale)
    cfg.palm_first_coarse_max_scale = float(args_cli.palm_first_coarse_max_scale)
    if args_cli.palm_first_ramp_steps is not None:
        cfg.palm_first_ramp_steps = int(args_cli.palm_first_ramp_steps)
    cfg.palm_first_motion_hold_enabled = bool(args_cli.palm_first_motion_hold)
    cfg.palm_first_motion_hold_scale = float(args_cli.palm_first_motion_hold_scale)
    cfg.palm_first_motion_hold_speed_m_s = float(args_cli.palm_first_motion_hold_speed)
    cfg.palm_first_motion_hold_offset_ratio = float(args_cli.palm_first_motion_hold_offset)
    cfg.per_digit_distal_gate = bool(args_cli.per_digit_distal_gate)
    if args_cli.per_digit_link3_threshold is not None:
        cfg.per_digit_distal_link3_threshold_n = float(args_cli.per_digit_link3_threshold)
    if args_cli.per_digit_link3_stable_steps is not None:
        cfg.per_digit_distal_link3_stable_steps = int(args_cli.per_digit_link3_stable_steps)
    if args_cli.per_digit_distal_min_episode_step is not None:
        cfg.per_digit_distal_min_episode_step = int(args_cli.per_digit_distal_min_episode_step)
    cfg.per_digit_distal_near_surface_enabled = bool(args_cli.per_digit_near_surface_gate)
    cfg.per_digit_distal_near_surface_distance_m = float(args_cli.per_digit_near_surface_distance)
    cfg.per_digit_distal_near_surface_min_episode_step = int(args_cli.per_digit_near_surface_min_episode_step)
    cfg.per_digit_distal_probe_enabled = bool(args_cli.per_digit_distal_probe)
    cfg.per_digit_distal_probe_distance_m = float(args_cli.per_digit_distal_probe_distance)
    cfg.per_digit_distal_probe_action = float(args_cli.per_digit_distal_probe_action)
    if args_cli.per_digit_distal_probe_action_per_digit is not None:
        values = tuple(float(item.strip()) for item in args_cli.per_digit_distal_probe_action_per_digit.split(",") if item.strip())
        if len(values) != 4 or any(abs(item) > 1.000001 for item in values):
            raise ValueError("--per-digit-distal-probe-action-per-digit requires four values in [-1,1]")
        cfg.per_digit_distal_probe_action_per_digit = values
    cfg.per_digit_distal_probe_start_episode_step = int(args_cli.per_digit_distal_probe_start_step)
    cfg.per_digit_distal_probe_ramp_steps = int(args_cli.per_digit_distal_probe_ramp_steps)
    cfg.per_digit_distal_probe_staggered = bool(args_cli.per_digit_distal_probe_staggered)
    cfg.per_digit_distal_probe_digit_interval_steps = int(args_cli.per_digit_distal_probe_digit_interval)
    if args_cli.tomato_contact_offset is not None:
        cfg.tomato_contact_offset_m = float(args_cli.tomato_contact_offset)
    if args_cli.tomato_rest_offset is not None:
        cfg.tomato_rest_offset_m = float(args_cli.tomato_rest_offset)
    if args_cli.proxy_contact_offset is not None:
        cfg.fingertip_proxy_contact_offset_m = float(args_cli.proxy_contact_offset)
    if args_cli.proxy_rest_offset is not None:
        cfg.fingertip_proxy_rest_offset_m = float(args_cli.proxy_rest_offset)
    if args_cli.palm_support_contact_offset is not None:
        cfg.palm_support_contact_offset_m = float(args_cli.palm_support_contact_offset)
    if args_cli.palm_support_rest_offset is not None:
        cfg.palm_support_rest_offset_m = float(args_cli.palm_support_rest_offset)
    if args_cli.reset_clearance is not None:
        cfg.tomato_reset_clearance_m = float(args_cli.reset_clearance)
    if args_cli.cup_half_extent is not None:
        cfg.palm_cup_inner_half_extent_m = parse_pair(args_cli.cup_half_extent, "--cup-half-extent")
    if args_cli.rim_height is not None:
        cfg.palm_cup_rim_height_m = float(args_cli.rim_height)
    cfg.debug_support_only_collision = bool(args_cli.debug_collision_digit)
    cfg.debug_collision_digit = args_cli.debug_collision_digit
    if args_cli.command_rate_rad_s is not None:
        cfg.command_rate_limit_rad_s = float(args_cli.command_rate_rad_s)
    if args_cli.mass_min is not None or args_cli.mass_max is not None:
        low = float(args_cli.mass_min if args_cli.mass_min is not None else cfg.accepted_mass_kg[0])
        high = float(args_cli.mass_max if args_cli.mass_max is not None else cfg.accepted_mass_kg[1])
        if not (0.0 < low <= high):
            raise ValueError("mass range must satisfy 0 < min <= max")
        cfg.accepted_mass_kg = (low, high)
        cfg.events["tomato_geometry"].params["accepted_mass_kg"] = (low, high)
    if args_cli.density_min is not None or args_cli.density_max is not None:
        low = float(args_cli.density_min if args_cli.density_min is not None else cfg.density_range_kg_m3[0])
        high = float(args_cli.density_max if args_cli.density_max is not None else cfg.density_range_kg_m3[1])
        if not (0.0 < low <= high):
            raise ValueError("density range must satisfy 0 < min <= max")
        cfg.density_range_kg_m3 = (low, high)
        cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (low, high)
    if args_cli.smoothing_alpha is not None:
        cfg.coarse_action_smoothing_alpha = float(args_cli.smoothing_alpha)
    if args_cli.distal_filter_alpha is not None:
        if not 0.0 < args_cli.distal_filter_alpha < 1.0:
            raise ValueError("--distal-filter-alpha must lie in (0,1)")
        cfg.distal_action_filter_alpha = float(args_cli.distal_filter_alpha)
    if args_cli.distal_filter_alpha_per_digit is not None:
        per_digit_alpha = parse_four(args_cli.distal_filter_alpha_per_digit, "--distal-filter-alpha-per-digit")
        if any(value >= 1.0 for value in per_digit_alpha):
            raise ValueError("--distal-filter-alpha-per-digit values must lie in [0,1)")
        cfg.distal_action_filter_alpha = per_digit_alpha
    if args_cli.assist_gain is not None:
        cfg.distal_proximal_reach_assist_gain = float(args_cli.assist_gain)
        cfg.distal_proximal_reach_assist_gain_per_digit = None
    if args_cli.assist_per_digit is not None:
        cfg.distal_proximal_reach_assist_gain_per_digit = parse_four(
            args_cli.assist_per_digit, "--assist-per-digit"
        )
        cfg.distal_proximal_reach_assist_gain = 0.0
    if args_cli.distal_filter_alpha_per_digit is not None:
        per_digit_alpha = parse_four(args_cli.distal_filter_alpha_per_digit, "--distal-filter-alpha-per-digit")
        if any(value >= 1.0 for value in per_digit_alpha):
            raise ValueError("--distal-filter-alpha-per-digit values must lie in [0,1)")
        cfg.distal_action_filter_alpha = per_digit_alpha
    if args_cli.distal_filter_alpha is not None:
        if not 0.0 < args_cli.distal_filter_alpha < 1.0:
            raise ValueError("--distal-filter-alpha must lie in (0,1)")
        cfg.distal_action_filter_alpha = float(args_cli.distal_filter_alpha)
    cfg = apply_profile(cfg, args_cli.profile, num_envs=args_cli.num_envs, seed=args_cli.seed)
    if args_cli.axis_range_m is not None:
        axis_range = parse_axis_range(args_cli.axis_range_m)
        cfg.axis_range_m = axis_range
        cfg.isotropic_axes = True
        if axis_range[1] > 0.040:
            cfg.accepted_mass_kg = (0.050, 0.500)
        cfg.events["tomato_geometry"].params["axis_range_m"] = axis_range
        cfg.events["tomato_geometry"].params["isotropic_axes"] = True
        cfg.events["tomato_geometry"].params["accepted_mass_kg"] = cfg.accepted_mass_kg
    if args_cli.contact_stiffness_n_m is not None:
        if args_cli.contact_stiffness_n_m <= 0.0:
            raise ValueError("--contact-stiffness-n-m must be positive")
        cfg.tomato_cfg.spawn.physics_material.compliant_contact_stiffness = float(args_cli.contact_stiffness_n_m)
    # Reapply explicit filter overrides after profile resolution.  Profiles
    # are allowed to define their own default, but a diagnostic command must
    # never silently lose its declared scalar/vector filter.
    if args_cli.distal_filter_alpha_per_digit is not None:
        per_digit_alpha = parse_four(args_cli.distal_filter_alpha_per_digit, "--distal-filter-alpha-per-digit")
        if any(value >= 1.0 for value in per_digit_alpha):
            raise ValueError("--distal-filter-alpha-per-digit values must lie in [0,1)")
        cfg.distal_action_filter_alpha = per_digit_alpha
    elif args_cli.distal_filter_alpha is not None:
        if not 0.0 < args_cli.distal_filter_alpha < 1.0:
            raise ValueError("--distal-filter-alpha must lie in (0,1)")
        cfg.distal_action_filter_alpha = float(args_cli.distal_filter_alpha)
    if args_cli.residual_base_controller:
        # The residual controller owns a 180-step coarse approach, four
        # per-digit insertion windows, and a hold window.  The historical
        # 12-second v2 evaluator resets at 240 policy steps, so it would
        # silently restart before the controller reaches the little finger.
        cfg.episode_length_s = max(
            24.0,
            (int(args_cli.residual_approach_steps)
             + 4 * int(args_cli.residual_per_digit_steps)
             + int(args_cli.residual_hold_steps) + 20) * cfg.policy_dt_s,
        )
    elif args_cli.action_prefix_override is not None:
        # A diagnostic action-prefix replay also needs a hold window.  The
        # historical 12-second episode is exactly 240 policy steps, so a
        # 240-step ramp would terminate on the first frame at the target and
        # report no distal contact even when the candidate is physically
        # reachable.  Keep this extension simulation-only and record the
        # resolved episode length in the report below.
        cfg.episode_length_s = max(
            12.0,
            (int(args_cli.action_prefix_ramp_steps) + 80 + 20) * cfg.policy_dt_s,
        )
    if args_cli.per_digit_root_span is not None:
        cfg.per_digit_root_residual_span_fraction = float(args_cli.per_digit_root_span)
    if args_cli.distal_near_surface_distance is not None:
        if args_cli.distal_near_surface_distance <= 0.0:
            raise ValueError("--distal-near-surface-distance must be positive")
        cfg.per_digit_distal_near_surface_distance_m = float(args_cli.distal_near_surface_distance)
    if args_cli.assist_gain is not None:
        cfg.distal_proximal_reach_assist_gain = float(args_cli.assist_gain)
        cfg.distal_proximal_reach_assist_gain_per_digit = None
    if args_cli.assist_per_digit is not None:
        cfg.distal_proximal_reach_assist_gain_per_digit = parse_four(
            args_cli.assist_per_digit, "--assist-per-digit"
        )
        cfg.distal_proximal_reach_assist_gain = 0.0
    # Profile application replaces the geometry prior, so apply explicit
    # matrix overrides after it as well as before it.
    if args_cli.mass_min is not None or args_cli.mass_max is not None:
        low = float(args_cli.mass_min if args_cli.mass_min is not None else cfg.accepted_mass_kg[0])
        high = float(args_cli.mass_max if args_cli.mass_max is not None else cfg.accepted_mass_kg[1])
        cfg.accepted_mass_kg = (low, high)
        cfg.events["tomato_geometry"].params["accepted_mass_kg"] = (low, high)
    if args_cli.density_min is not None or args_cli.density_max is not None:
        low = float(args_cli.density_min if args_cli.density_min is not None else cfg.density_range_kg_m3[0])
        high = float(args_cli.density_max if args_cli.density_max is not None else cfg.density_range_kg_m3[1])
        cfg.density_range_kg_m3 = (low, high)
        cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (low, high)
    return cfg


@torch.inference_mode()
def main():
    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=args_cli.device, weights_only=False)
    if (checkpoint.get("actor_observation_dim"), checkpoint.get("action_dim")) != (V2_ACTOR_OBSERVATION_DIM, V2_ACTION_DIM):
        raise ValueError("checkpoint does not match the v2 actor contract")
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    device = torch.device(args_cli.device)
    actor = Actor().to(device)
    model = checkpoint["model"]
    actor.load_state_dict({
        "body.0.weight": model["actor_body.0.weight"], "body.0.bias": model["actor_body.0.bias"],
        "body.2.weight": model["actor_body.2.weight"], "body.2.bias": model["actor_body.2.bias"],
        "body.4.weight": model["actor_body.4.weight"], "body.4.bias": model["actor_body.4.bias"],
        "mean.weight": model["actor_mean.weight"], "mean.bias": model["actor_mean.bias"],
    })
    actor.eval()
    # The prestartup PhysX events sample geometry/material state while the
    # environment is constructed, before ``env.reset(seed=...)`` can reseed
    # the episode.  Actor construction consumes torch RNG state, so reseed
    # immediately before scene construction to keep evaluator geometry and
    # controller probes bitwise-comparable for the same declared seed.
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    env = V2EnvelopingPhysxTomatoGraspEnv(make_cfg())
    obs, _ = env.reset(seed=args_cli.seed)
    residual_mask = residual_action_mask(args_cli.residual_mask, device)
    residual_controller = None
    if args_cli.residual_base_controller:
        target = parse_prefix(args_cli.residual_target_action)
        if target is None or len(target) != 14:
            raise ValueError("--residual-target-action requires 14 values")
        lower_force = (
            parse_four(args_cli.residual_lower_force_per_digit, "--residual-lower-force-per-digit")
            if args_cli.residual_lower_force_per_digit is not None
            else (0.18, 0.18, 0.18, 0.18)
        )
        upper_force = (
            parse_four(args_cli.residual_upper_force_per_digit, "--residual-upper-force-per-digit")
            if args_cli.residual_upper_force_per_digit is not None
            else (0.80, 0.80, 0.80, 0.80)
        )
        if any(upper < lower for lower, upper in zip(lower_force, upper_force)):
            raise ValueError("each residual force upper band must be >= its lower band")
        if not 0.0 < args_cli.residual_insertion_step_fraction <= 1.0:
            raise ValueError("--residual-insertion-step-fraction must lie in (0,1]")
        if not 0.0 < args_cli.residual_retreat_fraction <= 1.0:
            raise ValueError("--residual-retreat-fraction must lie in (0,1]")
        insertion_per_digit = (
            parse_four(args_cli.residual_insertion_step_per_digit, "--residual-insertion-step-per-digit")
            if args_cli.residual_insertion_step_per_digit is not None
            else None
        )
        retreat_per_digit = (
            parse_four(args_cli.residual_retreat_per_digit, "--residual-retreat-per-digit")
            if args_cli.residual_retreat_per_digit is not None
            else None
        )
        for name, values in (
            ("--residual-insertion-step-per-digit", insertion_per_digit),
            ("--residual-retreat-per-digit", retreat_per_digit),
        ):
            if values is not None and any(value <= 0.0 or value > 1.0 for value in values):
                raise ValueError(f"{name} values must lie in (0,1]")
        residual_controller = ResidualBaseController(
            args_cli.num_envs,
            device,
            target,
            ResidualBaseControllerCfg(
                approach_steps=int(args_cli.residual_approach_steps),
                per_digit_steps=int(args_cli.residual_per_digit_steps),
                approach_smoothing=float(args_cli.residual_approach_smoothing),
                thumb_delay_steps=int(args_cli.residual_thumb_delay_steps),
                thumb_ramp_steps=int(args_cli.residual_thumb_ramp_steps),
                thumb_first_steps=int(args_cli.residual_thumb_first_steps),
                thumb_first_global_scale=float(args_cli.residual_thumb_first_global_scale),
                simultaneous_reach=bool(args_cli.residual_simultaneous_reach),
                lower_force_n=lower_force,
                upper_force_n=upper_force,
                insertion_step_fraction=float(args_cli.residual_insertion_step_fraction),
                retreat_fraction=float(args_cli.residual_retreat_fraction),
                insertion_step_fraction_per_digit=insertion_per_digit,
                retreat_fraction_per_digit=retreat_per_digit,
                thumb_lower_force_n=float(args_cli.residual_thumb_lower_force),
                thumb_upper_force_n=float(args_cli.residual_thumb_upper_force),
                thumb_hold_target_offset=parse_signed_pair(
                    args_cli.residual_thumb_hold_offset, "--residual-thumb-hold-offset"
                ),
                thumb_search_amplitude=parse_signed_pair(
                    args_cli.residual_thumb_search_amplitude, "--residual-thumb-search-amplitude"
                ),
                thumb_search_period_steps=max(int(args_cli.residual_thumb_search_period), 1),
                thumb_first_contact_gate_steps=max(int(args_cli.residual_thumb_contact_gate_steps), 0),
                thumb_contact_latch_force_n=max(float(args_cli.residual_thumb_contact_latch_force), 0.0),
                thumb_contact_latch_steps=max(int(args_cli.residual_thumb_contact_latch_steps), 0),
                thumb_contact_latch_freeze_coarse=bool(args_cli.residual_thumb_contact_latch_freeze_coarse),
            ),
        )
    residual_teacher = None
    if float(args_cli.residual_teacher_coverage_blend) > 0.0:
        if float(args_cli.residual_teacher_coverage_blend) > 1.0:
            raise ValueError("--residual-teacher-coverage-blend must lie in [0,1]")
        teacher_target = parse_prefix(args_cli.residual_target_action)
        if teacher_target is None or len(teacher_target) != 14:
            raise ValueError("teacher coverage blend requires a 14D residual target")
        teacher_cfg = EnvelopingTeacherCfg(
            distal_prior_action=tuple(teacher_target[10:]),
            thumb_first_phase_steps=max(0, int(args_cli.teacher_thumb_first_steps)),
            thumb_first_global_scale=float(args_cli.teacher_thumb_first_global_scale),
            thumb_first_coverage_delta=parse_signed_four(args_cli.teacher_coverage_delta, "--teacher-coverage-delta"),
            motion_smoothing_alpha=float(args_cli.teacher_motion_smoothing_alpha),
        )
        residual_teacher = EnvelopingTeacher(args_cli.num_envs, device, teacher_cfg)
        target_tensor = torch.tensor(teacher_target, device=device, dtype=torch.float32)
        residual_teacher.root_prior[:, :10] = target_tensor[:10]
        residual_teacher.residual_prior[:] = target_tensor[5:10]
    completed = torch.zeros((), dtype=torch.long, device=device)
    names = ("success", "drop", "overload", "damage", "timeout", "five_digit", "five_digit_observed", "palm_supported", "measured_palm_supported", "distal_all", "distal_contact_all", "measured_distal_contact_all", "palm_first", "measured_geometry", "post_break_hold")
    counts = {name: torch.zeros((), dtype=torch.long, device=device) for name in names}
    force_sum = torch.zeros(5, device=device)
    peak_force_sum = torch.zeros(5, device=device)
    peak_palm_force_sum = torch.zeros((), device=device)
    peak_force_max = torch.zeros(5, device=device)
    peak_palm_force_max = torch.zeros((), device=device)
    first_contact_sum = torch.zeros(5, device=device)
    contact_occupancy_sum = torch.zeros(5, device=device)
    pre_break_occupancy_sum = torch.zeros(5, device=device)
    post_break_occupancy_sum = torch.zeros(5, device=device)
    distal_observed_count = torch.zeros(4, device=device)
    measured_distal_observed_count = torch.zeros(4, device=device)
    link3_force_sum = torch.zeros(5, device=device)
    measured_link4_force_sum = torch.zeros(5, device=device)
    link4_force_sum = torch.zeros(5, device=device)
    measured_link4_integral_sum = torch.zeros(4, device=device)
    measured_contact_steps_sum = torch.zeros(4, device=device)
    measured_max_streak_sum = torch.zeros(4, device=device)
    measured_palm_force_sum = torch.zeros((), device=device)
    strict_hold_max_sum = torch.zeros((), device=device)
    strict_hold_max_max = torch.zeros((), device=device)
    strict_coverage_max_sum = torch.zeros((), device=device)
    max_gap_sum = torch.zeros(5, device=device)
    load_share_sum = torch.zeros(5, device=device)
    action_sum = torch.zeros(14, device=device)
    action_abs_sum = torch.zeros(14, device=device)
    action_max = torch.full((14,), -float("inf"), device=device)
    trace_steps = set(parse_trace_steps(args_cli.trace_action_steps))
    action_trace: dict[str, dict[str, list[float] | int]] = {}
    prefix_override = parse_prefix(args_cli.action_prefix_override)
    for _ in range(args_cli.steps):
        policy_action = (
            torch.zeros((args_cli.num_envs, V2_ACTION_DIM), device=device)
            if args_cli.skip_actor_forward
            else actor(obs["policy"])
        )
        action = policy_action
        if residual_controller is not None:
            base_action = residual_controller.command(
                env.episode_length_buf,
                env._v2_measured_link4_force,
            )
            phase_gate = (
                (env.episode_length_buf >= int(args_cli.residual_approach_steps)).float()
                if not args_cli.residual_no_phase_gate
                else torch.ones(args_cli.num_envs, device=device)
            )
            action = (
                base_action
                + float(args_cli.residual_scale)
                * (policy_action * residual_mask.unsqueeze(0) * phase_gate.unsqueeze(-1))
            ).clamp(-1.0, 1.0)
            if residual_teacher is not None:
                teacher_action = residual_teacher.command(
                    env.contact_forces,
                    env.palm_force,
                    episode_step=env.episode_length_buf,
                    measured_digit_force_n=env._v2_measured_link4_force,
                )
                teacher_gate = phase_gate * (residual_teacher.phase >= 1).float()
                teacher_delta = (teacher_action[:, 2:] - base_action[:, 2:]).clamp(-0.30, 0.30)
                action[:, 2:] = (
                    action[:, 2:]
                    + float(args_cli.residual_teacher_coverage_blend)
                    * teacher_gate.unsqueeze(-1)
                    * teacher_delta
                ).clamp(-1.0, 1.0)
        if prefix_override is not None:
            action = action.clone()
            fraction = (
                env.episode_length_buf.float()
                / max(float(args_cli.action_prefix_ramp_steps), 1.0)
            ).clamp(0.0, 1.0).unsqueeze(-1)
            target = action.new_tensor(prefix_override).unsqueeze(0)
            count = len(prefix_override)
            # Match the environment's canonical open-hand semantic action
            # used by the action-search diagnostics.  Ramping every channel
            # from -1 silently drives the synergy/proximal channels through
            # a different trajectory and makes a valid search candidate
            # incomparable in this evaluator.
            open_action = action.new_zeros((1, V2_ACTION_DIM))
            open_action[:, :10] = action.new_tensor(
                (-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
            )
            action[:, :count] = (
                open_action[:, :count]
                + fraction * (target - open_action[:, :count])
            ).clamp(-1.0, 1.0)
        if args_cli.thumb_opposition_override is not None:
            action = action.clone()
            fraction = (
                env.episode_length_buf.float()
                / max(float(args_cli.thumb_opposition_ramp_steps), 1.0)
            ).clamp(0.0, 1.0)
            target = float(args_cli.thumb_opposition_override)
            action[:, 4] = (-1.0 + fraction * (target + 1.0)).clamp(-1.0, 1.0)
            if args_cli.thumb_residual_override is not None:
                action[:, 5] = float(args_cli.thumb_residual_override)
        action_sum += action.sum(0)
        action_abs_sum += action.abs().sum(0)
        action_max = torch.maximum(action_max, action.max(0).values)
        trace_step = int(env.episode_length_buf.max().item())
        if trace_step in trace_steps:
            action_trace[str(trace_step)] = {
                "episode_step": trace_step,
                "action_env0": [float(v) for v in action[0].detach().cpu().tolist()],
                "measured_link4_force_env0": [
                    float(v) for v in env._v2_measured_link4_force[0].detach().cpu().tolist()
                ],
                "contact_forces_env0": [
                    float(v) for v in env.contact_forces[0].detach().cpu().tolist()
                ],
                "tip_distance_env0_m": [
                    float(v) for v in torch.linalg.vector_norm(
                        env._tip_positions()[0, 1:] - env.tomato.data.root_pos_w[0].unsqueeze(0), dim=-1
                    ).detach().cpu().tolist()
                ],
                "thumb_tip_distance_env0_m": float(
                    torch.linalg.vector_norm(
                        env._tip_positions()[0, 0] - env.tomato.data.root_pos_w[0]
                    ).detach().cpu()
                ),
                "tomato_pos_env0_m": [
                    float(v) for v in env.tomato.data.root_pos_w[0].detach().cpu().tolist()
                ],
                "filtered_distal_env0": [
                    float(v) for v in env._v2_distal_action[0].detach().cpu().tolist()
                ],
                "active_joint_position_env0": [
                    float(v) for v in env.robot.data.joint_pos[0, env.active_joint_ids].detach().cpu().tolist()
                ],
                "prev_target_env0": [
                    float(v) for v in env.prev_targets[0].detach().cpu().tolist()
                ],
            }
        if args_cli.ablation == "freeze_distal":
            action = action.clone()
            action[:, 10:] = 0.0
        elif args_cli.ablation == "roots_only":
            action = action.clone()
            action[:, 5:] = 0.0
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated | truncated
        if bool(done.any()):
            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            if residual_controller is not None:
                # IsaacLab resets completed environments inside ``step``;
                # keep the staged controller's internal phase/freeze state in
                # lock-step with that reset.
                residual_controller.reset(done_ids)
            if residual_teacher is not None:
                residual_teacher.reset(done_ids)
            snapshot = env.consume_terminal_snapshots(done)
            code = snapshot["termination_code"]
            completed += done.sum()
            counts["success"] += (code == 1).sum()
            counts["drop"] += (code == 2).sum()
            counts["overload"] += (code == 3).sum()
            counts["damage"] += (code == 4).sum()
            counts["timeout"] += (code == 5).sum()
            counts["five_digit"] += (snapshot["contact_forces"] >= env.cfg.contact_threshold_n).all(-1).sum()
            # Terminal ``contact_forces`` answers whether all five are loaded
            # on the final frame.  The peak-force conjunction answers the
            # separate question of whether the episode ever observed all five
            # channels, which is essential for diagnosing transient thumb
            # contact without mislabeling it as sustained enclosure.
            counts["five_digit_observed"] += (
                snapshot["peak_force"] >= env.cfg.contact_threshold_n
            ).all(-1).sum()
            counts["palm_supported"] += (snapshot["palm_force"] >= env.cfg.palm_support_force_n).sum()
            counts["measured_palm_supported"] += (snapshot["measured_palm_force"] >= env.cfg.palm_support_force_n).sum()
            counts["distal_all"] += snapshot["strict_distal_participation"].all(-1).sum()
            counts["distal_contact_all"] += snapshot["distal_contact_observed"].all(-1).sum()
            # Keep the strict real-sensor conjunction separate from the
            # distance/virtual proxy conjunction above.  A per-digit observed
            # vector is not enough to know whether all four digits were
            # simultaneously observed in the same terminal episode.
            counts["measured_distal_contact_all"] += snapshot["measured_distal_contact_observed"].all(-1).sum()
            counts["palm_first"] += ((snapshot["strict_first_palm"] >= 0) & (snapshot["strict_first_three"] >= 0) & (snapshot["strict_first_palm"] + env.cfg.strict_palm_first_margin_steps <= snapshot["strict_first_three"])).sum()
            counts["measured_geometry"] += snapshot["strict_contact_geometry_measured"].any(-1).sum()
            counts["post_break_hold"] += (snapshot["stem_broken"] & (snapshot["post_break_hold"] >= env.cfg.success_hold_steps)).sum()
            force_sum += snapshot["contact_forces"].max(0).values
            # ``peak_force`` is the episode-level contact maximum retained by
            # the base environment; unlike the post-reset live tensor it is
            # safe to attribute to the terminal episode.
            peak_force_sum += (
                snapshot["peak_force"].sum(0)
                if "peak_force" in snapshot
                else snapshot["contact_forces"].sum(0)
            )
            peak_palm_force_sum += snapshot.get("peak_palm_force", snapshot["palm_force"]).sum()
            peak_force_max = torch.maximum(
                peak_force_max,
                (snapshot["peak_force"] if "peak_force" in snapshot else snapshot["contact_forces"]).max(0).values,
            )
            peak_palm_force_max = torch.maximum(
                peak_palm_force_max,
                snapshot.get("peak_palm_force", snapshot["palm_force"]).max(),
            )
            first_contact_sum += snapshot["first_contact_step"].clamp_min(0).float().sum(0)
            episode_steps = env.episode_length_buf.new_tensor(float(env.max_episode_length)).clamp_min(1.0)
            contact_occupancy_sum += snapshot["contact_steps"].sum(0) / episode_steps
            pre_break_occupancy_sum += snapshot["pre_break_contact_steps"].sum(0) / episode_steps
            post_break_occupancy_sum += snapshot["post_break_contact_steps"].sum(0) / episode_steps
            distal_observed_count += snapshot["distal_contact_observed"].float().sum(0)
            measured_distal_observed_count += snapshot["measured_distal_contact_observed"].float().sum(0)
            link3_force_sum += snapshot["link3_force"].mean(0)
            measured_link4_force_sum += snapshot["measured_link4_force"].mean(0)
            link4_force_sum += snapshot["link4_force"].mean(0)
            measured_link4_integral_sum += snapshot["measured_link4_force_integral"].sum(0)
            measured_contact_steps_sum += snapshot["measured_distal_contact_steps"].sum(0)
            measured_max_streak_sum += snapshot["measured_distal_max_streak"].sum(0)
            measured_palm_force_sum += snapshot["measured_palm_force"].mean()
            strict_hold_max_sum += snapshot["strict_hold_max"].float().sum()
            strict_hold_max_max = torch.maximum(
                strict_hold_max_max, snapshot["strict_hold_max"].float().max()
            )
            strict_coverage_max_sum += snapshot["strict_coverage_max"].float().sum()
            max_gap_sum += snapshot["max_lost_contact_gap"].mean(0)
            load = snapshot["force_integral"]
            load_share_sum += (load / load.sum(-1, keepdim=True).clamp_min(1.0e-6)).sum(0)
        if int(completed) >= args_cli.num_envs:
            break
    den = completed.float().clamp_min(1.0)
    report = {
        "version": "v2_independent_eval_1", "checkpoint": str(checkpoint_path), "checkpoint_version": checkpoint.get("version"),
        "profile": args_cli.profile, "profile_values": profile_definition(args_cli.profile), "seed": args_cli.seed,
        "ablation": args_cli.ablation,
        "residual_base_controller": bool(args_cli.residual_base_controller),
        "residual_scale": float(args_cli.residual_scale),
        "residual_teacher_coverage_blend": float(args_cli.residual_teacher_coverage_blend),
        "teacher_thumb_first_steps": int(args_cli.teacher_thumb_first_steps),
        "teacher_thumb_first_global_scale": float(args_cli.teacher_thumb_first_global_scale),
        "teacher_coverage_delta": list(parse_signed_four(args_cli.teacher_coverage_delta, "--teacher-coverage-delta")),
        "teacher_motion_smoothing_alpha": float(args_cli.teacher_motion_smoothing_alpha),
        "residual_mask": args_cli.residual_mask,
        "residual_phase_gate": not bool(args_cli.residual_no_phase_gate),
        "residual_approach_steps": int(args_cli.residual_approach_steps),
        "residual_per_digit_steps": int(args_cli.residual_per_digit_steps),
        "residual_hold_steps": int(args_cli.residual_hold_steps),
        "residual_approach_smoothing": float(args_cli.residual_approach_smoothing),
        "residual_thumb_delay_steps": int(args_cli.residual_thumb_delay_steps),
        "residual_thumb_ramp_steps": int(args_cli.residual_thumb_ramp_steps),
        "residual_thumb_first_steps": int(args_cli.residual_thumb_first_steps),
        "residual_thumb_first_global_scale": float(args_cli.residual_thumb_first_global_scale),
        "residual_lower_force_per_digit": (
            [float(v) for v in parse_four(args_cli.residual_lower_force_per_digit, "--residual-lower-force-per-digit")]
            if args_cli.residual_lower_force_per_digit is not None else [0.18, 0.18, 0.18, 0.18]
        ),
        "residual_upper_force_per_digit": (
            [float(v) for v in parse_four(args_cli.residual_upper_force_per_digit, "--residual-upper-force-per-digit")]
            if args_cli.residual_upper_force_per_digit is not None else [0.80, 0.80, 0.80, 0.80]
        ),
        "residual_insertion_step_fraction": float(args_cli.residual_insertion_step_fraction),
        "residual_retreat_fraction": float(args_cli.residual_retreat_fraction),
        "residual_insertion_step_per_digit": (
            [float(v) for v in parse_four(args_cli.residual_insertion_step_per_digit, "--residual-insertion-step-per-digit")]
            if args_cli.residual_insertion_step_per_digit is not None else None
        ),
        "residual_retreat_per_digit": (
            [float(v) for v in parse_four(args_cli.residual_retreat_per_digit, "--residual-retreat-per-digit")]
            if args_cli.residual_retreat_per_digit is not None else None
        ),
        "residual_thumb_hold_target_offset": list(parse_signed_pair(
            args_cli.residual_thumb_hold_offset, "--residual-thumb-hold-offset"
        )),
        "residual_thumb_lower_force_n": float(args_cli.residual_thumb_lower_force),
        "residual_thumb_upper_force_n": float(args_cli.residual_thumb_upper_force),
        "residual_thumb_search_amplitude": list(parse_signed_pair(
            args_cli.residual_thumb_search_amplitude, "--residual-thumb-search-amplitude"
        )),
        "residual_thumb_search_period_steps": int(args_cli.residual_thumb_search_period),
        "residual_thumb_contact_gate_steps": int(args_cli.residual_thumb_contact_gate_steps),
        "residual_thumb_contact_latch_force_n": float(args_cli.residual_thumb_contact_latch_force),
        "residual_thumb_contact_latch_steps": int(args_cli.residual_thumb_contact_latch_steps),
        "residual_thumb_contact_latch_freeze_coarse": bool(args_cli.residual_thumb_contact_latch_freeze_coarse),
        "residual_simultaneous_reach": bool(args_cli.residual_simultaneous_reach),
        "resolved_contact_geometry": {
            "tomato_contact_offset_m": float(env.cfg.tomato_contact_offset_m),
            "tomato_rest_offset_m": float(env.cfg.tomato_rest_offset_m),
            "fingertip_proxy_radius_m": float(env.cfg.fingertip_collision_proxy_radius_m),
            "physical_fingertip_proxy": bool(env.cfg.fingertip_collision_proxy_physical),
            "fingertip_proxy_contact_offset_m": float(env.cfg.fingertip_proxy_contact_offset_m),
            "fingertip_proxy_rest_offset_m": float(env.cfg.fingertip_proxy_rest_offset_m),
            "proxy_only_collision": bool(env.cfg.proxy_only_collision),
            "accepted_mass_kg": [float(v) for v in env.cfg.accepted_mass_kg],
            "density_range_kg_m3": [float(v) for v in env.cfg.density_range_kg_m3],
            "axis_range_m": [float(v) for v in env.cfg.axis_range_m],
            "isotropic_axes": bool(env.cfg.isotropic_axes),
            "contact_stiffness_n_m": float(env.cfg.tomato_cfg.spawn.physics_material.compliant_contact_stiffness),
            "per_digit_root_residual_span_fraction": (
                None
                if env.cfg.per_digit_root_residual_span_fraction is None
                else float(env.cfg.per_digit_root_residual_span_fraction)
            ),
            "pregrasp_settle_steps": int(getattr(env.cfg, "pregrasp_settle_steps", 0)),
            "tomato_reset_clearance_m": float(getattr(env.cfg, "tomato_reset_clearance_m", 0.0)),
            "palm_cup_inner_half_extent_m": [float(v) for v in env.cfg.palm_cup_inner_half_extent_m],
            "palm_cup_rim_height_m": float(env.cfg.palm_cup_rim_height_m),
            "palm_first_distal_gate": bool(getattr(env.cfg, "palm_first_distal_gate", False)),
            "per_digit_distal_gate": bool(getattr(env.cfg, "per_digit_distal_gate", False)),
            "coarse_action_smoothing_alpha": float(getattr(env.cfg, "coarse_action_smoothing_alpha", 1.0)),
            "distal_action_filter_alpha": (
                [float(v) for v in env.cfg.distal_action_filter_alpha]
                if isinstance(getattr(env.cfg, "distal_action_filter_alpha", 0.70), (tuple, list))
                else float(getattr(env.cfg, "distal_action_filter_alpha", 0.70))
            ),
            "command_rate_limit_rad_s": float(getattr(env.cfg, "command_rate_limit_rad_s", 0.0)),
            "safe_force_n": float(env.cfg.safe_force_n),
            "force_ceiling_n": [float(v) for v in env.cfg.force_ceiling_n],
            "distal_proximal_reach_assist_gain_per_digit": (
                None
                if getattr(env.cfg, "distal_proximal_reach_assist_gain_per_digit", None) is None
                else [float(v) for v in env.cfg.distal_proximal_reach_assist_gain_per_digit]
            ),
        },
        "resolved_object_randomization": {
            "object_mass_kg": {
                "min": float(env.object_mass.min().cpu()),
                "max": float(env.object_mass.max().cpu()),
                "mean": float(env.object_mass.mean().cpu()),
            },
            "tomato_axes_m": {
                "min": [float(v) for v in env.tomato_axes.amin(0).cpu().tolist()],
                "max": [float(v) for v in env.tomato_axes.amax(0).cpu().tolist()],
                "mean": [float(v) for v in env.tomato_axes.mean(0).cpu().tolist()],
            },
            "tomato_position_env0_m": [float(v) for v in env.tomato.data.root_pos_w[0].cpu().tolist()],
            "palm_support_point_env0_m": [float(v) for v in env.palm_support_point[0].cpu().tolist()],
            "active_joint_position_env0": [float(v) for v in env.robot.data.joint_pos[0, env.active_joint_ids].cpu().tolist()],
        },
        "num_envs": args_cli.num_envs, "steps": args_cli.steps,
        "episode_length_s": float(env.cfg.episode_length_s),
        "max_episode_length": int(env.max_episode_length),
        "stage": args_cli.stage, "completed_episodes": int(completed),
        **{f"{name}_rate": float(value / den) for name, value in counts.items()},
        "digit_peak_force_n_mean": [float(value / den) for value in force_sum],
        "episode_peak_force_n_mean": [float(value / den) for value in peak_force_sum],
        "episode_peak_palm_force_n_mean": float(peak_palm_force_sum / den),
        "episode_peak_force_n_max": [float(value) for value in peak_force_max],
        "episode_peak_palm_force_n_max": float(peak_palm_force_max),
        "first_contact_step_mean": [float(value / den) for value in first_contact_sum],
        "contact_occupancy_mean": [float(value / den) for value in contact_occupancy_sum],
        "pre_break_contact_occupancy_mean": [float(value / den) for value in pre_break_occupancy_sum],
        "post_break_contact_occupancy_mean": [float(value / den) for value in post_break_occupancy_sum],
        "distal_contact_observed_rate": [float(value / den) for value in distal_observed_count],
        "measured_distal_contact_observed_rate": [float(value / den) for value in measured_distal_observed_count],
        "link3_force_mean_n": [float(value / den) for value in link3_force_sum],
        "measured_link4_force_mean_n": [float(value / den) for value in measured_link4_force_sum],
        "link4_force_mean_n": [float(value / den) for value in link4_force_sum],
        "measured_link4_force_integral_n_s_mean": [float(value / den) for value in measured_link4_integral_sum],
        "measured_distal_contact_steps_mean": [float(value / den) for value in measured_contact_steps_sum],
        "measured_distal_max_streak_mean": [float(value / den) for value in measured_max_streak_sum],
        "measured_palm_force_mean_n": float(measured_palm_force_sum / den),
        "strict_hold_max_mean_steps": float(strict_hold_max_sum / den),
        "strict_hold_max_steps": int(strict_hold_max_max.item()),
        "strict_coverage_max_mean_deg": float(strict_coverage_max_sum / den),
        "max_lost_contact_gap_steps_mean": [float(value / den) for value in max_gap_sum],
        "force_integral_share_mean": [float(value / den) for value in load_share_sum],
        "actor_action_mean": [float(value / (max(args_cli.steps, 1) * args_cli.num_envs)) for value in action_sum],
        "actor_action_abs_mean": [float(value / (max(args_cli.steps, 1) * args_cli.num_envs)) for value in action_abs_sum],
        "actor_action_max": [float(value) for value in action_max],
        "hardware_command_allowed": False,
        "action_trace": action_trace,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
