"""Vectorized action-space search for a physically reachable link4 enclosure.

The search deliberately optimizes measured straight-finger link4 contact, not
the strict fallback geometry or a learned reward.  It is used to distinguish a
bad policy prior from an unreachable collision/sensor model before PPO spends
updates on the wrong manifold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--profile", choices=("v1_nominal_reliable", "v2_band1_match", "v2_band2_shape", "v2_band3_sensor_actuator", "v2_band4_stress"), default="v1_nominal_reliable")
parser.add_argument("--assist-gain", type=float, default=None, help="override the v2 distal-to-proximal reach coupling gain")
parser.add_argument("--assist-per-digit", type=str, default=None,
                    help="override four per-digit distal-to-proximal reach gains")
parser.add_argument("--smoothing-alpha", type=float, default=None, help="override v2 coarse action smoothing")
parser.add_argument("--palm-first-gate", action="store_true",
                    help="gate distal residuals until the palm support age is stable")
parser.add_argument("--palm-first-coarse-max-scale", type=float, default=None,
                    help="optional bound on coarse/root action after palm support")
parser.add_argument("--palm-first-ramp-steps", type=int, default=None)
parser.add_argument("--palm-first-motion-hold", action="store_true",
                    help="damp fresh coarse motion when the fruit moves outward")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--approach-steps", type=int, default=150)
parser.add_argument("--hold-steps", type=int, default=80)
parser.add_argument("--seed", type=int, default=20260831)
parser.add_argument("--distal-mode", choices=("signed", "positive"), default="signed")
parser.add_argument("--base-action", type=str, default=None, help="14 comma-separated values for a local search")
parser.add_argument("--noise-scale", type=float, default=0.25)
parser.add_argument("--vary-dims", type=str, default=None, help="comma-separated action indices to vary around --base-action")
parser.add_argument("--physical-fingertip-proxy", action="store_true")
parser.add_argument("--proxy-only-collision", action="store_true",
                    help="diagnostic: disable vendor hand colliders and keep only v2 fingertip proxy spheres")
parser.add_argument("--distal-filter-alpha", type=float, default=None,
                    help="override v2 distal residual smoothing to match a geometry probe")
parser.add_argument("--disable-virtual-distal-contact", action="store_true")
parser.add_argument("--fingertip-proxy-radius", type=float, default=None)
parser.add_argument("--fingertip-proxy-contact-offset", type=float, default=None,
                    help="simulation-only PhysX contact offset for the authored proxy")
parser.add_argument("--fingertip-proxy-rest-offset", type=float, default=None,
                    help="simulation-only PhysX rest offset for the authored proxy")
parser.add_argument("--tomato-contact-offset", type=float, default=None,
                    help="simulation-only tomato contact offset")
parser.add_argument("--tomato-rest-offset", type=float, default=None,
                    help="simulation-only tomato rest offset")
parser.add_argument("--per-digit-root-span", type=float, default=None,
                    help="optional v2-only root residual span for reachability diagnostics")
parser.add_argument("--cup-half-extent", type=str, default=None,
                    help="simulation-only palm rim inner half extents x,y in metres")
parser.add_argument("--palm-support-contact-offset", type=float, default=None)
parser.add_argument("--palm-support-rest-offset", type=float, default=None)
parser.add_argument("--reset-clearance", type=float, default=None)
parser.add_argument("--maximum-stem-force-n", type=float, default=None,
                    help="optional stem force termination threshold; None preserves profile default")
parser.add_argument("--maximum-stem-torque-nm", type=float, default=None,
                    help="optional stem torque termination threshold; None preserves profile default")
parser.add_argument("--safe-force-n", type=float, default=None,
                    help="optional palm safety reference in newtons")
parser.add_argument("--force-ceiling", type=str, default=None,
                    help="optional five comma-separated force ceilings")
parser.add_argument("--debug-collision-digit", choices=("", "palm", "all", "index", "middle", "ring", "little"), default="")
parser.add_argument("--fixed-fruit", action="store_true",
                    help="hold the tomato kinematic for a pure physical contact audit")
parser.add_argument("--axis-range-m", type=str, default=None,
                    help="override isotropic tomato semi-axis range as one or two metres")
parser.add_argument("--contact-stiffness-n-m", type=float, default=None,
                    help="override the authored tomato compliant-contact stiffness for this diagnostic")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import (
    V2EnvelopingPhysxTomatoGraspEnv,
    V2EnvelopingPhysxTomatoGraspEnvCfg,
)


def candidate_bank(n: int, device: torch.device, seed: int, distal_mode: str, base_action: str | None, noise_scale: float, vary_dims: str | None) -> torch.Tensor:
    """Sample structured synergy/residual actions without a fixed pose script."""
    gen = torch.Generator(device=device).manual_seed(seed)
    actions = 2.0 * torch.rand((n, 14), generator=gen, device=device) - 1.0
    if base_action is not None:
        values = [float(v.strip()) for v in base_action.split(",") if v.strip()]
        if len(values) != 14:
            raise ValueError("--base-action requires exactly 14 comma-separated values")
        center = torch.tensor(values, dtype=torch.float32, device=device)
        noise = torch.randn((n, 14), generator=gen, device=device)
        if vary_dims is not None:
            mask = torch.zeros(14, dtype=torch.bool, device=device)
            for value in vary_dims.split(","):
                if value.strip():
                    index = int(value.strip())
                    if not 0 <= index < 14:
                        raise ValueError("--vary-dims indices must lie in [0, 13]")
                    mask[index] = True
            noise *= mask
        actions = (center.unsqueeze(0) + noise_scale * noise).clamp(-1.0, 1.0)
    if base_action is None:
        # Put half of the samples in a physically plausible enclosure
        # corridor; retain the uniform half so a reachable branch outside our
        # prior is not silently excluded.
        half = n // 2
        actions[:half, 0] = torch.empty(half, device=device).uniform_(0.15, 1.0, generator=gen)
        actions[:half, 1] = torch.empty(half, device=device).uniform_(-0.8, 1.0, generator=gen)
        actions[:half, 2] = torch.empty(half, device=device).uniform_(-1.0, 1.0, generator=gen)
        actions[:half, 3] = torch.empty(half, device=device).uniform_(-1.0, 1.0, generator=gen)
        actions[:half, 4] = torch.empty(half, device=device).uniform_(0.0, 1.0, generator=gen)
        actions[:half, 5:10] = torch.empty((half, 5), device=device).uniform_(-1.0, 1.0, generator=gen)
        low = 0.0 if distal_mode == "positive" else -1.0
        actions[:half, 10:] = torch.empty((half, 4), device=device).uniform_(low, 1.0, generator=gen)
    return actions.clamp(-1.0, 1.0)


def parse_pair(raw: str) -> tuple[float, float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 2 or any(value <= 0.0 for value in values):
        raise ValueError("--cup-half-extent requires two positive metre values")
    return float(values[0]), float(values[1])


def parse_four(raw: str, *, name: str) -> tuple[float, float, float, float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 4 or any(value < 0.0 for value in values):
        raise ValueError(f"{name} requires four non-negative values")
    return tuple(float(value) for value in values)


def parse_force_ceiling(raw: str) -> tuple[float, float, float, float, float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 5 or any(value <= 0.0 for value in values):
        raise ValueError("--force-ceiling requires five positive values")
    return tuple(float(value) for value in values)


def parse_axis_range(raw: str) -> tuple[float, float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) == 1:
        values = [values[0], values[0]]
    if len(values) != 2 or any(value <= 0.0 for value in values) or values[0] > values[1]:
        raise ValueError("--axis-range-m requires one positive value or min,max with 0 < min <= max")
    return float(values[0]), float(values[1])


@torch.inference_mode()
def main() -> None:
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.episode_length_s = 20.0
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)
    cfg = apply_profile(cfg, args.profile, num_envs=args.num_envs, seed=args.seed)
    if args.axis_range_m is not None:
        axis_range = parse_axis_range(args.axis_range_m)
        cfg.axis_range_m = axis_range
        cfg.isotropic_axes = True
        # A 45--50 mm semi-axis fruit is heavier than the nominal 37.5 mm
        # reference at the same density.  Extend only the diagnostic mass
        # prior for this explicit geometry experiment; normal profiles and
        # all training defaults remain unchanged.
        if axis_range[1] > 0.040:
            cfg.accepted_mass_kg = (0.050, 0.500)
        cfg.events["tomato_geometry"].params["axis_range_m"] = axis_range
        cfg.events["tomato_geometry"].params["isotropic_axes"] = True
        cfg.events["tomato_geometry"].params["accepted_mass_kg"] = cfg.accepted_mass_kg
    if args.contact_stiffness_n_m is not None:
        if args.contact_stiffness_n_m <= 0.0:
            raise ValueError("--contact-stiffness-n-m must be positive")
        material = cfg.tomato_cfg.spawn.physics_material
        material.compliant_contact_stiffness = float(args.contact_stiffness_n_m)
    if args.fixed_fruit:
        cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
        cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    if args.assist_gain is not None:
        cfg.distal_proximal_reach_assist_gain = float(args.assist_gain)
        cfg.distal_proximal_reach_assist_gain_per_digit = None
    if args.assist_per_digit is not None:
        cfg.distal_proximal_reach_assist_gain_per_digit = parse_four(
            args.assist_per_digit, name="--assist-per-digit"
        )
        cfg.distal_proximal_reach_assist_gain = 0.0
    if args.smoothing_alpha is not None:
        cfg.coarse_action_smoothing_alpha = float(args.smoothing_alpha)
    cfg.palm_first_distal_gate = bool(args.palm_first_gate)
    if args.palm_first_coarse_max_scale is not None:
        cfg.palm_first_coarse_max_scale = float(args.palm_first_coarse_max_scale)
    if args.palm_first_ramp_steps is not None:
        cfg.palm_first_ramp_steps = int(args.palm_first_ramp_steps)
    cfg.palm_first_motion_hold_enabled = bool(args.palm_first_motion_hold)
    cfg.fingertip_collision_proxy_physical = bool(args.physical_fingertip_proxy)
    cfg.proxy_only_collision = bool(args.proxy_only_collision)
    if args.distal_filter_alpha is not None:
        cfg.distal_action_filter_alpha = float(args.distal_filter_alpha)
    if args.fingertip_proxy_radius is not None:
        cfg.fingertip_collision_proxy_radius_m = float(args.fingertip_proxy_radius)
    if args.fingertip_proxy_contact_offset is not None:
        cfg.fingertip_proxy_contact_offset_m = float(args.fingertip_proxy_contact_offset)
    if args.fingertip_proxy_rest_offset is not None:
        cfg.fingertip_proxy_rest_offset_m = float(args.fingertip_proxy_rest_offset)
    if args.tomato_contact_offset is not None:
        cfg.tomato_contact_offset_m = float(args.tomato_contact_offset)
    if args.tomato_rest_offset is not None:
        cfg.tomato_rest_offset_m = float(args.tomato_rest_offset)
    if args.per_digit_root_span is not None:
        cfg.per_digit_root_residual_span_fraction = float(args.per_digit_root_span)
    if args.disable_virtual_distal_contact:
        cfg.virtual_distal_contact_enabled = False
    if args.cup_half_extent is not None:
        cfg.palm_cup_inner_half_extent_m = parse_pair(args.cup_half_extent)
    if args.palm_support_contact_offset is not None:
        cfg.palm_support_contact_offset_m = float(args.palm_support_contact_offset)
    if args.palm_support_rest_offset is not None:
        cfg.palm_support_rest_offset_m = float(args.palm_support_rest_offset)
    if args.reset_clearance is not None:
        cfg.tomato_reset_clearance_m = float(args.reset_clearance)
    if args.maximum_stem_force_n is not None:
        cfg.maximum_stem_force_n = float(args.maximum_stem_force_n)
    if args.maximum_stem_torque_nm is not None:
        cfg.maximum_stem_torque_nm = float(args.maximum_stem_torque_nm)
    if args.safe_force_n is not None:
        cfg.safe_force_n = float(args.safe_force_n)
    if args.force_ceiling is not None:
        cfg.force_ceiling_n = parse_force_ceiling(args.force_ceiling)
    cfg.debug_support_only_collision = bool(args.debug_collision_digit)
    cfg.debug_collision_digit = args.debug_collision_digit
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)
    if args.fixed_fruit:
        cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
        cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)
    actions = candidate_bank(args.num_envs, env.device, args.seed, args.distal_mode, args.base_action, args.noise_scale, args.vary_dims)
    open_action = torch.zeros_like(actions)
    open_action[:, :10] = torch.tensor(
        (-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        device=env.device,
    )
    max_link4 = torch.zeros(args.num_envs, 4, device=env.device)
    max_link3 = torch.zeros(args.num_envs, 4, device=env.device)
    max_group = torch.zeros(args.num_envs, 5, device=env.device)
    max_thumb_sensor = torch.zeros(args.num_envs, 2, device=env.device)
    min_distance = torch.full((args.num_envs, 4), float("inf"), device=env.device)
    max_palm = torch.zeros(args.num_envs, device=env.device)
    near_surface = torch.zeros(args.num_envs, 4, dtype=torch.bool, device=env.device)
    link4_seen = torch.zeros(args.num_envs, 4, dtype=torch.bool, device=env.device)
    measured_link4_seen = torch.zeros(args.num_envs, 4, dtype=torch.bool, device=env.device)
    # A max/seen contact is not enough for a grasp: a candidate may only
    # brush a fingertip once while the fruit settles.  Track occupancy over
    # the explicit hold window so ranking prefers a sustained envelope.
    hold_contact_steps = torch.zeros(args.num_envs, 4, device=env.device)
    hold_all_four_steps = torch.zeros(args.num_envs, device=env.device)
    overload_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
    max_coverage = torch.zeros(args.num_envs, device=env.device)
    min_thumb_distance = torch.full((args.num_envs,), float("inf"), device=env.device)
    max_hold = torch.zeros(args.num_envs, device=env.device)
    all_five_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
    measured_all_five_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
    invalid_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
    min_height = torch.full((args.num_envs,), float("inf"), device=env.device)
    max_center_distance = torch.zeros(args.num_envs, device=env.device)
    total = args.approach_steps + args.hold_steps
    for step in range(total):
        f = float(step + 1) / max(args.approach_steps, 1) if step < args.approach_steps else 1.0
        env.step(open_action + f * (actions - open_action))
        tip = env._tip_positions()[:, 1:]
        dist = torch.linalg.vector_norm(tip - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1)
        thumb_dist = torch.linalg.vector_norm(
            env._tip_positions()[:, 0] - env.tomato.data.root_pos_w, dim=-1
        )
        min_thumb_distance = torch.minimum(min_thumb_distance, thumb_dist)
        min_distance = torch.minimum(min_distance, dist)
        near_surface |= dist < 0.045
        max_link3 = torch.maximum(max_link3, env._v2_link3_force[:, 1:])
        max_link4 = torch.maximum(max_link4, env._v2_link4_force[:, 1:])
        max_group = torch.maximum(max_group, env.contact_forces)
        max_thumb_sensor[:, 0] = torch.maximum(
            max_thumb_sensor[:, 0],
            torch.linalg.vector_norm(env.scene.sensors["thumb_contact"].data.net_forces_w[:, 0], dim=-1).nan_to_num(0.0),
        )
        max_thumb_sensor[:, 1] = torch.maximum(
            max_thumb_sensor[:, 1],
            torch.linalg.vector_norm(env.scene.sensors["thumb_contact_link4"].data.net_forces_w[:, 0], dim=-1).nan_to_num(0.0),
        )
        max_palm = torch.maximum(max_palm, env.palm_force)
        link4_seen |= env._v2_link4_force[:, 1:] >= env.cfg.contact_threshold_n
        measured_link4_seen |= env._v2_measured_link4_force[:, 1:] >= env.cfg.contact_threshold_n
        if step >= args.approach_steps:
            measured_active = env._v2_measured_link4_force[:, 1:] >= env.cfg.contact_threshold_n
            hold_contact_steps += measured_active.float()
            hold_all_four_steps += measured_active.all(-1).float()
        overload_seen |= env.overloaded
        max_coverage = torch.maximum(max_coverage, env.strict_coverage_current)
        max_hold = torch.maximum(max_hold, env.strict_hold_streak.float())
        # The aggregate contact vector can include a legacy/virtual channel;
        # strict five-finger evidence must use the v2 measured sensor vector,
        # including the corrected thumb_contact mapping.
        all_five_seen |= (env.contact_forces >= env.cfg.contact_threshold_n).all(-1)
        measured_all_five_seen |= (
            env._v2_measured_link4_force >= env.cfg.contact_threshold_n
        ).all(-1)
        tomato_pos = env.tomato.data.root_pos_w
        center_distance = torch.linalg.vector_norm(tomato_pos - env.grasp_center, dim=-1)
        min_height = torch.minimum(min_height, tomato_pos[:, 2])
        max_center_distance = torch.maximum(max_center_distance, center_distance)
        invalid_seen |= (tomato_pos[:, 2] < env.cfg.drop_height_m) | (center_distance > env.cfg.max_center_distance_m)

    # Reward measured contact coverage while penalizing excessive force and
    # retaining a distance tie-breaker.  This ranking is diagnostic only.
    contact_score = (max_link4 >= env.cfg.contact_threshold_n).float().sum(-1)
    force_penalty = (max_link4.clamp_min(0.0) / 8.0).sum(-1)
    distance_score = (0.06 - min_distance).clamp_min(0.0).sum(-1) / 0.06
    # Prefer an actually retained five-finger envelope, not just a single
    # link4 impulse.  Palm overload is penalized separately so a high-force
    # candidate cannot outrank a lower-load hold corridor.
    palm_penalty = (max_palm / 12.0).clamp_min(0.0)
    hold_occupancy = hold_contact_steps / max(float(args.hold_steps), 1.0)
    # Use the weakest digit as the envelope bottleneck.  This keeps a high
    # score from being produced by three stable fingers plus one transient
    # contact, while still retaining the old reach/force tie breakers.
    weakest_hold = hold_occupancy.min(-1).values
    score = (
        contact_score * 4.0
        + 40.0 * weakest_hold
        + 10.0 * (hold_all_four_steps / max(float(args.hold_steps), 1.0))
        + distance_score
        + 0.05 * max_coverage
        + 1.0 * max_hold
        - force_penalty
        - 4.0 * palm_penalty
        - 20.0 * overload_seen.float()
        - 30.0 * invalid_seen.float()
    )
    order = torch.argsort(score, descending=True)
    best_hold_idx = int(torch.argmax(weakest_hold).item())
    safe_weakest = torch.where(
        ~overload_seen & ~invalid_seen,
        weakest_hold,
        torch.full_like(weakest_hold, -1.0),
    )
    best_safe_idx = int(torch.argmax(safe_weakest).item())
    best_coverage_idx = int(torch.argmax(max_coverage).item())
    best_thumb_idx = int(torch.argmin(min_thumb_distance).item())
    all_five_coverage = torch.where(
        measured_all_five_seen,
        max_coverage,
        torch.full_like(max_coverage, -1.0),
    )
    best_all_five_coverage_idx = int(torch.argmax(all_five_coverage).item())

    def _stats(values: torch.Tensor) -> dict:
        values = values.detach().float()
        return {
            "min": [float(v) for v in values.amin(0).cpu().reshape(-1)],
            "max": [float(v) for v in values.amax(0).cpu().reshape(-1)],
            "mean": [float(v) for v in values.mean(0).cpu().reshape(-1)],
        }

    def _candidate_env_spec(idx: int) -> dict:
        return {
            "object_mass_kg": float(env.object_mass[idx].cpu()),
            "tomato_axes_m": [float(v) for v in env.tomato_axes[idx].cpu()],
            "tomato_density_kg_m3": float(env._tomato_density_kg_m3[idx].cpu()),
            "object_friction": float(env.object_friction[idx].cpu()),
            "object_stiffness_n_m": float(env.object_stiffness[idx].cpu()),
            "tomato_position_m": [float(v) for v in env.tomato.data.root_pos_w[idx].cpu()],
        }

    def candidate_summary(idx: int) -> dict:
        return {
            "candidate_id": int(idx),
            "action": [float(v) for v in actions[idx].cpu()],
            "hold_contact_occupancy": [float(v) for v in hold_occupancy[idx].cpu()],
            "hold_all_four_occupancy": float((hold_all_four_steps[idx] / max(float(args.hold_steps), 1.0)).cpu()),
            "weakest_hold_occupancy": float(weakest_hold[idx].cpu()),
            "max_link4_force_n": [float(v) for v in max_link4[idx].cpu()],
            "max_palm_force_n": float(max_palm[idx].cpu()),
            "max_coverage_deg": float(max_coverage[idx].cpu()),
            "min_thumb_distance_m": float(min_thumb_distance[idx].cpu()),
            "max_strict_hold_steps": float(max_hold[idx].cpu()),
            "all_five_contact_seen": bool(measured_all_five_seen[idx].cpu()),
            "aggregate_all_five_contact_seen": bool(all_five_seen[idx].cpu()),
            "overload_seen": bool(overload_seen[idx].cpu()),
            "invalid_in_hand_seen": bool(invalid_seen[idx].cpu()),
            "resolved_env": _candidate_env_spec(idx),
        }
    rows = []
    for idx in order[: min(30, args.num_envs)].tolist():
        rows.append({
            "candidate_id": int(idx),
            "score": float(score[idx].cpu()),
            "action": [float(v) for v in actions[idx].cpu()],
            "max_link3_force_n": [float(v) for v in max_link3[idx].cpu()],
            "max_link4_force_n": [float(v) for v in max_link4[idx].cpu()],
            "max_aggregate_force_n": [float(v) for v in max_group[idx].cpu()],
            "max_thumb_sensor_force_n": [float(v) for v in max_thumb_sensor[idx].cpu()],
            "min_tip_distance_m": [float(v) for v in min_distance[idx].cpu()],
            "near_surface": [bool(v) for v in near_surface[idx].cpu()],
            "link4_contact_seen": [bool(v) for v in link4_seen[idx].cpu()],
            "measured_link4_contact_seen": [bool(v) for v in measured_link4_seen[idx].cpu()],
            "hold_contact_occupancy": [float(v) for v in hold_occupancy[idx].cpu()],
            "hold_all_four_occupancy": float((hold_all_four_steps[idx] / max(float(args.hold_steps), 1.0)).cpu()),
            "weakest_hold_occupancy": float(weakest_hold[idx].cpu()),
            "max_palm_force_n": float(max_palm[idx].cpu()),
            "overload_seen": bool(overload_seen[idx].cpu()),
            "max_coverage_deg": float(max_coverage[idx].cpu()),
            "max_strict_hold_steps": float(max_hold[idx].cpu()),
            "all_five_contact_seen": bool(measured_all_five_seen[idx].cpu()),
            "aggregate_all_five_contact_seen": bool(all_five_seen[idx].cpu()),
            "min_thumb_distance_m": float(min_thumb_distance[idx].cpu()),
            "min_tomato_height_m": float(min_height[idx].cpu()),
            "max_center_distance_m": float(max_center_distance[idx].cpu()),
            "invalid_in_hand_seen": bool(invalid_seen[idx].cpu()),
        })
    report = {
        "version": "link4_action_search_v1",
        "profile": args.profile,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "approach_steps": args.approach_steps,
        "hold_steps": args.hold_steps,
        "resolved_config": {
            "maximum_stem_force_n": float(cfg.maximum_stem_force_n),
            "maximum_stem_torque_nm": float(cfg.maximum_stem_torque_nm),
            "virtual_distal_contact_enabled": bool(cfg.virtual_distal_contact_enabled),
            "virtual_distal_contact_gain_n_per_m": float(cfg.virtual_distal_contact_gain_n_per_m),
            "distal_proximal_reach_assist_gain": float(cfg.distal_proximal_reach_assist_gain),
            "distal_proximal_reach_assist_gain_per_digit": None if cfg.distal_proximal_reach_assist_gain_per_digit is None else [float(v) for v in cfg.distal_proximal_reach_assist_gain_per_digit],
            "distal_action_filter_alpha": float(cfg.distal_action_filter_alpha),
            "coarse_action_smoothing_alpha": float(cfg.coarse_action_smoothing_alpha),
            "palm_first_distal_gate": bool(cfg.palm_first_distal_gate),
            "palm_first_coarse_max_scale": float(cfg.palm_first_coarse_max_scale),
            "palm_first_ramp_steps": int(cfg.palm_first_ramp_steps),
            "palm_first_motion_hold_enabled": bool(cfg.palm_first_motion_hold_enabled),
            "fingertip_collision_proxy_physical": bool(cfg.fingertip_collision_proxy_physical),
            "fingertip_collision_proxy_radius_m": float(cfg.fingertip_collision_proxy_radius_m),
            "contact_threshold_n": float(cfg.contact_threshold_n),
            "safe_force_n": float(cfg.safe_force_n),
            "force_ceiling_n": [float(v) for v in cfg.force_ceiling_n],
            "axis_range_m": [float(v) for v in cfg.axis_range_m],
            "isotropic_axes": bool(cfg.isotropic_axes),
            "contact_stiffness_n_m": float(cfg.tomato_cfg.spawn.physics_material.compliant_contact_stiffness),
        },
        "resolved_object_randomization": {
            "object_mass_kg": _stats(env.object_mass),
            "tomato_axes_m": _stats(env.tomato_axes),
            "tomato_density_kg_m3": _stats(env._tomato_density_kg_m3),
            "object_friction": _stats(env.object_friction),
            "object_stiffness_n_m": _stats(env.object_stiffness),
        },
        "ranked_candidates": rows,
        "best_hold_candidate": candidate_summary(best_hold_idx),
        "best_safe_hold_candidate": candidate_summary(best_safe_idx),
        "best_coverage_candidate": candidate_summary(best_coverage_idx),
        "best_thumb_distance_candidate": candidate_summary(best_thumb_idx),
        "best_all_five_coverage_candidate": candidate_summary(best_all_five_coverage_idx),
        "all_five_contact_seen_count": int(measured_all_five_seen.sum().cpu()),
        "aggregate_all_five_contact_seen_count": int(all_five_seen.sum().cpu()),
        "any_link4_contact_rate": [float(v) for v in link4_seen.float().amax(0).cpu()],
        "any_measured_link4_contact_rate": [float(v) for v in measured_link4_seen.float().amax(0).cpu()],
        "hold_contact_occupancy_mean": [float(v) for v in hold_occupancy.mean(0).cpu()],
        "hold_contact_occupancy_best_min": float(weakest_hold.max().cpu()),
        "hold_all_four_occupancy_best": float((hold_all_four_steps / max(float(args.hold_steps), 1.0)).max().cpu()),
        "best_link4_force_n": [float(v) for v in max_link4.amax(0).cpu()],
        "max_thumb_sensor_force_n": [float(v) for v in max_thumb_sensor.amax(0).cpu()],
        "hardware_command_allowed": False,
        "interpretation": "diagnostic_only_no_training_or_hardware_command",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
