"""Simulation-only low-force contact/hold/retreat primitive.

This probe is deliberately not a policy, demonstration, or hardware command.
It compares a fixed multi-channel reach reference with a conservative
per-digit feedback loop using raw link4 force.  A finger advances toward its
weak reach target only while its measured force is below a low band; once the
force reaches the band, that finger is held, and an upper-band violation
retreats the matching proximal/distal channels toward the open action.

The object is dynamic and a physical fingertip proxy is enabled by default so
the result diagnoses contact impulse, retention, and force feedback rather than
only virtual shell reachability.  The proxy settings remain simulation-only.
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
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--approach-steps", type=int, default=120)
parser.add_argument("--per-digit-steps", type=int, default=50)
parser.add_argument("--hold-steps", type=int, default=100)
parser.add_argument("--contact-target-n", type=float, default=0.18)
parser.add_argument("--retreat-limit-n", type=float, default=0.80)
parser.add_argument("--contact-target-per-digit", type=str, default="",
                    help="optional four lower force bands: index,middle,ring,little")
parser.add_argument("--retreat-limit-per-digit", type=str, default="",
                    help="optional four upper force bands: index,middle,ring,little")
parser.add_argument("--contact-streak", type=int, default=5)
parser.add_argument("--step-fraction", type=float, default=0.08)
parser.add_argument("--retreat-fraction", type=float, default=0.12)
parser.add_argument("--command-rate-rad-s", type=float, default=0.45)
parser.add_argument("--approach-smoothing", type=float, default=0.10,
                    help="fraction of the reference ramp applied per step (diagnostic)")
parser.add_argument("--distal-filter-alpha", type=float, default=0.999,
                    help="diagnostic v2 distal command filter alpha")
parser.add_argument("--assist-gain", type=float, default=0.0,
                    help="diagnostic per-digit proximal reach assist scalar; 0 disables")
parser.add_argument("--assist-per-digit", type=str, default="",
                    help="optional four per-digit reach assist gains")
parser.add_argument("--proxy-radius", type=float, default=0.004)
parser.add_argument("--proxy-only-collision", action="store_true",
                    help="simulation-only: disable vendor hand colliders and keep only v2 fingertip proxy spheres")
parser.add_argument("--tomato-contact-offset", type=float, default=None,
                    help="simulation-only tomato mesh contact offset in metres")
parser.add_argument("--tomato-rest-offset", type=float, default=None,
                    help="simulation-only tomato mesh rest offset in metres")
parser.add_argument("--palm-support-contact-offset", type=float, default=None,
                    help="simulation-only palm support contact offset in metres")
parser.add_argument("--palm-support-rest-offset", type=float, default=None,
                    help="simulation-only palm support rest offset in metres")
parser.add_argument("--reset-clearance", type=float, default=None,
                    help="simulation-only fruit-to-palm reset clearance in metres")
parser.add_argument("--tomato-stiffness", type=float, default=None,
                    help="simulation-only constant compliant contact stiffness in N/m")
parser.add_argument("--tomato-friction", type=float, default=None,
                    help="simulation-only constant tomato friction coefficient")
parser.add_argument("--cup-half-extent", type=str, default="",
                    help="simulation-only palm rim inner half extents x,y in metres")
parser.add_argument("--rim-height", type=float, default=None,
                    help="simulation-only palm rim height in metres")
parser.add_argument("--root-span", type=float, default=None)
parser.add_argument("--fixed-fruit", action="store_true",
                    help="freeze the fruit at the palm center for contact-window diagnosis")
parser.add_argument("--retention-only", action="store_true",
                    help="start from the retention seed with a dynamic, already-released fruit")
parser.add_argument("--maximum-stem-force-n", type=float, default=0.0,
                    help="diagnostic stem force termination threshold; 0 disables stem termination")
parser.add_argument("--maximum-stem-torque-nm", type=float, default=0.0,
                    help="diagnostic stem torque termination threshold; 0 disables stem termination")
parser.add_argument("--diagnostic-env-id", type=int, default=0,
                    help="include this vectorized environment's per-window metrics in the report")
parser.add_argument("--trace-action-steps", type=str, default=None,
                    help="diagnostic comma-separated episode steps to record")
parser.add_argument("--safe-force-n", type=float, default=12.0,
                    help="diagnostic palm safety reference in newtons")
parser.add_argument("--force-ceiling", type=str, default="12,9.75,9.75,8.25,8.25",
                    help="five comma-separated diagnostic force ceilings: thumb,index,middle,ring,little")
parser.add_argument("--kinematic-bootstrap-steps", type=int, default=0,
                    help="simulation-only: keep the fruit kinematic through the approach, then release it")
parser.add_argument("--debug-collision-digit", choices=("", "palm", "all", "index", "middle", "ring", "little"), default="")
parser.add_argument("--simultaneous-reach", action="store_true",
                    help="ramp the full weak multi-channel reference before the hold controller")
parser.add_argument("--static-hold", action="store_true",
                    help="diagnostic only: hold the final reference without force-band retreats")
parser.add_argument("--digit-scales", type=str, default="1,1,1,1",
                    help="four residual target scales for index,middle,ring,little")
parser.add_argument("--seed", type=int, default=20261120)
parser.add_argument("--profile", default="v1_nominal_reliable")
parser.add_argument("--candidate", type=str, default=(
    "0.864469110965,0.517134726048,-0.583076238632,0.901439428329,"
    "0.113878421485,-0.767623305321,0.276706576348,-0.15316349268,"
    "0.629932641983,-0.107332348824,0.584053039551,0.197868227959,"
    "-0.820022583008,0.175772547722"
))
parser.add_argument("--virtual-distal", action="store_true",
                    help="diagnostic comparison only; disabled by default")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import (
    V2EnvelopingPhysxTomatoGraspEnv,
    V2EnvelopingPhysxTomatoGraspEnvCfg,
)


def parse_candidate(raw: str) -> torch.Tensor:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 14:
        raise ValueError("--candidate requires exactly 14 comma-separated values")
    return torch.tensor(values, dtype=torch.float32)


def parse_digit_scales(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 4 or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("--digit-scales requires four values in [0,1]")
    return values


def parse_four(raw: str, *, name: str) -> tuple[float, float, float, float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 4 or any(value < 0.0 for value in values):
        raise ValueError(f"{name} requires four non-negative values")
    return tuple(float(value) for value in values)


def parse_trace_steps(raw: str | None) -> set[int]:
    if raw is None:
        return set()
    values = {int(value.strip()) for value in raw.split(",") if value.strip()}
    if any(value < 0 for value in values):
        raise ValueError("--trace-action-steps requires non-negative integer steps")
    return values


def parse_pair(raw: str) -> tuple[float, float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 2 or any(value <= 0.0 for value in values):
        raise ValueError("--cup-half-extent requires two positive metre values")
    return float(values[0]), float(values[1])


def parse_force_ceiling(raw: str) -> tuple[float, float, float, float, float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 5 or any(value <= 0.0 for value in values):
        raise ValueError("--force-ceiling requires five positive values")
    return tuple(float(value) for value in values)


def release_kinematic_fruit(env) -> None:
    """Release the diagnostic bootstrap fruit without changing task defaults."""
    from pxr import PhysxSchema, UsdPhysics

    for env_id in range(env.num_envs):
        prim = env.sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Tomato")
        if not prim.IsValid():
            raise RuntimeError(f"missing tomato prim for env {env_id}")
        UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(False)
        PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Set(False)


def apply_tomato_offsets(cfg) -> None:
    if args.tomato_contact_offset is None and args.tomato_rest_offset is None:
        return
    contact = 0.002 if args.tomato_contact_offset is None else float(args.tomato_contact_offset)
    rest = 0.0 if args.tomato_rest_offset is None else float(args.tomato_rest_offset)
    if contact < 0.0 or rest < 0.0 or rest > contact:
        raise ValueError("tomato collision offsets require 0 <= rest <= contact")
    cfg.tomato_cfg.spawn.collision_props.contact_offset = contact
    cfg.tomato_cfg.spawn.collision_props.rest_offset = rest
    event = cfg.events.get("tomato_geometry")
    if event is not None:
        event.params["contact_offset_m"] = contact
        event.params["rest_offset_m"] = rest


def apply_support_offsets(cfg) -> None:
    if args.palm_support_contact_offset is None and args.palm_support_rest_offset is None:
        return
    contact = 0.002 if args.palm_support_contact_offset is None else float(args.palm_support_contact_offset)
    rest = 0.0 if args.palm_support_rest_offset is None else float(args.palm_support_rest_offset)
    if contact < 0.0 or rest < 0.0 or rest > contact:
        raise ValueError("palm support offsets require 0 <= rest <= contact")
    cfg.palm_support_contact_offset_m = contact
    cfg.palm_support_rest_offset_m = rest


def apply_reset_clearance(cfg) -> None:
    if args.reset_clearance is None:
        return
    if args.reset_clearance < 0.0:
        raise ValueError("reset clearance must be non-negative")
    cfg.tomato_reset_clearance_m = float(args.reset_clearance)


def apply_tomato_material(cfg) -> None:
    if args.tomato_stiffness is not None:
        if args.tomato_stiffness <= 0.0:
            raise ValueError("tomato stiffness must be positive")
        cfg.stiffness_range_n_m = (float(args.tomato_stiffness), float(args.tomato_stiffness))
    if args.tomato_friction is not None:
        if args.tomato_friction < 0.0:
            raise ValueError("tomato friction must be non-negative")
        cfg.friction_range = (float(args.tomato_friction), float(args.tomato_friction))


def step_metrics(env, max_link4, max_palm, max_center, min_height, seen, dropped, overloaded):
    """Update sticky probe metrics from the current simulator state."""
    max_link4 = torch.maximum(max_link4, env._v2_measured_link4_force[:, 1:])
    max_palm = torch.maximum(max_palm, env.palm_force)
    center = torch.linalg.vector_norm(env.tomato.data.root_pos_w - env.grasp_center, dim=-1)
    max_center = torch.maximum(max_center, center)
    min_height = torch.minimum(min_height, env.tomato.data.root_pos_w[:, 2])
    seen |= env._v2_measured_link4_force[:, 1:] >= env.cfg.contact_threshold_n
    dropped |= env.dropped
    overloaded |= env.overloaded
    return max_link4, max_palm, max_center, min_height, seen, dropped, overloaded


def post_release_metrics(env, seen, dropped, overloaded, max_center, max_link4):
    seen |= env._v2_measured_link4_force[:, 1:] >= env.cfg.contact_threshold_n
    dropped |= env.dropped
    overloaded |= env.overloaded
    max_center = torch.maximum(
        max_center,
        torch.linalg.vector_norm(env.tomato.data.root_pos_w - env.grasp_center, dim=-1),
    )
    max_link4 = torch.maximum(max_link4, env._v2_measured_link4_force[:, 1:])
    return seen, dropped, overloaded, max_center, max_link4


@torch.inference_mode()
def main() -> None:
    if args.num_envs < 8:
        raise ValueError("--num-envs must be at least 8")
    if not 0 <= args.diagnostic_env_id < args.num_envs:
        raise ValueError("--diagnostic-env-id must lie in [0, num-envs)")
    if args.kinematic_bootstrap_steps < 0:
        raise ValueError("--kinematic-bootstrap-steps must be non-negative")
    if args.kinematic_bootstrap_steps > 0:
        raise RuntimeError(
            "kinematic bootstrap is intentionally blocked: Isaac Sim 4.5 direct-GPU PhysX "
            "does not permit a runtime dynamic/kinematic switch; use a separate spawned "
            "dynamic scene for this counterfactual instead"
        )
    candidate_cpu = parse_candidate(args.candidate)
    if not 0.0 < args.approach_smoothing <= 1.0:
        raise ValueError("--approach-smoothing must lie in (0,1]")
    if not 0.0 < args.distal_filter_alpha <= 1.0:
        raise ValueError("--distal-filter-alpha must lie in (0,1]")
    digit_scales = parse_digit_scales(args.digit_scales)
    assist_per_digit = parse_four(args.assist_per_digit, name="--assist-per-digit") if args.assist_per_digit else None
    contact_target_per_digit = parse_four(args.contact_target_per_digit, name="--contact-target-per-digit") if args.contact_target_per_digit else tuple(float(args.contact_target_n) for _ in range(4))
    retreat_limit_per_digit = parse_four(args.retreat_limit_per_digit, name="--retreat-limit-per-digit") if args.retreat_limit_per_digit else tuple(float(args.retreat_limit_n) for _ in range(4))
    if any(upper < lower for lower, upper in zip(contact_target_per_digit, retreat_limit_per_digit)):
        raise ValueError("each retreat force band must be >= its contact target")
    force_ceiling = parse_force_ceiling(args.force_ceiling)
    retention_seed_cpu = candidate_cpu.clone()
    for digit, scale in enumerate(digit_scales):
        retention_seed_cpu[6 + digit] *= float(scale)
        retention_seed_cpu[10 + digit] *= float(scale)
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.retention_only = bool(args.retention_only)
    cfg.episode_length_s = max(
        20.0,
        (args.approach_steps + 4 * args.per_digit_steps + args.hold_steps + 20)
        * cfg.policy_dt_s,
    )
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)
    cfg.fingertip_collision_proxy_physical = True
    cfg.fingertip_collision_proxy_radius_m = float(args.proxy_radius)
    cfg.proxy_only_collision = bool(args.proxy_only_collision)
    cfg.fingertip_proxy_contact_offset_m = 0.0003
    cfg.fingertip_proxy_rest_offset_m = 0.0001
    cfg.virtual_distal_contact_enabled = bool(args.virtual_distal)
    cfg.palm_first_distal_gate = False
    cfg.per_digit_distal_gate = False
    cfg.per_digit_distal_probe_enabled = False
    cfg.distal_proximal_reach_assist_gain = float(args.assist_gain)
    cfg.distal_proximal_reach_assist_gain_per_digit = assist_per_digit
    cfg.distal_action_filter_alpha = float(args.distal_filter_alpha)
    cfg.coarse_action_smoothing_alpha = 1.0
    cfg.command_rate_limit_rad_s = float(args.command_rate_rad_s)
    if args.root_span is not None:
        cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    cfg.debug_support_only_collision = bool(args.debug_collision_digit)
    cfg.debug_collision_digit = args.debug_collision_digit
    if args.cup_half_extent:
        cfg.palm_cup_inner_half_extent_m = parse_pair(args.cup_half_extent)
    if args.rim_height is not None:
        if args.rim_height <= 0.0:
            raise ValueError("--rim-height must be positive")
        cfg.palm_cup_rim_height_m = float(args.rim_height)
    if args.fixed_fruit:
        cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
        cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    elif args.kinematic_bootstrap_steps > 0:
        cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
        cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    cfg.maximum_stem_force_n = float(args.maximum_stem_force_n)
    cfg.maximum_stem_torque_nm = float(args.maximum_stem_torque_nm)
    # Use the audited v2 evaluation safety envelope.  The inherited v1
    # defaults (6 N palm ceiling) classify the normal support-surface reaction
    # as overload before the fingers are even tested.
    cfg.safe_force_n = float(args.safe_force_n)
    cfg.force_ceiling_n = force_ceiling
    cfg = apply_profile(cfg, args.profile, num_envs=args.num_envs, seed=args.seed)
    # Re-assert the diagnostic contact model after profile resolution.
    cfg.episode_length_s = max(
        20.0,
        (args.approach_steps + 4 * args.per_digit_steps + args.hold_steps + 20)
        * cfg.policy_dt_s,
    )
    cfg.retention_only = bool(args.retention_only)
    if args.retention_only:
        # The retention diagnostic begins from the audited weak reach
        # reference itself.  This tests whether a dynamic fruit can remain
        # supported under a fixed grasp, without conflating approach failure
        # with post-contact retention.
        cfg.retention_seed_joint_positions = None
        cfg.retention_start_action = tuple(float(v) for v in retention_seed_cpu.tolist())
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)
    cfg.fingertip_collision_proxy_physical = True
    cfg.fingertip_collision_proxy_radius_m = float(args.proxy_radius)
    cfg.proxy_only_collision = bool(args.proxy_only_collision)
    cfg.fingertip_proxy_contact_offset_m = 0.0003
    cfg.fingertip_proxy_rest_offset_m = 0.0001
    cfg.virtual_distal_contact_enabled = bool(args.virtual_distal)
    cfg.palm_first_distal_gate = False
    cfg.per_digit_distal_gate = False
    cfg.per_digit_distal_probe_enabled = False
    cfg.distal_proximal_reach_assist_gain = float(args.assist_gain)
    cfg.distal_proximal_reach_assist_gain_per_digit = assist_per_digit
    cfg.distal_action_filter_alpha = float(args.distal_filter_alpha)
    cfg.coarse_action_smoothing_alpha = 1.0
    cfg.command_rate_limit_rad_s = float(args.command_rate_rad_s)
    if args.root_span is not None:
        cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    cfg.debug_support_only_collision = bool(args.debug_collision_digit)
    cfg.debug_collision_digit = args.debug_collision_digit
    if args.cup_half_extent:
        cfg.palm_cup_inner_half_extent_m = parse_pair(args.cup_half_extent)
    if args.rim_height is not None:
        if args.rim_height <= 0.0:
            raise ValueError("--rim-height must be positive")
        cfg.palm_cup_rim_height_m = float(args.rim_height)
    if args.fixed_fruit:
        cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
        cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    elif args.kinematic_bootstrap_steps > 0:
        cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
        cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    cfg.maximum_stem_force_n = float(args.maximum_stem_force_n)
    cfg.maximum_stem_torque_nm = float(args.maximum_stem_torque_nm)
    cfg.safe_force_n = float(args.safe_force_n)
    cfg.force_ceiling_n = force_ceiling
    apply_tomato_offsets(cfg)
    apply_support_offsets(cfg)
    apply_reset_clearance(cfg)
    apply_tomato_material(cfg)

    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)
    device = env.device
    candidate = candidate_cpu.to(device)
    scaled_candidate = candidate.clone()
    for digit, scale in enumerate(digit_scales):
        scaled_candidate[6 + digit] *= float(scale)
        scaled_candidate[10 + digit] *= float(scale)
    open_action = torch.zeros((args.num_envs, 14), device=device)
    open_action[:, :10] = torch.tensor(
        (-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        device=device,
    )
    current = open_action.clone()
    if args.retention_only:
        current = retention_seed_cpu.to(device).unsqueeze(0).expand(args.num_envs, -1).clone()
    max_link4 = torch.zeros(args.num_envs, 4, device=device)
    max_thumb = torch.zeros(args.num_envs, device=device)
    max_palm = torch.zeros(args.num_envs, device=device)
    max_center = torch.zeros(args.num_envs, device=device)
    min_height = torch.full((args.num_envs,), float("inf"), device=device)
    seen = torch.zeros(args.num_envs, 4, dtype=torch.bool, device=device)
    dropped = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    overloaded = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    contact_steps = torch.zeros(args.num_envs, 4, device=device)
    hold_contact_steps = torch.zeros(args.num_envs, 4, device=device)
    hold_all_four_steps = torch.zeros(args.num_envs, device=device)
    contact_streak = torch.zeros(args.num_envs, 4, device=device)
    contact_first_step = torch.full((args.num_envs, 4), -1, dtype=torch.long, device=device)
    post_release_seen = torch.zeros(args.num_envs, 4, dtype=torch.bool, device=device)
    post_release_dropped = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    post_release_overloaded = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    post_release_max_center = torch.zeros(args.num_envs, device=device)
    post_release_max_link4 = torch.zeros(args.num_envs, 4, device=device)
    global_step = 0
    released_at_step = None
    trace_steps = parse_trace_steps(args.trace_action_steps)
    action_trace: dict[str, dict[str, object]] = {}

    def record_trace(step: int, action: torch.Tensor) -> None:
        if step not in trace_steps:
            return
        env_id = args.diagnostic_env_id
        tip_distance = torch.linalg.vector_norm(
            env._tip_positions()[env_id, 1:]
            - env.tomato.data.root_pos_w[env_id].unsqueeze(0), dim=-1
        )
        action_trace[str(step)] = {
            "episode_step": int(step),
            "action_env0": [float(v) for v in action[env_id].detach().cpu().tolist()],
            "measured_link4_force_env0": [float(v) for v in env._v2_measured_link4_force[env_id].detach().cpu().tolist()],
            "contact_forces_env0": [float(v) for v in env.contact_forces[env_id].detach().cpu().tolist()],
            "tip_distance_env0_m": [float(v) for v in tip_distance.detach().cpu().tolist()],
            "filtered_distal_env0": [float(v) for v in env._v2_distal_action[env_id].detach().cpu().tolist()],
            "active_joint_position_env0": [
                float(v) for v in env.robot.data.joint_pos[env_id, env.active_joint_ids].detach().cpu().tolist()
            ],
            "prev_target_env0": [float(v) for v in env.prev_targets[env_id].detach().cpu().tolist()],
        }

    # Palm/root approach: no distal motion is injected while the palm settles.
    coarse_target = scaled_candidate.clone()
    if not args.simultaneous_reach:
        coarse_target[10:] = 0.0
    for local_step in range(args.approach_steps):
        fraction = float(local_step + 1) / max(args.approach_steps, 1)
        if args.retention_only:
            desired = coarse_target.unsqueeze(0).expand(args.num_envs, -1)
            current = desired.clone()
        else:
            desired = open_action + fraction * (coarse_target.unsqueeze(0) - open_action)
            current = current + float(args.approach_smoothing) * (desired - current)
        record_trace(global_step, current)
        _ = env.step(current)
        max_thumb = torch.maximum(max_thumb, env._v2_measured_link4_force[:, 0])
        max_link4, max_palm, max_center, min_height, seen, dropped, overloaded = step_metrics(
            env, max_link4, max_palm, max_center, min_height, seen, dropped, overloaded
        )
        contact_steps += (env._v2_measured_link4_force[:, 1:] >= env.cfg.contact_threshold_n).float()
        global_step += 1

    if args.kinematic_bootstrap_steps > 0:
        # Counterfactual only: establish a reachable contact pose while the
        # fruit is kinematic, then restore gravity and dynamics to test hold.
        release_kinematic_fruit(env)
        released_at_step = global_step

    # Sequential low-force distal insertion.  Each finger has an independent
    # frozen target once a measured contact streak is reached.
    digit_labels = ("index", "middle", "ring", "little")
    for digit in range(4):
        channels = (6 + digit, 10 + digit)
        target = candidate[list(channels)]
        frozen = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        for local_step in range(args.per_digit_steps):
            force = env._v2_measured_link4_force[:, digit + 1]
            active = force >= float(contact_target_per_digit[digit])
            contact_streak[:, digit] = torch.where(
                active, contact_streak[:, digit] + 1.0, torch.zeros_like(contact_streak[:, digit])
            )
            newly = (contact_streak[:, digit] >= float(args.contact_streak)) & (contact_first_step[:, digit] < 0)
            contact_first_step[:, digit] = torch.where(
                newly, torch.full_like(contact_first_step[:, digit], global_step), contact_first_step[:, digit]
            )
            frozen |= contact_streak[:, digit] >= float(args.contact_streak)
            # Advance only the currently active digit; all others retain their
            # previous command.  A high force retreats symmetrically toward the
            # open value, preventing a fixed target from accumulating impulse.
            for channel_index, channel in enumerate(channels):
                desired = current[:, channel] + float(args.step_fraction) * (target[channel_index] - current[:, channel])
                retreat = current[:, channel] + float(args.retreat_fraction) * (open_action[:, channel] - current[:, channel])
                high = force >= float(retreat_limit_per_digit[digit])
                next_value = torch.where(high, retreat, desired)
                next_value = torch.where(frozen, current[:, channel], next_value)
                current[:, channel] = next_value.clamp(-1.0, 1.0)
            record_trace(global_step, current)
            _ = env.step(current)
            max_thumb = torch.maximum(max_thumb, env._v2_measured_link4_force[:, 0])
            max_link4, max_palm, max_center, min_height, seen, dropped, overloaded = step_metrics(
                env, max_link4, max_palm, max_center, min_height, seen, dropped, overloaded
            )
            contact_steps += (env._v2_measured_link4_force[:, 1:] >= env.cfg.contact_threshold_n).float()
            if released_at_step is not None:
                (
                    post_release_seen,
                    post_release_dropped,
                    post_release_overloaded,
                    post_release_max_center,
                    post_release_max_link4,
                ) = post_release_metrics(
                    env,
                    post_release_seen,
                    post_release_dropped,
                    post_release_overloaded,
                    post_release_max_center,
                    post_release_max_link4,
                )
            global_step += 1

    # Hold with an independent force band.  This is the key counterfactual:
    # a finger below the target advances in tiny increments, while a finger
    # above the upper band retreats without disturbing the other digits.
    for _ in range(args.hold_steps):
        if args.static_hold:
            # Static reference is intentionally a diagnostic baseline.  It
            # separates a reachable, dynamically stable pose from the
            # feedback controller's force-band tuning; it is never used as a
            # policy or hardware command.
            current = scaled_candidate.unsqueeze(0).expand(args.num_envs, -1).clone()
        else:
            force = env._v2_measured_link4_force[:, 1:]
            for digit in range(4):
                for channel in (6 + digit, 10 + digit):
                    low = force[:, digit] < float(contact_target_per_digit[digit])
                    high = force[:, digit] >= float(retreat_limit_per_digit[digit])
                    advance = current[:, channel] + float(args.step_fraction) * (scaled_candidate[channel] - current[:, channel])
                    retreat = current[:, channel] + float(args.retreat_fraction) * (open_action[:, channel] - current[:, channel])
                    current[:, channel] = torch.where(high, retreat, torch.where(low, advance, current[:, channel])).clamp(-1.0, 1.0)
        record_trace(global_step, current)
        _ = env.step(current)
        max_thumb = torch.maximum(max_thumb, env._v2_measured_link4_force[:, 0])
        max_link4, max_palm, max_center, min_height, seen, dropped, overloaded = step_metrics(
            env, max_link4, max_palm, max_center, min_height, seen, dropped, overloaded
        )
        contact_steps += (env._v2_measured_link4_force[:, 1:] >= env.cfg.contact_threshold_n).float()
        measured_active = env._v2_measured_link4_force[:, 1:] >= env.cfg.contact_threshold_n
        hold_contact_steps += measured_active.float()
        hold_all_four_steps += measured_active.all(-1).float()
        if released_at_step is not None:
            (
                post_release_seen,
                post_release_dropped,
                post_release_overloaded,
                post_release_max_center,
                post_release_max_link4,
            ) = post_release_metrics(
                env,
                post_release_seen,
                post_release_dropped,
                post_release_overloaded,
                post_release_max_center,
                post_release_max_link4,
            )
        global_step += 1

    strict = env.strict_success_latched.clone()
    final_center_offset = env.tomato.data.root_pos_w - env.grasp_center
    base_quat_world = env.robot.data.body_quat_w[:, env.base_id]
    base_pos_world = env.robot.data.body_pos_w[:, env.base_id]
    report = {
        "version": "v2_contact_closed_loop_probe_v1",
        "profile": args.profile,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": global_step,
        "candidate": candidate_cpu.tolist(),
        "controller": {
            "approach_steps": args.approach_steps,
            "per_digit_steps": args.per_digit_steps,
            "hold_steps": args.hold_steps,
            "contact_target_n": args.contact_target_n,
            "retreat_limit_n": args.retreat_limit_n,
            "contact_target_per_digit": [float(v) for v in contact_target_per_digit],
            "retreat_limit_per_digit": [float(v) for v in retreat_limit_per_digit],
            "contact_streak": args.contact_streak,
            "step_fraction": args.step_fraction,
            "retreat_fraction": args.retreat_fraction,
            "command_rate_rad_s": args.command_rate_rad_s,
            "approach_smoothing": args.approach_smoothing,
            "distal_filter_alpha": args.distal_filter_alpha,
            "assist_gain": args.assist_gain,
            "assist_per_digit": assist_per_digit,
            "digit_order": list(digit_labels),
            "simultaneous_reach": bool(args.simultaneous_reach),
            "static_hold": bool(args.static_hold),
            "retention_only": bool(args.retention_only),
            "kinematic_bootstrap_steps": int(args.kinematic_bootstrap_steps),
            "released_at_step": released_at_step,
            "digit_scales": digit_scales,
            "maximum_stem_force_n": float(args.maximum_stem_force_n),
            "maximum_stem_torque_nm": float(args.maximum_stem_torque_nm),
            "diagnostic_env_id": int(args.diagnostic_env_id),
        },
        "resolved_contact_geometry": {
            "physical_fingertip_proxy": True,
            "proxy_only_collision": bool(args.proxy_only_collision),
            "radius_m": float(args.proxy_radius),
            "contact_offset_m": 0.0003,
            "rest_offset_m": 0.0001,
            "virtual_distal_contact": bool(args.virtual_distal),
            "fixed_fruit": bool(args.fixed_fruit),
            "retention_only": bool(args.retention_only),
            "debug_collision_digit": args.debug_collision_digit,
            "palm_cup_inner_half_extent_m": list(cfg.palm_cup_inner_half_extent_m),
            "palm_cup_rim_height_m": float(cfg.palm_cup_rim_height_m),
            "tomato_contact_offset_m": float(cfg.tomato_cfg.spawn.collision_props.contact_offset),
            "tomato_rest_offset_m": float(cfg.tomato_cfg.spawn.collision_props.rest_offset),
            "palm_support_contact_offset_m": float(cfg.palm_support_contact_offset_m),
            "palm_support_rest_offset_m": float(cfg.palm_support_rest_offset_m),
            "tomato_reset_clearance_m": float(cfg.tomato_reset_clearance_m),
            "tomato_stiffness_range_n_m": list(cfg.stiffness_range_n_m),
            "tomato_friction_range": list(cfg.friction_range),
            "maximum_stem_force_n": float(cfg.maximum_stem_force_n),
            "maximum_stem_torque_nm": float(cfg.maximum_stem_torque_nm),
            "virtual_distal_contact_gain_n_per_m": float(cfg.virtual_distal_contact_gain_n_per_m),
            "distal_proximal_reach_assist_gain_per_digit": None if cfg.distal_proximal_reach_assist_gain_per_digit is None else [float(v) for v in cfg.distal_proximal_reach_assist_gain_per_digit],
            "contact_threshold_n": float(cfg.contact_threshold_n),
            "safe_force_n": float(cfg.safe_force_n),
            "force_ceiling_n": [float(v) for v in cfg.force_ceiling_n],
        },
        "resolved_object_randomization": {
            "object_mass_kg": {
                "min": float(env.object_mass.min().cpu()),
                "max": float(env.object_mass.max().cpu()),
                "mean": float(env.object_mass.mean().cpu()),
            },
            "tomato_axes_m": {
                "min": [float(v) for v in env.tomato_axes.amin(0).cpu()],
                "max": [float(v) for v in env.tomato_axes.amax(0).cpu()],
                "mean": [float(v) for v in env.tomato_axes.mean(0).cpu()],
            },
            "tomato_density_kg_m3": {
                "min": float(env._tomato_density_kg_m3.min().cpu()),
                "max": float(env._tomato_density_kg_m3.max().cpu()),
                "mean": float(env._tomato_density_kg_m3.mean().cpu()),
            },
            "object_friction": {
                "min": float(env.object_friction.min().cpu()),
                "max": float(env.object_friction.max().cpu()),
                "mean": float(env.object_friction.mean().cpu()),
            },
            "object_stiffness_n_m": {
                "min": float(env.object_stiffness.min().cpu()),
                "max": float(env.object_stiffness.max().cpu()),
                "mean": float(env.object_stiffness.mean().cpu()),
            },
            "sample_env_0": {
                "object_mass_kg": float(env.object_mass[0].cpu()),
                "tomato_axes_m": [float(v) for v in env.tomato_axes[0].cpu()],
                "tomato_density_kg_m3": float(env._tomato_density_kg_m3[0].cpu()),
                "object_friction": float(env.object_friction[0].cpu()),
                "object_stiffness_n_m": float(env.object_stiffness[0].cpu()),
                "tomato_position_m": [float(v) for v in env.tomato.data.root_pos_w[0].cpu()],
            },
        },
        "per_digit_measured_contact_rate": seen.float().mean(0).cpu().tolist(),
        "per_digit_contact_occupancy_rate": (contact_steps / max(global_step, 1)).mean(0).cpu().tolist(),
        "hold_contact_occupancy_rate": (hold_contact_steps / max(float(args.hold_steps), 1.0)).mean(0).cpu().tolist(),
        "hold_all_four_occupancy_rate": (hold_all_four_steps / max(float(args.hold_steps), 1.0)).mean().cpu().item(),
        "diagnostic_env_metrics": {
            "env_id": int(args.diagnostic_env_id),
            "hold_contact_occupancy_rate": (hold_contact_steps[args.diagnostic_env_id] / max(float(args.hold_steps), 1.0)).cpu().tolist(),
            "hold_all_four_occupancy_rate": float((hold_all_four_steps[args.diagnostic_env_id] / max(float(args.hold_steps), 1.0)).cpu()),
            "contact_occupancy_rate_all_steps": (contact_steps[args.diagnostic_env_id] / max(float(global_step), 1.0)).cpu().tolist(),
            "measured_contact_seen": [bool(v) for v in seen[args.diagnostic_env_id].cpu()],
            "drop": bool(dropped[args.diagnostic_env_id].cpu()),
            "overload": bool(overloaded[args.diagnostic_env_id].cpu()),
            "max_link4_force_n": [float(v) for v in max_link4[args.diagnostic_env_id].cpu()],
            "max_thumb_force_n": float(max_thumb[args.diagnostic_env_id].cpu()),
            "max_palm_force_n": float(max_palm[args.diagnostic_env_id].cpu()),
            "max_center_distance_m": float(max_center[args.diagnostic_env_id].cpu()),
            "resolved_env": {
                "object_mass_kg": float(env.object_mass[args.diagnostic_env_id].cpu()),
                "tomato_axes_m": [float(v) for v in env.tomato_axes[args.diagnostic_env_id].cpu()],
                "tomato_density_kg_m3": float(env._tomato_density_kg_m3[args.diagnostic_env_id].cpu()),
                "object_friction": float(env.object_friction[args.diagnostic_env_id].cpu()),
                "object_stiffness_n_m": float(env.object_stiffness[args.diagnostic_env_id].cpu()),
            },
        },
        "measured_distal_all_rate": seen.all(-1).float().mean().cpu().item(),
        "strict_success_rate": strict.float().mean().cpu().item(),
        "drop_rate": dropped.float().mean().cpu().item(),
        "overload_rate": overloaded.float().mean().cpu().item(),
        "palm_support_rate": (max_palm >= cfg.palm_support_force_n).float().mean().cpu().item(),
        "max_link4_force_n_mean": max_link4.mean(0).cpu().tolist(),
        "max_link4_force_n_max": max_link4.amax(0).cpu().tolist(),
        "max_thumb_force_n_mean": float(max_thumb.mean().cpu()),
        "max_thumb_force_n_max": float(max_thumb.max().cpu()),
        "max_palm_force_n_mean": max_palm.mean().cpu().item(),
        "min_height_m_min": min_height.min().cpu().item(),
        "max_center_distance_m_max": max_center.max().cpu().item(),
        "final_center_offset_mean_m": final_center_offset.mean(0).cpu().tolist(),
        "final_center_offset_max_abs_m": final_center_offset.abs().amax(0).cpu().tolist(),
        "base_link_world_position_mean_m": base_pos_world.mean(0).cpu().tolist(),
        "base_link_world_quaternion_mean": base_quat_world.mean(0).cpu().tolist(),
        "post_release": None if released_at_step is None else {
            "per_digit_measured_contact_rate": post_release_seen.float().mean(0).cpu().tolist(),
            "measured_distal_all_rate": post_release_seen.all(-1).float().mean().cpu().item(),
            "drop_rate": post_release_dropped.float().mean().cpu().item(),
            "overload_rate": post_release_overloaded.float().mean().cpu().item(),
            "max_center_distance_m_max": post_release_max_center.max().cpu().item(),
            "max_link4_force_n_mean": post_release_max_link4.mean(0).cpu().tolist(),
            "max_link4_force_n_max": post_release_max_link4.amax(0).cpu().tolist(),
        },
        "contact_first_step_mean": torch.where(
            contact_first_step >= 0, contact_first_step.float(), torch.full_like(contact_first_step.float(), float(global_step))
        ).mean(0).cpu().tolist(),
        "hardware_command_allowed": False,
        "action_trace": action_trace,
        "interpretation": "simulation_only_low_force_contact_hold_retreat_diagnostic",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
