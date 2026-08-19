"""P2-05/P2-06 PhysX event-window and pre-pick dwell diagnostic.

This script is intentionally open-loop and simulation-only.  It uses one
audited action candidate to measure (a) the stem-break transition and its
local +/-20 policy-step window, and (b) the effect of 3/6/10/20 pre-pick
stable-contact dwell steps.  It is not a policy, a success claim, or a
hardware command path.
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--approach-steps", type=int, default=240)
parser.add_argument("--post-steps", type=int, default=240)
parser.add_argument("--prepick-hold-steps", type=str, default="3,6,10,20")
parser.add_argument("--window-radius", type=int, default=20)
parser.add_argument("--candidate", type=str, default=(
    "0.6376953125,-0.6281427145,-0.6301738620,-0.2590247393,1.0,"
    "0.0078130066,1.0,0.4984259009,0.7332278490,0.3807550669,"
    "-0.5072317719,0.2533326745,-0.5880369544,0.7160025835"
))
parser.add_argument("--seed", type=int, default=20260819)
parser.add_argument("--profile", default="v1_nominal_reliable")
parser.add_argument("--proxy-radius", type=float, default=0.006)
parser.add_argument("--tomato-contact-offset", type=float, default=0.0003)
parser.add_argument("--tomato-rest-offset", type=float, default=0.0001)
parser.add_argument("--proxy-contact-offset", type=float, default=0.0003)
parser.add_argument("--proxy-rest-offset", type=float, default=0.0001)
parser.add_argument("--root-span", type=float, default=0.40)
parser.add_argument("--distal-filter-alpha", type=float, default=0.999)
parser.add_argument("--assist-per-digit", type=str, default="0.45,0.45,0.30,0.25")
parser.add_argument("--cup-half-extent", type=str, default="0.040,0.035",
                    help="simulation-only palm rim inner half extents x,y in metres")
parser.add_argument("--command-rate-rad-s", type=float, default=None)
parser.add_argument("--fixed-fruit", action="store_true",
                    help="keep the fruit kinematic; useful as a contact-only control")
parser.add_argument("--kinematic-approach", action="store_true",
                    help="hold the fruit kinematic through approach, then release once before the event window")
parser.add_argument("--dynamic-no-gravity", action="store_true",
                    help="start dynamic but disable gravity; avoids unsupported direct-GPU kinematic switching")
parser.add_argument("--stem-limit-n", type=float, default=None,
                    help="optional diagnostic fixed stem tensile limit; omitted means the profile prior")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import (
    V2EnvelopingPhysxTomatoGraspEnv,
    V2EnvelopingPhysxTomatoGraspEnvCfg,
)


def parse_values(raw: str, n: int, name: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != n:
        raise ValueError(f"{name} requires {n} comma-separated values")
    return values


def parse_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("--prepick-hold-steps requires positive integers")
    return values


def parse_pair(raw: str) -> tuple[float, float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 2 or any(value <= 0.0 for value in values):
        raise ValueError("--cup-half-extent requires two positive metre values")
    return float(values[0]), float(values[1])


def make_cfg(prepick_hold_steps: int, *, num_envs: int, seed: int):
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 2
    cfg.strict_enveloping_enabled = True
    # Event-window measurement must continue after the break; terminal strict
    # success is not the purpose of this probe.
    cfg.strict_enveloping_require_stem_break = False
    cfg.strict_require_measured_distal_contact = False
    cfg.episode_length_s = max(12.0, (args.approach_steps + args.post_steps + 80) * cfg.policy_dt_s)
    cfg.prepick_hold_steps = int(prepick_hold_steps)
    cfg.pregrasp_drop_grace_steps = max(90, args.approach_steps)
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = bool(args.fixed_fruit or args.kinematic_approach)
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = bool(args.fixed_fruit or args.kinematic_approach or args.dynamic_no_gravity)
    cfg.fingertip_collision_proxy_physical = True
    cfg.proxy_only_collision = True
    cfg.fingertip_collision_proxy_radius_m = float(args.proxy_radius)
    cfg.fingertip_proxy_contact_offset_m = float(args.proxy_contact_offset)
    cfg.fingertip_proxy_rest_offset_m = float(args.proxy_rest_offset)
    cfg.tomato_contact_offset_m = float(args.tomato_contact_offset)
    cfg.tomato_rest_offset_m = float(args.tomato_rest_offset)
    cfg.virtual_distal_contact_enabled = False
    cfg.palm_first_distal_gate = False
    cfg.per_digit_distal_gate = False
    cfg.per_digit_distal_probe_enabled = False
    cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    cfg.distal_action_filter_alpha = float(args.distal_filter_alpha)
    cfg.distal_proximal_reach_assist_gain_per_digit = tuple(parse_values(args.assist_per_digit, 4, "--assist-per-digit"))
    cfg.distal_proximal_reach_assist_gain = 0.30
    cfg.coarse_action_smoothing_alpha = 1.0
    cfg.palm_cup_inner_half_extent_m = parse_pair(args.cup_half_extent)
    cfg.debug_support_only_collision = False
    cfg.debug_collision_digit = ""
    if args.command_rate_rad_s is not None:
        cfg.command_rate_limit_rad_s = float(args.command_rate_rad_s)
    if args.stem_limit_n is not None:
        limit = float(args.stem_limit_n)
        if limit <= 0.0:
            raise ValueError("--stem-limit-n must be positive")
        cfg.stem_tensile_limit_n = (limit, limit)
    cfg = apply_profile(cfg, args.profile, num_envs=num_envs, seed=seed)
    # Reapply all diagnostic-only values after the profile, since the profile
    # is allowed to replace physical ranges but must not erase this audit.
    cfg.training_stage = 2
    cfg.strict_enveloping_enabled = True
    cfg.strict_enveloping_require_stem_break = False
    cfg.prepick_hold_steps = int(prepick_hold_steps)
    cfg.pregrasp_drop_grace_steps = max(90, args.approach_steps)
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = bool(args.fixed_fruit or args.kinematic_approach)
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = bool(args.fixed_fruit or args.kinematic_approach or args.dynamic_no_gravity)
    cfg.fingertip_collision_proxy_physical = True
    cfg.proxy_only_collision = True
    cfg.fingertip_collision_proxy_radius_m = float(args.proxy_radius)
    cfg.fingertip_proxy_contact_offset_m = float(args.proxy_contact_offset)
    cfg.fingertip_proxy_rest_offset_m = float(args.proxy_rest_offset)
    cfg.tomato_contact_offset_m = float(args.tomato_contact_offset)
    cfg.tomato_rest_offset_m = float(args.tomato_rest_offset)
    cfg.virtual_distal_contact_enabled = False
    cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    cfg.distal_action_filter_alpha = float(args.distal_filter_alpha)
    cfg.distal_proximal_reach_assist_gain_per_digit = tuple(parse_values(args.assist_per_digit, 4, "--assist-per-digit"))
    cfg.distal_proximal_reach_assist_gain = 0.30
    cfg.coarse_action_smoothing_alpha = 1.0
    cfg.palm_cup_inner_half_extent_m = parse_pair(args.cup_half_extent)
    if args.stem_limit_n is not None:
        limit = float(args.stem_limit_n)
        cfg.stem_tensile_limit_n = (limit, limit)
    return cfg


@torch.inference_mode()
def run_case(prepick_hold_steps: int) -> dict:
    cfg = make_cfg(prepick_hold_steps, num_envs=args.num_envs, seed=args.seed + prepick_hold_steps)
    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed + prepick_hold_steps)
    device = env.device
    candidate = torch.tensor(parse_values(args.candidate, 14, "--candidate"), device=device, dtype=torch.float32)
    open_action = torch.zeros((args.num_envs, 14), device=device)
    action = open_action.clone()
    target = candidate.unsqueeze(0).expand(args.num_envs, -1)
    open_action[:, :10] = torch.tensor(
        (-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        device=device,
    )
    total_steps = int(args.approach_steps + args.post_steps)
    diag = min(args.num_envs - 1, 0)
    # step x env x [linvel3,angvel3,center3,forces5,palm,stem_damage,prev_targets16]
    width = 3 + 3 + 3 + 5 + 1 + 1 + int(env.prev_targets.shape[-1])
    trace = torch.zeros((total_steps, args.num_envs, width), device=device)
    was_broken = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    event_steps: list[int] = []
    dropped = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    overloaded = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    peak_force = torch.zeros((args.num_envs, 5), device=device)
    max_center = torch.zeros(args.num_envs, device=device)
    pick_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    for step in range(total_steps):
        if step < args.approach_steps:
            fraction = float(step + 1) / float(max(args.approach_steps, 1))
            action = open_action + fraction * (target - open_action)
        else:
            action = target
        if args.kinematic_approach and step == args.approach_steps - 1:
            # This is a diagnostic bootstrap only: it separates initial
            # reach/geometry from the subsequent dynamic break window without
            # changing the policy or production environment defaults.
            from pxr import PhysxSchema, UsdPhysics
            for env_id in range(args.num_envs):
                prim = env.sim.stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Tomato")
                UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(False)
                PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Set(False)
        prior_broken = env.stem_broke.clone()
        env.step(action)
        linvel = getattr(env.tomato.data, "root_lin_vel_w", torch.zeros_like(env.tomato.data.root_pos_w))
        angvel = getattr(env.tomato.data, "root_ang_vel_w", torch.zeros_like(env.tomato.data.root_pos_w))
        center = env.tomato.data.root_pos_w - env.grasp_center
        forces = env._v2_measured_link4_force
        row = torch.cat((linvel, angvel, center, forces, env.palm_force.unsqueeze(-1), env.stem_damage.unsqueeze(-1), env.prev_targets), dim=-1)
        trace[step] = row
        new_break = env.stem_broke & ~prior_broken
        new_break |= env.last_break_event
        if bool(new_break.any()):
            event_steps.extend([step] * int(new_break.sum()))
        was_broken |= env.stem_broke
        dropped |= env.dropped
        overloaded |= env.overloaded
        peak_force = torch.maximum(peak_force, forces)
        max_center = torch.maximum(max_center, torch.linalg.vector_norm(center, dim=-1))
        pick_seen |= env.pick_started
    events = []
    radius = max(0, int(args.window_radius))
    for step in sorted(set(event_steps))[:16]:
        lo = max(0, step - radius)
        hi = min(total_steps, step + radius + 1)
        window = trace[lo:hi, diag].detach().cpu()
        events.append({
            "diagnostic_env": int(diag),
            "event_step": int(step),
            "window_start_step": int(lo),
            "window_end_step_exclusive": int(hi),
            "rows": window.tolist(),
            "columns": [
                "linvel_x", "linvel_y", "linvel_z", "angvel_x", "angvel_y", "angvel_z",
                "center_x", "center_y", "center_z", "thumb_force", "index_force",
                "middle_force", "ring_force", "little_force", "palm_force", "stem_damage",
            ] + [f"prev_target_{i}" for i in range(int(env.prev_targets.shape[-1]))],
        })
    report = {
        "prepick_hold_steps": int(prepick_hold_steps),
        "num_envs": int(args.num_envs),
        "approach_steps": int(args.approach_steps),
        "post_steps": int(args.post_steps),
        "stem_break_event_count": int(len(event_steps)),
        "stem_break_event_env_rate": float((was_broken.float().mean()).cpu()),
        "pick_started_rate": float(pick_seen.float().mean().cpu()),
        "drop_rate": float(dropped.float().mean().cpu()),
        "overload_rate": float(overloaded.float().mean().cpu()),
        "peak_force_n_mean": peak_force.mean(0).cpu().tolist(),
        "peak_force_n_max": peak_force.max(0).values.cpu().tolist(),
        "max_center_distance_m_mean": float(max_center.mean().cpu()),
        "max_center_distance_m_max": float(max_center.max().cpu()),
        "event_windows": events,
        "resolved_contact_geometry": {
            "proxy_only_collision": True,
            "physical_fingertip_proxy": True,
            "proxy_radius_m": float(args.proxy_radius),
            "proxy_contact_offset_m": float(args.proxy_contact_offset),
            "proxy_rest_offset_m": float(args.proxy_rest_offset),
            "tomato_contact_offset_m": float(args.tomato_contact_offset),
            "tomato_rest_offset_m": float(args.tomato_rest_offset),
            "root_span": float(args.root_span),
            "distal_filter_alpha": float(args.distal_filter_alpha),
            "assist_per_digit": parse_values(args.assist_per_digit, 4, "--assist-per-digit"),
            "fixed_fruit": bool(args.fixed_fruit),
            "kinematic_approach": bool(args.kinematic_approach),
            "dynamic_no_gravity": bool(args.dynamic_no_gravity),
            "stem_limit_n": None if args.stem_limit_n is None else float(args.stem_limit_n),
        },
        "hardware_command_allowed": False,
    }
    env.close()
    return report


def main() -> None:
    cases = [run_case(value) for value in parse_ints(args.prepick_hold_steps)]
    report = {
        "version": "v2_p2_event_window_probe_v1",
        "profile": args.profile,
        "seed": args.seed,
        "candidate": parse_values(args.candidate, 14, "--candidate"),
        "cases": cases,
        "interpretation": "simulation-only open-loop diagnostic; event windows are not PPO success evidence",
        "hardware_command_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
