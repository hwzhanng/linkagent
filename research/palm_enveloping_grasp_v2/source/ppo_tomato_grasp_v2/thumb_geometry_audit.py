"""Simulation-only thumb geometry/action audit.

This diagnostic does not train, alter the URDF, or send hardware commands.  It
replays one 14-D action against a fixed tomato and reports the imported
``thumb_link5`` body pose, the authored v2 proxy pose, and the measured force.
The purpose is to distinguish a kinematic reach problem from a collision/proxy
attachment problem before another PPO run is started.
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
parser.add_argument("--candidate", type=str, required=True)
parser.add_argument("--approach-steps", type=int, default=180)
parser.add_argument("--hold-steps", type=int, default=80)
parser.add_argument("--proxy-radius", type=float, default=0.006)
parser.add_argument("--physical-proxy", action="store_true")
parser.add_argument("--proxy-only-collision", action="store_true",
                    help="match the search that disables uncalibrated vendor colliders")
parser.add_argument("--proxy-contact-offset", type=float, default=0.0003)
parser.add_argument("--proxy-rest-offset", type=float, default=0.0001)
parser.add_argument("--tomato-contact-offset", type=float, default=0.0003)
parser.add_argument("--tomato-rest-offset", type=float, default=0.0001)
parser.add_argument("--root-span", type=float, default=None,
                    help="match the per-digit root residual span used by a search")
parser.add_argument("--distal-filter-alpha", type=float, default=0.999)
parser.add_argument("--coarse-smoothing-alpha", type=float, default=1.0)
parser.add_argument("--assist-gains", type=str, default="",
                    help="optional four per-digit distal-to-proximal gains")
parser.add_argument("--assist-gain", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=20261295)
parser.add_argument("--profile", default="v1_nominal_reliable")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from pxr import UsdGeom

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import (
    V2EnvelopingPhysxTomatoGraspEnv,
    V2EnvelopingPhysxTomatoGraspEnvCfg,
)


def parse_candidate(raw: str) -> torch.Tensor:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 14:
        raise ValueError("--candidate requires exactly 14 values")
    return torch.tensor(values, dtype=torch.float32)


def parse_four(raw: str) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 4 or any(value < 0.0 for value in values):
        raise ValueError("--assist-gains requires four non-negative values")
    return tuple(values)  # type: ignore[return-value]


@torch.inference_mode()
def main() -> None:
    candidate = parse_candidate(args.candidate)
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.episode_length_s = max(20.0, (args.approach_steps + args.hold_steps + 20) * cfg.policy_dt_s)
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)
    cfg.fingertip_collision_proxy_physical = bool(args.physical_proxy)
    cfg.proxy_only_collision = bool(args.proxy_only_collision)
    cfg.fingertip_collision_proxy_radius_m = float(args.proxy_radius)
    cfg.fingertip_proxy_contact_offset_m = float(args.proxy_contact_offset)
    cfg.fingertip_proxy_rest_offset_m = float(args.proxy_rest_offset)
    cfg.tomato_contact_offset_m = float(args.tomato_contact_offset)
    cfg.tomato_rest_offset_m = float(args.tomato_rest_offset)
    cfg.distal_action_filter_alpha = float(args.distal_filter_alpha)
    cfg.coarse_action_smoothing_alpha = float(args.coarse_smoothing_alpha)
    cfg.distal_proximal_reach_assist_gain = float(args.assist_gain)
    cfg.distal_proximal_reach_assist_gain_per_digit = parse_four(args.assist_gains)
    if args.root_span is not None:
        cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    cfg.virtual_distal_contact_enabled = False
    cfg.virtual_palm_support_enabled = True
    cfg.debug_support_only_collision = False
    cfg = apply_profile(cfg, args.profile, num_envs=1, seed=args.seed)
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)
    cfg.fingertip_collision_proxy_physical = bool(args.physical_proxy)
    cfg.proxy_only_collision = bool(args.proxy_only_collision)
    cfg.fingertip_collision_proxy_radius_m = float(args.proxy_radius)
    cfg.fingertip_proxy_contact_offset_m = float(args.proxy_contact_offset)
    cfg.fingertip_proxy_rest_offset_m = float(args.proxy_rest_offset)
    cfg.tomato_contact_offset_m = float(args.tomato_contact_offset)
    cfg.tomato_rest_offset_m = float(args.tomato_rest_offset)
    cfg.distal_action_filter_alpha = float(args.distal_filter_alpha)
    cfg.coarse_action_smoothing_alpha = float(args.coarse_smoothing_alpha)
    cfg.distal_proximal_reach_assist_gain = float(args.assist_gain)
    cfg.distal_proximal_reach_assist_gain_per_digit = parse_four(args.assist_gains)
    if args.root_span is not None:
        cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    cfg.virtual_distal_contact_enabled = False
    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)

    open_action = torch.zeros((1, 14), device=env.device)
    open_action[:, :10] = torch.tensor(
        (-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        device=env.device,
    )
    target = candidate.to(env.device).unsqueeze(0)
    tomato = env.tomato.data.root_pos_w.detach().clone()
    names = ("thumb_link5", "index_link4", "middle_link4", "ring_link4", "little_link4")
    min_body = torch.full((5,), float("inf"), device=env.device)
    min_proxy = torch.full((5,), float("inf"), device=env.device)
    final_body = torch.zeros((5, 3), device=env.device)
    final_proxy = torch.zeros((5, 3), device=env.device)
    max_force = torch.zeros(5, device=env.device)

    for step in range(args.approach_steps + args.hold_steps):
        fraction = min(float(step + 1) / max(args.approach_steps, 1), 1.0)
        action = open_action + fraction * (target - open_action)
        env.step(action)
        body = env._tip_positions()[0]
        # The proxy is authored at each link's local origin.  Body pose is the
        # same world-space origin for a sphere with zero local offset; retain a
        # separate array in the report so a future calibrated offset is
        # explicit rather than silently conflated with the visual fingertip.
        proxy = body.clone()
        dist_body = torch.linalg.vector_norm(body - env.tomato.data.root_pos_w[0], dim=-1)
        dist_proxy = torch.linalg.vector_norm(proxy - env.tomato.data.root_pos_w[0], dim=-1)
        min_body = torch.minimum(min_body, dist_body)
        min_proxy = torch.minimum(min_proxy, dist_proxy)
        final_body = body
        final_proxy = proxy
        max_force = torch.maximum(max_force, env._v2_measured_link4_force[0])

    report = {
        "version": "v2_thumb_geometry_audit_v1",
        "candidate": [float(v) for v in candidate.tolist()],
        "seed": args.seed,
        "approach_steps": args.approach_steps,
        "hold_steps": args.hold_steps,
        "tomato_center_w": [float(v) for v in tomato[0].cpu().tolist()],
        "tip_names": list(names),
        "min_body_distance_m": [float(v) for v in min_body.cpu().tolist()],
        "min_proxy_distance_m": [float(v) for v in min_proxy.cpu().tolist()],
        "final_body_pos_w": final_body.cpu().tolist(),
        "final_proxy_origin_w": final_proxy.cpu().tolist(),
        "max_measured_force_n": [float(v) for v in max_force.cpu().tolist()],
        "proxy_radius_m": float(args.proxy_radius),
        "physical_proxy": bool(args.physical_proxy),
        "proxy_only_collision": bool(args.proxy_only_collision),
        "proxy_contact_offset_m": float(args.proxy_contact_offset),
        "proxy_rest_offset_m": float(args.proxy_rest_offset),
        "tomato_contact_offset_m": float(args.tomato_contact_offset),
        "tomato_rest_offset_m": float(args.tomato_rest_offset),
        "per_digit_root_residual_span_fraction": args.root_span,
        "distal_filter_alpha": float(args.distal_filter_alpha),
        "coarse_smoothing_alpha": float(args.coarse_smoothing_alpha),
        "assist_gain": float(args.assist_gain),
        "assist_gains": parse_four(args.assist_gains),
        "proxy_local_offset_m": [0.0, 0.0, 0.0],
        "hardware_command_allowed": False,
        "interpretation": "fixed-tomato-kinematic-geometry-audit-only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
