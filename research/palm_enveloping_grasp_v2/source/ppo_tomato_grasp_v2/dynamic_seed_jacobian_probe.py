"""Signed finite-difference audit around the dynamic retention seed.

The probe evaluates one fixed 14-D action channel at a time around the
scaled candidate-57 retention seed.  It records fingertip distances, raw
link4 forces and fruit displacement with a dynamic, released fruit.  This
separates an action-map reachability problem from a contact-controller
problem before any PPO update is allowed.
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
parser.add_argument("--num-envs", type=int, default=29)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--delta", type=float, default=0.18)
parser.add_argument("--seed", type=int, default=20261170)
parser.add_argument("--command-rate-rad-s", type=float, default=0.10)
parser.add_argument("--proxy-radius", type=float, default=0.008)
parser.add_argument("--root-span", type=float, default=0.40)
parser.add_argument("--profile", default="v1_nominal_reliable")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import (
    V2EnvelopingPhysxTomatoGraspEnv,
    V2EnvelopingPhysxTomatoGraspEnvCfg,
)


SEED_ACTION = torch.tensor((
    0.864469110965, 0.517134726048, -0.583076238632, 0.901439428329,
    0.113878421485, -0.767623305321, 0.276706576348, -0.153163492680,
    0.629932641983, -0.107332348824, 0.584053039551, 0.197868227959,
    -0.820022583008, 0.175772547723,
), dtype=torch.float32)
CHANNEL_NAMES = (
    "global_flexion", "curl_shape", "balance", "spread", "thumb_opposition", "thumb_residual",
    "index_root", "middle_root", "ring_root", "little_root",
    "index_distal", "middle_distal", "ring_distal", "little_distal",
)


@torch.inference_mode()
def main() -> None:
    labels = [("baseline", None, 0.0)]
    for channel in range(14):
        labels.append((f"{CHANNEL_NAMES[channel]}_plus", channel, 1.0))
        labels.append((f"{CHANNEL_NAMES[channel]}_minus", channel, -1.0))
    labels = labels[:args.num_envs]
    if len(labels) < 29:
        raise ValueError("--num-envs must be at least 29 for the full 14-D signed audit")
    seed_action = SEED_ACTION.clone()
    seed_action[7] *= 0.70
    seed_action[11] *= 0.70

    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.retention_only = True
    cfg.retention_start_action = tuple(float(v) for v in seed_action.tolist())
    cfg.episode_length_s = max(30.0, (args.steps + 20) * cfg.policy_dt_s)
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = False
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = False
    cfg.maximum_stem_force_n = 0.0
    cfg.maximum_stem_torque_nm = 0.0
    cfg.fingertip_collision_proxy_physical = True
    cfg.fingertip_collision_proxy_radius_m = float(args.proxy_radius)
    cfg.fingertip_proxy_contact_offset_m = 0.0003
    cfg.fingertip_proxy_rest_offset_m = 0.0001
    cfg.virtual_distal_contact_enabled = False
    cfg.debug_support_only_collision = True
    cfg.debug_collision_digit = "all"
    cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    cfg.distal_action_filter_alpha = 0.999
    cfg.coarse_action_smoothing_alpha = 1.0
    cfg.command_rate_limit_rad_s = float(args.command_rate_rad_s)
    cfg.safe_force_n = 12.0
    cfg.force_ceiling_n = (12.0, 9.75, 9.75, 8.25, 8.25)
    cfg = apply_profile(cfg, args.profile, num_envs=args.num_envs, seed=args.seed)
    cfg.retention_only = True
    cfg.retention_start_action = tuple(float(v) for v in seed_action.tolist())
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = False
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = False
    cfg.fingertip_collision_proxy_physical = True
    cfg.fingertip_collision_proxy_radius_m = float(args.proxy_radius)
    cfg.fingertip_proxy_contact_offset_m = 0.0003
    cfg.fingertip_proxy_rest_offset_m = 0.0001
    cfg.virtual_distal_contact_enabled = False
    cfg.debug_support_only_collision = True
    cfg.debug_collision_digit = "all"
    cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    cfg.command_rate_limit_rad_s = float(args.command_rate_rad_s)
    cfg.safe_force_n = 12.0
    cfg.force_ceiling_n = (12.0, 9.75, 9.75, 8.25, 8.25)

    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)
    device = env.device
    actions = seed_action.to(device).unsqueeze(0).expand(args.num_envs, -1).clone()
    for env_id, (_label, channel, sign) in enumerate(labels):
        if channel is not None:
            actions[env_id, channel] = (seed_action[channel] + sign * float(args.delta)).clamp(-1.0, 1.0)
    max_force = torch.zeros(args.num_envs, 5, device=device)
    max_palm = torch.zeros(args.num_envs, device=device)
    min_distance = torch.full((args.num_envs, 4), float("inf"), device=device)
    max_center = torch.zeros(args.num_envs, device=device)
    measured_steps = torch.zeros(args.num_envs, 4, device=device)
    dropped = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    overloaded = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    for _ in range(args.steps):
        _, _, terminated, truncated, _ = env.step(actions)
        tips = env._tip_positions()[:, 1:]
        distance = torch.linalg.vector_norm(tips - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1)
        min_distance = torch.minimum(min_distance, distance)
        max_force = torch.maximum(max_force, env.contact_forces)
        max_palm = torch.maximum(max_palm, env.palm_force)
        center = torch.linalg.vector_norm(env.tomato.data.root_pos_w - env.grasp_center, dim=-1)
        max_center = torch.maximum(max_center, center)
        measured_steps += (env._v2_measured_link4_force[:, 1:] >= cfg.contact_threshold_n).float()
        dropped |= env.dropped | (center > cfg.max_center_distance_m) | (env.tomato.data.root_pos_w[:, 2] < cfg.drop_height_m)
        overloaded |= env.overloaded | (env.palm_force >= cfg.safe_force_n)
        ceiling = torch.tensor(cfg.force_ceiling_n, device=device)
        overloaded |= (env._v2_measured_link4_force >= ceiling.unsqueeze(0)).any(-1)
        _ = terminated | truncated

    rows = []
    for env_id, (label, channel, sign) in enumerate(labels):
        rows.append({
            "label": label,
            "channel": None if channel is None else int(channel),
            "channel_name": None if channel is None else CHANNEL_NAMES[channel],
            "sign": float(sign),
            "delta": float(args.delta),
            "min_tip_distance_m": min_distance[env_id].cpu().tolist(),
            "peak_measured_link4_force_n": max_force[env_id, 1:].cpu().tolist(),
            "measured_link4_contact_fraction": (measured_steps[env_id] / max(float(args.steps), 1.0)).cpu().tolist(),
            "peak_palm_force_n": float(max_palm[env_id].cpu()),
            "max_center_distance_m": float(max_center[env_id].cpu()),
            "dropped": bool(dropped[env_id]),
            "overloaded": bool(overloaded[env_id]),
        })
    report = {
        "version": "v2_dynamic_seed_jacobian_probe_v1",
        "profile": args.profile,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed_action": seed_action.tolist(),
        "physical_fingertip_proxy": True,
        "proxy_radius_m": float(args.proxy_radius),
        "root_span": float(args.root_span),
        "rows": rows,
        "hardware_command_allowed": False,
        "interpretation": "dynamic_seed_signed_action_jacobian_diagnostic",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
