"""Scripted per-digit contact probe for link3/link4 sensor coverage.

This is a diagnostic, not a training environment.  Each clone is assigned a
digit and one of the three weak enclosure branches.  The hand is moved from
the safe open pose to that branch, then only the assigned straight-finger
distal residual is swept.  The report keeps the measured link3/link4 forces,
tip-to-centre distance and contact-point validity separate so a zero link4
signal cannot be mistaken for a successful proximal contact.
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
parser.add_argument("--num-envs", type=int, default=15)
parser.add_argument("--approach-steps", type=int, default=80)
parser.add_argument("--sweep-steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=20260830)
parser.add_argument(
    "--physical-proxy",
    action="store_true",
    help="enable the v2-only authored fingertip collision pad for a measured-contact audit",
)
parser.add_argument("--proxy-radius", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp.enveloping_teacher import EnvelopingTeacher, EnvelopingTeacherCfg
from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import (
    V2EnvelopingPhysxTomatoGraspEnv,
    V2EnvelopingPhysxTomatoGraspEnvCfg,
)


@torch.inference_mode()
def main() -> None:
    branches = 3
    digits = 4  # index, middle, ring, little; thumb has no distal residual here
    required = branches * digits
    if args.num_envs < required:
        raise ValueError(f"num-envs must be at least {required}")

    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.episode_length_s = 20.0
    cfg.strict_enveloping_enabled = True
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)
    cfg.fingertip_collision_proxy_physical = bool(args.physical_proxy)
    if args.proxy_radius is not None:
        cfg.fingertip_collision_proxy_radius_m = float(args.proxy_radius)
    cfg = apply_profile(cfg, "v1_nominal_reliable", num_envs=args.num_envs, seed=args.seed)
    # apply_profile restores its explicit deterministic ranges; these are the
    # probe invariants and are intentionally repeated after profile resolution.
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)

    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)
    teacher = EnvelopingTeacher(
        args.num_envs,
        env.device,
        EnvelopingTeacherCfg(distal_base_action=0.40),
    )

    branch_id = torch.arange(args.num_envs, device=env.device) % branches
    digit_id = (torch.arange(args.num_envs, device=env.device) // branches) % digits
    # Reuse the three known branch priors, but make them explicit and hold the
    # roots constant during the distal sweep.
    root_scale = torch.tensor((0.60, 0.75, 1.00), device=env.device)[branch_id].unsqueeze(-1)
    teacher.root_prior[:, :5] = (teacher.root_prior[:, :5] * root_scale).clamp(-1.0, 1.0)
    teacher.residual_prior = (teacher.residual_prior * root_scale).clamp(-1.0, 1.0)
    open_action = teacher.open_action.clone()
    target = teacher.root_prior.clone()
    target[:, 5:10] = teacher.residual_prior
    target[:, 10:] = 0.0
    max_link3 = torch.zeros(args.num_envs, 5, device=env.device)
    max_measured_link4 = torch.zeros(args.num_envs, 5, device=env.device)
    max_link4 = torch.zeros(args.num_envs, 5, device=env.device)
    min_tip_distance = torch.full((args.num_envs, 5), float("inf"), device=env.device)
    max_group_force = torch.zeros(args.num_envs, 5, device=env.device)
    max_palm_force = torch.zeros(args.num_envs, device=env.device)
    max_measured_palm_force = torch.zeros(args.num_envs, device=env.device)
    contact_point_seen = torch.zeros(args.num_envs, 5, dtype=torch.bool, device=env.device)
    link4_contact_seen = torch.zeros(args.num_envs, 5, dtype=torch.bool, device=env.device)
    measured_link4_contact_seen = torch.zeros(args.num_envs, 5, dtype=torch.bool, device=env.device)
    near_surface_seen = torch.zeros(args.num_envs, 4, dtype=torch.bool, device=env.device)

    total_steps = args.approach_steps + args.sweep_steps
    for step in range(total_steps):
        if step < args.approach_steps:
            fraction = float(step + 1) / max(args.approach_steps, 1)
            action = open_action + fraction * (target - open_action)
        else:
            fraction = float(step - args.approach_steps + 1) / max(args.sweep_steps, 1)
            action = target.clone()
            # Sweep only the assigned straight-finger residual.  The V2 gate
            # may intentionally hold this at zero while the fingertip is far
            # from the surface; near_surface_seen records that fact.
            action[:, 10:] = 0.0
            action[torch.arange(args.num_envs, device=env.device), 10 + digit_id] = fraction

        env.step(action)
        tips = env._tip_positions()[:, 1:]
        distance = torch.linalg.vector_norm(
            tips - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1
        )
        min_tip_distance[:, 1:] = torch.minimum(min_tip_distance[:, 1:], distance)
        near_surface_seen |= distance < 0.045
        max_link3 = torch.maximum(max_link3, env._v2_link3_force)
        max_measured_link4 = torch.maximum(max_measured_link4, env._v2_measured_link4_force)
        max_link4 = torch.maximum(max_link4, env._v2_link4_force)
        max_group_force = torch.maximum(max_group_force, env.contact_forces)
        max_palm_force = torch.maximum(max_palm_force, env.palm_force)
        max_measured_palm_force = torch.maximum(max_measured_palm_force, env._v2_measured_palm_force)
        contact_point_seen |= env.contact_points_valid
        link4_contact_seen |= env._v2_link4_force >= env.cfg.contact_threshold_n
        measured_link4_contact_seen |= env._v2_measured_link4_force >= env.cfg.contact_threshold_n

    rows = []
    for i in range(args.num_envs):
        rows.append({
            "env_id": i,
            "branch_id": int(branch_id[i]),
            "digit": ("index", "middle", "ring", "little")[int(digit_id[i])],
            "max_link3_force_n": [float(v) for v in max_link3[i].cpu()],
            "max_measured_link4_force_n": [float(v) for v in max_measured_link4[i].cpu()],
            "max_link4_force_n": [float(v) for v in max_link4[i].cpu()],
            "max_aggregate_force_n": [float(v) for v in max_group_force[i].cpu()],
            "max_palm_force_n": float(max_palm_force[i].cpu()),
            "max_measured_palm_force_n": float(max_measured_palm_force[i].cpu()),
            "min_tip_distance_m": [float(v) for v in min_tip_distance[i].cpu()],
            "contact_point_seen": [bool(v) for v in contact_point_seen[i].cpu()],
            "link4_contact_seen": [bool(v) for v in link4_contact_seen[i].cpu()],
            "measured_link4_contact_seen": [bool(v) for v in measured_link4_contact_seen[i].cpu()],
            "near_surface_seen": [bool(v) for v in near_surface_seen[i].cpu()],
        })
    report = {
        "version": "link3_link4_contact_probe_v1",
        "profile": "v1_nominal_reliable",
        "seed": args.seed,
        "approach_steps": args.approach_steps,
        "sweep_steps": args.sweep_steps,
        "physical_fingertip_proxy": bool(args.physical_proxy),
        "fingertip_proxy_radius_m": float(cfg.fingertip_collision_proxy_radius_m),
        "num_envs": args.num_envs,
        "rows": rows,
        "link4_contact_any_rate": [float(v) for v in link4_contact_seen[:, 1:].any(0).float().cpu()],
        "measured_link4_contact_any_rate": [float(v) for v in measured_link4_contact_seen[:, 1:].any(0).float().cpu()],
        "measured_link4_max_force_n": [float(v) for v in max_measured_link4[:, 1:].amax(0).cpu()],
        "link4_max_force_n": [float(v) for v in max_link4[:, 1:].amax(0).cpu()],
        "measured_palm_max_force_n": float(max_measured_palm_force.max().cpu()),
        "palm_max_force_n": float(max_palm_force.max().cpu()),
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
