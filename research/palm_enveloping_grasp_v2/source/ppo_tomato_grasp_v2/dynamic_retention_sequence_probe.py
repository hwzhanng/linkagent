"""Dynamic retention-only candidate-switch sequence diagnostic.

This is a simulation-only, non-policy probe for A-12b.  It starts from the
fixed-fruit candidate-57 seed, releases the fruit into PhysX, and then gives
one straight finger at a time a digit-specific signed root/distal reference.
The selected pair is advanced under raw link4 force feedback while all other
channels are held at their current targets.  The probe measures whether a
candidate switch can restore a missing distal contact without moving the
fruit or overloading another digit; it is not a PPO teacher or hardware path.
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
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps-per-digit", type=int, default=180)
parser.add_argument("--hold-steps", type=int, default=120)
parser.add_argument("--seed", type=int, default=20261160)
parser.add_argument("--command-rate-rad-s", type=float, default=0.10)
parser.add_argument("--step-fraction", type=float, default=0.08)
parser.add_argument("--retreat-fraction", type=float, default=0.20)
parser.add_argument("--contact-target-n", type=float, default=0.18)
parser.add_argument("--retreat-limit-n", type=float, default=0.50)
parser.add_argument("--proxy-radius", type=float, default=0.008)
parser.add_argument("--root-span", type=float, default=0.40)
parser.add_argument("--profile", default="v1_nominal_reliable")
parser.add_argument("--full-candidate-switch", action="store_true",
                    help="move all 14 channels toward each digit candidate (diagnostic ablation)")
parser.add_argument("--candidate", type=str, default=(
    "0.864469110965,0.517134726048,-0.583076238632,0.901439428329,"
    "0.113878421485,-0.767623305321,0.276706576348,-0.15316349268,"
    "0.629932641983,-0.107332348824,0.584053039551,0.197868227959,"
    "-0.820022583008,0.175772547722"
))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import (
    V2EnvelopingPhysxTomatoGraspEnv,
    V2EnvelopingPhysxTomatoGraspEnvCfg,
)


def parse_action(raw: str) -> torch.Tensor:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != 14:
        raise ValueError("--candidate requires exactly 14 comma-separated values")
    return torch.tensor(values, dtype=torch.float32)


# Per-digit physical candidates obtained from the preceding 256-env static
# reach searches.  Only each candidate's matching root/distal pair is used;
# its global channels are intentionally not copied into the sequence.
DIGIT_CANDIDATES = {
    "index": (0.616881966591, 0.900050759315, 0.935884475708, 0.882205128670,
              0.478024005890, -0.457377374172, -0.551750659943, 0.625029921532,
              0.469964861870, 0.122303366661, 0.204585552216, 0.536499261856,
              -0.603124082089, 0.493593335151),
    "middle": (0.741239905357, 0.527686357498, 0.183400630951, 0.204107165317,
               -0.339919686317, -0.103223383427, 0.254480481148, 0.998529314995,
               0.025300741196, 0.378131270409, -0.963099122047, -0.450703382492,
               0.310962319374, -0.617253363133),
    "ring": (0.448410272598, 0.992792725563, 0.009776473045, -0.641432106494,
             -0.025105893612, 0.441104054451, -0.832883238792, -0.679682016373,
             0.737260580063, -0.713629961014, 0.546782374383, 0.154125809670,
             0.263229846954, -0.042645931244),
    "little": (-0.194267570972, 0.662258028985, -0.782237708569, -0.885605096817,
               -0.995427370071, -0.304447352886, 0.048097729683, -0.451059997082,
               0.155187010765, -0.375882506371, 0.934023141861, -0.796382069588,
               0.645989537239, -0.588658332825),
}
DIGITS = ("index", "middle", "ring", "little")


@torch.inference_mode()
def main() -> None:
    if args.num_envs < 8:
        raise ValueError("--num-envs must be at least 8")
    candidate_cpu = parse_action(args.candidate)
    # The A-12a middle scale is retained as the fixed seed because it was the
    # only scale that passed the fixed-fruit force ceiling.
    seed_cpu = candidate_cpu.clone()
    seed_cpu[7] *= 0.70
    seed_cpu[11] *= 0.70

    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.retention_only = True
    cfg.retention_start_action = tuple(float(v) for v in seed_cpu.tolist())
    cfg.episode_length_s = max(100.0, (4 * args.steps_per_digit + args.hold_steps + 20) * cfg.policy_dt_s)
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
    cfg.palm_first_distal_gate = False
    cfg.per_digit_distal_gate = False
    cfg.per_digit_distal_probe_enabled = False
    cfg.distal_proximal_reach_assist_gain = 0.0
    cfg.distal_proximal_reach_assist_gain_per_digit = None
    cfg.coarse_action_smoothing_alpha = 1.0
    cfg.distal_action_filter_alpha = 0.999
    cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    cfg.debug_support_only_collision = True
    cfg.debug_collision_digit = "all"
    cfg.safe_force_n = 12.0
    cfg.force_ceiling_n = (12.0, 9.75, 9.75, 8.25, 8.25)
    cfg.command_rate_limit_rad_s = float(args.command_rate_rad_s)
    cfg = apply_profile(cfg, args.profile, num_envs=args.num_envs, seed=args.seed)
    # apply_profile returns a fresh config resolution; keep the diagnostic
    # settings explicit in the resolved run.
    cfg.retention_only = True
    cfg.retention_start_action = tuple(float(v) for v in seed_cpu.tolist())
    cfg.position_jitter = (0.0, 0.0, 0.0)
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
    current = seed_cpu.to(device).unsqueeze(0).expand(args.num_envs, -1).clone()
    seed = current.clone()
    max_force = torch.zeros(args.num_envs, 4, device=device)
    contact_steps = torch.zeros(args.num_envs, 4, device=device)
    window_steps = torch.zeros(4, dtype=torch.long, device=device)
    max_palm = torch.zeros(args.num_envs, device=device)
    max_center = torch.zeros(args.num_envs, device=device)
    min_tip_distance = torch.full((args.num_envs, 4), float("inf"), device=device)
    min_height = torch.full((args.num_envs,), float("inf"), device=device)
    dropped = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    overloaded = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    done_count = 0
    sequence_terminated = False
    digit_reports = []

    for digit_index, digit in enumerate(DIGITS):
        digit_target_cpu = torch.tensor(DIGIT_CANDIDATES[digit], dtype=torch.float32)
        target = current.clone()
        if args.full_candidate_switch:
            target[:] = digit_target_cpu.to(device)
            target[:, 7] *= 0.70
            target[:, 11] *= 0.70
        else:
            target[:, 6 + digit_index] = digit_target_cpu[6 + digit_index].to(device)
            target[:, 10 + digit_index] = digit_target_cpu[10 + digit_index].to(device)
        before_steps = contact_steps[:, digit_index].clone()
        before_force = max_force[:, digit_index].clone()
        before_center = max_center.clone()
        streak = torch.zeros(args.num_envs, device=device)
        digit_max_center = torch.zeros(args.num_envs, device=device)
        digit_max_force = torch.zeros(args.num_envs, device=device)
        digit_min_tip = torch.full((args.num_envs,), float("inf"), device=device)
        for _ in range(args.steps_per_digit):
            force = env._v2_measured_link4_force[:, digit_index + 1]
            low = force < float(args.contact_target_n)
            high = force >= float(args.retreat_limit_n)
            controlled = tuple(range(14)) if args.full_candidate_switch else (6 + digit_index, 10 + digit_index)
            advance = current[:, controlled] + float(args.step_fraction) * (target[:, controlled] - current[:, controlled])
            retreat = current[:, controlled] + float(args.retreat_fraction) * (seed[:, controlled] - current[:, controlled])
            active = streak >= 5.0
            next_values = torch.where(high.unsqueeze(-1), retreat, torch.where(active.unsqueeze(-1), current[:, controlled], advance))
            current[:, controlled] = next_values.clamp(-1.0, 1.0)
            _, _, terminated, truncated, _ = env.step(current)
            force_after = env._v2_measured_link4_force[:, digit_index + 1]
            active_after = force_after >= env.cfg.contact_threshold_n
            streak = torch.where(active_after, streak + 1.0, torch.zeros_like(streak))
            contact_steps[:, digit_index] += active_after.float()
            max_force[:, digit_index] = torch.maximum(max_force[:, digit_index], force_after)
            max_palm = torch.maximum(max_palm, env.palm_force)
            center = torch.linalg.vector_norm(env.tomato.data.root_pos_w - env.grasp_center, dim=-1)
            tip_distance = torch.linalg.vector_norm(
                env._tip_positions()[:, digit_index + 1] - env.tomato.data.root_pos_w,
                dim=-1,
            )
            min_tip_distance[:, digit_index] = torch.minimum(min_tip_distance[:, digit_index], tip_distance)
            digit_min_tip = torch.minimum(digit_min_tip, tip_distance)
            max_center = torch.maximum(max_center, center)
            digit_max_center = torch.maximum(digit_max_center, center)
            digit_max_force = torch.maximum(digit_max_force, force_after)
            min_height = torch.minimum(min_height, env.tomato.data.root_pos_w[:, 2])
            dropped |= env.dropped | (center > cfg.max_center_distance_m) | (env.tomato.data.root_pos_w[:, 2] < cfg.drop_height_m)
            overloaded |= env.overloaded | (force_after >= cfg.force_ceiling_n[digit_index + 1]) | (env.palm_force >= cfg.safe_force_n)
            done = terminated | truncated
            if bool(done.any()):
                done_count += int(done.sum())
                sequence_terminated = True
                break
        window_steps[digit_index] += int(args.steps_per_digit)
        if sequence_terminated:
            break
        for _ in range(args.hold_steps):
            _, _, terminated, truncated, _ = env.step(current)
            force_after = env._v2_measured_link4_force[:, digit_index + 1]
            active_after = force_after >= env.cfg.contact_threshold_n
            contact_steps[:, digit_index] += active_after.float()
            max_force[:, digit_index] = torch.maximum(max_force[:, digit_index], force_after)
            max_palm = torch.maximum(max_palm, env.palm_force)
            center = torch.linalg.vector_norm(env.tomato.data.root_pos_w - env.grasp_center, dim=-1)
            tip_distance = torch.linalg.vector_norm(
                env._tip_positions()[:, digit_index + 1] - env.tomato.data.root_pos_w,
                dim=-1,
            )
            min_tip_distance[:, digit_index] = torch.minimum(min_tip_distance[:, digit_index], tip_distance)
            digit_min_tip = torch.minimum(digit_min_tip, tip_distance)
            max_center = torch.maximum(max_center, center)
            min_height = torch.minimum(min_height, env.tomato.data.root_pos_w[:, 2])
            dropped |= env.dropped | (center > cfg.max_center_distance_m) | (env.tomato.data.root_pos_w[:, 2] < cfg.drop_height_m)
            overloaded |= env.overloaded | (force_after >= cfg.force_ceiling_n[digit_index + 1]) | (env.palm_force >= cfg.safe_force_n)
            done = terminated | truncated
            if bool(done.any()):
                done_count += int(done.sum())
                sequence_terminated = True
                break
        window_steps[digit_index] += int(args.hold_steps)
        digit_reports.append({
            "digit": digit,
            "contact_fraction": float((contact_steps[:, digit_index] / max(float(window_steps[digit_index]), 1.0)).mean().cpu()),
            "max_force_mean_n": float(max_force[:, digit_index].mean().cpu()),
            "max_force_max_n": float(max_force[:, digit_index].max().cpu()),
            "max_center_distance_m": float(digit_max_center.max().cpu()),
            "min_tip_distance_m": float(digit_min_tip.min().cpu()),
            "target_root": float(target[0, 6 + digit_index].cpu()),
            "target_distal": float(target[0, 10 + digit_index].cpu()),
        })
        if sequence_terminated:
            break

    report = {
        "version": "v2_dynamic_retention_sequence_probe_v1",
        "profile": args.profile,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps_per_digit": args.steps_per_digit,
        "hold_steps": args.hold_steps,
        "full_candidate_switch": bool(args.full_candidate_switch),
        "digit_reports": digit_reports,
        "per_digit_contact_fraction": (contact_steps / window_steps.clamp_min(1).unsqueeze(0)).mean(0).cpu().tolist(),
        "per_digit_max_force_mean_n": max_force.mean(0).cpu().tolist(),
        "per_digit_max_force_max_n": max_force.max(0).values.cpu().tolist(),
        "drop_rate": float(dropped.float().mean().cpu()),
        "overload_rate": float(overloaded.float().mean().cpu()),
        "retained_rate": float((~dropped & ~overloaded).float().mean().cpu()),
        "palm_support_rate": float((max_palm >= cfg.palm_support_force_n).float().mean().cpu()),
        "max_palm_force_n": float(max_palm.max().cpu()),
        "max_center_distance_m": float(max_center.max().cpu()),
        "min_tip_distance_m_per_digit": min_tip_distance.min(0).values.cpu().tolist(),
        "min_height_m": float(min_height.min().cpu()),
        "done_count": done_count,
        "sequence_terminated": sequence_terminated,
        "resolved_contact_geometry": {
            "retention_only": True,
            "dynamic_fruit": True,
            "physical_fingertip_proxy": True,
            "radius_m": float(args.proxy_radius),
            "contact_offset_m": 0.0003,
            "rest_offset_m": 0.0001,
            "debug_collision_digit": "all",
            "root_span": float(args.root_span),
        },
        "hardware_command_allowed": False,
        "interpretation": "candidate_switch_sequence_diagnostic_not_policy",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
