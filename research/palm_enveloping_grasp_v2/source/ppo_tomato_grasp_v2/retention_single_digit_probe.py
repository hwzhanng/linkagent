"""Retention-only single-digit contact-window probe.

The fruit is initialized in the palm-supported retention curriculum with the
stem already released.  Only one straight finger is allowed to advance toward
a weak multi-channel reach reference; the other straight fingers remain in the
open command.  The active digit uses measured link4 force feedback: it advances
in small increments below the target band, holds inside the band, and retreats
toward open above the soft ceiling.  This is a diagnostic curriculum primitive,
not a PPO policy, demonstration, or hardware command.
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
parser.add_argument("--digit", choices=("index", "middle", "ring", "little"), required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=20261130)
parser.add_argument("--profile", default="v1_nominal_reliable")
parser.add_argument("--contact-target-n", type=float, default=0.18)
parser.add_argument("--retreat-limit-n", type=float, default=0.80)
parser.add_argument("--approach-steps", type=int, default=80)
parser.add_argument("--step-fraction", type=float, default=1.0)
parser.add_argument("--retreat-fraction", type=float, default=0.08)
parser.add_argument("--hold-contact-streak", type=int, default=5,
                    help="consecutive raw-force samples before latching selected channels")
parser.add_argument("--command-rate-rad-s", type=float, default=0.45,
                    help="diagnostic joint target rate limit in rad/s")
parser.add_argument("--root-span", type=float, default=None,
                    help="simulation-only straight-finger root residual span")
parser.add_argument("--proxy-radius", type=float, default=0.004,
                    help="simulation-only physical fingertip proxy radius (m)")
parser.add_argument("--dynamic-fruit", action="store_true",
                    help="leave the fruit dynamic for retention-only contact testing")
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


@torch.inference_mode()
def main() -> None:
    if args.num_envs < 8:
        raise ValueError("--num-envs must be at least 8")
    digit_index = ("index", "middle", "ring", "little").index(args.digit)
    candidate_cpu = parse_action(args.candidate)

    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    # A-12 first stage is a fixed-fruit contact window, not dynamic retention;
    # the second stage will reuse this controller with retention_only enabled
    # only after contact reachability is established.
    cfg.retention_only = bool(args.dynamic_fruit)
    cfg.strict_enveloping_enabled = True
    cfg.episode_length_s = max(12.0, (args.steps + 20) * cfg.policy_dt_s)
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
    # Contact-window audit intentionally releases the distal command in the
    # known 100 mm reach corridor; the learned v2 gate remains at its audited
    # 45 mm default and is not changed by this diagnostic.
    cfg.per_digit_distal_near_surface_distance_m = 0.10
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = not args.dynamic_fruit
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = not args.dynamic_fruit
    cfg.maximum_stem_force_n = 0.0
    cfg.maximum_stem_torque_nm = 0.0
    if args.root_span is not None:
        cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    # Single-digit isolation: retain the palm and only the selected finger's
    # authored collision bodies.  This prevents an untested neighboring finger
    # from masking whether the target digit can carry a low-force window.
    cfg.debug_support_only_collision = True
    cfg.debug_collision_digit = args.digit
    cfg.safe_force_n = 12.0
    cfg.force_ceiling_n = (12.0, 9.75, 9.75, 8.25, 8.25)
    cfg.command_rate_limit_rad_s = float(args.command_rate_rad_s)
    cfg = apply_profile(cfg, args.profile, num_envs=args.num_envs, seed=args.seed)
    # The profile intentionally changes only the object/randomization prior;
    # retain deterministic retention and audited safety settings.
    cfg.retention_only = bool(args.dynamic_fruit)
    cfg.strict_enveloping_enabled = True
    cfg.episode_length_s = max(12.0, (args.steps + 20) * cfg.policy_dt_s)
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
    cfg.per_digit_distal_near_surface_distance_m = 0.10
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = not args.dynamic_fruit
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = not args.dynamic_fruit
    cfg.maximum_stem_force_n = 0.0
    cfg.maximum_stem_torque_nm = 0.0
    if args.root_span is not None:
        cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    cfg.debug_support_only_collision = True
    cfg.debug_collision_digit = args.digit
    cfg.safe_force_n = 12.0
    cfg.force_ceiling_n = (12.0, 9.75, 9.75, 8.25, 8.25)
    cfg.command_rate_limit_rad_s = float(args.command_rate_rad_s)

    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)
    device = env.device
    candidate = candidate_cpu.to(device)
    open_action = torch.zeros((args.num_envs, 14), device=device)
    open_action[:, :10] = torch.tensor(
        (-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        device=device,
    )

    # Keep the same palm/thumb pre-shape used by the retention start prior,
    # but explicitly open all four straight fingers.  Only the selected digit
    # receives the signed root+distal reach channels.
    retention = torch.tensor(cfg.retention_start_action, device=device)
    common = open_action[0].clone()
    # Use the signed global/thumb channels from the kinematic reach candidate;
    # otherwise the selected root/distal residual alone can remain centimetres
    # away from the fruit even though the full multi-channel combination is
    # reachable.  Only the selected finger's two residual channels are then
    # allowed to make physical contact.
    common[:6] = candidate[:6]
    # Keep the full signed multi-channel reach combination while physically
    # enabling only the selected digit.  The earlier two-channel-only trial
    # stayed ~52 mm from the fruit; the Jacobian audit already indicated that
    # global/residual coupling is required to reach the distal shell.  Since
    # non-target collision bodies are disabled, this still measures the target
    # digit's contact/retention window rather than a hidden four-finger pinch.
    target = candidate.clone()
    if args.dynamic_fruit:
        # Dynamic retention starts from the authored valid enclosure seed.  A
        # single digit is then allowed to move toward its candidate while the
        # palm/common channels and the other straight-finger residuals remain
        # at that seed.  This avoids interpreting an open-hand-to-grasp
        # transient as a retention result and keeps the diagnostic causal.
        retention_target = retention.clone()
        retention_target[:6] = retention[:6]
        retention_target[6 + digit_index] = target[6 + digit_index]
        retention_target[10 + digit_index] = target[10 + digit_index]
        current = retention.unsqueeze(0).expand(args.num_envs, -1).clone()
        target = retention_target
        open_action = current.clone()
    # The palm support is the retention fixture in this isolated probe.  Start
    # from the open action and use the same open→candidate path as the static
    # reach search; this avoids trapping the distal gate in the retention seed
    # pose while the fruit remains supported by the palm.
    if not args.dynamic_fruit:
        current = open_action.clone()
    target_batch = target.unsqueeze(0).expand(args.num_envs, -1)

    done_count = 0
    retained_count = 0
    target_seen = 0
    target_contact_steps = 0
    target_total_steps = 0
    target_hold_steps = 0.0
    target_max_streak = 0.0
    target_force_integral = 0.0
    max_target_force = 0.0
    max_palm_force = 0.0
    max_center_distance = 0.0
    min_height = float("inf")
    min_target_tip_distance = float("inf")
    drop_count = 0
    overload_count = 0
    timeout_count = 0
    target_contact_windows = []
    termination_codes: dict[str, int] = {}
    contact_streak = torch.zeros(args.num_envs, device=device)
    contact_steps_per_env = torch.zeros(args.num_envs, device=device)
    max_target_force_per_env = torch.zeros(args.num_envs, device=device)
    max_palm_force_per_env = torch.zeros(args.num_envs, device=device)
    max_center_distance_per_env = torch.zeros(args.num_envs, device=device)
    min_height_per_env = torch.full((args.num_envs,), float("inf"), device=device)
    window_drop_per_env = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    window_overload_per_env = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    episode_step = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    ever_invalid = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

    for step in range(args.steps):
        # Common palm/thumb pre-shape ramps first.  The selected finger then
        # advances slowly; all other straight fingers remain open.
        # Ramp from open to the signed reach combination, with raw link4
        # feedback controlling only the selected finger's two reach channels.
        fraction = min(float(step + 1) / max(args.approach_steps, 1), 1.0)
        desired = open_action + fraction * (target_batch - open_action)

        force = env._v2_measured_link4_force[:, digit_index + 1]
        active = force >= float(args.contact_target_n)
        contact_streak = torch.where(active, contact_streak + 1.0, torch.zeros_like(contact_streak))
        # Once the digit reaches the target band, hold its current command;
        # above the soft ceiling, retreat toward open.  Below the band, keep
        # moving toward the weak reach reference in small bounded increments.
        channels = (6 + digit_index, 10 + digit_index)
        hold = contact_streak >= float(max(args.hold_contact_streak, 1))
        for channel in channels:
            advance = current[:, channel] + float(args.step_fraction) * (desired[:, channel] - current[:, channel])
            retreat = current[:, channel] + float(args.retreat_fraction) * (open_action[:, channel] - current[:, channel])
            high = force >= float(args.retreat_limit_n)
            next_value = torch.where(high, retreat, torch.where(hold, current[:, channel], advance))
            current[:, channel] = next_value.clamp(-1.0, 1.0)
        # Ramp only common channels; the active digit was updated above.
        for channel in tuple(range(6)):
            current[:, channel] = desired[:, channel]
        # Other residual channels are allowed as kinematic coupling aids, but
        # their collision bodies are disabled by the single-digit isolation
        # switch, so they cannot contribute hidden physical contact.
        for channel in tuple(range(6, 14)):
            if channel not in channels:
                current[:, channel] = desired[:, channel]
        _, _, terminated, truncated, _ = env.step(current)
        done = terminated | truncated

        force_after = env._v2_measured_link4_force[:, digit_index + 1]
        tip_distance = torch.linalg.vector_norm(
            env._tip_positions()[:, digit_index + 1] - env.tomato.data.root_pos_w,
            dim=-1,
        )
        min_target_tip_distance = min(min_target_tip_distance, float(tip_distance.min().cpu()))
        target_seen += int((force_after >= env.cfg.contact_threshold_n).sum())
        target_contact_steps += int((force_after >= env.cfg.contact_threshold_n).sum())
        target_total_steps += int(force_after.numel())
        contact_steps_per_env += (force_after >= env.cfg.contact_threshold_n).float()
        target_force_integral += float(force_after.sum().cpu()) * float(cfg.policy_dt_s)
        target_max_streak = max(target_max_streak, float(env._v2_measured_distal_contact_streak[:, digit_index].max().cpu()))
        max_target_force = max(max_target_force, float(force_after.max().cpu()))
        max_palm_force = max(max_palm_force, float(env.palm_force.max().cpu()))
        center_distance = torch.linalg.vector_norm(env.tomato.data.root_pos_w - env.grasp_center, dim=-1)
        max_center_distance = max(max_center_distance, float(center_distance.max().cpu()))
        min_height = min(min_height, float(env.tomato.data.root_pos_w[:, 2].min().cpu()))
        max_target_force_per_env = torch.maximum(max_target_force_per_env, force_after)
        max_palm_force_per_env = torch.maximum(max_palm_force_per_env, env.palm_force)
        max_center_distance_per_env = torch.maximum(max_center_distance_per_env, center_distance)
        min_height_per_env = torch.minimum(min_height_per_env, env.tomato.data.root_pos_w[:, 2])
        window_drop_per_env |= (
            (center_distance > float(cfg.max_center_distance_m))
            | (env.tomato.data.root_pos_w[:, 2] < float(cfg.drop_height_m))
            | env.dropped
        )
        window_overload_per_env |= (
            env.overloaded
            | (force_after >= env.cfg.force_ceiling_n[digit_index + 1])
            | (env.palm_force >= env.cfg.safe_force_n)
        )
        ever_invalid |= (center_distance > cfg.max_center_distance_m) | (env.tomato.data.root_pos_w[:, 2] < cfg.drop_height_m)
        episode_step += 1

        if bool(done.any()):
            snapshot = env.consume_terminal_snapshots(done)
            count = int(done.sum())
            done_count += count
            codes = snapshot["termination_code"]
            for code in codes.detach().cpu().tolist():
                termination_codes[str(int(code))] = termination_codes.get(str(int(code)), 0) + 1
            drop_count += int((codes == 2).sum())
            overload_count += int((codes == 3).sum())
            timeout_count += int((codes == 5).sum())
            retained = (~snapshot["dropped"]) & (~snapshot["overloaded"])
            retained_count += int(retained.sum())
            target_observed = snapshot["measured_distal_contact_observed"][:, digit_index]
            target_seen += int(target_observed.sum())
            target_hold_steps += float(snapshot["measured_distal_contact_steps"][:, digit_index].sum())
            target_contact_windows.extend(snapshot["measured_distal_contact_steps"][:, digit_index].detach().cpu().tolist())
            # Reset controller state only for completed vector slots; the
            # environment has already reset the corresponding physics state.
            ids = torch.nonzero(done, as_tuple=False).flatten()
            current[ids] = open_action[ids]
            contact_streak[ids] = 0.0
            episode_step[ids] = 0

    # A retention probe is intentionally allowed to run without terminal
    # resets: a fruit that remains on the palm is the desired outcome.  When
    # no vector slot reaches the environment timeout, report horizon-level
    # window rates as the primary safety evidence rather than silently
    # returning zero completed episodes.
    den = max(done_count, 1)
    window_den = max(args.num_envs, 1)
    window_drop_rate = float(window_drop_per_env.float().mean().cpu())
    window_overload_rate = float(window_overload_per_env.float().mean().cpu())
    window_retained_rate = float((~window_drop_per_env & ~window_overload_per_env).float().mean().cpu())
    window_contact_fraction_per_env = contact_steps_per_env / max(float(args.steps), 1.0)
    report = {
        "version": "v2_retention_single_digit_probe_v1",
        "profile": args.profile,
        "seed": args.seed,
        "digit": args.digit,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "completed_episodes": done_count,
        "window_env_count": args.num_envs,
        "window_retained_rate": window_retained_rate,
        "window_drop_rate": window_drop_rate,
        "window_overload_rate": window_overload_rate,
        "retained_rate": retained_count / den,
        "drop_rate": drop_count / den,
        "overload_rate": overload_count / den,
        "timeout_rate": timeout_count / den,
        "target_observed_rate": float(sum(1 for v in target_contact_windows if v > 0.0) / max(len(target_contact_windows), 1)),
        "target_contact_fraction": target_contact_steps / max(target_total_steps, 1),
        "target_contact_fraction_per_env_mean": float(window_contact_fraction_per_env.mean().cpu()),
        "target_contact_fraction_per_env_min": float(window_contact_fraction_per_env.min().cpu()),
        "target_contact_fraction_per_env_max": float(window_contact_fraction_per_env.max().cpu()),
        "target_contact_hold_steps_mean": target_hold_steps / den,
        "target_max_contact_streak": target_max_streak,
        "target_force_integral_n_s_mean": target_force_integral / max(args.num_envs, 1),
        "target_max_force_n": max_target_force,
        "max_palm_force_n": max_palm_force,
        "max_center_distance_m": max_center_distance,
        "min_height_m": min_height,
        "max_target_force_n_per_env_max": float(max_target_force_per_env.max().cpu()),
        "max_palm_force_n_per_env_max": float(max_palm_force_per_env.max().cpu()),
        "max_center_distance_m_per_env_max": float(max_center_distance_per_env.max().cpu()),
        "min_height_m_per_env_min": float(min_height_per_env.min().cpu()),
        "min_target_tip_distance_m": min_target_tip_distance,
        "final_action_env0": current[0].detach().cpu().tolist(),
        "final_active_joint_pos_env0": env.robot.data.joint_pos[0, env.active_joint_ids].detach().cpu().tolist(),
        "final_target_pos_env0": env.prev_targets[0].detach().cpu().tolist(),
        "termination_code_counts": termination_codes,
        "controller": {
            "approach_steps": args.approach_steps,
            "contact_target_n": args.contact_target_n,
            "retreat_limit_n": args.retreat_limit_n,
            "step_fraction": args.step_fraction,
            "retreat_fraction": args.retreat_fraction,
            "command_rate_rad_s": args.command_rate_rad_s,
            "hold_contact_streak": args.hold_contact_streak,
            "non_target_straight_fingers": "open",
        },
        "resolved_contact_geometry": {
            "retention_only": False,
            "physical_fingertip_proxy": True,
            "radius_m": float(args.proxy_radius),
            "contact_offset_m": 0.0003,
            "rest_offset_m": 0.0001,
            "virtual_distal_contact": False,
            "dynamic_fruit": bool(args.dynamic_fruit),
            "per_digit_root_residual_span_fraction": args.root_span,
        },
        "hardware_command_allowed": False,
        "interpretation": (
            "dynamic_fruit_single_digit_retention_window_diagnostic"
            if args.dynamic_fruit
            else "fixed_fruit_single_digit_contact_window_diagnostic"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
