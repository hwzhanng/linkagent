"""Simulation-only per-digit multi-channel reach primitive audit.

The probe deliberately disables finger collision response and virtual distal
contact.  It therefore measures the local kinematic action-to-tip-distance
map without allowing a bad action direction to launch the tomato.  It is a
diagnostic for choosing a later single-digit curriculum, never a controller
or a terminal success signal.
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
parser.add_argument("--base-action", type=str, required=True,
                    help="14 comma-separated action values around which to perturb")
parser.add_argument("--channels", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
parser.add_argument("--delta", type=float, default=0.15)
parser.add_argument("--per-digit-root-span", type=float, default=0.40,
                    help="temporary v2-only root residual span used by this audit")
parser.add_argument("--approach-steps", type=int, default=120)
parser.add_argument("--hold-steps", type=int, default=60)
parser.add_argument("--physical-fingertip-proxy", action="store_true",
                    help="enable physical link4 proxy collisions for a geometry comparison")
parser.add_argument("--debug-collision-digit", choices=("", "index", "middle", "ring", "little"), default="",
                    help="retain only this digit's link4 collision bodies")
parser.add_argument("--seed", type=int, default=20261090)
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


def parse_values(raw: str, expected: int) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != expected:
        raise ValueError(f"expected {expected} comma-separated values, got {len(values)}")
    return values


@torch.inference_mode()
def main() -> None:
    base_values = parse_values(args.base_action, 14)
    channels = [int(value.strip()) for value in args.channels.split(",") if value.strip()]
    if not channels or any(channel < 0 or channel >= 14 for channel in channels):
        raise ValueError("channels must be a non-empty subset of [0, 13]")
    labels = [("baseline", None, 0.0)]
    for channel in channels:
        labels.extend(((f"ch{channel}_plus", channel, 1.0), (f"ch{channel}_minus", channel, -1.0)))

    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.episode_length_s = max(8.0, (args.approach_steps + args.hold_steps + 20) * cfg.policy_dt_s)
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)
    # Kinematic audit only: the support surface holds the fruit while finger
    # collision response is removed.  This does not alter the training cfg.
    cfg.debug_support_only_collision = True
    cfg.debug_collision_digit = args.debug_collision_digit
    cfg.fingertip_collision_proxy_physical = bool(args.physical_fingertip_proxy)
    cfg.virtual_distal_contact_enabled = False
    cfg.virtual_palm_support_enabled = True
    cfg.palm_first_distal_gate = False
    cfg.per_digit_distal_gate = False
    cfg.per_digit_distal_probe_enabled = False
    cfg.distal_proximal_reach_assist_gain = 0.0
    cfg.distal_proximal_reach_assist_gain_per_digit = None
    cfg.per_digit_root_residual_span_fraction = float(args.per_digit_root_span)
    cfg.distal_action_filter_alpha = 0.999
    cfg.coarse_action_smoothing_alpha = 1.0
    # Freeze the object for a pure kinematic finite difference.  Otherwise
    # the stem/gravity/support dynamics can roll the tomato while the hand is
    # being probed, contaminating the action Jacobian.
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    cfg.maximum_stem_force_n = 0.0
    cfg.maximum_stem_torque_nm = 0.0
    cfg = apply_profile(cfg, args.profile, num_envs=len(labels), seed=args.seed)
    # apply_profile may restore its own diagnostic defaults; re-assert the
    # isolation switches after profile resolution.
    cfg.episode_length_s = max(20.0, (args.approach_steps + args.hold_steps + 20) * cfg.policy_dt_s)
    cfg.debug_support_only_collision = True
    cfg.debug_collision_digit = args.debug_collision_digit
    cfg.fingertip_collision_proxy_physical = bool(args.physical_fingertip_proxy)
    cfg.virtual_distal_contact_enabled = False
    cfg.palm_first_distal_gate = False
    cfg.per_digit_distal_gate = False
    cfg.per_digit_distal_probe_enabled = False
    cfg.distal_proximal_reach_assist_gain = 0.0
    cfg.distal_proximal_reach_assist_gain_per_digit = None
    cfg.per_digit_root_residual_span_fraction = float(args.per_digit_root_span)
    cfg.distal_action_filter_alpha = 0.999
    cfg.coarse_action_smoothing_alpha = 1.0
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    cfg.maximum_stem_force_n = 0.0
    cfg.maximum_stem_torque_nm = 0.0
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0

    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)
    device = env.device
    open_action = torch.zeros((len(labels), 14), device=device)
    open_action[:, :10] = torch.tensor(
        (-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        device=device,
    )
    base = torch.tensor(base_values, dtype=torch.float32, device=device)
    targets = base.unsqueeze(0).expand(len(labels), -1).clone()
    for env_id, (_label, channel, sign) in enumerate(labels):
        if channel is not None:
            targets[env_id, channel] = (targets[env_id, channel] + sign * args.delta).clamp(-1.0, 1.0)

    # Keep the thumb in the kinematic map as well.  The original audit only
    # reported the four straight fingertips, which hid a missing thumb
    # opposition direction even though v2 strict success requires all five.
    min_distance = torch.full((len(labels), 5), float("inf"), device=device)
    final_distance = torch.full((len(labels), 5), float("inf"), device=device)
    max_tomato_drift = torch.zeros(len(labels), device=device)
    baseline_tomato = env.tomato.data.root_pos_w.detach().clone()
    for step in range(args.approach_steps + args.hold_steps):
        fraction = min(float(step + 1) / max(args.approach_steps, 1), 1.0)
        action = open_action + fraction * (targets - open_action)
        env.step(action)
        tips = env._tip_positions()
        distance = torch.linalg.vector_norm(
            tips - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1
        )
        min_distance = torch.minimum(min_distance, distance)
        final_distance = distance
        drift = torch.linalg.vector_norm(env.tomato.data.root_pos_w - baseline_tomato, dim=-1)
        max_tomato_drift = torch.maximum(max_tomato_drift, drift)

    baseline = min_distance[0]
    rows = []
    for env_id, (label, channel, sign) in enumerate(labels):
        row = {
            "label": label,
            "channel": channel,
            "sign": sign,
            "min_tip_distance_m": min_distance[env_id].cpu().tolist(),
            "final_tip_distance_m": final_distance[env_id].cpu().tolist(),
            "max_tomato_drift_m": float(max_tomato_drift[env_id].cpu()),
        }
        if channel is not None:
            row["distance_change_vs_baseline_m"] = (min_distance[env_id] - baseline).cpu().tolist()
            row["closure_gain_vs_baseline_m"] = (baseline - min_distance[env_id]).cpu().tolist()
            final_baseline = final_distance[0]
            row["final_distance_change_vs_baseline_m"] = (final_distance[env_id] - final_baseline).cpu().tolist()
            row["final_closure_gain_vs_baseline_m"] = (final_baseline - final_distance[env_id]).cpu().tolist()
        rows.append(row)

    report = {
        "version": "v2_multichannel_reach_primitive_v1",
        "profile": args.profile,
        "seed": args.seed,
        "base_action": base_values,
        "channels": channels,
        "delta": float(args.delta),
        "per_digit_root_residual_span_fraction": float(args.per_digit_root_span),
        "approach_steps": args.approach_steps,
        "hold_steps": args.hold_steps,
        "isolation": {
            "finger_collision_response": bool(args.physical_fingertip_proxy),
            "physical_fingertip_proxy": bool(args.physical_fingertip_proxy),
            "debug_collision_digit": args.debug_collision_digit,
            "virtual_distal_contact": False,
            "support_surface": True,
        },
        "max_tomato_drift_m": float(max_tomato_drift.max().cpu()),
        "final_active_joint_pos_env0": env.robot.data.joint_pos[0, env.active_joint_ids].detach().cpu().tolist(),
        "rows": rows,
        "hardware_command_allowed": False,
        "interpretation": "kinematic_action_map_only_no_training_no_success_no_hardware",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
