"""Simulation-only fixed-fruit thumb/straight-finger workspace search.

The search varies only the six thumb/global channels while holding a candidate
set of straight-finger residuals.  The tomato is kinematic and all fingertip
collisions are disabled, so a candidate is a geometric reach hypothesis only;
it is never a grasp, success event, or hardware command.
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
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--approach-steps", type=int, default=160)
parser.add_argument("--hold-steps", type=int, default=40)
parser.add_argument("--seed", type=int, default=20261296)
parser.add_argument("--fixed-straight", type=str, required=True,
                    help="eight values for action channels 6..13")
parser.add_argument("--random-all", action="store_true",
                    help="sample all 14 action channels instead of fixing 6..13")
parser.add_argument("--base-action", type=str, default=None,
                    help="optional 14-value local-search center")
parser.add_argument("--noise-scale", type=float, default=0.25)
parser.add_argument("--thumb-range", type=float, default=1.0)
parser.add_argument("--root-span", type=float, default=0.40)
parser.add_argument("--profile", default="v1_nominal_reliable")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv, V2EnvelopingPhysxTomatoGraspEnvCfg


def parse_values(raw: str, expected: int) -> torch.Tensor:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if len(values) != expected:
        raise ValueError(f"expected {expected} comma-separated values")
    return torch.tensor(values, dtype=torch.float32)


@torch.inference_mode()
def main() -> None:
    if args.num_envs < 8:
        raise ValueError("--num-envs must be at least 8")
    straight = parse_values(args.fixed_straight, 8)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    thumb = (2.0 * torch.rand((args.num_envs, 6), generator=generator) - 1.0) * float(args.thumb_range)
    actions = torch.zeros((args.num_envs, 14), dtype=torch.float32)
    actions[:, :6] = thumb
    actions[:, 6:] = straight.unsqueeze(0)
    if args.base_action is not None:
        base_action = parse_values(args.base_action, 14)
        actions = base_action.unsqueeze(0) + args.noise_scale * torch.randn(
            (args.num_envs, 14), generator=generator
        )
        actions = actions.clamp(-1.0, 1.0)
    elif args.random_all:
        actions = 2.0 * torch.rand((args.num_envs, 14), generator=generator) - 1.0

    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.episode_length_s = max(20.0, (args.approach_steps + args.hold_steps + 20) * cfg.policy_dt_s)
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    cfg.fingertip_collision_proxy_physical = False
    cfg.virtual_distal_contact_enabled = False
    cfg.virtual_palm_support_enabled = True
    # Match the dedicated kinematic primitive: remove all robot collision
    # response so self-collision cannot mask the reachable joint workspace.
    cfg.debug_support_only_collision = True
    cfg.debug_collision_digit = ""
    cfg.distal_action_filter_alpha = 0.999
    cfg.coarse_action_smoothing_alpha = 1.0
    cfg.distal_proximal_reach_assist_gain = 0.0
    cfg.distal_proximal_reach_assist_gain_per_digit = None
    cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    cfg = apply_profile(cfg, args.profile, num_envs=args.num_envs, seed=args.seed)
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    cfg.fingertip_collision_proxy_physical = False
    cfg.virtual_distal_contact_enabled = False
    cfg.virtual_palm_support_enabled = True
    cfg.debug_support_only_collision = True
    cfg.debug_collision_digit = ""
    cfg.distal_action_filter_alpha = 0.999
    cfg.coarse_action_smoothing_alpha = 1.0
    cfg.distal_proximal_reach_assist_gain = 0.0
    cfg.distal_proximal_reach_assist_gain_per_digit = None
    cfg.per_digit_root_residual_span_fraction = float(args.root_span)

    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)
    open_action = torch.zeros((args.num_envs, 14), device=env.device)
    open_action[:, :10] = torch.tensor(
        (-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        device=env.device,
    )
    target = actions.to(env.device)
    min_distance = torch.full((args.num_envs, 5), float("inf"), device=env.device)
    final_distance = torch.full((args.num_envs, 5), float("inf"), device=env.device)
    tomato_center = env.tomato.data.root_pos_w[0].detach().clone()
    total_steps = args.approach_steps + args.hold_steps
    for step in range(total_steps):
        fraction = min(float(step + 1) / max(args.approach_steps, 1), 1.0)
        env.step(open_action + fraction * (target - open_action))
        distance = torch.linalg.vector_norm(
            env._tip_positions() - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1
        )
        min_distance = torch.minimum(min_distance, distance)
        final_distance = distance

    # Penalize the worst straight finger, then the thumb.  A shell threshold
    # is reported for inspection only; it is not a contact/success criterion.
    shell = 0.0375 + float(cfg.fingertip_collision_proxy_radius_m)
    score = torch.maximum(min_distance[:, 1:].amax(-1), min_distance[:, 0])
    feasible = (min_distance <= shell).all(-1)
    order = torch.argsort(score)
    rows = []
    for index in order[: min(20, args.num_envs)].tolist():
        rows.append({
            "candidate_id": int(index),
            "thumb_channels_0_5": target[index, :6].cpu().tolist(),
            "fixed_straight_channels_6_13": target[index, 6:].cpu().tolist(),
            "min_tip_distance_m": min_distance[index].cpu().tolist(),
            "final_tip_distance_m": final_distance[index].cpu().tolist(),
            "worst_distance_m": float(score[index].cpu()),
            "inside_shell_all_five": bool(feasible[index].cpu()),
        })
    best = rows[0] if rows else None
    report = {
        "version": "v2_thumb_workspace_search_v1",
        "seed": args.seed,
        "num_envs": args.num_envs,
        "approach_steps": args.approach_steps,
        "hold_steps": args.hold_steps,
        "profile": args.profile,
        "fixed_straight_channels_6_13": straight.tolist(),
        "random_all": bool(args.random_all),
        "base_action": None if args.base_action is None else base_action.tolist(),
        "noise_scale": float(args.noise_scale),
        "per_digit_root_residual_span_fraction": float(args.root_span),
        "tomato_center_w": tomato_center.cpu().tolist(),
        "shell_threshold_m": shell,
        "feasible_all_five_shell_count": int(feasible.sum().cpu()),
        "best": best,
        "ranked_candidates": rows,
        "hardware_command_allowed": False,
        "interpretation": "fixed-tomato-kinematic-workspace-only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
