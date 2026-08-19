"""Search kinematic multi-channel reach primitives (simulation diagnostics only).

The tomato and support are held fixed and all finger collision response is
disabled.  Random action combinations are therefore ranked only by fingertip
distance.  The resulting actions are weak reach references for a later
single-digit curriculum; they are not terminal poses, demonstrations, or
hardware commands.
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
parser.add_argument("--approach-steps", type=int, default=140)
parser.add_argument("--hold-steps", type=int, default=60)
parser.add_argument("--seed", type=int, default=20261110)
parser.add_argument("--profile", default="v1_nominal_reliable")
parser.add_argument("--root-span", type=float, default=None)
parser.add_argument("--shell-distance", type=float, default=0.039)
parser.add_argument("--physical-fingertip-proxy", action="store_true",
                    help="enable physical fingertip proxy collision response")
parser.add_argument("--debug-collision-digit", choices=("", "index", "middle", "ring", "little"), default="",
                    help="retain only this digit's authored collision bodies")
parser.add_argument("--center-action", type=str, default=None,
                    help="optional 14D action around which to search (dynamic-seed audit)")
parser.add_argument("--delta-span", type=float, default=1.0,
                    help="uniform action perturbation half-span around --center-action")
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
        raise ValueError("--center-action requires exactly 14 comma-separated values")
    return torch.tensor(values, dtype=torch.float32)


@torch.inference_mode()
def main() -> None:
    if args.num_envs < 8:
        raise ValueError("--num-envs must be at least 8")
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.retention_only = args.center_action is not None
    cfg.episode_length_s = max(20.0, (args.approach_steps + args.hold_steps + 20) * cfg.policy_dt_s)
    cfg.position_jitter = (0.0, 0.0, 0.0)
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
    cfg.distal_action_filter_alpha = 0.999
    cfg.coarse_action_smoothing_alpha = 1.0
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    cfg.maximum_stem_force_n = 0.0
    cfg.maximum_stem_torque_nm = 0.0
    if args.root_span is not None:
        cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    center_cpu = parse_action(args.center_action) if args.center_action is not None else None
    if center_cpu is not None:
        center_cpu[7] *= 0.70
        center_cpu[11] *= 0.70
        cfg.retention_start_action = tuple(float(value) for value in center_cpu.tolist())
    cfg = apply_profile(cfg, args.profile, num_envs=args.num_envs, seed=args.seed)
    # Re-assert diagnostic isolation after profile resolution.
    cfg.episode_length_s = max(20.0, (args.approach_steps + args.hold_steps + 20) * cfg.policy_dt_s)
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.debug_support_only_collision = True
    cfg.debug_collision_digit = args.debug_collision_digit
    cfg.fingertip_collision_proxy_physical = bool(args.physical_fingertip_proxy)
    cfg.virtual_distal_contact_enabled = False
    cfg.palm_first_distal_gate = False
    cfg.per_digit_distal_gate = False
    cfg.per_digit_distal_probe_enabled = False
    cfg.distal_proximal_reach_assist_gain = 0.0
    cfg.distal_proximal_reach_assist_gain_per_digit = None
    cfg.distal_action_filter_alpha = 0.999
    cfg.coarse_action_smoothing_alpha = 1.0
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = True
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = True
    cfg.maximum_stem_force_n = 0.0
    cfg.maximum_stem_torque_nm = 0.0
    if args.root_span is not None:
        cfg.per_digit_root_residual_span_fraction = float(args.root_span)
    cfg.retention_only = args.center_action is not None
    if center_cpu is not None:
        cfg.retention_start_action = tuple(float(value) for value in center_cpu.tolist())

    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)
    device = env.device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if center_cpu is None:
        actions = 2.0 * torch.rand((args.num_envs, 14), generator=generator, device=device) - 1.0
    else:
        center = center_cpu.to(device).unsqueeze(0)
        perturbation = (2.0 * torch.rand((args.num_envs, 14), generator=generator, device=device) - 1.0) * float(args.delta_span)
        actions = (center + perturbation).clamp(-1.0, 1.0)
    # Retain a structured closure half so the search does not spend all its
    # budget in open poses that are known to be irrelevant to fingertip reach.
    if center_cpu is None:
        half = args.num_envs // 2
        actions[:half, 0] = torch.empty(half, device=device).uniform_(0.1, 1.0, generator=generator)
        actions[:half, 1] = torch.empty(half, device=device).uniform_(-0.8, 1.0, generator=generator)
        actions[:half, 4] = torch.empty(half, device=device).uniform_(0.0, 1.0, generator=generator)
        actions[:half, 5:10] = torch.empty((half, 5), device=device).uniform_(-1.0, 1.0, generator=generator)
        actions[:half, 10:] = torch.empty((half, 4), device=device).uniform_(-1.0, 1.0, generator=generator)
    actions.clamp_(-1.0, 1.0)
    if center_cpu is None:
        open_action = torch.zeros_like(actions)
        open_action[:, :10] = torch.tensor(
            (-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
            device=device,
        )
    else:
        open_action = center_cpu.to(device).unsqueeze(0).expand(args.num_envs, -1).clone()
    min_distance = torch.full((args.num_envs, 4), float("inf"), device=device)
    final_distance = torch.full((args.num_envs, 4), float("inf"), device=device)
    tomato_start = env.tomato.data.root_pos_w.detach().clone()
    for step in range(args.approach_steps + args.hold_steps):
        fraction = min(float(step + 1) / max(args.approach_steps, 1), 1.0)
        env.step(open_action + fraction * (actions - open_action))
        tips = env._tip_positions()[:, 1:]
        distance = torch.linalg.vector_norm(
            tips - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1
        )
        min_distance = torch.minimum(min_distance, distance)
        final_distance = distance
    drift = torch.linalg.vector_norm(env.tomato.data.root_pos_w - tomato_start, dim=-1)

    shell = float(args.shell_distance)
    composite_mask = min_distance.max(-1).values < shell
    composite_order = torch.argsort(min_distance.max(-1).values)
    rows = []
    for idx in composite_order[: min(20, args.num_envs)].tolist():
        rows.append({
            "candidate_id": int(idx),
            "action": actions[idx].cpu().tolist(),
            "min_tip_distance_m": min_distance[idx].cpu().tolist(),
            "final_tip_distance_m": final_distance[idx].cpu().tolist(),
            "max_tip_distance_m": float(min_distance[idx].max().cpu()),
            "all_four_inside_shell": bool(composite_mask[idx]),
            "tomato_drift_m": float(drift[idx].cpu()),
        })
    per_digit = []
    for digit in range(4):
        order = torch.argsort(min_distance[:, digit])
        per_digit.append({
            "digit": ("index", "middle", "ring", "little")[digit],
            "best_distance_m": float(min_distance[order[0], digit].cpu()),
            "top_candidates": [
                {
                    "candidate_id": int(idx),
                    "distance_m": float(min_distance[idx, digit].cpu()),
                    "action": actions[idx].cpu().tolist(),
                    "all_four_min_distance_m": min_distance[idx].cpu().tolist(),
                }
                for idx in order[:5].tolist()
            ],
        })
    report = {
        "version": "v2_reach_primitive_search_v1",
        "profile": args.profile,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "approach_steps": args.approach_steps,
        "hold_steps": args.hold_steps,
        "root_span": args.root_span,
        "shell_distance_m": shell,
        "physical_fingertip_proxy": bool(args.physical_fingertip_proxy),
        "debug_collision_digit": args.debug_collision_digit,
        "center_action": None if center_cpu is None else center_cpu.cpu().tolist(),
        "delta_span": float(args.delta_span),
        "all_four_inside_shell_rate": float(composite_mask.float().mean().cpu()),
        "max_tomato_drift_m": float(drift.max().cpu()),
        "per_digit": per_digit,
        "top_composite": rows,
        "hardware_command_allowed": False,
        "interpretation": "kinematic_reach_reference_only_no_training_no_success_no_hardware",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
