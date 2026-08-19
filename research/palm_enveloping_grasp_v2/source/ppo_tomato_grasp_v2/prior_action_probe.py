"""Evaluate one staged action prior without PPO or hardware commands."""

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
parser.add_argument("--action", required=True, help="14 comma-separated normalized action values")
parser.add_argument("--profile", default="v2_band1_match")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--approach-steps", type=int, default=140)
parser.add_argument("--hold-steps", type=int, default=140)
parser.add_argument("--distal-start", type=int, default=70)
parser.add_argument("--physical-fingertip-proxy", action="store_true")
parser.add_argument("--fingertip-proxy-radius", type=float, default=None)
parser.add_argument("--disable-virtual-distal-contact", action="store_true")
parser.add_argument("--seed", type=int, default=20260913)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv, V2EnvelopingPhysxTomatoGraspEnvCfg


@torch.inference_mode()
def main() -> None:
    values = [float(v.strip()) for v in args.action.split(",") if v.strip()]
    if len(values) != 14:
        raise ValueError("--action requires 14 values")
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.episode_length_s = 16.0
    cfg.safe_force_n = 12.0
    cfg.force_ceiling_n = (12.0, 9.75, 9.75, 8.25, 8.25)
    cfg = apply_profile(cfg, args.profile, num_envs=args.num_envs, seed=args.seed)
    cfg.fingertip_collision_proxy_physical = bool(args.physical_fingertip_proxy)
    if args.fingertip_proxy_radius is not None:
        cfg.fingertip_collision_proxy_radius_m = float(args.fingertip_proxy_radius)
    if args.disable_virtual_distal_contact:
        cfg.virtual_distal_contact_enabled = False
    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)
    target = torch.tensor(values, dtype=torch.float32, device=env.device).expand(args.num_envs, -1)
    open_action = torch.zeros_like(target)
    open_action[:, :10] = torch.tensor((-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0), device=env.device)
    done_count = 0
    success = drop = overload = timeout = distal_all = distal_contact_all = 0
    strict_hold_max = strict_coverage_max = contact_geometry = 0.0
    first_palm = first_three = 0.0
    termination_counts = {}
    max_force = torch.zeros(5, device=env.device)
    max_palm_force = torch.zeros(args.num_envs, device=env.device)
    max_measured_palm_force = torch.zeros(args.num_envs, device=env.device)
    steps = args.approach_steps + args.hold_steps
    for step in range(steps):
        if step < args.approach_steps:
            f = float(step + 1) / max(args.approach_steps, 1)
        else:
            f = 1.0
        action = open_action + f * (target - open_action)
        if step < args.distal_start:
            action[:, 10:] = 0.0
        _obs, _reward, terminated, truncated, _info = env.step(action)
        max_force = torch.maximum(max_force, env.contact_forces.max(0).values)
        max_palm_force = torch.maximum(max_palm_force, env.palm_force)
        max_measured_palm_force = torch.maximum(max_measured_palm_force, env._v2_measured_palm_force)
        done = terminated | truncated
        if bool(done.any()):
            snap = env.consume_terminal_snapshots(done)
            n = int(done.sum())
            code = snap["termination_code"]
            done_count += n
            success += int((code == 1).sum())
            drop += int((code == 2).sum())
            overload += int((code == 3).sum())
            timeout += int((code == 5).sum())
            distal_all += int(snap["strict_distal_participation"].all(-1).sum())
            distal_contact_all += int(snap["distal_contact_observed"].all(-1).sum())
            strict_hold_max += float(snap["strict_hold_max"].float().sum())
            strict_coverage_max += float(snap["strict_coverage_max"].float().sum())
            contact_geometry += float(snap["strict_contact_geometry_measured"].float().sum())
            first_palm += float((snap["strict_first_palm"] >= 0).float().sum())
            first_three += float((snap["strict_first_three"] >= 0).float().sum())
            for value in code.detach().cpu().tolist():
                termination_counts[str(int(value))] = termination_counts.get(str(int(value)), 0) + 1
        if done_count >= args.num_envs:
            break
    den = max(done_count, 1)
    report = {
        "version": "staged_prior_probe_v1", "profile": args.profile, "seed": args.seed,
        "action": values, "approach_steps": args.approach_steps, "hold_steps": args.hold_steps,
        "distal_start": args.distal_start, "completed_episodes": done_count,
        "success_rate": success / den, "drop_rate": drop / den, "overload_rate": overload / den,
        "timeout_rate": timeout / den, "distal_all_rate": distal_all / den,
        "distal_contact_all_rate": distal_contact_all / den,
        "strict_hold_max_mean": strict_hold_max / den,
        "strict_coverage_max_mean_deg": strict_coverage_max / den,
        "strict_contact_geometry_rate": contact_geometry / den,
        "palm_support_seen_rate": first_palm / den,
        "three_digit_seen_rate": first_three / den,
        "termination_code_counts": termination_counts,
        "max_contact_force_n": [float(v) for v in max_force.cpu()],
        "max_palm_force_n_mean": float(max_palm_force.mean().cpu()),
        "max_palm_force_n_max": float(max_palm_force.max().cpu()),
        "max_measured_palm_force_n_mean": float(max_measured_palm_force.mean().cpu()),
        "max_measured_palm_force_n_max": float(max_measured_palm_force.max().cpu()),
        "hardware_command_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
