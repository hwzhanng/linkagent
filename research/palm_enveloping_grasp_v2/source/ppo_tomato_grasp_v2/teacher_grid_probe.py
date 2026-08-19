"""Vectorized search over staged-teacher closure scales.

This is a feasibility probe, not a learned policy.  It identifies whether the
current geometry needs a different root/distal approach before spending PPO
updates on an unreachable prior.
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
parser.add_argument("--profile", choices=("v1_training_small", "v1_nominal_reliable", "v2_band1_match"), default="v1_training_small")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=20260820)
parser.add_argument("--distal-base", type=float, default=0.15)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp.enveloping_teacher import EnvelopingTeacher, EnvelopingTeacherCfg
from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv, V2EnvelopingPhysxTomatoGraspEnvCfg


@torch.inference_mode()
def main() -> None:
    variants = [(r, d) for r in (0.60, 0.75, 1.0, 1.25) for d in (0.5, 1.0, 2.0, 3.0)]
    if args.num_envs < len(variants):
        raise ValueError(f"num-envs must be at least {len(variants)}")
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.episode_length_s = 12.0
    cfg.safe_force_n = 12.0
    cfg.force_ceiling_n = (12.0, 9.75, 9.75, 8.25, 8.25)
    cfg = apply_profile(cfg, args.profile, num_envs=args.num_envs, seed=args.seed)
    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    obs, _ = env.reset(seed=args.seed)
    teacher = EnvelopingTeacher(args.num_envs, env.device, EnvelopingTeacherCfg(distal_base_action=args.distal_base))
    variant_ids = torch.arange(args.num_envs, device=env.device) % len(variants)
    root_scale = torch.tensor([variants[int(i)][0] for i in variant_ids.cpu()], device=env.device).unsqueeze(-1)
    distal_scale = torch.tensor([variants[int(i)][1] for i in variant_ids.cpu()], device=env.device).unsqueeze(-1)
    teacher.root_prior[:, :5] = (teacher.root_prior[:, :5] * root_scale).clamp(-1.0, 1.0)
    teacher.residual_prior = (teacher.residual_prior * root_scale).clamp(-1.0, 1.0)
    teacher.distal_trim.zero_()
    success = torch.zeros(len(variants), device=env.device)
    completed = torch.zeros(len(variants), device=env.device)
    drop = torch.zeros(len(variants), device=env.device)
    distal_contact = torch.zeros(len(variants), 4, device=env.device)
    for _ in range(args.steps):
        action = teacher.command(env.contact_forces, env.palm_force)
        action[:, 10:] = (action[:, 10:] * distal_scale).clamp(-1.0, 1.0)
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated | truncated
        if bool(done.any()):
            snap = env.consume_terminal_snapshots(done)
            ids = torch.nonzero(done, as_tuple=False).flatten()
            vid = variant_ids[ids]
            for v in range(len(variants)):
                mask = vid == v
                completed[v] += mask.sum()
                success[v] += (mask & snap["strict_success"]).sum()
                drop[v] += (mask & (snap["termination_code"] == 2)).sum()
                distal_contact[v] += snap["distal_contact_observed"][mask].float().sum(0)
            teacher.reset(ids)
        if bool((completed.sum() >= args.num_envs).item()):
            break
    report = []
    for v, (r, d) in enumerate(variants):
        den = completed[v].clamp_min(1.0)
        report.append({"root_scale": r, "distal_scale": d, "completed": int(completed[v]), "success_rate": float(success[v] / den), "drop_rate": float(drop[v] / den), "distal_contact_observed_rate": [float(x / den) for x in distal_contact[v]]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"version": "teacher_grid_probe_1", "profile": args.profile, "seed": args.seed, "variants": report, "hardware_command_allowed": False}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
