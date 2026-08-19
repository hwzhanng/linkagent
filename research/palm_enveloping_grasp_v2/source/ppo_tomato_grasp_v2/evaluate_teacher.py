"""Evaluate the weak staged teacher itself as a diagnostic, never as a release policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--profile", choices=("v1_training_small", "v1_nominal_reliable", "v2_band1_match"), default="v1_nominal_reliable")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=20260820)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch

from simulation.ppo_tomato_grasp.enveloping_teacher import EnvelopingTeacher
from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv, V2EnvelopingPhysxTomatoGraspEnvCfg


@torch.inference_mode()
def main():
    torch.manual_seed(args_cli.seed)
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args_cli.device
    cfg.training_stage = 1
    cfg.strict_enveloping_enabled = True
    cfg.strict_enveloping_require_stem_break = False
    cfg.episode_length_s = 12.0
    cfg.safe_force_n = 12.0
    cfg.force_ceiling_n = (12.0, 9.75, 9.75, 8.25, 8.25)
    cfg = apply_profile(cfg, args_cli.profile, num_envs=args_cli.num_envs, seed=args_cli.seed)
    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    obs, _ = env.reset(seed=args_cli.seed)
    teacher = EnvelopingTeacher(args_cli.num_envs, env.device)
    completed = torch.zeros((), dtype=torch.long, device=env.device)
    success = torch.zeros((), dtype=torch.long, device=env.device)
    drop = torch.zeros((), dtype=torch.long, device=env.device)
    for _ in range(args_cli.steps):
        action = teacher.command(env.contact_forces, env.palm_force)
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated | truncated
        if bool(done.any()):
            snap = env.consume_terminal_snapshots(done)
            completed += done.sum()
            success += snap["strict_success"].sum()
            drop += (snap["termination_code"] == 2).sum()
            ids = torch.nonzero(done, as_tuple=False).flatten()
            teacher.reset(ids)
        if int(completed) >= args_cli.num_envs:
            break
    den = completed.float().clamp_min(1.0)
    report = {"version": "v2_teacher_diagnostic_1", "profile": args_cli.profile, "seed": args_cli.seed, "completed_episodes": int(completed), "success_rate": float(success / den), "drop_rate": float(drop / den), "hardware_command_allowed": False}
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
