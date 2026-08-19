"""GPU smoke test for state-consistent retention capture/restore."""

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
parser.add_argument("--num-envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import V2EnvelopingPhysxTomatoGraspEnv, V2EnvelopingPhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp_v2.retention_snapshot import capture_retention_snapshot, restore_retention_snapshot


def main() -> None:
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.episode_length_s = 12.0
    cfg.safe_force_n = 12.0
    cfg.force_ceiling_n = (12.0, 9.75, 9.75, 8.25, 8.25)
    cfg = apply_profile(cfg, "v2_band1_match", num_envs=args.num_envs, seed=20260826)
    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    obs, _ = env.reset(seed=20260826)
    zero = torch.zeros((args.num_envs, 14), device=env.device)
    env.step(zero)
    snapshot = capture_retention_snapshot(env, [0])
    before = {name: value[0].detach().clone() for name, value in snapshot["tensor_fields"].items()}
    env.step(torch.ones_like(zero).clamp(-0.5, 0.5))
    restore_retention_snapshot(env, snapshot, [0])
    max_error = 0.0
    for name, expected in before.items():
        actual = getattr(env, name)[0].detach()
        if actual.dtype == torch.bool:
            error = float(torch.logical_xor(actual, expected).any().item())
        else:
            error = float((actual - expected).abs().max().item())
        max_error = max(max_error, error)
    report = {
        "version": "v2_retention_snapshot_smoke_1",
        "num_envs": args.num_envs,
        "tensor_field_count": len(before),
        "max_tensor_error": max_error,
        "passed": max_error < 1.0e-6,
        "hardware_command_allowed": False,
    }
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
