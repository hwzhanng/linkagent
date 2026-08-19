"""Headless integration smoke for the 14D strict enclosure environment."""

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
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from simulation.ppo_tomato_grasp.physx_enveloping_env import (
    EnvelopingPhysxTomatoGraspEnv,
    EnvelopingPhysxTomatoGraspEnvCfg,
)


def main() -> None:
    cfg = EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.events["tomato_geometry"].params["axis_range_m"] = (0.035, 0.050)
    cfg.events["tomato_geometry"].params["isotropic_axes"] = False
    cfg.events["tomato_geometry"].params["accepted_mass_kg"] = (0.02, 1.20)
    cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (700.0, 1050.0)
    cfg.position_jitter = (0.001, 0.001, 0.0005)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.tactile_gain_range = (1.0, 1.0)
    cfg.tactile_bias_range_n = (0.0, 0.0)
    cfg.friction_range = (0.70, 0.70)
    cfg.restitution_range = (0.0, 0.0)
    cfg.stiffness_range_n_m = (18000.0, 18000.0)
    env = EnvelopingPhysxTomatoGraspEnv(cfg)
    obs, _ = env.reset()
    action = torch.zeros((env.num_envs, 14), device=env.device)
    # A non-zero distal command proves the new path is accepted end-to-end.
    action[:, 10:] = 0.20
    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    measured_geometry_rate_max = 0.0
    for _ in range(args_cli.steps):
        obs, reward, term, trunc, info = env.step(action)
        terminated |= term | trunc
        measured_geometry_rate_max = max(
            measured_geometry_rate_max,
            float(info.get("log", {}).get("strict_contact_geometry_measured_rate", 0.0)),
        )
    report = {
        "schema_version": 1,
        "action_dim": env.action_dim,
        "observation_shape": list(obs["policy"].shape),
        "reward_shape": list(reward.shape),
        "terminated_count": int(terminated.sum()),
        "strict_enabled": bool(cfg.strict_enveloping_enabled),
        "strict_success_count": int(env.strict_success_latched.sum()),
        "contact_geometry_measured_rate_max": measured_geometry_rate_max,
        "contact_geometry_proxy": measured_geometry_rate_max <= 0.0,
        "hardware_command_allowed": False,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
