"""Physical enclosure self-check for a hand-only BC initializer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=32000)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from simulation.ppo_tomato_grasp.physx_tomato_env import PhysxTomatoGraspEnv, PhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp.pretrain_teacher_bc import BCActor


def main() -> None:
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if (payload.get("actor_observation_dim"), payload.get("action_dim"), payload.get("converged")) != (110, 10, True):
        raise ValueError("invalid BC initializer")
    model = BCActor().to(args.device)
    model.load_state_dict(payload["actor_state_dict"])
    model.eval()
    cfg = PhysxTomatoGraspEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.training_stage = 1
    cfg.events["tomato_geometry"].params["axis_range_m"] = (0.032, 0.035)
    cfg.events["tomato_geometry"].params["isotropic_axes"] = True
    cfg.position_jitter = (0.0005, 0.0005, 0.00025)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    env = PhysxTomatoGraspEnv(cfg)
    try:
        observation, _ = env.reset(seed=args.seed)
        actor = observation["policy"]
        max_digits = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
        five_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        palm_seen = torch.zeros_like(five_seen)
        success_seen = torch.zeros_like(five_seen)
        overload_seen = torch.zeros_like(five_seen)
        for _ in range(args.steps):
            with torch.inference_mode():
                action = model(actor)
            observation, _, _, _, _ = env.step(action)
            actor = observation["policy"]
            count = (env.contact_forces >= cfg.contact_threshold_n).sum(dim=-1)
            max_digits = torch.maximum(max_digits, count)
            five_seen |= count == 5
            palm_seen |= env.palm_force >= cfg.palm_support_force_n
            success_seen |= env.success
            overload_seen |= env.overloaded
        result = {
            "seed": args.seed,
            "environments": args.num_envs,
            "steps": args.steps,
            "max_simultaneous_digits": int(max_digits.max()),
            "mean_max_simultaneous_digits": float(max_digits.float().mean()),
            "five_digit_environment_rate": float(five_seen.float().mean()),
            "palm_support_environment_rate": float(palm_seen.float().mean()),
            "stage1_success_environment_rate": float(success_seen.float().mean()),
            "overload_environment_rate": float(overload_seen.float().mean()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("BC_PHYSICS=" + json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
