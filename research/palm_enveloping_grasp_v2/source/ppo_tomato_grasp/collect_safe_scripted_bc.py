"""Collect first-episode safe left-hand demonstrations for BC stabilization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--seed", type=int, default=33000)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from simulation.ppo_tomato_grasp.physx_tomato_env import PhysxTomatoGraspEnv, PhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp.stage0_scripted_controller import Stage0ControllerCfg, Stage0ScriptedController


def main() -> None:
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
        controller = Stage0ScriptedController(
            args.num_envs, env.device,
            Stage0ControllerCfg(
                contact_threshold_n=cfg.contact_threshold_n,
                palm_support_force_n=cfg.palm_support_force_n,
                target_force_n=cfg.target_force_n,
                prelatch_seek_increment=0.003,
                seek_increment=0.001,
                retention_seek_increment=0.002,
                missing_hysteresis_steps=4,
                trim_limit=0.8,
            ),
        )
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
        observations = []
        actions = []
        weights = []
        five_seen = torch.zeros_like(alive)
        for _ in range(args.steps):
            action = controller.command(
                env.contact_forces,
                env.palm_force,
                center_error_m=env.tomato.data.root_pos_w - env.grasp_center,
                object_radius_m=env.tomato_axes.mean(dim=-1),
            )
            contact_count = (env.contact_forces >= cfg.contact_threshold_n).sum(dim=-1)
            safe = alive & (env.contact_forces <= cfg.safe_force_n).all(dim=-1) & (env.damage_score <= cfg.damage_limit)
            useful = safe & (env.palm_force >= cfg.palm_support_force_n) & (contact_count >= 2)
            background = safe & ((env.episode_length_buf % 6) == 0)
            selected = useful | background
            if bool(selected.any()):
                observations.append(actor[selected].detach().cpu())
                actions.append(action[selected].detach().cpu())
                weights.append(torch.where(contact_count[selected] == 5, 8.0, torch.where(useful[selected], 4.0, 1.0)).cpu())
            five_seen |= alive & (contact_count == 5)
            observation, _, terminated, truncated, _ = env.step(action)
            alive &= ~(terminated | truncated)
            actor = observation["policy"]
            if not bool(alive.any()):
                break
        obs, act, weight = map(torch.cat, (observations, actions, weights))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "version": "safe_scripted_left_hand_bc_v1",
            "seed": args.seed,
            "observation_dim": 110,
            "action_dim": 10,
            "observation": obs,
            "action": act,
            "weight": weight,
        }, args.output)
        summary = {
            "samples": len(obs),
            "five_digit_environment_rate": float(five_seen.float().mean()),
            "dataset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "output": str(args.output.resolve()),
        }
        args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print("SCRIPTED_BC=" + json.dumps(summary), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
