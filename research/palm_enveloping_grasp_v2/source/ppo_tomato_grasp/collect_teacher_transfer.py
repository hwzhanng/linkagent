"""Collect physically screened Clean-v5 teacher labels in the left-hand task."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=31000)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from simulation.ppo_tomato_grasp.physx_tomato_env import (
    PhysxTomatoGraspEnv,
    PhysxTomatoGraspEnvCfg,
)
from simulation.ppo_tomato_grasp.teacher_transfer import (
    CleanV5Teacher,
    hand_actor_to_teacher_observation,
    teacher_action_to_hand_action,
)


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
        teacher = CleanV5Teacher.load(args.checkpoint, env.device)
        observation, _ = env.reset(seed=args.seed)
        actor = observation["policy"]
        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        useful_count = safe_count = 0
        max_simultaneous = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
        enclosure_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        palm_seen = torch.zeros_like(enclosure_seen)
        for _ in range(args.steps):
            with torch.inference_mode():
                source_obs = hand_actor_to_teacher_observation(actor)
                source_mean = teacher(source_obs)
                action = teacher_action_to_hand_action(source_mean)
            contact_count = (env.contact_forces >= cfg.contact_threshold_n).sum(dim=-1)
            safe = (
                (env.contact_forces <= cfg.safe_force_n).all(dim=-1)
                & (env.damage_score <= cfg.damage_limit)
                & ~env.dropped
            )
            useful = safe & (env.palm_force >= cfg.palm_support_force_n) & (contact_count >= 2)
            # Keep a smaller background sample so BC learns how the teacher
            # approaches contact, while weighting contact-bearing samples.
            background = safe & ((env.episode_length_buf % 8) == 0)
            selected = useful | background
            if bool(selected.any()):
                observations.append(actor[selected].detach().cpu())
                actions.append(action[selected].detach().cpu())
                weights.append(torch.where(useful[selected], 4.0, 1.0).detach().cpu())
            useful_count += int(useful.sum())
            safe_count += int(safe.sum())
            max_simultaneous = torch.maximum(max_simultaneous, contact_count)
            enclosure_seen |= contact_count == 5
            palm_seen |= env.palm_force >= cfg.palm_support_force_n
            observation, _, _, _, _ = env.step(action)
            actor = observation["policy"]

        if not observations:
            raise RuntimeError("teacher transfer produced no safe samples")
        obs = torch.cat(observations)
        act = torch.cat(actions)
        weight = torch.cat(weights)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": "clean_v5_to_left_hand_bc_v1",
                "teacher_checkpoint": str(args.checkpoint.resolve()),
                "teacher_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "seed": args.seed,
                "observation_dim": 110,
                "action_dim": 10,
                "observation": obs,
                "action": act,
                "weight": weight,
            },
            args.output,
        )
        summary = {
            "samples": len(obs),
            "useful_samples": useful_count,
            "safe_state_visits": safe_count,
            "environments": args.num_envs,
            "steps": args.steps,
            "max_simultaneous_digits": int(max_simultaneous.max()),
            "five_digit_environment_rate": float(enclosure_seen.float().mean()),
            "palm_support_environment_rate": float(palm_seen.float().mean()),
            "output": str(args.output.resolve()),
            "dataset_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print("TEACHER_TRANSFER=" + json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
