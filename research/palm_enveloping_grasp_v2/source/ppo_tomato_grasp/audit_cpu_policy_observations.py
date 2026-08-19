#!/usr/bin/env python3
"""Measure CPU-surrogate policy reliance on each deployable sensor block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.ppo_tomato_grasp.cpu_surrogate_env import CpuHandGraspEnv
from simulation.ppo_tomato_grasp.cpu_surrogate_train import CpuActorCritic
from simulation.ppo_tomato_grasp.hand_manifold import (
    LEFT_SAFE_LOWER_RAD,
    LEFT_SAFE_UPPER_RAD,
    hybrid_action_to_joint_targets,
)
from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    ACTION_DIM,
    ACTOR_OBSERVATION_BLOCKS,
    ACTOR_OBSERVATION_DIM,
    CRITIC_STATE_DIM,
    TACTILE_FEATURES_PER_FINGER,
)


EXPECTED_CONTRACT = {
    "actor_observation_dim": ACTOR_OBSERVATION_DIM,
    "critic_state_dim": CRITIC_STATE_DIM,
    "action_dim": ACTION_DIM,
    "simulation_only": True,
}


def alter_observation(observation: torch.Tensor, mode: str) -> torch.Tensor:
    altered = observation.clone()
    if mode == "nominal":
        return altered
    if mode == "no_tactile":
        altered[:, ACTOR_OBSERVATION_BLOCKS["tactile_matrix_features"]] = 0.0
    elif mode == "tactile_stale":
        block = altered[:, ACTOR_OBSERVATION_BLOCKS["tactile_matrix_features"]]
        block.zero_()
        block[:, 9::TACTILE_FEATURES_PER_FINGER] = 1.0
    elif mode == "no_object_estimate":
        for name in (
            "fruit_center_in_palm",
            "fruit_semi_axes",
            "fruit_orientation_6d",
            "fruit_twist_in_palm",
        ):
            altered[:, ACTOR_OBSERVATION_BLOCKS[name]] = 0.0
    elif mode == "no_joint_velocity":
        altered[:, ACTOR_OBSERVATION_BLOCKS["hand_joint_velocity"]] = 0.0
    elif mode == "no_previous_action":
        altered[:, ACTOR_OBSERVATION_BLOCKS["previous_action"]] = 0.0
    elif mode == "proprioception_only":
        altered.zero_()
        altered[:, ACTOR_OBSERVATION_BLOCKS["hand_joint_position"]] = observation[
            :, ACTOR_OBSERVATION_BLOCKS["hand_joint_position"]
        ]
        altered[:, ACTOR_OBSERVATION_BLOCKS["hand_joint_velocity"]] = observation[
            :, ACTOR_OBSERVATION_BLOCKS["hand_joint_velocity"]
        ]
    else:
        raise ValueError(f"unknown ablation {mode!r}")
    return altered


@torch.no_grad()
def evaluate_mode(
    policy: CpuActorCritic,
    *,
    mode: str,
    envs: int,
    seed: int,
    fraction: float,
) -> dict[str, float | int | str]:
    env = CpuHandGraspEnv(
        envs, seed=seed, curriculum_fraction=fraction, auto_reset=False
    )
    observation, _ = env.reset()
    active = torch.ones(envs, dtype=torch.bool)
    success = torch.zeros(envs, dtype=torch.bool)
    drop = torch.zeros(envs, dtype=torch.bool)
    peak_force = torch.zeros(envs)
    episode_return = torch.zeros(envs)
    action_sum = torch.zeros(ACTION_DIM)
    action_square_sum = torch.zeros(ACTION_DIM)
    action_count = 0
    action_near_limit = 0
    joint_near_limit = 0
    total_joint_samples = 0
    lower = torch.tensor(LEFT_SAFE_LOWER_RAD)
    upper = torch.tensor(LEFT_SAFE_UPPER_RAD)
    span = upper - lower
    for _ in range(env.episode_steps):
        policy_observation = alter_observation(observation, mode)
        action = policy.deterministic(policy_observation)
        result = env.step(action)
        observation = result.observation
        active_count = int(active.sum())
        if active_count:
            active_action = action[active]
            action_sum += active_action.sum(dim=0)
            action_square_sum += active_action.square().sum(dim=0)
            action_count += active_count
            action_near_limit += int((active_action.abs() >= 0.98).sum())
            target = hybrid_action_to_joint_targets(active_action)
            normalized_margin = torch.minimum(target - lower, upper - target) / span
            joint_near_limit += int((normalized_margin <= 0.01).sum())
            total_joint_samples += int(target.numel())
        success |= active & result.success
        drop |= active & result.metrics["drop"]
        peak_force = torch.where(
            active, torch.maximum(peak_force, result.metrics["max_force_n"]), peak_force
        )
        episode_return += active.float() * result.reward
        active &= ~(result.terminated | result.truncated)
        if not active.any():
            break
    mean = action_sum / max(action_count, 1)
    variance = action_square_sum / max(action_count, 1) - mean.square()
    return {
        "mode": mode,
        "envs": envs,
        "success_rate": float(success.float().mean()),
        "drop_rate": float(drop.float().mean()),
        "return_mean": float(episode_return.mean()),
        "peak_force_mean_n": float(peak_force.mean()),
        "action_absolute_mean": float(mean.abs().mean()),
        "action_temporal_std_mean": float(variance.clamp_min(0.0).sqrt().mean()),
        "action_near_limit_fraction": action_near_limit / max(action_count * ACTION_DIM, 1),
        "joint_near_limit_fraction": joint_near_limit / max(total_joint_samples, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20264811)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.envs < 32 or not 0.0 <= args.fraction <= 1.0:
        raise ValueError("invalid environment count or curriculum fraction")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != EXPECTED_CONTRACT:
        raise ValueError("checkpoint does not match frozen CPU-surrogate contract")
    policy = CpuActorCritic()
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    modes = (
        "nominal",
        "no_tactile",
        "tactile_stale",
        "no_object_estimate",
        "no_joint_velocity",
        "no_previous_action",
        "proprioception_only",
    )
    results = [
        evaluate_mode(
            policy,
            mode=mode,
            envs=args.envs,
            seed=args.seed,
            fraction=args.fraction,
        )
        for mode in modes
    ]
    nominal = results[0]["success_rate"]
    for result in results:
        result["success_rate_delta_from_nominal"] = result["success_rate"] - nominal
    report = {
        "schema_version": 1,
        "status": "cpu_surrogate_observation_ablation_not_physical_evidence",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_update": int(checkpoint["update"]),
        "device_used": "cpu",
        "same_environment_seed_for_all_modes": args.seed,
        "curriculum_fraction": args.fraction,
        "results": results,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
