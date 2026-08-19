#!/usr/bin/env python3
"""Evaluate one CPU-surrogate checkpoint by curriculum and robustness bucket."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.ppo_tomato_grasp.cpu_surrogate_env import CpuHandGraspEnv
from simulation.ppo_tomato_grasp.cpu_surrogate_train import CpuActorCritic
from simulation.ppo_tomato_grasp.domain_randomization import (
    ROBUSTNESS_BUCKET_ORDER,
    robustness_bucket_masks,
)
from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    ACTION_DIM,
    ACTION_ORDER,
    ACTOR_OBSERVATION_DIM,
    CRITIC_STATE_DIM,
)


EXPECTED_CONTRACT = {
    "actor_observation_dim": ACTOR_OBSERVATION_DIM,
    "critic_state_dim": CRITIC_STATE_DIM,
    "action_dim": ACTION_DIM,
    "simulation_only": True,
}


@torch.no_grad()
def evaluate_once(
    policy: CpuActorCritic,
    *,
    envs: int,
    seed: int,
    curriculum_fraction: float,
) -> dict[str, object]:
    env = CpuHandGraspEnv(
        envs,
        seed=seed,
        curriculum_fraction=curriculum_fraction,
        auto_reset=False,
    )
    observation, _ = env.reset()
    masks = robustness_bucket_masks(env.profiles)
    initial_action = policy.deterministic(observation)
    active = torch.ones(envs, dtype=torch.bool)
    success = torch.zeros(envs, dtype=torch.bool)
    drop = torch.zeros(envs, dtype=torch.bool)
    peak_force = torch.zeros(envs)
    peak_damage = torch.zeros(envs)
    episode_return = torch.zeros(envs)
    for _ in range(env.episode_steps):
        action = policy.deterministic(observation)
        result = env.step(action)
        observation = result.observation
        success |= active & result.success
        drop |= active & result.metrics["drop"]
        peak_force = torch.where(
            active, torch.maximum(peak_force, result.metrics["max_force_n"]), peak_force
        )
        peak_damage = torch.where(
            active, torch.maximum(peak_damage, result.metrics["damage_score"]), peak_damage
        )
        episode_return += active.float() * result.reward
        active &= ~(result.terminated | result.truncated)
        if not active.any():
            break

    def aggregate(mask: torch.Tensor) -> dict[str, float | int | None]:
        count = int(mask.sum())
        if count == 0:
            return {"count": 0, "success_rate": None, "drop_rate": None}
        return {
            "count": count,
            "success_rate": float(success[mask].float().mean()),
            "drop_rate": float(drop[mask].float().mean()),
            "return_mean": float(episode_return[mask].mean()),
            "peak_force_mean_n": float(peak_force[mask].mean()),
            "damage_score_mean": float(peak_damage[mask].mean()),
        }

    return {
        "curriculum_fraction": curriculum_fraction,
        "seed": seed,
        "overall": aggregate(torch.ones(envs, dtype=torch.bool)),
        "buckets": {name: aggregate(masks[name]) for name in ROBUSTNESS_BUCKET_ORDER},
        "initial_deterministic_action": {
            name: {
                "mean": float(initial_action[:, index].mean()),
                "std_across_environments": float(initial_action[:, index].std(unbiased=False)),
            }
            for index, name in enumerate(ACTION_ORDER)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261811)
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=(0.0, 0.25, 0.50, 0.75, 1.0)
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.envs < 32 or args.threads < 1:
        raise ValueError("require at least 32 environments and one CPU thread")
    if any(not 0.0 <= value <= 1.0 for value in args.fractions):
        raise ValueError("fractions must lie in [0, 1]")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != EXPECTED_CONTRACT:
        raise ValueError("checkpoint does not match the frozen CPU-surrogate contract")
    policy = CpuActorCritic()
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    started = time.monotonic()
    evaluations = [
        evaluate_once(
            policy,
            envs=args.envs,
            seed=args.seed + index,
            curriculum_fraction=fraction,
        )
        for index, fraction in enumerate(args.fractions)
    ]
    report = {
        "schema_version": 1,
        "status": "cpu_surrogate_only_not_physics_or_hardware",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_update": int(checkpoint["update"]),
        "device_used": "cpu",
        "elapsed_seconds": time.monotonic() - started,
        "evaluations": evaluations,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
