#!/usr/bin/env python3
"""Benchmark CPU surrogate rollout throughput for local batch selection."""

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
from simulation.ppo_tomato_grasp.hand_ppo_contract import ACTION_DIM


@torch.no_grad()
def benchmark_size(
    envs: int,
    *,
    steps: int,
    warmup: int,
    seed: int,
) -> dict[str, float | int]:
    if envs < 1 or steps < 1 or warmup < 0:
        raise ValueError("benchmark sizes and steps must be positive")
    environment = CpuHandGraspEnv(
        envs, seed=seed, curriculum_fraction=1.0, auto_reset=True
    )
    observation, critic = environment.reset()
    prior = torch.zeros(ACTION_DIM)
    prior[0] = 0.70
    prior[4] = 0.60
    policy = CpuActorCritic(initial_std=0.40, weak_prior_action=prior).eval()
    for _ in range(warmup):
        action = policy.deterministic(observation)
        result = environment.step(action)
        observation, critic = result.observation, result.critic_state
    del critic
    started = time.perf_counter()
    for _ in range(steps):
        action = policy.deterministic(observation)
        result = environment.step(action)
        observation = result.observation
    elapsed = time.perf_counter() - started
    transitions = envs * steps
    return {
        "num_envs": envs,
        "measured_steps": steps,
        "elapsed_seconds": elapsed,
        "policy_environment_steps_per_second": steps / elapsed,
        "environment_transitions_per_second": transitions / elapsed,
        "milliseconds_per_policy_step": 1000.0 * elapsed / steps,
        "all_finite": bool(torch.isfinite(observation).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=(64, 256, 1024, 4096))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20269811)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if len(set(args.sizes)) != len(args.sizes) or any(value < 1 for value in args.sizes):
        raise ValueError("sizes must be unique positive integers")
    torch.set_num_threads(args.threads)
    results = [
        benchmark_size(size, steps=args.steps, warmup=args.warmup, seed=args.seed + index)
        for index, size in enumerate(args.sizes)
    ]
    best = max(results, key=lambda item: item["environment_transitions_per_second"])
    report = {
        "schema_version": 1,
        "status": "cpu_surrogate_benchmark_not_isaac_performance",
        "torch_threads": args.threads,
        "results": results,
        "highest_throughput_num_envs": best["num_envs"],
        "highest_transitions_per_second": best["environment_transitions_per_second"],
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if all(item["all_finite"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
