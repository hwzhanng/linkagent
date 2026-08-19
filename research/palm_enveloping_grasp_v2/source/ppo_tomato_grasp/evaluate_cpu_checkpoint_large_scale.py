#!/usr/bin/env python3
"""Streaming large-sample evaluation with confidence intervals and buckets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.ppo_tomato_grasp.cpu_surrogate_env import SURROGATE_ENV_VERSION, CpuHandGraspEnv
from simulation.ppo_tomato_grasp.cpu_surrogate_train import CpuActorCritic
from simulation.ppo_tomato_grasp.domain_randomization import (
    RANDOMIZATION_VERSION,
    ROBUSTNESS_BUCKET_ORDER,
    robustness_bucket_masks,
)
from simulation.ppo_tomato_grasp.hand_grasp_mdp import REWARD_VERSION
from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_STATE_DIM,
)


TENSOR_CONTRACT = {
    "actor_observation_dim": ACTOR_OBSERVATION_DIM,
    "critic_state_dim": CRITIC_STATE_DIM,
    "action_dim": ACTION_DIM,
    "simulation_only": True,
}
ENVIRONMENT_CONTRACT = {
    "surrogate_env_version": SURROGATE_ENV_VERSION,
    "reward_version": REWARD_VERSION,
    "randomization_version": RANDOMIZATION_VERSION,
}
PROFILE_FIELDS = (
    "mean_radius_m",
    "mass_kg",
    "ripeness",
    "wetness",
    "contact_stiffness_n_m",
    "static_friction",
    "mean_actuator_response",
    "failed_tactile_sensor_count",
)
STRESS_SLICE_ORDER = (
    "small_and_ripe",
    "small_ripe_and_wet_low_friction",
)


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count < 1 or successes < 0 or successes > count:
        raise ValueError("invalid binomial counts")
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    ) / denominator
    return center - radius, center + radius


def _empty_counter() -> dict[str, float | int]:
    return {
        "count": 0,
        "successes": 0,
        "drops": 0,
        "peak_force_sum_n": 0.0,
        "damage_sum": 0.0,
    }


def _profile_values(profiles: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "mean_radius_m": profiles["axes_m"][:, :2].mean(dim=1),
        "mass_kg": profiles["mass_kg"],
        "ripeness": profiles["ripeness"],
        "wetness": profiles["wetness"],
        "contact_stiffness_n_m": profiles["contact_stiffness_n_m"],
        "static_friction": profiles["static_friction"],
        "mean_actuator_response": profiles["actuator_response_scale"].mean(dim=1),
        "failed_tactile_sensor_count": profiles["tactile_failure_mask"].sum(dim=1).float(),
    }


@torch.no_grad()
def evaluate_batch(
    policy: CpuActorCritic,
    *,
    count: int,
    seed: int,
) -> dict[str, object]:
    env = CpuHandGraspEnv(count, seed=seed, curriculum_fraction=1.0, auto_reset=False)
    observation, _ = env.reset()
    masks = robustness_bucket_masks(env.profiles)
    stress_slices = {
        "small_and_ripe": masks["small"] & masks["ripe_soft"],
        "small_ripe_and_wet_low_friction": (
            masks["small"] & masks["ripe_soft"] & masks["wet_low_friction"]
        ),
    }
    active = torch.ones(count, dtype=torch.bool)
    success = torch.zeros(count, dtype=torch.bool)
    drop = torch.zeros(count, dtype=torch.bool)
    peak_force = torch.zeros(count)
    peak_damage = torch.zeros(count)
    for _ in range(env.episode_steps):
        result = env.step(policy.deterministic(observation))
        observation = result.observation
        success |= active & result.success
        drop |= active & result.metrics["drop"]
        peak_force = torch.where(
            active, torch.maximum(peak_force, result.metrics["max_force_n"]), peak_force
        )
        peak_damage = torch.where(
            active, torch.maximum(peak_damage, result.metrics["damage_score"]), peak_damage
        )
        active &= ~(result.terminated | result.truncated)
        if not active.any():
            break
    return {
        "success": success,
        "drop": drop,
        "peak_force": peak_force,
        "peak_damage": peak_damage,
        "masks": masks,
        "stress_slices": stress_slices,
        "profiles": _profile_values(env.profiles),
    }


def _accumulate(
    counter: dict[str, float | int],
    mask: torch.Tensor,
    batch: dict[str, object],
) -> None:
    success = batch["success"]
    drop = batch["drop"]
    peak_force = batch["peak_force"]
    peak_damage = batch["peak_damage"]
    assert isinstance(success, torch.Tensor)
    assert isinstance(drop, torch.Tensor)
    assert isinstance(peak_force, torch.Tensor)
    assert isinstance(peak_damage, torch.Tensor)
    counter["count"] += int(mask.sum())
    counter["successes"] += int(success[mask].sum())
    counter["drops"] += int(drop[mask].sum())
    counter["peak_force_sum_n"] += float(peak_force[mask].sum())
    counter["damage_sum"] += float(peak_damage[mask].sum())


def _finalize(counter: dict[str, float | int]) -> dict[str, float | int | list[float]]:
    count = int(counter["count"])
    successes = int(counter["successes"])
    drops = int(counter["drops"])
    if count < 1:
        return {"count": 0, "successes": 0, "drops": 0}
    return {
        "count": count,
        "successes": successes,
        "drops": drops,
        "success_rate": successes / count,
        "success_rate_wilson_95": list(wilson_interval(successes, count)),
        "drop_rate": drops / count,
        "drop_rate_wilson_95": list(wilson_interval(drops, count)),
        "peak_force_mean_n": float(counter["peak_force_sum_n"]) / count,
        "damage_score_mean": float(counter["damage_sum"]) / count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20270811)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.batch_size < 32 or args.threads < 1:
        raise ValueError("invalid large-scale evaluation settings")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != TENSOR_CONTRACT:
        raise ValueError("checkpoint tensor contract is incompatible")
    if checkpoint.get("environment_contract") != ENVIRONMENT_CONTRACT:
        raise ValueError("checkpoint environment contract is incompatible")
    policy = CpuActorCritic()
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    overall = _empty_counter()
    buckets = {name: _empty_counter() for name in ROBUSTNESS_BUCKET_ORDER}
    stress_slices = {name: _empty_counter() for name in STRESS_SLICE_ORDER}
    profile_sum = {
        outcome: {name: 0.0 for name in PROFILE_FIELDS}
        for outcome in ("success", "failure")
    }
    profile_count = {"success": 0, "failure": 0}
    started = time.monotonic()
    completed = 0
    batch_index = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while completed < args.episodes:
        count = min(args.batch_size, args.episodes - completed)
        batch = evaluate_batch(policy, count=count, seed=args.seed + batch_index)
        all_mask = torch.ones(count, dtype=torch.bool)
        _accumulate(overall, all_mask, batch)
        masks = batch["masks"]
        assert isinstance(masks, dict)
        for name in ROBUSTNESS_BUCKET_ORDER:
            _accumulate(buckets[name], masks[name], batch)
        batch_stress_slices = batch["stress_slices"]
        assert isinstance(batch_stress_slices, dict)
        for name in STRESS_SLICE_ORDER:
            _accumulate(stress_slices[name], batch_stress_slices[name], batch)
        success = batch["success"]
        profiles = batch["profiles"]
        assert isinstance(success, torch.Tensor)
        assert isinstance(profiles, dict)
        for outcome, mask in (("success", success), ("failure", ~success)):
            profile_count[outcome] += int(mask.sum())
            for name in PROFILE_FIELDS:
                profile_sum[outcome][name] += float(profiles[name][mask].sum())
        completed += count
        batch_index += 1
        progress = {
            "status": "running" if completed < args.episodes else "complete",
            "episodes_completed": completed,
            "episodes_requested": args.episodes,
            "batches_completed": batch_index,
            "elapsed_seconds": time.monotonic() - started,
            "overall": _finalize(overall),
        }
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(args.output)
        print(json.dumps(progress, sort_keys=True), flush=True)
    profile_means = {
        outcome: {
            name: profile_sum[outcome][name] / max(profile_count[outcome], 1)
            for name in PROFILE_FIELDS
        }
        for outcome in ("success", "failure")
    }
    report = {
        "schema_version": 1,
        "status": "cpu_surrogate_large_scale_only_not_physical_evidence",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_update": int(checkpoint["update"]),
        "environment_contract": ENVIRONMENT_CONTRACT,
        "episodes": completed,
        "batches": batch_index,
        "elapsed_seconds": time.monotonic() - started,
        "overall": _finalize(overall),
        "buckets": {name: _finalize(counter) for name, counter in buckets.items()},
        "stress_slices": {
            name: _finalize(counter) for name, counter in stress_slices.items()
        },
        "profile_means_by_outcome": profile_means,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
