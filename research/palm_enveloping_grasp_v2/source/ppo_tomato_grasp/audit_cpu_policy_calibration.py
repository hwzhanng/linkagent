#!/usr/bin/env python3
"""Stress a CPU policy with deployable-observation calibration errors."""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
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


SCENARIOS = (
    "baseline",
    "joint_zero_bias_positive_2deg",
    "joint_zero_bias_negative_2deg",
    "vision_center_bias_x_5mm",
    "vision_center_bias_z_5mm",
    "tactile_gain_0p70",
    "tactile_gain_1p30",
    "observation_delay_2_steps",
    "observation_delay_5_steps",
)
EXTENDED_SCENARIOS = (
    "joint_zero_bias_positive_5deg",
    "joint_zero_bias_negative_5deg",
    "vision_center_bias_x_10mm",
    "vision_center_bias_z_10mm",
    "vision_center_bias_x_20mm",
    "tactile_gain_0p50",
    "tactile_gain_1p50",
    "observation_delay_10_steps",
)


def _delay_steps(scenario: str) -> int:
    if not scenario.startswith("observation_delay_"):
        return 0
    try:
        return int(scenario.removeprefix("observation_delay_").removesuffix("_steps"))
    except ValueError as error:
        raise ValueError(f"unknown calibration scenario {scenario!r}") from error


def perturb_observation(observation: torch.Tensor, scenario: str) -> torch.Tensor:
    result = observation.clone()
    if scenario == "baseline" or scenario.startswith("observation_delay_"):
        _delay_steps(scenario)
        return result
    if scenario.startswith("joint_zero_bias_"):
        sign = 1.0 if "positive" in scenario else -1.0
        degrees = 5.0 if scenario.endswith("5deg") else 2.0 if scenario.endswith("2deg") else None
        if degrees is None:
            raise ValueError(f"unknown calibration scenario {scenario!r}")
        span = torch.tensor(LEFT_SAFE_UPPER_RAD) - torch.tensor(LEFT_SAFE_LOWER_RAD)
        normalized_bias = sign * 2.0 * math.radians(degrees) / span
        section = ACTOR_OBSERVATION_BLOCKS["hand_joint_position"]
        result[:, section] = (result[:, section] + normalized_bias).clamp(-1.0, 1.0)
    elif scenario.startswith("vision_center_bias_"):
        parts = scenario.split("_")
        axis = parts[-2]
        millimeters_text = parts[-1]
        if axis not in ("x", "z") or not millimeters_text.endswith("mm"):
            raise ValueError(f"unknown calibration scenario {scenario!r}")
        millimeters = float(millimeters_text.removesuffix("mm"))
        section = ACTOR_OBSERVATION_BLOCKS["fruit_center_in_palm"]
        offset = section.start if axis == "x" else section.start + 2
        result[:, offset] = (result[:, offset] + (millimeters / 1000.0) / 0.08).clamp(-1.0, 1.0)
    elif scenario.startswith("tactile_gain_"):
        try:
            gain = float(scenario.removeprefix("tactile_gain_").replace("p", "."))
        except ValueError as error:
            raise ValueError(f"unknown calibration scenario {scenario!r}") from error
        if gain <= 0.0:
            raise ValueError(f"unknown calibration scenario {scenario!r}")
        tactile = result[:, ACTOR_OBSERVATION_BLOCKS["tactile_matrix_features"]]
        for offset in (0, 1, 7):
            tactile[:, offset::TACTILE_FEATURES_PER_FINGER] = (
                tactile[:, offset::TACTILE_FEATURES_PER_FINGER] * gain
            ).clamp(-1.0, 1.0)
    else:
        raise ValueError(f"unknown calibration scenario {scenario!r}")
    return result


@torch.no_grad()
def evaluate_scenario(
    policy: CpuActorCritic,
    *,
    scenario: str,
    envs: int,
    seed: int,
) -> dict[str, float | int | str]:
    env = CpuHandGraspEnv(envs, seed=seed, curriculum_fraction=1.0, auto_reset=False)
    observation, _ = env.reset()
    delay = _delay_steps(scenario)
    history: deque[torch.Tensor] = deque(
        (observation.clone() for _ in range(delay + 1)), maxlen=delay + 1
    )
    active = torch.ones(envs, dtype=torch.bool)
    success = torch.zeros(envs, dtype=torch.bool)
    drop = torch.zeros(envs, dtype=torch.bool)
    peak_force = torch.zeros(envs)
    episode_return = torch.zeros(envs)
    action_limit_count = 0
    joint_limit_count = 0
    action_samples = 0
    joint_samples = 0
    lower = torch.tensor(LEFT_SAFE_LOWER_RAD)
    upper = torch.tensor(LEFT_SAFE_UPPER_RAD)
    span = upper - lower
    for _ in range(env.episode_steps):
        delayed = history[0]
        policy_observation = perturb_observation(delayed, scenario)
        action = policy.deterministic(policy_observation)
        result = env.step(action)
        history.append(result.observation.clone())
        active_action = action[active]
        action_limit_count += int((active_action.abs() >= 0.98).sum())
        action_samples += int(active_action.numel())
        target = hybrid_action_to_joint_targets(active_action)
        margin = torch.minimum(target - lower, upper - target) / span
        joint_limit_count += int((margin <= 0.01).sum())
        joint_samples += int(target.numel())
        success |= active & result.success
        drop |= active & result.metrics["drop"]
        peak_force = torch.where(
            active, torch.maximum(peak_force, result.metrics["max_force_n"]), peak_force
        )
        episode_return += active.float() * result.reward
        active &= ~(result.terminated | result.truncated)
        if not active.any():
            break
    return {
        "scenario": scenario,
        "envs": envs,
        "success_rate": float(success.float().mean()),
        "drop_rate": float(drop.float().mean()),
        "peak_force_mean_n": float(peak_force.mean()),
        "return_mean": float(episode_return.mean()),
        "action_near_limit_fraction": action_limit_count / max(action_samples, 1),
        "joint_near_limit_fraction": joint_limit_count / max(joint_samples, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20267811)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--extended",
        action="store_true",
        help="also scan larger biases/gain errors and ten-step delay to find breakpoints",
    )
    args = parser.parse_args()
    if args.envs < 32 or args.threads < 1:
        raise ValueError("invalid environment or CPU thread count")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != EXPECTED_CONTRACT:
        raise ValueError("checkpoint does not match frozen CPU-surrogate contract")
    policy = CpuActorCritic()
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    scenario_names = SCENARIOS + EXTENDED_SCENARIOS if args.extended else SCENARIOS
    results = [
        evaluate_scenario(policy, scenario=name, envs=args.envs, seed=args.seed)
        for name in scenario_names
    ]
    baseline = results[0]
    for result in results:
        result["success_rate_delta_from_baseline"] = (
            result["success_rate"] - baseline["success_rate"]
        )
        result["peak_force_delta_from_baseline_n"] = (
            result["peak_force_mean_n"] - baseline["peak_force_mean_n"]
        )
    report = {
        "schema_version": 1,
        "status": "cpu_surrogate_calibration_stress_not_physical_evidence",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_update": int(checkpoint["update"]),
        "same_environment_seed_for_all_scenarios": args.seed,
        "extended_breakpoint_scan": args.extended,
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
