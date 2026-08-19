#!/usr/bin/env python3
"""Export a checked deterministic Actor for offline inference validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.ppo_tomato_grasp.cpu_surrogate_env import SURROGATE_ENV_VERSION
from simulation.ppo_tomato_grasp.cpu_surrogate_train import CpuActorCritic
from simulation.ppo_tomato_grasp.domain_randomization import RANDOMIZATION_VERSION
from simulation.ppo_tomato_grasp.hand_grasp_mdp import REWARD_VERSION
from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    ACTION_DIM,
    ACTION_LAYOUT_ID,
    ACTOR_OBSERVATION_DIM,
    ACTOR_OBSERVATION_LAYOUT_ID,
    CRITIC_STATE_DIM,
    INTERFACE_VERSION,
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


class CheckedDeterministicActor(nn.Module):
    def __init__(self, actor: nn.Sequential) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.dim() != 2 or observation.size(1) != 110:
            raise RuntimeError("Actor input must have shape (N, 110)")
        if not bool(torch.isfinite(observation).all()):
            raise RuntimeError("Actor input contains NaN or Inf")
        return torch.tanh(self.actor(observation))


def export_actor(
    checkpoint_path: Path,
    output_path: Path,
    *,
    parity_samples: int = 10000,
    seed: int = 20268811,
) -> dict[str, object]:
    if parity_samples < 1:
        raise ValueError("parity_samples must be positive")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("contract") != TENSOR_CONTRACT:
        raise ValueError("checkpoint tensor contract is incompatible")
    if checkpoint.get("environment_contract") != ENVIRONMENT_CONTRACT:
        raise ValueError("checkpoint environment contract is missing or incompatible")
    policy = CpuActorCritic()
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    checked = CheckedDeterministicActor(policy.actor).eval()
    scripted = torch.jit.script(checked)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    observation = 2.0 * torch.rand(
        (parity_samples, ACTOR_OBSERVATION_DIM), generator=generator
    ) - 1.0
    with torch.no_grad():
        expected = policy.deterministic(observation)
        actual = scripted(observation)
    maximum_error = float((actual - expected).abs().max())
    if not torch.isfinite(actual).all() or maximum_error > 1.0e-6:
        raise RuntimeError("TorchScript parity validation failed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.jit.save(scripted, temporary)
    temporary.replace(output_path)
    metadata = {
        "schema_version": 1,
        "status": "cpu_surrogate_only_hardware_release_forbidden",
        "source_checkpoint": str(checkpoint_path.resolve()),
        "source_update": int(checkpoint["update"]),
        "interface_version": INTERFACE_VERSION,
        "actor_observation_layout": ACTOR_OBSERVATION_LAYOUT_ID,
        "action_layout": ACTION_LAYOUT_ID,
        "tensor_contract": TENSOR_CONTRACT,
        "environment_contract": ENVIRONMENT_CONTRACT,
        "parity_samples": parity_samples,
        "parity_seed": seed,
        "maximum_absolute_error": maximum_error,
        "formal_training_allowed": False,
        "hardware_command_allowed": False,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parity-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20268811)
    args = parser.parse_args()
    metadata = export_actor(
        args.checkpoint.resolve(),
        args.output.resolve(),
        parity_samples=args.parity_samples,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
