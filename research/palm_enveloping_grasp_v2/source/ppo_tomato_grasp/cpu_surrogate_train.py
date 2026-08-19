#!/usr/bin/env python3
"""Small asymmetric PPO runner for the pure-CPU hand surrogate.

The resulting checkpoint is an interface/learnability artifact only.  It is
explicitly incompatible with hardware release and cannot substitute for an
Isaac/PhysX checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.distributions import Normal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.ppo_tomato_grasp.cpu_surrogate_env import (
    SURROGATE_ENV_VERSION,
    CpuHandGraspEnv,
)
from simulation.ppo_tomato_grasp.domain_randomization import RANDOMIZATION_VERSION
from simulation.ppo_tomato_grasp.hand_grasp_mdp import REWARD_VERSION
from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_STATE_DIM,
)


@dataclass(frozen=True)
class CpuPpoConfig:
    num_envs: int = 256
    horizon: int = 32
    updates: int = 80
    epochs: int = 4
    minibatch_size: int = 2048
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.003
    max_grad_norm: float = 1.0
    initial_std: float = 0.60
    minimum_std: float = 0.12
    weak_prior_global_closure: float = 0.70
    weak_prior_thumb_opposition: float = 0.60
    curriculum_fraction: float = 0.20
    tail_focus_fraction: float = 0.0
    seed: int = 42
    evaluation_interval: int = 20
    evaluation_envs: int = 512
    threads: int = 8
    device: str = "cpu"
    wandb_mode: str = "disabled"
    wandb_project: str = "linkbot-tomato-grasp-v3-10d110d"
    wandb_run_name: str | None = None


class CpuActorCritic(nn.Module):
    def __init__(
        self,
        *,
        initial_std: float = 0.60,
        minimum_std: float = 0.12,
        weak_prior_action: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < minimum_std <= initial_std:
            raise ValueError("require 0 < minimum_std <= initial_std")
        self.actor = nn.Sequential(
            nn.Linear(ACTOR_OBSERVATION_DIM, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, ACTION_DIM),
        )
        self.critic = nn.Sequential(
            nn.Linear(CRITIC_STATE_DIM, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), math.log(initial_std)))
        self.minimum_log_std = math.log(minimum_std)
        if weak_prior_action is not None:
            if weak_prior_action.shape != (ACTION_DIM,) or not torch.isfinite(weak_prior_action).all():
                raise ValueError(f"weak_prior_action must be finite shape {(ACTION_DIM,)}")
            if bool((weak_prior_action.abs() >= 0.95).any()):
                raise ValueError("weak_prior_action must lie strictly inside (-0.95, 0.95)")
            # This broad semantic prior avoids sparse-contact cold starts while
            # leaving all network weights and all ten action dimensions trainable.
            # It is not a target trajectory or a frozen demonstration policy.
            output = self.actor[-1]
            assert isinstance(output, nn.Linear)
            nn.init.normal_(output.weight, mean=0.0, std=1.0e-3)
            with torch.no_grad():
                output.bias.copy_(torch.atanh(weak_prior_action))

    def distribution(self, observation: torch.Tensor) -> Normal:
        mean = self.actor(observation)
        std = self.log_std.clamp_min(self.minimum_log_std).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _squashed_log_prob(distribution: Normal, raw: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        correction = torch.log(1.0 - action.square() + 1.0e-6)
        return (distribution.log_prob(raw) - correction).sum(dim=-1)

    def act(
        self, observation: torch.Tensor, critic_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observation)
        raw = distribution.rsample()
        action = torch.tanh(raw)
        log_prob = self._squashed_log_prob(distribution, raw, action)
        value = self.critic(critic_state).squeeze(-1)
        return action, raw, log_prob, value

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        critic_state: torch.Tensor,
        raw_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action = torch.tanh(raw_action)
        distribution = self.distribution(observation)
        log_prob = self._squashed_log_prob(distribution, raw_action, action)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.critic(critic_state).squeeze(-1)
        return log_prob, entropy, value

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(observation))


def _validate_config(config: CpuPpoConfig) -> None:
    integer_positive = (
        config.num_envs,
        config.horizon,
        config.updates,
        config.epochs,
        config.minibatch_size,
        config.evaluation_interval,
        config.evaluation_envs,
        config.threads,
    )
    if any(value < 1 for value in integer_positive):
        raise ValueError("PPO integer settings must be positive")
    if not 0.0 <= config.curriculum_fraction <= 1.0:
        raise ValueError("curriculum_fraction must lie in [0, 1]")
    if not 0.0 <= config.tail_focus_fraction <= 1.0:
        raise ValueError("tail_focus_fraction must lie in [0, 1]")
    if not 0.0 <= config.weak_prior_global_closure < 0.95:
        raise ValueError("weak_prior_global_closure must lie in [0, 0.95)")
    if not 0.0 <= config.weak_prior_thumb_opposition < 0.95:
        raise ValueError("weak_prior_thumb_opposition must lie in [0, 0.95)")
    if config.minibatch_size > config.num_envs * config.horizon:
        raise ValueError("minibatch_size exceeds one rollout")


@torch.no_grad()
def evaluate_policy(
    policy: CpuActorCritic,
    *,
    num_envs: int,
    seed: int,
    curriculum_fraction: float,
    tail_focus_fraction: float = 0.0,
) -> dict[str, float]:
    env = CpuHandGraspEnv(
        num_envs,
        seed=seed,
        curriculum_fraction=curriculum_fraction,
        tail_focus_fraction=tail_focus_fraction,
        auto_reset=False,
        device=next(policy.parameters()).device,
    )
    observation, critic = env.reset()
    del critic
    device = next(policy.parameters()).device
    success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    dropped = torch.zeros(num_envs, dtype=torch.bool, device=device)
    maximum_force = torch.zeros(num_envs, device=device)
    damage = torch.zeros(num_envs, device=device)
    return_sum = torch.zeros(num_envs, device=device)
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    for _ in range(env.episode_steps):
        action = policy.deterministic(observation)
        result = env.step(action)
        observation = result.observation
        success |= active & result.success
        dropped |= active & result.metrics["drop"]
        maximum_force = torch.where(
            active,
            torch.maximum(maximum_force, result.metrics["max_force_n"]),
            maximum_force,
        )
        damage = torch.where(
            active,
            torch.maximum(damage, result.metrics["damage_score"]),
            damage,
        )
        return_sum += result.reward * active.float()
        active &= ~(result.terminated | result.truncated)
        if not active.any():
            break
    return {
        "success_rate": float(success.float().mean()),
        "drop_rate": float(dropped.float().mean()),
        "return_mean": float(return_sum.mean()),
        "maximum_force_mean_n": float(maximum_force.mean()),
        "damage_score_mean": float(damage.mean()),
    }


def train_cpu_ppo(
    config: CpuPpoConfig,
    *,
    output_dir: Path | None = None,
    resume_path: Path | None = None,
    allow_environment_migration: bool = False,
) -> dict[str, object]:
    _validate_config(config)
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for 10D/110D surrogate training but is unavailable")
        torch.cuda.manual_seed_all(config.seed)
    env = CpuHandGraspEnv(
        config.num_envs,
        seed=config.seed,
        curriculum_fraction=config.curriculum_fraction,
        tail_focus_fraction=config.tail_focus_fraction,
        auto_reset=True,
        device=device,
    )
    observation, critic_state = env.reset()
    weak_prior_action = torch.zeros(ACTION_DIM, device=device)
    weak_prior_action[0] = config.weak_prior_global_closure
    weak_prior_action[4] = config.weak_prior_thumb_opposition
    policy = CpuActorCritic(
        initial_std=config.initial_std,
        minimum_std=config.minimum_std,
        weak_prior_action=weak_prior_action,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    start_update = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        contract = checkpoint.get("contract", {})
        if contract != {
            "actor_observation_dim": ACTOR_OBSERVATION_DIM,
            "critic_state_dim": CRITIC_STATE_DIM,
            "action_dim": ACTION_DIM,
            "simulation_only": True,
        }:
            raise ValueError("resume checkpoint contract does not match CPU surrogate")
        expected_environment = {
            "surrogate_env_version": SURROGATE_ENV_VERSION,
            "reward_version": REWARD_VERSION,
            "randomization_version": RANDOMIZATION_VERSION,
        }
        if (
            checkpoint.get("environment_contract") != expected_environment
            and not allow_environment_migration
        ):
            raise ValueError(
                "resume checkpoint environment changed; pass explicit migration approval"
            )
        policy.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = int(checkpoint["update"])
        if start_update >= config.updates:
            raise ValueError("resume checkpoint already reached requested updates")
    run_started = time.monotonic()
    wandb_run = None
    if config.wandb_mode != "disabled":
        if config.wandb_mode not in {"online", "offline"}:
            raise ValueError("wandb_mode must be disabled, online, or offline")
        import wandb

        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            mode=config.wandb_mode,
            config=asdict(config),
            resume="allow" if resume_path is not None else None,
        )
    history: list[dict[str, float | int]] = []
    latest_evaluation: dict[str, float] = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "cpu_surrogate_only_not_physics_or_hardware",
                    "resume_path": str(resume_path.resolve()) if resume_path else None,
                    "starting_update": start_update,
                    "environment_migration_explicitly_allowed": allow_environment_migration,
                    "environment_contract": {
                        "surrogate_env_version": SURROGATE_ENV_VERSION,
                        "reward_version": REWARD_VERSION,
                        "randomization_version": RANDOMIZATION_VERSION,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def save_checkpoint(update: int) -> None:
        if output_dir is None:
            return
        payload = {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "config": asdict(config),
            "contract": {
                "actor_observation_dim": ACTOR_OBSERVATION_DIM,
                "critic_state_dim": CRITIC_STATE_DIM,
                "action_dim": ACTION_DIM,
                "simulation_only": True,
            },
            "environment_contract": {
                "surrogate_env_version": SURROGATE_ENV_VERSION,
                "reward_version": REWARD_VERSION,
                "randomization_version": RANDOMIZATION_VERSION,
            },
        }
        temporary = output_dir / "latest.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(output_dir / "latest.pt")

    for update in range(start_update + 1, config.updates + 1):
        rollout_observation = []
        rollout_critic = []
        rollout_raw_action = []
        rollout_log_prob = []
        rollout_value = []
        rollout_reward = []
        rollout_done = []
        success_events = 0
        enclosure_sum = 0.0
        for _ in range(config.horizon):
            with torch.no_grad():
                action, raw, log_prob, value = policy.act(observation, critic_state)
            result = env.step(action)
            done = result.terminated | result.truncated
            rollout_observation.append(observation)
            rollout_critic.append(critic_state)
            rollout_raw_action.append(raw)
            rollout_log_prob.append(log_prob)
            rollout_value.append(value)
            rollout_reward.append(result.reward)
            rollout_done.append(done)
            observation = result.observation
            critic_state = result.critic_state
            success_events += int(result.success.sum())
            enclosure_sum += float(result.metrics["secure_enclosure"].float().mean())

        with torch.no_grad():
            bootstrap = policy.critic(critic_state).squeeze(-1)
        rewards = torch.stack(rollout_reward)
        values = torch.stack(rollout_value)
        dones = torch.stack(rollout_done)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(config.num_envs, device=device)
        next_value = bootstrap
        for step in reversed(range(config.horizon)):
            nonterminal = (~dones[step]).float()
            delta = rewards[step] + config.gamma * next_value * nonterminal - values[step]
            gae = delta + config.gamma * config.gae_lambda * nonterminal * gae
            advantages[step] = gae
            next_value = values[step]
        returns = advantages + values

        flat_observation = torch.cat(rollout_observation)
        flat_critic = torch.cat(rollout_critic)
        flat_raw = torch.cat(rollout_raw_action)
        flat_old_log_prob = torch.cat(rollout_log_prob)
        flat_advantages = advantages.flatten()
        flat_returns = returns.flatten()
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std(unbiased=False) + 1.0e-8
        )
        sample_count = flat_observation.shape[0]
        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        entropy_sum = 0.0
        batches = 0
        for _ in range(config.epochs):
            permutation = torch.randperm(sample_count, device=device)
            for start in range(0, sample_count, config.minibatch_size):
                indices = permutation[start : start + config.minibatch_size]
                new_log_prob, entropy, new_value = policy.evaluate_actions(
                    flat_observation[indices], flat_critic[indices], flat_raw[indices]
                )
                ratio = torch.exp(new_log_prob - flat_old_log_prob[indices])
                surrogate = ratio * flat_advantages[indices]
                clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * flat_advantages[indices]
                actor_loss = -torch.minimum(surrogate, clipped).mean()
                critic_loss = (new_value - flat_returns[indices]).square().mean()
                entropy_mean = entropy.mean()
                loss = (
                    actor_loss
                    + config.value_coefficient * critic_loss
                    - config.entropy_coefficient * entropy_mean
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    policy.log_std.clamp_(min=policy.minimum_log_std, max=math.log(1.5))
                actor_loss_sum += float(actor_loss)
                critic_loss_sum += float(critic_loss)
                entropy_sum += float(entropy_mean)
                batches += 1

        record: dict[str, float | int] = {
            "update": update,
            "rollout_reward_mean": float(rewards.mean()),
            "success_events": success_events,
            "secure_enclosure_rate": enclosure_sum / config.horizon,
            "actor_loss": actor_loss_sum / batches,
            "critic_loss": critic_loss_sum / batches,
            "entropy": entropy_sum / batches,
            "action_std_mean": float(policy.log_std.exp().mean()),
            "elapsed_seconds": time.monotonic() - run_started,
        }
        if update % config.evaluation_interval == 0 or update == config.updates:
            latest_evaluation = evaluate_policy(
                policy,
                num_envs=config.evaluation_envs,
                seed=config.seed + 10000 + update,
                curriculum_fraction=config.curriculum_fraction,
                tail_focus_fraction=config.tail_focus_fraction,
            )
            record.update({f"eval_{key}": value for key, value in latest_evaluation.items()})
            save_checkpoint(update)
        history.append(record)
        if output_dir is not None:
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        if wandb_run is not None:
            wandb_run.log({key: value for key, value in record.items() if isinstance(value, (float, int))}, step=update)
        if update == 1 or update % 10 == 0 or update == config.updates:
            print(json.dumps(record, sort_keys=True), flush=True)

    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "cpu_surrogate_only_not_physics_or_hardware",
        "starting_update": start_update,
        "resumed_from": str(resume_path.resolve()) if resume_path else None,
        "environment_contract": {
            "surrogate_env_version": SURROGATE_ENV_VERSION,
            "reward_version": REWARD_VERSION,
            "randomization_version": RANDOMIZATION_VERSION,
        },
        "config": asdict(config),
        "elapsed_seconds": time.monotonic() - run_started,
        "latest": history[-1],
        "evaluation": latest_evaluation,
        "all_finite": all(
            math.isfinite(float(value))
            for record in history
            for value in record.values()
            if isinstance(value, (float, int))
        ),
        "wandb_url": getattr(wandb_run, "url", None),
    }
    if wandb_run is not None:
        wandb_run.finish()
    if output_dir is not None:
        save_checkpoint(config.updates)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--curriculum-fraction", type=float, default=0.20)
    parser.add_argument(
        "--tail-focus-fraction",
        type=float,
        default=0.0,
        help="fraction of episodes biased toward joint small/ripe/wet tail cases",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--evaluation-interval", type=int, default=20)
    parser.add_argument("--evaluation-envs", type=int, default=512)
    parser.add_argument("--initial-std", type=float, default=0.60)
    parser.add_argument("--minimum-std", type=float, default=0.12)
    parser.add_argument("--weak-prior-global-closure", type=float, default=0.70)
    parser.add_argument("--weak-prior-thumb-opposition", type=float, default=0.60)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation/ppo_tomato_grasp/artifacts/cpu_surrogate"),
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--allow-environment-migration", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wandb-project", default="linkbot-tomato-grasp-v3-10d110d")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()
    config = CpuPpoConfig(
        num_envs=args.num_envs,
        horizon=args.horizon,
        updates=args.updates,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        curriculum_fraction=args.curriculum_fraction,
        tail_focus_fraction=args.tail_focus_fraction,
        seed=args.seed,
        threads=args.threads,
        evaluation_interval=args.evaluation_interval,
        evaluation_envs=args.evaluation_envs,
        initial_std=args.initial_std,
        minimum_std=args.minimum_std,
        weak_prior_global_closure=args.weak_prior_global_closure,
        weak_prior_thumb_opposition=args.weak_prior_thumb_opposition,
        device=args.device,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    summary = train_cpu_ppo(
        config,
        output_dir=args.output_dir.resolve(),
        resume_path=args.resume.resolve() if args.resume is not None else None,
        allow_environment_migration=args.allow_environment_migration,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if bool(summary["all_finite"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
