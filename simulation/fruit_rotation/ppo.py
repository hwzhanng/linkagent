"""TTSD-CPO: Task-Tangent Stress Descent + Constrained PPO.

Three independent optimization signals:
  (1) reward advantage from continuous physically supported rotation;
  (2) Lagrangian cost advantage from *measured* simulated hotspot pressure;
  (3) locally task-preserving actor correction using gradients of a held-out-
      validated ensemble pressure dynamics model wrt actual hand actions.

The actor sees only five tactile arrays and six sparse 2x2 fruit pressure
patches plus proprioception/estimated object motion. Simulator-dense 96-site
pressure targets are privileged labels for the critic, belief and dynamics,
never direct actor observations. No PhysX differentiability is assumed.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from .core import ACTION_DIM, POLICY_DIM, CRITIC_DIM
from .pressure_guidance import (
    ContactDynamicsEnsemble, GuidanceConfig, ensemble_update,
    stress_guidance_loss, validate_ensemble,
)


@dataclass
class PPOSettings:
    rollout_steps: int = 64
    updates: int = 6000
    epochs: int = 4
    minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    learning_rate: float = 3.0e-4
    predictor_learning_rate: float = 5.0e-4
    entropy: float = 0.002
    value_coef: float = 0.5
    cost_value_coef: float = 0.5
    belief_coef: float = 0.08
    guidance_coef: float = 1.0
    max_grad_norm: float = 1.0
    target_kl: float = 0.025
    pressure_cost_budget: float = 0.035
    dual_lr: float = 0.06
    dual_initial: float = 0.1
    dual_max: float = 5.0
    max_predictor_pressure_mae: float = 0.25
    max_predictor_omega_mae: float = 0.18
    max_predictor_support_mae: float = 0.25
    max_predictor_risk_mae: float = 0.18
    minimum_samples_for_guidance: int = 100_000
    checkpoint_every: int = 100
    disable_guidance: bool = False
    disable_pressure_constraint: bool = False
    disable_dense_belief: bool = False


def mlp(input_dim, layers, output_dim):
    modules = []
    for hidden in layers:
        modules += [nn.Linear(input_dim, hidden), nn.ELU()]
        input_dim = hidden
    modules.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*modules)


class ActorCritic(nn.Module):
    """Only sparse policy features enter the Actor or its pressure belief."""
    def __init__(self):
        super().__init__()
        self.taxel_encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(64, 32), nn.ELU(),
        )
        self.pressure_encoder = mlp(72, (96,), 64)
        self.feature_dim = 5 * 32 + 64 + (POLICY_DIM - 432)
        # Dense sensor privilege is a TRAINING TARGET, not an inference input.
        self.pressure_belief_head = mlp(self.feature_dim, (256, 128), 96)
        self.actor_body = mlp(self.feature_dim + 96, (256, 256), 128)
        self.actor_mean = nn.Linear(128, ACTION_DIM)
        self.log_std = nn.Parameter(
            torch.full((ACTION_DIM,), math.log(0.45))
        )
        self.critic = mlp(CRITIC_DIM, (512, 256, 128), 1)
        self.cost_critic = mlp(CRITIC_DIM, (512, 256, 128), 1)

    def encode_sparse(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != POLICY_DIM:
            raise ValueError("Only the 502D sparse observation is accepted by Actor")
        taxels = observation[:, :360].reshape(-1, 1, 12, 6)
        fingertips = self.taxel_encoder(taxels).reshape(-1, 5 * 32)
        patches = self.pressure_encoder(observation[:, 360:432])
        return torch.cat((fingertips, patches, observation[:, 432:]), -1)

    def _belief(self, features: torch.Tensor):
        return F.softplus(self.pressure_belief_head(features)).clamp(0.0, 8.0)

    def dense_belief(self, observation: torch.Tensor):
        return self._belief(self.encode_sparse(observation))

    def mean(self, observation):
        features = self.encode_sparse(observation)
        belief = self._belief(features)
        return self.actor_mean(self.actor_body(torch.cat((features, belief), -1)))

    def value(self, critic_observation):
        return self.critic(critic_observation).squeeze(-1)

    def cost_value(self, critic_observation):
        return self.cost_critic(critic_observation).squeeze(-1)

    def act(self, observation, critic_observation):
        mean = self.mean(observation)
        std = self.log_std.clamp(-3.0, 0.5).exp()
        distribution = Normal(mean, std)
        latent = distribution.sample()
        action = latent.tanh()
        log_prob = (
            distribution.log_prob(latent)
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(-1)
        return (
            action, latent, log_prob,
            self.value(critic_observation),
            self.cost_value(critic_observation),
        )

    def evaluate(self, observation, latent):
        mean = self.mean(observation)
        std = self.log_std.clamp(-3.0, 0.5).exp()
        dist = Normal(mean, std)
        action = latent.tanh()
        log_prob = (
            dist.log_prob(latent)
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(-1)
        return log_prob, dist.entropy().sum(-1), mean.tanh()


# Retained for external import convenience; this is now an ensemble rather
# than the original single-point forecast.
ContactDynamics = ContactDynamicsEnsemble


@dataclass
class DualState:
    multiplier: float = 0.1

    def update(self, measured_cost: float, settings: PPOSettings):
        if settings.disable_pressure_constraint:
            self.multiplier = 0.0
            return
        self.multiplier = min(
            settings.dual_max,
            max(
                0.0,
                self.multiplier
                + settings.dual_lr * (
                    measured_cost - settings.pressure_cost_budget
                ),
            ),
        )


def compute_gae(reward, done, value, final_value, gamma=0.99, lam=0.95):
    """Correct finite-horizon GAE: physical failure and time limit terminate."""
    steps = reward.shape[0]
    advantage = torch.zeros_like(reward)
    gae = torch.zeros_like(final_value)
    for t in range(steps - 1, -1, -1):
        next_value = final_value if t == steps - 1 else value[t + 1]
        alive = 1.0 - done[t].float()
        delta = reward[t] + gamma * next_value * alive - value[t]
        gae = delta + gamma * lam * alive * gae
        advantage[t] = gae
    return advantage, advantage + value


def sample_rollout(env, model: ActorCritic, settings: PPOSettings, current):
    policy, critic = current["policy"], current["critic"]
    names = (
        "policy", "critic", "latent", "logp", "value", "cost_value",
        "reward", "cost", "done", "next_pressure", "next_omega",
        "next_support", "next_slip", "predictor_mask",
    )
    records = {name: [] for name in names}
    reward_sum = 0.0
    cost_sum = 0.0
    terminal_count = 0
    support_sum = 0.0
    rotation_sum = 0.0
    for _ in range(settings.rollout_steps):
        with torch.no_grad():
            action, latent, logp, value, cost_value = model.act(policy, critic)
            new_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        extras = info.get("step", {})
        required = (
            "rotation_rate", "pressure_risk", "secure_support",
            "slip_speed", "unwrapped_rotation_rad",
        )
        missing = [name for name in required if name not in extras]
        if missing:
            raise RuntimeError(f"Missing current-step physical evidence: {missing}")
        cost = extras["pressure_risk"].clamp_min(0.0)
        omega = extras["rotation_rate"] / 6.0
        support = extras["secure_support"].float()
        slip = extras["slip_speed"].clamp_min(0.0)
        # The poststep observation MAY be auto-reset on done. Mask such
        # transitions from model training, never pretend terminal -> reset
        # is a physically meaningful pressure dynamics transition.
        pressure_next = new_observation["critic"][
            :, POLICY_DIM:POLICY_DIM + 96
        ].clone()
        for name, tensor in (
            ("policy", policy), ("critic", critic), ("latent", latent),
            ("logp", logp), ("value", value), ("cost_value", cost_value),
            ("reward", reward), ("cost", cost), ("done", done),
            ("next_pressure", pressure_next), ("next_omega", omega),
            ("next_support", support), ("next_slip", slip),
            ("predictor_mask", ~done),
        ):
            records[name].append(tensor.detach().clone())
        policy, critic = new_observation["policy"], new_observation["critic"]
        reward_sum += reward.mean().item()
        cost_sum += cost.mean().item()
        terminal_count += int(terminated.sum())
        support_sum += support.float().mean().item()
        rotation_sum += extras["unwrapped_rotation_rad"].abs().mean().item()

    with torch.no_grad():
        final_value = model.value(critic)
        final_cost_value = model.cost_value(critic)
    data = {k: torch.stack(v) for k, v in records.items()}
    data["advantages"], data["returns"] = compute_gae(
        data["reward"], data["done"], data["value"], final_value,
        settings.gamma, settings.gae_lambda,
    )
    data["cost_advantages"], data["cost_returns"] = compute_gae(
        data["cost"], data["done"], data["cost_value"],
        final_cost_value, settings.gamma, settings.gae_lambda,
    )
    return data, {"policy": policy, "critic": critic}, {
        "rollout_reward_mean": reward_sum / settings.rollout_steps,
        "rollout_pressure_cost": cost_sum / settings.rollout_steps,
        "failure_terminations": terminal_count,
        "mean_supported_fraction": support_sum / settings.rollout_steps,
        "mean_abs_rotation_rad_per_step": rotation_sum / settings.rollout_steps,
    }


def _flatten(data):
    return {
        key: value.reshape((-1,) + value.shape[2:])
        for key, value in data.items()
    }


def update_predictor(predictor, optimizer, flat, settings):
    total = flat["policy"].shape[0]
    perm = torch.randperm(total, device=flat["policy"].device)
    # Held-out validation is never fitted in this update.
    split = max(128, int(0.15 * total))
    split = min(split, total - 1)
    validation_ids = perm[:split]
    train_ids = perm[split:]
    losses = []
    for ids in torch.chunk(train_ids, settings.minibatches):
        mask = flat["predictor_mask"][ids]
        ids = ids[mask]
        if ids.numel() == 0:
            continue
        loss = ensemble_update(
            predictor, optimizer,
            flat["policy"][ids], flat["latent"][ids].tanh(),
            flat["next_pressure"][ids], flat["next_omega"][ids],
            flat["next_support"][ids], flat["next_slip"][ids],
        )
        losses.append(loss)
    validation_ids = validation_ids[flat["predictor_mask"][validation_ids]]
    if validation_ids.numel() == 0:
        scores = {
            "pressure_mae": float("inf"), "omega_mae": float("inf"),
            "support_mae": float("inf"), "risk_mae": float("inf"),
        }
    else:
        scores = validate_ensemble(
            predictor,
            flat["policy"][validation_ids],
            flat["latent"][validation_ids].tanh(),
            flat["next_pressure"][validation_ids],
            flat["next_omega"][validation_ids],
            flat["next_support"][validation_ids],
        )
    return (torch.stack(losses).mean().item() if losses else float("inf")), scores


def ppo_update(
    model: ActorCritic,
    predictor: ContactDynamicsEnsemble,
    optimizer,
    dynamics_optimizer,
    data,
    settings: PPOSettings,
    prior_samples: int,
    dual: DualState | None = None,
):
    flat = _flatten(data)
    count = flat["policy"].shape[0]
    if count % settings.minibatches:
        raise ValueError("rollout size must be divisible by minibatches")
    batch = count // settings.minibatches
    predictor_loss, validation = update_predictor(
        predictor, dynamics_optimizer, flat, settings
    )
    predictor_ready = (
        prior_samples >= settings.minimum_samples_for_guidance
        and validation["pressure_mae"] <= settings.max_predictor_pressure_mae
        and validation["omega_mae"] <= settings.max_predictor_omega_mae
        and validation["support_mae"] <= settings.max_predictor_support_mae
        and validation["risk_mae"] <= settings.max_predictor_risk_mae
        and not settings.disable_guidance
    )
    reward_adv = flat["advantages"]
    reward_adv = (reward_adv - reward_adv.mean()) / reward_adv.std().clamp_min(1.0e-6)
    cost_adv = flat["cost_advantages"]
    cost_adv = (cost_adv - cost_adv.mean()) / cost_adv.std().clamp_min(1.0e-6)

    if dual is None:
        dual = DualState(settings.dual_initial)
    multiplier = 0.0 if settings.disable_pressure_constraint else dual.multiplier
    stats = {
        "predictor_loss": predictor_loss,
        "predictor_pressure_mae": validation["pressure_mae"],
        "predictor_omega_mae": validation["omega_mae"],
        "predictor_support_mae": validation["support_mae"],
        "predictor_risk_mae": validation["risk_mae"],
        "guidance_ready": float(predictor_ready),
        "guidance_active_fraction": 0.0,
        "guidance_accepted_improvement": 0.0,
        "guidance_gradient_agreement": 0.0,
        "guidance_predicted_mean_risk": 0.0,
        "guidance_model_disagreement": 0.0,
        "dual_pressure_multiplier": multiplier,
    }
    updated = 0
    for _epoch in range(settings.epochs):
        all_ids = torch.randperm(count, device=flat["policy"].device)
        for ids in all_ids.split(batch):
            logp, entropy, action_mean = model.evaluate(
                flat["policy"][ids], flat["latent"][ids]
            )
            ratio = (logp - flat["logp"][ids]).exp()
            clipped_ratio = ratio.clamp(
                1.0 - settings.clip, 1.0 + settings.clip
            )
            reward_surrogate = torch.minimum(
                ratio * reward_adv[ids],
                clipped_ratio * reward_adv[ids],
            ).mean()
            # Conservative (upper) surrogate when cost advantage is positive.
            cost_surrogate = torch.maximum(
                ratio * cost_adv[ids],
                clipped_ratio * cost_adv[ids],
            ).mean()
            loss_reward_policy = -reward_surrogate
            loss_cost_policy = multiplier * cost_surrogate
            loss_value = F.mse_loss(
                model.value(flat["critic"][ids]), flat["returns"][ids]
            )
            loss_cost_value = F.mse_loss(
                model.cost_value(flat["critic"][ids]),
                flat["cost_returns"][ids],
            )
            loss = (
                loss_reward_policy + loss_cost_policy
                + settings.value_coef * loss_value
                + settings.cost_value_coef * loss_cost_value
                - settings.entropy * entropy.mean()
            )
            if not settings.disable_dense_belief:
                belief = model.dense_belief(flat["policy"][ids])
                teacher_dense = flat["critic"][
                    ids, POLICY_DIM:POLICY_DIM + 96
                ]
                loss_belief = F.smooth_l1_loss(belief, teacher_dense)
                loss = loss + settings.belief_coef * loss_belief
            else:
                loss_belief = torch.zeros((), device=flat["policy"].device)

            if predictor_ready:
                guidance, report = stress_guidance_loss(
                    action_mean, flat["policy"][ids],
                    predictor, GuidanceConfig()
                )
                loss = loss + settings.guidance_coef * guidance
                stats.update(report)
            else:
                guidance = torch.zeros((), device=flat["policy"].device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            optimizer.step()
            updated += 1
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - ratio.log()).mean()
                stats.update({
                    "ppo_reward_policy_loss": float(loss_reward_policy),
                    "ppo_cost_policy_loss": float(loss_cost_policy),
                    "ppo_reward_value_loss": float(loss_value),
                    "ppo_cost_value_loss": float(loss_cost_value),
                    "dense_belief_loss": float(loss_belief),
                    "task_tangent_guidance_loss": float(guidance),
                    "ppo_approx_kl": float(approx_kl),
                })
            if float(approx_kl) > settings.target_kl:
                break
        else:
            continue
        break
    stats["ppo_minibatch_updates"] = updated
    return stats


METRIC_FIELDS = (
    "update", "environment_samples", "rollout_reward_mean",
    "rollout_pressure_cost", "failure_terminations",
    "mean_supported_fraction", "mean_abs_rotation_rad_per_step",
    "predictor_loss", "predictor_pressure_mae", "predictor_omega_mae",
    "predictor_support_mae", "predictor_risk_mae", "guidance_ready",
    "guidance_active_fraction", "guidance_accepted_improvement",
    "guidance_gradient_agreement", "guidance_predicted_mean_risk",
    "guidance_model_disagreement", "dual_pressure_multiplier",
    "ppo_reward_policy_loss", "ppo_cost_policy_loss",
    "ppo_reward_value_loss", "ppo_cost_value_loss", "dense_belief_loss",
    "task_tangent_guidance_loss", "ppo_approx_kl", "ppo_minibatch_updates",
)


def save_training_state(
    path: Path, model, predictor, optimizer,
    predictor_optimizer, settings, update, samples, dual: DualState,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save({
        "actor_critic": model.state_dict(),
        "contact_dynamics": predictor.state_dict(),
        "ppo_optimizer": optimizer.state_dict(),
        "dynamics_optimizer": predictor_optimizer.state_dict(),
        "settings": asdict(settings), "update": update, "samples": samples,
        "dual_multiplier": dual.multiplier,
        "algorithm": "tactile_task_tangent_stress_descent_constrained_ppo_v2",
        "policy_dim": POLICY_DIM, "critic_dim": CRITIC_DIM,
        "action_dim": ACTION_DIM,
    }, temp)
    os.replace(temp, path)


def train(
    env, settings: PPOSettings, run_dir: Path,
    seed: int, resume: Path | None = None,
):
    torch.manual_seed(seed)
    device = env.device
    model = ActorCritic().to(device)
    predictor = ContactDynamicsEnsemble(members=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    predictor_optimizer = torch.optim.Adam(
        predictor.parameters(), lr=settings.predictor_learning_rate
    )
    dual = DualState(settings.dual_initial)
    start, previous_samples = 0, 0
    if resume is not None:
        # Trusted checkpoints only: torch pickle can execute Python.
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        if (ckpt["policy_dim"], ckpt["critic_dim"], ckpt["action_dim"]) != (
            POLICY_DIM, CRITIC_DIM, ACTION_DIM
        ) or ckpt.get("algorithm") != "tactile_task_tangent_stress_descent_constrained_ppo_v2":
            raise RuntimeError("Incompatible pressure-guided PPO checkpoint")
        model.load_state_dict(ckpt["actor_critic"])
        predictor.load_state_dict(ckpt["contact_dynamics"])
        optimizer.load_state_dict(ckpt["ppo_optimizer"])
        predictor_optimizer.load_state_dict(ckpt["dynamics_optimizer"])
        start, previous_samples = ckpt["update"], ckpt["samples"]
        dual.multiplier = float(ckpt["dual_multiplier"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ppo_settings.json").write_text(json.dumps({
        **asdict(settings), "seed": seed, "num_envs": env.num_envs,
        "reward_version": "continuous_rotation_cppo_task_tangent_v2",
        "pressure_type": "contact_gaussian_proxy_not_tissue_damage",
    }, indent=2) + "\n")
    observation, _ = env.reset()
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        for update in range(start, settings.updates):
            data, observation, rollout_metrics = sample_rollout(
                env, model, settings, observation
            )
            metrics = ppo_update(
                model, predictor, optimizer, predictor_optimizer,
                data, settings, previous_samples, dual,
            )
            dual.update(rollout_metrics["rollout_pressure_cost"], settings)
            previous_samples += env.num_envs * settings.rollout_steps
            row = {
                "update": update + 1,
                "environment_samples": previous_samples,
                **rollout_metrics, **metrics,
            }
            writer.writerow(row)
            handle.flush()
            if (update + 1) % 10 == 0:
                print(json.dumps(row, default=str), flush=True)
            if (update + 1) % settings.checkpoint_every == 0:
                save_training_state(
                    run_dir / f"model_{update + 1:06d}.pt",
                    model, predictor, optimizer, predictor_optimizer,
                    settings, update + 1, previous_samples, dual,
                )
    save_training_state(
        run_dir / "model_final.pt", model, predictor, optimizer,
        predictor_optimizer, settings, settings.updates, previous_samples, dual,
    )
