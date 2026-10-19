"""Self-contained on-policy PPO with an online differentiable pressure predictor.

No RSL-RL patching or privileged pressure in the actor. The auxiliary step
uses d predicted pressure / d action, locally projected away from predicted
rotation change. Gradients through the simulator are neither used nor claimed.
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

from .core import (
    ACTION_DIM, POLICY_DIM, CRITIC_DIM, POLICY_SLICES,
    pressure_risk, tangent_project,
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
    learning_rate: float = 3e-4
    predictor_learning_rate: float = 1e-3
    entropy: float = 0.002
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    target_kl: float = 0.025
    aux_coef: float = 0.30
    aux_step: float = 0.04
    aux_max_examples: int = 256
    max_predictor_validation_mae: float = 0.15
    minimum_samples_for_aux: int = 50000
    checkpoint_every: int = 100


def mlp(input_dim, layers, output_dim):
    modules = []
    for hidden in layers:
        modules += [nn.Linear(input_dim, hidden), nn.ELU()]
        input_dim = hidden
    modules.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*modules)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.taxel_encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
            nn.Linear(64, 32), nn.ELU(),
        )
        self.pressure_encoder = mlp(72, (96,), 64)
        self.actor_body = mlp(5 * 32 + 64 + (POLICY_DIM - 432), (256, 256), 128)
        self.actor_mean = nn.Linear(128, ACTION_DIM)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), math.log(0.45)))
        self.critic = mlp(CRITIC_DIM, (512, 256, 128), 1)

    def mean(self, observation):
        taxels = observation[:, :360].reshape(-1, 1, 12, 6)
        fingertips = self.taxel_encoder(taxels).reshape(-1, 5 * 32)
        object_patches = self.pressure_encoder(observation[:, 360:432])
        rest = observation[:, 432:]
        merged = torch.cat((fingertips, object_patches, rest), -1)
        return self.actor_mean(self.actor_body(merged))

    def value(self, critic_observation):
        return self.critic(critic_observation).squeeze(-1)

    def act(self, observation, critic_observation):
        mean = self.mean(observation)
        std = self.log_std.clamp(-3.0, 0.5).exp()
        distribution = Normal(mean, std)
        latent = distribution.sample()
        action = latent.tanh()
        log_prob = (distribution.log_prob(latent)
                    - torch.log(1.0 - action.square() + 1e-6)).sum(-1)
        return action, latent, log_prob, self.value(critic_observation)

    def evaluate(self, observation, latent):
        mean = self.mean(observation)
        std = self.log_std.clamp(-3.0, 0.5).exp()
        dist = Normal(mean, std)
        action = latent.tanh()
        log_prob = (dist.log_prob(latent)
                    - torch.log(1.0 - action.square() + 1e-6)).sum(-1)
        return log_prob, dist.entropy().sum(-1), mean.tanh()


class ContactDynamics(nn.Module):
    """One-step action-conditional model: next 96D proxy pressure + omega_z."""
    def __init__(self):
        super().__init__()
        self.network = mlp(POLICY_DIM + ACTION_DIM, (512, 256, 256), 97)

    def forward(self, observations, actions):
        raw = self.network(torch.cat((observations, actions), -1))
        pressure = F.softplus(raw[:, :96]).clamp(0.0, 8.0)
        angular_velocity = raw[:, 96].tanh()
        return pressure, angular_velocity


def compute_gae(reward, done, value, final_value, gamma=0.99, lam=0.95):
    """The 24-second episodes are explicitly finite horizon (no timeout bootstrap)."""
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
    records = {name: [] for name in (
        "policy", "critic", "latent", "logp", "value",
        "reward", "done", "next_pressure", "next_omega", "predictor_mask"
    )}
    reward_sum = 0.0
    terminal_count = 0
    rotation_sum = 0.0
    for _ in range(settings.rollout_steps):
        with torch.no_grad():
            action, latent, logp, value = model.act(policy, critic)
            new_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        extras = info.get("step", {})
        omega = extras.get("rotation_rate", torch.zeros_like(reward)) / 6.0
        # The next observation is reset for terminal environments. Explicit
        # mask prevents accidentally teaching model a reset pressure transition.
        pressure_next = new_observation["critic"][
            :, POLICY_DIM:POLICY_DIM + 96
        ].clone()
        for name, tensor in (
            ("policy", policy), ("critic", critic), ("latent", latent),
            ("logp", logp), ("value", value), ("reward", reward),
            ("done", done), ("next_pressure", pressure_next),
            ("next_omega", omega), ("predictor_mask", ~done),
        ):
            records[name].append(tensor.detach())
        policy, critic = new_observation["policy"], new_observation["critic"]
        reward_sum += reward.mean().item()
        terminal_count += terminated.sum().item()
        rotation_sum += extras.get("unwrapped_rotation_rad", torch.zeros_like(reward)).abs().mean().item()
    with torch.no_grad():
        final_value = model.value(critic)
    data = {k: torch.stack(v) for k, v in records.items()}
    advantages, returns = compute_gae(
        data["reward"], data["done"], data["value"], final_value,
        settings.gamma, settings.gae_lambda,
    )
    data["advantages"] = advantages
    data["returns"] = returns
    return data, {"policy": policy, "critic": critic}, {
        "rollout_reward_mean": reward_sum / settings.rollout_steps,
        "failure_terminations": terminal_count,
        "mean_abs_rotation_rad_per_step": rotation_sum / settings.rollout_steps,
    }


def _flatten(data):
    return {k: v.reshape((-1,) + v.shape[2:]) for k, v in data.items()}


def predictor_update(predictor, optimizer, flat, indices):
    valid = flat["predictor_mask"][indices]
    if not bool(valid.any()):
        return torch.zeros((), device=indices.device)
    inp = flat["policy"][indices][valid]
    # Actual environment action is tanh(latent).
    a = flat["latent"][indices][valid].tanh()
    target_p = flat["next_pressure"][indices][valid]
    target_w = flat["next_omega"][indices][valid]
    predicted_p, predicted_w = predictor(inp, a)
    loss = F.smooth_l1_loss(predicted_p, target_p) + (
        0.40 * F.smooth_l1_loss(predicted_w, target_w)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
    optimizer.step()
    return loss.detach()


@torch.no_grad()
def prediction_validation_mae(predictor, flat, indices):
    # The validation split is separate from the online predictor updates.
    valid = flat["predictor_mask"][indices]
    if not bool(valid.any()):
        return float("inf")
    idx = indices[valid]
    predicted, _ = predictor(
        flat["policy"][idx], flat["latent"][idx].tanh()
    )
    target = flat["next_pressure"][idx]
    contact = target.amax(-1) > 0.25
    if not bool(contact.any()):
        return float("inf")
    return (predicted[contact] - target[contact]).abs().mean().item()


def auxiliary_loss(model, predictor, policy_obs, settings):
    """Projected local stress descent, preserving the predicted task tangent.

    Only applies where predicted upper-tail pressure exceeds safe proxy and
    both task and pressure gradients exist. No model gradients reach the Actor
    except through the explicit fixed, trust-region-scaled action target.
    """
    count = min(policy_obs.shape[0], settings.aux_max_examples)
    obs = policy_obs[:count]
    mu = model.mean(obs).tanh()
    probe = mu.detach().clone().requires_grad_(True)
    predicted_p, predicted_omega = predictor(obs.detach(), probe)
    damage = pressure_risk(predicted_p)
    gp = torch.autograd.grad(damage.sum(), probe, retain_graph=True)[0]
    gw = torch.autograd.grad(predicted_omega.sum(), probe)[0]
    direction = tangent_project(gp, gw).detach()
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(0.10)
    active = (damage.detach() > 0.01).float()
    target = (mu.detach() - settings.aux_step * direction).clamp(-0.95, 0.95)
    return (((mu - target).square().mean(-1)) * active).mean(), active.mean().item()


def ppo_update(model, predictor, optimizer, dynamics_optimizer,
               data, settings: PPOSettings, prior_samples: int):
    flat = _flatten(data)
    count = flat["policy"].shape[0]
    if count % settings.minibatches:
        raise ValueError("rollout size must be divisible by number of minibatches")
    batch = count // settings.minibatches
    permutation = torch.randperm(count, device=flat["policy"].device)
    split = max(batch, int(count * 0.10))
    held_out = permutation[:split]
    train_indices = permutation[split:]
    # Predictors are fit only on nonterminal on-policy transitions, separate
    # from PPO gradient epochs. The held-out split is never used for fitting.
    train_loss = []
    for parts in torch.chunk(train_indices, settings.minibatches):
        train_loss.append(predictor_update(predictor, dynamics_optimizer, flat, parts))
    validation_mae = prediction_validation_mae(predictor, flat, held_out)
    aux_enabled = (
        prior_samples >= settings.minimum_samples_for_aux
        and validation_mae < settings.max_predictor_validation_mae
    )
    raw_adv = flat["advantages"]
    advantages = (raw_adv - raw_adv.mean()) / raw_adv.std().clamp_min(1e-6)
    metrics = {"predictor_loss": torch.stack(train_loss).mean().item(),
               "predictor_validation_mae": validation_mae,
               "aux_enabled": float(aux_enabled)}
    n_updates = 0
    for epoch in range(settings.epochs):
        all_ids = torch.randperm(count, device=flat["policy"].device)
        for ids in all_ids.split(batch):
            logp, entropy, _ = model.evaluate(
                flat["policy"][ids], flat["latent"][ids]
            )
            ratio = (logp - flat["logp"][ids]).exp()
            surr_a = ratio * advantages[ids]
            surr_b = ratio.clamp(1.0 - settings.clip,
                                 1.0 + settings.clip) * advantages[ids]
            loss_policy = -torch.minimum(surr_a, surr_b).mean()
            values = model.value(flat["critic"][ids])
            loss_value = F.mse_loss(values, flat["returns"][ids])
            loss = loss_policy + settings.value_coef * loss_value - (
                settings.entropy * entropy.mean()
            )
            if aux_enabled:
                guidance, active_fraction = auxiliary_loss(
                    model, predictor, flat["policy"][ids], settings
                )
                loss = loss + settings.aux_coef * guidance
                metrics["aux_active_fraction"] = active_fraction
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            optimizer.step()
            n_updates += 1
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - ratio.log()).mean()
                metrics.update({
                    "ppo_policy_loss": float(loss_policy),
                    "ppo_value_loss": float(loss_value),
                    "ppo_approx_kl": float(approx_kl),
                })
            if approx_kl.item() > settings.target_kl:
                break
        else:
            continue
        break
    metrics["ppo_minibatch_updates"] = n_updates
    return metrics


def save_training_state(path: Path, model, predictor, optimizer,
                        predictor_optimizer, settings, update, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save({
        "actor_critic": model.state_dict(),
        "contact_dynamics": predictor.state_dict(),
        "ppo_optimizer": optimizer.state_dict(),
        "dynamics_optimizer": predictor_optimizer.state_dict(),
        "settings": asdict(settings), "update": update, "samples": samples,
        "policy_dim": POLICY_DIM, "critic_dim": CRITIC_DIM,
        "action_dim": ACTION_DIM,
    }, temp)
    os.replace(temp, path)


def train(env, settings: PPOSettings, run_dir: Path, seed: int, resume: Path | None = None):
    torch.manual_seed(seed)
    device = env.device
    model = ActorCritic().to(device)
    predictor = ContactDynamics().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    predictor_optimizer = torch.optim.Adam(
        predictor.parameters(), lr=settings.predictor_learning_rate
    )
    start, previous_samples = 0, 0
    if resume is not None:
        # Load checkpoints only from sources you trust.
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        if (ckpt["policy_dim"], ckpt["critic_dim"], ckpt["action_dim"]) != (
            POLICY_DIM, CRITIC_DIM, ACTION_DIM
        ):
            raise RuntimeError("Incompatible checkpoint observation or action layout")
        model.load_state_dict(ckpt["actor_critic"])
        predictor.load_state_dict(ckpt["contact_dynamics"])
        optimizer.load_state_dict(ckpt["ppo_optimizer"])
        predictor_optimizer.load_state_dict(ckpt["dynamics_optimizer"])
        start, previous_samples = ckpt["update"], ckpt["samples"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ppo_settings.json").write_text(json.dumps({
        **asdict(settings), "seed": seed, "num_envs": env.num_envs,
        "reward_version": "continuous_rotation_v1",
        "pressure_type": "contact_gaussian_proxy_not_tissue_damage",
    }, indent=2) + "\n")
    observation, _ = env.reset()
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("a", newline="") as handle:
        writer = None
        for update in range(start, settings.updates):
            data, observation, rollout_metrics = sample_rollout(
                env, model, settings, observation
            )
            metrics = ppo_update(
                model, predictor, optimizer, predictor_optimizer, data,
                settings, previous_samples,
            )
            previous_samples += env.num_envs * settings.rollout_steps
            row = {"update": update + 1, "environment_samples": previous_samples,
                   **rollout_metrics, **metrics}
            if writer is None:
                fields = [
                    "update", "environment_samples", "rollout_reward_mean",
                    "failure_terminations", "mean_abs_rotation_rad_per_step",
                    "predictor_loss", "predictor_validation_mae",
                    "aux_enabled", "aux_active_fraction",
                    "ppo_policy_loss", "ppo_value_loss",
                    "ppo_approx_kl", "ppo_minibatch_updates",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                if handle.tell() == 0:
                    writer.writeheader()
            writer.writerow(row)
            handle.flush()
            if (update + 1) % 10 == 0:
                print(json.dumps(row, default=str), flush=True)
            if (update + 1) % settings.checkpoint_every == 0:
                save_training_state(
                    run_dir / f"model_{update+1:06d}.pt",
                    model, predictor, optimizer, predictor_optimizer,
                    settings, update + 1, previous_samples,
                )
    save_training_state(
        run_dir / "model_final.pt", model, predictor, optimizer,
        predictor_optimizer, settings, settings.updates, previous_samples,
    )
