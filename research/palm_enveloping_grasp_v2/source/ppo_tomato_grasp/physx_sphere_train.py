"""Asymmetric PPO trainer for the final first-pass PhysX sphere task.

The Actor receives the deployable 110D observation.  The Critic receives the
148D state containing simulator-only force, contact, object and sensor-truth
fields.  This is the same asymmetric interface intended for later tomato and
sim-to-real work; only the object model is currently a compliant-contact
sphere.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Final first-pass PhysX sphere PPO")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--horizon", type=int, default=128)
parser.add_argument("--updates", type=int, default=2000)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--minibatch-size", type=int, default=8192)
parser.add_argument("--learning-rate", type=float, default=3e-4)
parser.add_argument("--entropy-coef", type=float, default=1e-3)
parser.add_argument("--value-coef", type=float, default=0.5)
parser.add_argument("--clip-eps", type=float, default=0.2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--initial-std", type=float, default=0.45)
parser.add_argument("--minimum-std", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save-interval", type=int, default=25)
parser.add_argument("--eval-interval", type=int, default=25)
parser.add_argument("--eval-steps", type=int, default=160)
parser.add_argument("--output-dir", type=Path, default=ROOT / "simulation/ppo_tomato_grasp/artifacts/physx_sphere_final")
parser.add_argument("--resume", type=Path, default=None)
parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
parser.add_argument("--wandb-project", default="linkbot-tomato-grasp-physx-final")
parser.add_argument("--wandb-run-name", default=None)
parser.add_argument("--metrics-file", type=Path, default=None)
parser.add_argument("--control-file", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from torch import nn
from torch.distributions import Normal

from simulation.ppo_tomato_grasp.physx_sphere_env import PhysxSphereGraspEnv, PhysxSphereGraspEnvCfg


class AsymmetricActorCritic(nn.Module):
    def __init__(self, actor_dim: int, critic_dim: int, action_dim: int, minimum_std: float):
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(actor_dim, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.critic_body = nn.Sequential(
            nn.Linear(critic_dim, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(args_cli.initial_std)))
        self.minimum_log_std = math.log(minimum_std)

    def actor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(self.actor_body(obs))

    def critic(self, state: torch.Tensor) -> torch.Tensor:
        return self.value(self.critic_body(state)).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.actor(obs), self.log_std.clamp_min(self.minimum_log_std).clamp_max(1.0).exp())


def squashed_log_prob(dist: Normal, raw: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return (dist.log_prob(raw) - torch.log(1.0 - action.square() + 1e-6)).sum(-1)


def sample_action(policy: AsymmetricActorCritic, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dist = policy.distribution(obs)
    raw = dist.rsample()
    action = torch.tanh(raw)
    return action, squashed_log_prob(dist, raw, action)


@torch.inference_mode()
def evaluate(env: PhysxSphereGraspEnv, policy: AsymmetricActorCritic, steps: int) -> dict[str, float]:
    obs, _ = env.reset()
    actor_obs = obs["policy"]
    returns = torch.zeros(env.num_envs, device=actor_obs.device)
    successes = torch.zeros(env.num_envs, dtype=torch.bool, device=actor_obs.device)
    force_sum = torch.zeros(env.num_envs, device=actor_obs.device)
    force_max = torch.zeros(env.num_envs, device=actor_obs.device)
    for _ in range(steps):
        action = torch.tanh(policy.actor(actor_obs))
        obs, reward, terminated, truncated, _ = env.step(action)
        returns += reward
        successes |= env.success
        force_sum += env.contact_forces.mean(dim=-1)
        force_max = torch.maximum(force_max, env.contact_forces.max(dim=-1).values)
        actor_obs = obs["policy"]
    result = {
        "eval/return_mean": returns.mean().item(),
        "eval/success_rate": successes.float().mean().item(),
        "eval/contact_force_mean_n": (force_sum / steps).mean().item(),
        "eval/contact_force_max_n": force_max.mean().item(),
        "eval/damage_score": env.damage_score.mean().item(),
    }
    return result


def save_checkpoint(path: Path, policy: AsymmetricActorCritic, optimizer: torch.optim.Optimizer, update: int, total_steps: int, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "version": "physx_sphere_final_v1",
            "actor_observation_dim": 110,
            "critic_state_dim": 148,
            "action_dim": 10,
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update": update,
            "total_steps": total_steps,
            "metrics": metrics,
        },
        tmp,
    )
    tmp.replace(path)


def _bounded(value, low, high, default):
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def load_runtime_overrides(path: Path | None) -> dict[str, float | int]:
    """Load monitor-written knobs with hard safety bounds.

    The monitor can shape learning, but it cannot disable termination, raise a
    force limit, or bypass the fixed action-rate cap.  Invalid or partially
    written JSON is ignored until the next update.
    """

    if path is None or not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        "proximity_reward_weight": _bounded(raw.get("proximity_reward_weight"), 0.5, 6.0, 2.0),
        "contact_reward_weight": _bounded(raw.get("contact_reward_weight"), 0.5, 8.0, 2.0),
        "secure_reward_weight": _bounded(raw.get("secure_reward_weight"), 0.5, 10.0, 2.0),
        "force_penalty_weight": _bounded(raw.get("force_penalty_weight"), 0.02, 2.0, 0.10),
        "speed_penalty_weight": _bounded(raw.get("speed_penalty_weight"), 0.05, 2.0, 0.50),
        "success_bonus": _bounded(raw.get("success_bonus"), 2.0, 30.0, 10.0),
        "target_force_n": _bounded(raw.get("target_force_n"), 0.25, 1.5, 0.65),
        "learning_rate": _bounded(raw.get("learning_rate"), 3e-5, 8e-4, args_cli.learning_rate),
        "entropy_coef": _bounded(raw.get("entropy_coef"), 1e-5, 5e-3, args_cli.entropy_coef),
        "policy_std_scale": _bounded(raw.get("policy_std_scale"), 0.60, 1.40, 1.0),
    }


def apply_runtime_overrides(
    cfg: PhysxSphereGraspEnvCfg,
    policy: AsymmetricActorCritic,
    optimizer: torch.optim.Optimizer,
    overrides: dict[str, float | int],
    base_log_std: torch.Tensor,
) -> tuple[float, float]:
    """Apply bounded monitor knobs and return effective LR/entropy values."""

    for name in (
        "proximity_reward_weight",
        "contact_reward_weight",
        "secure_reward_weight",
        "force_penalty_weight",
        "speed_penalty_weight",
        "success_bonus",
        "target_force_n",
    ):
        if name in overrides:
            setattr(cfg, name, overrides[name])
    learning_rate = float(overrides.get("learning_rate", args_cli.learning_rate))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    entropy_coef = float(overrides.get("entropy_coef", args_cli.entropy_coef))
    std_scale = float(overrides.get("policy_std_scale", 1.0))
    with torch.no_grad():
        policy.log_std.copy_((base_log_std + math.log(std_scale)).clamp(policy.minimum_log_std, 1.0))
    return learning_rate, entropy_coef


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.horizon <= 0 or args_cli.updates <= 0:
        raise ValueError("num_envs, horizon and updates must be positive")
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    cfg = PhysxSphereGraspEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    env = PhysxSphereGraspEnv(cfg, render_mode=None)
    obs, _ = env.reset()
    actor_obs = obs["policy"]
    critic_state = env._get_states()
    device = actor_obs.device
    policy = AsymmetricActorCritic(actor_obs.shape[-1], critic_state.shape[-1], cfg.action_space, args_cli.minimum_std).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.learning_rate)
    start_update = 0
    if args_cli.resume is not None:
        checkpoint = torch.load(args_cli.resume.expanduser().resolve(), map_location=device, weights_only=False)
        if (checkpoint.get("actor_observation_dim"), checkpoint.get("critic_state_dim"), checkpoint.get("action_dim")) != (110, 148, 10):
            raise ValueError("checkpoint does not match the final PhysX 10D/110D/148D contract")
        policy.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = int(checkpoint.get("update", 0))
        print(f"[physx-ppo] resumed={args_cli.resume} update={start_update}", flush=True)

    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args_cli.metrics_file or (args_cli.output_dir / "metrics.jsonl")
    control_path = args_cli.control_file or (args_cli.output_dir / "adaptive_overrides.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    control_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path = args_cli.output_dir / "trainer.pid"
    pid_path.write_text(str(os.getpid()), encoding="utf-8")
    base_log_std = policy.log_std.detach().clone()
    runtime_entropy_coef = args_cli.entropy_coef

    run = None
    if args_cli.wandb_mode != "disabled":
        try:
            import wandb

            run = wandb.init(
                project=args_cli.wandb_project,
                name=args_cli.wandb_run_name,
                mode=args_cli.wandb_mode,
                dir=str(args_cli.output_dir),
                config={
                    "environment": "physx_compliant_sphere_final_v2",
                    "actor_observation_dim": 110,
                    "critic_state_dim": 148,
                    "action_dim": 10,
                    "num_envs": args_cli.num_envs,
                    "horizon": args_cli.horizon,
                    "physics_dt": cfg.sim.dt,
                    "decimation": cfg.decimation,
                    "object_mass_prior_kg": cfg.mass_range_kg,
                    "object_stiffness_prior_n_m": cfg.stiffness_range_n_m,
                    "sensor_delay_steps_max": cfg.sensor_delay_steps_max,
                },
            )
        except Exception as exc:
            print(f"[physx-ppo] W&B unavailable ({type(exc).__name__}: {exc}); continuing with local JSONL", flush=True)
            run = None

    stop = {"value": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("value", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("value", True))
    best_success = -float("inf")
    total_steps = int(checkpoint.get("total_steps", 0)) if args_cli.resume is not None else 0
    start_time = time.time()

    try:
        for update in range(start_update + 1, start_update + args_cli.updates + 1):
            overrides = load_runtime_overrides(control_path)
            learning_rate, runtime_entropy_coef = apply_runtime_overrides(
                cfg, policy, optimizer, overrides, base_log_std
            )
            obs_buf = torch.zeros((args_cli.horizon, args_cli.num_envs, 110), device=device)
            state_buf = torch.zeros((args_cli.horizon, args_cli.num_envs, 148), device=device)
            action_buf = torch.zeros((args_cli.horizon, args_cli.num_envs, cfg.action_space), device=device)
            logp_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), device=device)
            reward_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), device=device)
            done_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), device=device)
            value_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), device=device)

            for step in range(args_cli.horizon):
                obs_buf[step] = actor_obs
                state_buf[step] = critic_state
                with torch.no_grad():
                    action, logp = sample_action(policy, actor_obs)
                    value = policy.critic(critic_state)
                action_buf[step] = action
                logp_buf[step] = logp
                value_buf[step] = value
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                reward_buf[step] = reward
                done_buf[step] = done.float()
                actor_obs = next_obs["policy"]
                critic_state = env._get_states()
                total_steps += args_cli.num_envs

            with torch.no_grad():
                next_value = policy.critic(critic_state)
                advantage = torch.zeros_like(reward_buf)
                gae = torch.zeros(args_cli.num_envs, device=device)
                for step in reversed(range(args_cli.horizon)):
                    nonterminal = 1.0 - done_buf[step]
                    next_val = next_value if step == args_cli.horizon - 1 else value_buf[step + 1]
                    delta = reward_buf[step] + args_cli.gamma * next_val * nonterminal - value_buf[step]
                    gae = delta + args_cli.gamma * args_cli.gae_lambda * nonterminal * gae
                    advantage[step] = gae
                returns = advantage + value_buf

            flat_obs = obs_buf.reshape(-1, 110)
            flat_state = state_buf.reshape(-1, 148)
            flat_action = action_buf.reshape(-1, cfg.action_space)
            flat_logp = logp_buf.reshape(-1)
            flat_adv = advantage.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1e-8)
            batch_size = flat_obs.shape[0]
            minibatch = min(args_cli.minibatch_size, batch_size)
            actor_loss_value = 0.0
            critic_loss_value = 0.0
            entropy_value = 0.0
            for _ in range(args_cli.epochs):
                permutation = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch):
                    idx = permutation[start : start + minibatch]
                    dist = policy.distribution(flat_obs[idx])
                    # Reconstruct a numerically stable pre-tanh action.
                    raw_action = torch.atanh(flat_action[idx].clamp(-0.999999, 0.999999))
                    new_logp = squashed_log_prob(dist, raw_action, flat_action[idx])
                    ratio = (new_logp - flat_logp[idx]).exp()
                    clipped = torch.clamp(ratio, 1.0 - args_cli.clip_eps, 1.0 + args_cli.clip_eps)
                    actor_loss = -torch.minimum(ratio * flat_adv[idx], clipped * flat_adv[idx]).mean()
                    value_pred = policy.critic(flat_state[idx])
                    critic_loss = 0.5 * (value_pred - flat_returns[idx]).square().mean()
                    entropy = dist.entropy().sum(dim=-1).mean()
                    loss = actor_loss + args_cli.value_coef * critic_loss - runtime_entropy_coef * entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    actor_loss_value += float(actor_loss.detach())
                    critic_loss_value += float(critic_loss.detach())
                    entropy_value += float(entropy.detach())

            metrics = {
                "update": update,
                "steps": total_steps,
                "rollout/reward_mean": reward_buf.mean().item(),
                "rollout/success_rate": env.success.float().mean().item(),
                "loss/actor": actor_loss_value,
                "loss/critic": critic_loss_value,
                "loss/entropy": entropy_value,
                "runtime/learning_rate": learning_rate,
                "runtime/entropy_coef": runtime_entropy_coef,
                "runtime/target_force_n": float(cfg.target_force_n),
                "runtime/force_penalty_weight": float(cfg.force_penalty_weight),
                "runtime/contact_reward_weight": float(cfg.contact_reward_weight),
                "runtime/secure_reward_weight": float(cfg.secure_reward_weight),
                "runtime/policy_std_mean": float(policy.log_std.exp().mean().detach()),
                "elapsed_seconds": time.time() - start_time,
            }
            if update % args_cli.eval_interval == 0:
                metrics.update(evaluate(env, policy, args_cli.eval_steps))
                evaluation_success = metrics["eval/success_rate"]
                is_best = evaluation_success > best_success
                best_success = max(best_success, evaluation_success)
                # Evaluation resets the vectorized environment.  Rebuild both
                # streams before the next rollout instead of continuing with
                # stale tensors from the pre-evaluation episode.
                obs, _ = env.reset()
                actor_obs = obs["policy"]
                critic_state = env._get_states()
            else:
                is_best = False
            if update % args_cli.save_interval == 0 or update == start_update + args_cli.updates:
                save_checkpoint(args_cli.output_dir / "latest.pt", policy, optimizer, update, total_steps, metrics)
            if is_best:
                save_checkpoint(args_cli.output_dir / "best.pt", policy, optimizer, update, total_steps, metrics)
            if run is not None:
                run.log(metrics, step=update)
            record = {"timestamp": time.time(), **metrics}
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
            if update == 1 or update % 10 == 0:
                print(metrics, flush=True)
            if stop["value"]:
                print("[physx-ppo] stop requested; checkpoint saved", flush=True)
                save_checkpoint(args_cli.output_dir / "latest.pt", policy, optimizer, update, total_steps, metrics)
                break
    finally:
        if run is not None:
            run.finish()
        env.close()
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        import traceback

        print(f"[physx-ppo] fatal: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
