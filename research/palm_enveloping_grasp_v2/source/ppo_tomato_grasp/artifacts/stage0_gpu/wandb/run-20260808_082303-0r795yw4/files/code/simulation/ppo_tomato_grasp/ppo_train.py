"""Compact PPO trainer for the Stage-0 rigid-sphere enclosure task.

The trainer intentionally stays dependency-light, but it supports the pieces
needed for a real local run: bounded actions, minibatches, wall-clock limits,
periodic deterministic evaluation, atomic checkpoints, resume, and W&B.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="PPO trainer for Stage-0 Linkbot sphere enclosure")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--updates", type=int, default=4, help="Total update target; 0 means no update limit")
parser.add_argument("--max-hours", type=float, default=0.0, help="Cumulative wall-clock budget; 0 disables it")
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--minibatch-size", type=int, default=4096)
parser.add_argument("--learning-rate", type=float, default=3.0e-4)
parser.add_argument("--entropy-coef", type=float, default=2.0e-3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save-interval", type=int, default=25)
parser.add_argument("--eval-interval", type=int, default=25)
parser.add_argument("--eval-steps", type=int, default=120)
parser.add_argument("--log-interval", type=int, default=5)
parser.add_argument("--early-stop-evals", type=int, default=0)
parser.add_argument("--early-stop-success", type=float, default=0.99)
parser.add_argument("--early-stop-max-load", type=float, default=0.75)
parser.add_argument("--linkbot", action="store_true")
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=ROOT / "simulation/ppo_tomato_grasp/artifacts/stage0_gpu/latest.pt",
)
parser.add_argument("--resume", type=Path, default=None)
parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
parser.add_argument("--wandb-project", default="linkbot-tomato-grasp-stage0")
parser.add_argument("--wandb-entity", default=None)
parser.add_argument("--wandb-run-name", default=None)
parser.add_argument("--wandb-run-id", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from torch import nn
from torch.distributions import Normal

from simulation.ppo_tomato_grasp.tomato_grasp_env import TomatoGraspEnv, TomatoGraspEnvCfg


def squashed_log_prob(dist: Normal, raw_action: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Log probability of a tanh-squashed Gaussian action."""

    correction = torch.log(1.0 - action.square() + 1.0e-6)
    return (dist.log_prob(raw_action) - correction).sum(-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mean, value = self(obs)
        return Normal(mean, self.log_std.clamp(-5.0, 1.0).exp()), value


@torch.inference_mode()
def evaluate_policy(
    env: TomatoGraspEnv, policy: ActorCritic, steps: int
) -> tuple[dict[str, float], torch.Tensor]:
    """Run a deterministic evaluation on the same vector environment."""

    obs, _ = env.reset()
    obs_tensor = obs["policy"]
    returns = torch.zeros(env.num_envs, device=obs_tensor.device)
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=obs_tensor.device)
    mean_load = torch.zeros(env.num_envs, device=obs_tensor.device)
    max_load = torch.zeros(env.num_envs, device=obs_tensor.device)
    for _ in range(steps):
        mean, _ = policy(obs_tensor)
        action = torch.tanh(mean)
        next_obs, reward, _, _, _ = env.step(action)
        returns += reward
        success |= env.success
        mean_load += env.contact_loads.mean(dim=-1)
        max_load = torch.maximum(max_load, env.contact_loads.max(dim=-1).values)
        obs_tensor = next_obs["policy"]

    metrics = {
        "eval/return_mean": returns.mean().item(),
        "eval/success_rate": success.float().mean().item(),
        "eval/mean_contact_load": (mean_load / steps).mean().item(),
        "eval/max_contact_load": max_load.mean().item(),
    }
    # Evaluation resets the vector environment. Start the next rollout from a
    # fresh, explicitly returned observation so no stale training state leaks.
    obs, _ = env.reset()
    return metrics, obs["policy"]


def atomic_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.horizon <= 0:
        raise ValueError("--num_envs and --horizon must be positive")
    if args_cli.updates < 0 or args_cli.max_hours < 0.0:
        raise ValueError("--updates and --max-hours cannot be negative")
    if args_cli.updates == 0 and args_cli.max_hours == 0.0:
        raise ValueError("set a positive --updates or --max-hours to bound the run")

    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    cfg = TomatoGraspEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.use_linkbot_asset = args_cli.linkbot
    env = TomatoGraspEnv(cfg=cfg, render_mode=None)
    obs, _ = env.reset()
    obs_tensor = obs["policy"]
    device = obs_tensor.device
    policy = ActorCritic(obs_tensor.shape[-1], cfg.action_space).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.learning_rate)

    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    resume_path = args_cli.resume.expanduser().resolve() if args_cli.resume else None
    update = 0
    total_steps = 0
    elapsed_before = 0.0
    resumed_wandb_id = None
    best_eval_score = float("-inf")
    if resume_path is not None:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        policy.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        update = int(state.get("update", 0))
        total_steps = int(state.get("total_steps", update * args_cli.horizon * args_cli.num_envs))
        elapsed_before = float(state.get("elapsed_seconds", 0.0))
        resumed_wandb_id = state.get("wandb_run_id")
        best_eval_score = float(state.get("best_eval_score", float("-inf")))
        print(f"[tomato-ppo] resumed={resume_path} update={update} steps={total_steps}", flush=True)

    run = None
    wandb_run_id = args_cli.wandb_run_id or resumed_wandb_id
    if args_cli.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args_cli.wandb_project,
            entity=args_cli.wandb_entity,
            name=args_cli.wandb_run_name,
            id=wandb_run_id,
            resume="allow" if wandb_run_id else None,
            mode=args_cli.wandb_mode,
            force=args_cli.wandb_mode == "online",
            dir=str(checkpoint_path.parent),
            save_code=True,
            config={
                "stage": "rigid_sphere_enclosure",
                "num_envs": args_cli.num_envs,
                "horizon": args_cli.horizon,
                "max_hours": args_cli.max_hours,
                "epochs": args_cli.epochs,
                "minibatch_size": args_cli.minibatch_size,
                "learning_rate": args_cli.learning_rate,
                "entropy_coefficient": args_cli.entropy_coef,
                "seed": args_cli.seed,
                "device": str(device),
                "use_linkbot_asset": args_cli.linkbot,
                "object_position_jitter_m": list(cfg.object_position_jitter),
                "observation_dim": obs_tensor.shape[-1],
                "action_dim": cfg.action_space,
            },
        )
        wandb_run_id = run.id
        print(f"[tomato-ppo] wandb_url={run.url}", flush=True)

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print("[tomato-ppo] graceful stop requested; finishing current update", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started_at = time.monotonic()
    last_metrics: dict[str, float] = {}
    successful_evaluations = 0

    def elapsed_seconds() -> float:
        return elapsed_before + (time.monotonic() - started_at)

    def save_checkpoint(reason: str, destination: Path = checkpoint_path) -> None:
        payload = {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_dim": obs_tensor.shape[-1],
            "action_dim": cfg.action_space,
            "update": update,
            "total_steps": total_steps,
            "elapsed_seconds": elapsed_seconds(),
            "wandb_run_id": wandb_run_id,
            "best_eval_score": best_eval_score,
            "reason": reason,
            "metrics": last_metrics,
        }
        atomic_save(destination, payload)
        print(f"[tomato-ppo] checkpoint={destination} reason={reason}", flush=True)

    try:
        while True:
            if args_cli.updates > 0 and update >= args_cli.updates:
                break
            if args_cli.max_hours > 0.0 and elapsed_seconds() >= args_cli.max_hours * 3600.0:
                break
            if stop_requested:
                break

            rollout_started = time.monotonic()
            observations, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
            rollout_logs: dict[str, float] = {}
            for _ in range(args_cli.horizon):
                with torch.no_grad():
                    dist, value = policy.distribution(obs_tensor)
                    raw_action = dist.sample()
                    action = torch.tanh(raw_action)
                    log_prob = squashed_log_prob(dist, raw_action, action)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                observations.append(obs_tensor)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(terminated | truncated)
                values.append(value)
                obs_tensor = next_obs["policy"]
                for key, value_item in env.extras.get("log", {}).items():
                    scalar = value_item.item() if torch.is_tensor(value_item) else float(value_item)
                    rollout_logs[key] = rollout_logs.get(key, 0.0) + scalar / args_cli.horizon

            with torch.no_grad():
                _, next_value = policy(obs_tensor)
            obs_batch = torch.stack(observations)
            action_batch = torch.stack(actions)
            old_log_prob = torch.stack(log_probs)
            reward_batch = torch.stack(rewards)
            done_batch = torch.stack(dones).float()
            value_batch = torch.stack(values)
            advantages = torch.zeros_like(reward_batch)
            gae = torch.zeros(args_cli.num_envs, device=device)
            for t in reversed(range(args_cli.horizon)):
                bootstrap = next_value if t == args_cli.horizon - 1 else value_batch[t + 1]
                not_done = 1.0 - done_batch[t]
                delta = reward_batch[t] + 0.99 * bootstrap * not_done - value_batch[t]
                gae = delta + 0.95 * 0.99 * not_done * gae
                advantages[t] = gae
            returns = advantages + value_batch
            flat_obs = obs_batch.reshape(-1, obs_batch.shape[-1])
            flat_actions = action_batch.reshape(-1, action_batch.shape[-1])
            flat_old_log = old_log_prob.reshape(-1)
            flat_adv = ((advantages.reshape(-1) - advantages.mean()) / (advantages.std() + 1.0e-8)).detach()
            flat_returns = returns.reshape(-1).detach()

            batch_size = flat_obs.shape[0]
            minibatch_size = min(args_cli.minibatch_size, batch_size)
            loss_sums = {"actor": 0.0, "critic": 0.0, "entropy": 0.0, "total": 0.0}
            optimization_steps = 0
            for _ in range(args_cli.epochs):
                permutation = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch_size):
                    indices = permutation[start : start + minibatch_size]
                    dist, value = policy.distribution(flat_obs[indices])
                    bounded_actions = flat_actions[indices].clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
                    raw_actions = torch.atanh(bounded_actions)
                    log_ratio = squashed_log_prob(dist, raw_actions, bounded_actions) - flat_old_log[indices]
                    ratio = log_ratio.exp().clamp(0.05, 20.0)
                    mb_adv = flat_adv[indices]
                    actor_loss = -torch.minimum(ratio * mb_adv, ratio.clamp(0.8, 1.2) * mb_adv).mean()
                    critic_loss = 0.5 * (value - flat_returns[indices]).square().mean()
                    entropy = dist.entropy().sum(-1).mean()
                    loss = actor_loss + 0.5 * critic_loss - args_cli.entropy_coef * entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    loss_sums["actor"] += actor_loss.item()
                    loss_sums["critic"] += critic_loss.item()
                    loss_sums["entropy"] += entropy.item()
                    loss_sums["total"] += loss.item()
                    optimization_steps += 1

            update += 1
            update_steps = args_cli.horizon * args_cli.num_envs
            total_steps += update_steps
            rollout_seconds = max(time.monotonic() - rollout_started, 1.0e-6)
            metrics = {
                "train/rollout_return_mean": reward_batch.sum(0).mean().item(),
                "train/reward_mean": reward_batch.mean().item(),
                "train/actor_loss": loss_sums["actor"] / optimization_steps,
                "train/critic_loss": loss_sums["critic"] / optimization_steps,
                "train/entropy": loss_sums["entropy"] / optimization_steps,
                "train/total_loss": loss_sums["total"] / optimization_steps,
                "train/action_std_mean": policy.log_std.clamp(-5.0, 1.0).exp().mean().item(),
                "train/steps_per_second": update_steps / rollout_seconds,
                "train/update": update,
                "train/total_steps": total_steps,
                "train/elapsed_hours": elapsed_seconds() / 3600.0,
            }
            metrics.update({f"env/{key}": value for key, value in rollout_logs.items()})

            if args_cli.eval_interval > 0 and update % args_cli.eval_interval == 0:
                eval_metrics, obs_tensor = evaluate_policy(env, policy, args_cli.eval_steps)
                metrics.update(eval_metrics)
                eval_score = eval_metrics["eval/success_rate"] - 0.1 * eval_metrics["eval/mean_contact_load"]
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    metrics["eval/best_score"] = best_eval_score
                    last_metrics = metrics
                    save_checkpoint("best", checkpoint_path.with_name("best.pt"))
                acceptable = (
                    eval_metrics["eval/success_rate"] >= args_cli.early_stop_success
                    and eval_metrics["eval/max_contact_load"] <= args_cli.early_stop_max_load
                )
                successful_evaluations = successful_evaluations + 1 if acceptable else 0

            last_metrics = metrics
            if run is not None:
                run.log(metrics, step=total_steps)
            if update == 1 or update % args_cli.log_interval == 0:
                print(
                    f"[tomato-ppo] update={update} steps={total_steps} "
                    f"return={metrics['train/rollout_return_mean']:.4f} "
                    f"success={metrics.get('env/grasp_success_rate', 0.0):.3f} "
                    f"sps={metrics['train/steps_per_second']:.0f} "
                    f"hours={metrics['train/elapsed_hours']:.3f}",
                    flush=True,
                )
            if args_cli.save_interval > 0 and update % args_cli.save_interval == 0:
                save_checkpoint("periodic")
            if args_cli.early_stop_evals > 0 and successful_evaluations >= args_cli.early_stop_evals:
                print(
                    f"[tomato-ppo] early_stop successful_evaluations={successful_evaluations}",
                    flush=True,
                )
                break

        save_checkpoint("completed" if not stop_requested else "signal")
    finally:
        env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
