"""Audit terminal-path returns and PPO GAE without updating a checkpoint."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
import math
from pathlib import Path


OUTCOME_NAMES = ("incomplete", "success", "drop", "excess_force", "timeout", "truncated")


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "min": min(values),
        "max": max(values),
    }


def _parser() -> argparse.ArgumentParser:
    app_launcher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sim2real-mode", choices=("full", "nominal"), default="full")
    parser.add_argument("--stochastic", action="store_true")
    app_launcher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    app_launcher = importlib.import_module("isaaclab.app").AppLauncher
    args = _parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("--num-envs and --steps must be positive")

    launcher = app_launcher(args)
    simulation_app = launcher.app
    wrapped = None
    try:
        gym = importlib.import_module("gymnasium")
        torch = importlib.import_module("torch")
        runner_module = importlib.import_module("rsl_rl.runners.on_policy_runner")
        wrapper_module = importlib.import_module("isaaclab_rl.rsl_rl")
        task = importlib.import_module("simulation.tomato_pick.isaac")
        cfg_module = importlib.import_module("simulation.tomato_pick.isaac.env_cfg")
        agent_module = importlib.import_module(
            "simulation.tomato_pick.isaac.agents.rsl_rl_ppo_cfg"
        )

        cfg = cfg_module.TomatoPickEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.enable_sim2real_randomization = args.sim2real_mode == "full"
        raw_gym_env = gym.make(task.TASK_ID, cfg=cfg)

        agent_cfg = agent_module.TomatoPickPPORunnerCfg()
        agent_cfg.device = args.device
        agent_cfg = wrapper_module.handle_deprecated_rsl_rl_cfg(
            agent_cfg, metadata.version("rsl-rl-lib")
        )
        wrapped = wrapper_module.RslRlVecEnvWrapper(
            raw_gym_env, clip_actions=agent_cfg.clip_actions
        )
        runner = runner_module.OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(args.checkpoint.resolve()))
        algorithm = runner.alg
        policy = runner.get_inference_policy(device=wrapped.device)
        raw = wrapped.unwrapped
        observations = wrapped.get_observations()

        episode_return = torch.zeros(args.num_envs, device=wrapped.device)
        discounted_return = torch.zeros_like(episode_return)
        discount = torch.ones_like(episode_return)
        episode_length = torch.zeros(args.num_envs, dtype=torch.long, device=wrapped.device)
        maximum_phase = torch.zeros_like(episode_length)
        initial_value = torch.zeros_like(episode_return)
        episode_term_sums: dict[str, object] = {}
        episode_rows: list[dict[str, object]] = []

        rewards_history = []
        values_history = []
        next_values_history = []
        dones_history = []
        phases_history = []
        outcomes_history = []
        actions_history = []
        latent_means_history = []
        stds_history = []

        gamma = float(algorithm.gamma)
        gae_lambda = float(algorithm.lam)
        rollout_length = int(agent_cfg.num_steps_per_env)

        with torch.inference_mode():
            for _ in range(args.steps):
                phase = raw._phase.clone()
                values = algorithm.critic(observations).squeeze(-1)
                starting = episode_length == 0
                initial_value[starting] = values[starting]
                maximum_phase = torch.maximum(maximum_phase, phase)

                actions = (
                    algorithm.actor(observations, stochastic_output=True)
                    if args.stochastic
                    else policy(observations)
                )
                if args.stochastic:
                    latent_mean, std = algorithm.actor.distribution.params
                    actions_history.append(actions.clone())
                    latent_means_history.append(latent_mean.clone())
                    stds_history.append(std.clone())
                next_observations, rewards, dones, extras = wrapped.step(actions)
                next_values = algorithm.critic(next_observations).squeeze(-1)
                terms = raw._last_reward_terms
                if not episode_term_sums:
                    episode_term_sums = {
                        name: torch.zeros(args.num_envs, device=wrapped.device)
                        for name in terms
                    }

                episode_return += rewards
                discounted_return += discount * rewards
                discount *= gamma
                episode_length += 1
                for name, term in terms.items():
                    episode_term_sums[name] += term

                outcome = torch.zeros(args.num_envs, dtype=torch.long, device=wrapped.device)
                outcome[terms["success"] != 0.0] = 1
                outcome[terms["drop"] != 0.0] = 2
                outcome[terms["excess_force"] != 0.0] = 3
                outcome[terms["timeout"] != 0.0] = 4
                time_outs = extras.get("time_outs") if isinstance(extras, dict) else None
                if time_outs is not None:
                    outcome[time_outs.bool() & (outcome == 0)] = 5

                rewards_history.append(rewards.clone())
                values_history.append(values.clone())
                next_values_history.append(next_values.clone())
                dones_history.append(dones.bool().clone())
                phases_history.append(phase.clone())
                outcomes_history.append(outcome.clone())

                done_ids = dones.nonzero(as_tuple=False).squeeze(1)
                for env_id in done_ids.detach().cpu().tolist():
                    code = int(outcome[env_id])
                    row = {
                        "outcome": OUTCOME_NAMES[code],
                        "maximum_phase": int(maximum_phase[env_id]),
                        "length": int(episode_length[env_id]),
                        "return": float(episode_return[env_id]),
                        "discounted_return": float(discounted_return[env_id]),
                        "initial_value": float(initial_value[env_id]),
                        "terms": {
                            name: float(values_by_env[env_id])
                            for name, values_by_env in episode_term_sums.items()
                        },
                    }
                    episode_rows.append(row)

                if done_ids.numel():
                    episode_return[done_ids] = 0.0
                    discounted_return[done_ids] = 0.0
                    discount[done_ids] = 1.0
                    episode_length[done_ids] = 0
                    maximum_phase[done_ids] = 0
                    initial_value[done_ids] = 0.0
                    for values_by_env in episode_term_sums.values():
                        values_by_env[done_ids] = 0.0
                observations = next_observations

        rewards_tensor = torch.stack(rewards_history)
        values_tensor = torch.stack(values_history)
        next_values_tensor = torch.stack(next_values_history)
        dones_tensor = torch.stack(dones_history)
        phases_tensor = torch.stack(phases_history)
        outcomes_tensor = torch.stack(outcomes_history)

        # Match RSL-RL's 24-step rollout boundary: bootstrap the last value in
        # each chunk, but never propagate an advantage across a true episode end.
        advantages = torch.zeros_like(rewards_tensor)
        for chunk_start in range(0, args.steps, rollout_length):
            chunk_stop = min(chunk_start + rollout_length, args.steps)
            running_advantage = torch.zeros(args.num_envs, device=wrapped.device)
            for step in reversed(range(chunk_start, chunk_stop)):
                not_done = (~dones_tensor[step]).float()
                delta = (
                    rewards_tensor[step]
                    + gamma * not_done * next_values_tensor[step]
                    - values_tensor[step]
                )
                running_advantage = (
                    delta + gamma * gae_lambda * not_done * running_advantage
                )
                advantages[step] = running_advantage

        # Label each transition with the terminal result reached later in the
        # same episode. Incomplete tail episodes remain zero and are reported.
        propagated_outcomes = torch.zeros_like(outcomes_tensor)
        future_outcome = torch.zeros(args.num_envs, dtype=torch.long, device=wrapped.device)
        for step in reversed(range(args.steps)):
            terminal = dones_tensor[step]
            future_outcome = torch.where(terminal, outcomes_tensor[step], future_outcome)
            propagated_outcomes[step] = future_outcome

        global_advantage_mean = float(advantages.mean())
        global_advantage_std = float(advantages.std(unbiased=False).clamp_min(1.0e-12))
        normalized_advantages = (
            advantages - global_advantage_mean
        ) / global_advantage_std

        policy_gradient_by_phase = {}
        if args.stochastic:
            actions_tensor = torch.stack(actions_history)
            latent_means_tensor = torch.stack(latent_means_history)
            stds_tensor = torch.stack(stds_history)
            latent_actions = torch.atanh(actions_tensor.clamp(-0.999999, 0.999999))
            mean_score = (latent_actions - latent_means_tensor) / stds_tensor.square()
            score_advantage = mean_score * normalized_advantages.unsqueeze(-1)
            for phase in range(4):
                mask = phases_tensor == phase
                policy_gradient_by_phase[str(phase)] = {
                    "transition_count": int(mask.sum()),
                    "mean_action": actions_tensor[mask].mean(dim=0).cpu().tolist(),
                    "mean_latent_mean": latent_means_tensor[mask].mean(dim=0).cpu().tolist(),
                    "mean_std": stds_tensor[mask].mean(dim=0).cpu().tolist(),
                    "mean_score_advantage": score_advantage[mask].mean(dim=0).cpu().tolist(),
                    "mean_score_advantage_by_outcome": {
                        name: (
                            score_advantage[mask & (propagated_outcomes == code)]
                            .mean(dim=0)
                            .cpu()
                            .tolist()
                            if bool((mask & (propagated_outcomes == code)).any())
                            else None
                        )
                        for code, name in enumerate(OUTCOME_NAMES)
                    },
                }

        def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
            return {
                "return": _summary([float(row["return"]) for row in rows]),
                "discounted_return": _summary(
                    [float(row["discounted_return"]) for row in rows]
                ),
                "length": _summary([float(row["length"]) for row in rows]),
                "initial_value": _summary([float(row["initial_value"]) for row in rows]),
                "mean_term_sums": {
                    name: sum(float(row["terms"][name]) for row in rows) / len(rows)
                    for name in episode_term_sums
                } if rows else {},
            }

        path_report = {}
        for outcome_name in OUTCOME_NAMES[1:]:
            rows = [row for row in episode_rows if row["outcome"] == outcome_name]
            path_report[outcome_name] = summarize_rows(rows)
            path_report[outcome_name]["by_maximum_phase"] = {
                str(phase): summarize_rows(
                    [row for row in rows if row["maximum_phase"] == phase]
                )
                for phase in range(4)
            }

        advantage_by_outcome = {}
        for code, name in enumerate(OUTCOME_NAMES):
            mask = propagated_outcomes == code
            advantage_by_outcome[name] = {
                "transition_count": int(mask.sum()),
                "raw": _summary(advantages[mask].detach().cpu().tolist()),
                "rollout_normalized": _summary(
                    normalized_advantages[mask].detach().cpu().tolist()
                ),
            }

        advantage_by_phase = {}
        for phase in range(4):
            mask = phases_tensor == phase
            advantage_by_phase[str(phase)] = {
                "transition_count": int(mask.sum()),
                "raw": _summary(advantages[mask].detach().cpu().tolist()),
                "rollout_normalized": _summary(
                    normalized_advantages[mask].detach().cpu().tolist()
                ),
            }

        report = {
            "schema_version": 1,
            "checkpoint": str(args.checkpoint.resolve()),
            "num_envs": args.num_envs,
            "steps": args.steps,
            "completed_episodes": len(episode_rows),
            "sim2real_mode": args.sim2real_mode,
            "stochastic": args.stochastic,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "rollout_length": rollout_length,
            "global_advantage": {
                "mean": global_advantage_mean,
                "std": global_advantage_std,
            },
            "terminal_paths": path_report,
            "advantage_by_eventual_outcome": advantage_by_outcome,
            "advantage_by_phase": advantage_by_phase,
            "policy_gradient_by_phase": policy_gradient_by_phase,
        }
        print("TOMATO_RETURN_AUDIT=" + json.dumps(report, sort_keys=True), flush=True)
        return 0
    finally:
        if wrapped is not None:
            wrapped.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
