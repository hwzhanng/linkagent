"""Evaluate one tomato PPO checkpoint with a finite deterministic rollout."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import time

import gymnasium as gym
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection, string_to_callable
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config


_isaac_root_value = os.environ.get("LINKAGENT_ISAACLAB_ROOT")
if not _isaac_root_value:
    raise RuntimeError("set LINKAGENT_ISAACLAB_ROOT to the Isaac Lab checkout")
ISAAC_ROOT = Path(_isaac_root_value).expanduser().resolve()
sys.path.insert(0, str(ISAAC_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"))
import cli_args  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--task", type=str, default="LinkAgent-TomatoPick-Clean-v5")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--stochastic-policy",
    action="store_true",
    help="Sample the checkpoint policy distribution as during PPO rollouts.",
)
parser.add_argument("--external_callback", default="simulation.tomato_pick.isaac.register_task")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

remaining_args_env_registration = None
if args_cli.external_callback:
    callback = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = callback()
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args
installed_version = metadata.version("rsl-rl-lib")
torch = importlib.import_module("torch")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required")
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    output = args_cli.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args_cli.steps <= 0 or args_cli.num_envs <= 0:
        raise ValueError("steps and num_envs must be positive")

    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.log_dir = str(checkpoint.parent)

        configure_seed(args_cli.seed, True)
        gym_env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(retrieve_file_path(str(checkpoint))))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()

        episode_returns = torch.zeros(env.num_envs, device=env.device)
        episode_lengths = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        completed_return_sum = 0.0
        completed_return_sq_sum = 0.0
        completed_length_sum = 0
        completed_episodes = 0
        success_events = 0
        drop_events = 0
        excess_force_events = 0
        out_of_bounds_events = 0
        timeout_events = 0
        logged_reset_count = 0.0
        logged_success_count = 0.0
        logged_drop_count = 0.0
        logged_detach_count = 0.0
        logged_retained_count = 0.0
        logged_hold_sum = 0.0
        logged_reward_term_sums: dict[str, float] = {}
        raw_action_abs_sum = 0.0
        action_abs_sum = 0.0
        action_sq_sum = 0.0
        action_count = 0
        saturated_actions = 0
        base_env = env.unwrapped
        has_stem_damage_diagnostics = all(
            hasattr(base_env, name)
            for name in (
                "_stem_damage",
                "_stem_combined_load_ratio",
                "_shadow_secure_support",
            )
        )
        stem_sample_count = 0
        stem_damage_sum = 0.0
        stem_damage_sq_sum = 0.0
        stem_load_sum = 0.0
        stem_load_sq_sum = 0.0
        stem_damage_threshold_counts = {
            "gt_0_01": 0,
            "gt_0_10": 0,
            "gt_0_25": 0,
            "gt_0_50": 0,
            "gt_0_75": 0,
            "gt_0_90": 0,
        }
        stem_load_threshold_counts = {
            "gt_0_55": 0,
            "gt_0_75": 0,
            "gt_0_90": 0,
            "gt_1_00": 0,
        }
        secure_support_count = 0
        secure_support_damage_sum = 0.0
        secure_support_load_sum = 0.0
        maximum_stem_damage = 0.0
        maximum_stem_load_ratio = 0.0
        physics_v1_break_audit_sums: dict[str, float] = {}
        started = time.perf_counter()

        for _ in range(args_cli.steps):
            with torch.inference_mode():
                actions = policy(obs, stochastic_output=args_cli.stochastic_policy)
                raw_action_abs_sum += actions.abs().sum().item()
                executed_actions = actions.clamp(-1.0, 1.0)
                action_abs_sum += executed_actions.abs().sum().item()
                action_sq_sum += executed_actions.square().sum().item()
                action_count += actions.numel()
                saturated_actions += (actions.abs() >= 0.99).sum().item()
                obs, rewards, dones, extras = env.step(actions)
                policy.reset(dones)

                if has_stem_damage_diagnostics:
                    damage = base_env._stem_damage
                    load_ratio = base_env._stem_combined_load_ratio
                    secure_support = base_env._shadow_secure_support
                    stem_sample_count += damage.numel()
                    stem_damage_sum += damage.sum().item()
                    stem_damage_sq_sum += damage.square().sum().item()
                    stem_load_sum += load_ratio.sum().item()
                    stem_load_sq_sum += load_ratio.square().sum().item()
                    maximum_stem_damage = max(
                        maximum_stem_damage, float(damage.max().item())
                    )
                    maximum_stem_load_ratio = max(
                        maximum_stem_load_ratio, float(load_ratio.max().item())
                    )
                    for name, threshold in (
                        ("gt_0_01", 0.01),
                        ("gt_0_10", 0.10),
                        ("gt_0_25", 0.25),
                        ("gt_0_50", 0.50),
                        ("gt_0_75", 0.75),
                        ("gt_0_90", 0.90),
                    ):
                        stem_damage_threshold_counts[name] += int(
                            (damage > threshold).sum().item()
                        )
                    for name, threshold in (
                        ("gt_0_55", 0.55),
                        ("gt_0_75", 0.75),
                        ("gt_0_90", 0.90),
                        ("gt_1_00", 1.00),
                    ):
                        stem_load_threshold_counts[name] += int(
                            (load_ratio > threshold).sum().item()
                        )
                    support_count = int(secure_support.sum().item())
                    secure_support_count += support_count
                    if support_count:
                        secure_support_damage_sum += damage[secure_support].sum().item()
                        secure_support_load_sum += load_ratio[secure_support].sum().item()

            episode_returns += rewards
            episode_lengths += 1
            done = dones.bool()
            done_count = int(done.sum().item())
            if done_count:
                finished_returns = episode_returns[done]
                completed_return_sum += finished_returns.sum().item()
                completed_return_sq_sum += finished_returns.square().sum().item()
                completed_length_sum += int(episode_lengths[done].sum().item())
                completed_episodes += done_count
                episode_returns[done] = 0.0
                episode_lengths[done] = 0

                log = extras.get("log", {})
                reset_count = float(log.get("Metrics/reset_count", done_count))
                logged_reset_count += reset_count
                logged_success_count += float(log.get("Metrics/success_rate", 0.0)) * reset_count
                logged_drop_count += float(log.get("Metrics/drop_rate", 0.0)) * reset_count
                logged_detach_count += float(log.get("Metrics/detach_rate", 0.0)) * reset_count
                logged_retained_count += float(log.get("Metrics/retained_rate", 0.0)) * reset_count
                logged_hold_sum += float(log.get("Metrics/mean_hold_steps", 0.0)) * reset_count
                for key, value in log.items():
                    if key.startswith("Episode_Reward/"):
                        name = key.removeprefix("Episode_Reward/")
                        logged_reward_term_sums[name] = (
                            logged_reward_term_sums.get(name, 0.0)
                            + float(value) * reset_count
                        )

            step = extras.get("step", {})
            for key, value in step.items():
                if key.startswith("physics_v1_break_"):
                    physics_v1_break_audit_sums[key] = (
                        physics_v1_break_audit_sums.get(key, 0.0) + float(value)
                    )
            success_events += int(step.get("success_count", 0))
            drop_events += int(step.get("drop_count", 0))
            excess_force_events += int(step.get("excess_force_count", 0))
            out_of_bounds_events += int(step.get("out_of_bounds_count", 0))
            timeouts = extras.get("time_outs")
            if timeouts is not None:
                timeout_events += int(timeouts.sum().item())

        elapsed = time.perf_counter() - started
        env.close()

        if completed_episodes <= 0 or logged_reset_count <= 0:
            raise RuntimeError("evaluation completed no episodes")
        if abs(logged_reset_count - completed_episodes) > 0.5:
            raise RuntimeError(
                f"reset accounting mismatch: log={logged_reset_count}, dones={completed_episodes}"
            )
        mean_return = completed_return_sum / completed_episodes
        return_variance = max(
            0.0, completed_return_sq_sum / completed_episodes - mean_return * mean_return
        )
        result = {
            "schema_version": 1,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "deterministic_policy": not args_cli.stochastic_policy,
            "completed_episodes": completed_episodes,
            "success_count": success_events,
            "success_rate": success_events / completed_episodes,
            "drop_count": drop_events,
            "drop_rate": drop_events / completed_episodes,
            "detach_count": logged_detach_count,
            "detach_rate": logged_detach_count / logged_reset_count,
            "retained_count": logged_retained_count,
            "retained_rate": logged_retained_count / logged_reset_count,
            "mean_hold_steps_at_reset": logged_hold_sum / logged_reset_count,
            "excess_force_count": excess_force_events,
            "out_of_bounds_count": out_of_bounds_events,
            "timeout_count": timeout_events,
            "mean_episode_return": mean_return,
            "std_episode_return": return_variance**0.5,
            "mean_episode_length": completed_length_sum / completed_episodes,
            "raw_mean_abs_action": raw_action_abs_sum / action_count,
            "mean_abs_action": action_abs_sum / action_count,
            "rms_action": (action_sq_sum / action_count) ** 0.5,
            "action_saturation_rate": saturated_actions / action_count,
            "elapsed_seconds": elapsed,
            "simulation_steps_per_second": args_cli.steps * args_cli.num_envs / elapsed,
            "logged_success_count": logged_success_count,
            "logged_drop_count": logged_drop_count,
            "mean_episode_reward_terms": {
                name: value / logged_reset_count
                for name, value in sorted(logged_reward_term_sums.items())
            },
        }
        if has_stem_damage_diagnostics:
            mean_damage = stem_damage_sum / stem_sample_count
            mean_load = stem_load_sum / stem_sample_count
            result["physics_v1_stem_diagnostics"] = {
                "sample_count": stem_sample_count,
                "mean_damage": mean_damage,
                "std_damage": max(
                    0.0, stem_damage_sq_sum / stem_sample_count - mean_damage**2
                )
                ** 0.5,
                "maximum_damage": maximum_stem_damage,
                "damage_sample_rates": {
                    name: count / stem_sample_count
                    for name, count in stem_damage_threshold_counts.items()
                },
                "mean_combined_load_ratio": mean_load,
                "std_combined_load_ratio": max(
                    0.0, stem_load_sq_sum / stem_sample_count - mean_load**2
                )
                ** 0.5,
                "maximum_combined_load_ratio": maximum_stem_load_ratio,
                "combined_load_sample_rates": {
                    name: count / stem_sample_count
                    for name, count in stem_load_threshold_counts.items()
                },
                "secure_support_sample_rate": secure_support_count / stem_sample_count,
                "mean_damage_when_securely_supported": (
                    secure_support_damage_sum / secure_support_count
                    if secure_support_count
                    else 0.0
                ),
                "mean_combined_load_when_securely_supported": (
                    secure_support_load_sum / secure_support_count
                    if secure_support_count
                    else 0.0
                ),
            }
            break_count = physics_v1_break_audit_sums.get(
                "physics_v1_break_count", 0.0
            )
            result["physics_v1_break_event_audit"] = {
                key.removeprefix("physics_v1_break_"): value
                for key, value in sorted(physics_v1_break_audit_sums.items())
            }
            if break_count:
                result["physics_v1_break_event_audit"].update(
                    {
                        "mean_supported_fingertips": physics_v1_break_audit_sums.get(
                            "physics_v1_break_supported_fingertip_sum", 0.0
                        )
                        / break_count,
                        "mean_distance_m": physics_v1_break_audit_sums.get(
                            "physics_v1_break_distance_sum_m", 0.0
                        )
                        / break_count,
                        "mean_relative_speed_m_s": physics_v1_break_audit_sums.get(
                            "physics_v1_break_relative_speed_sum_m_s", 0.0
                        )
                        / break_count,
                        "mean_damage": physics_v1_break_audit_sums.get(
                            "physics_v1_break_damage_sum", 0.0
                        )
                        / break_count,
                        "mean_load_ratio": physics_v1_break_audit_sums.get(
                            "physics_v1_break_load_ratio_sum", 0.0
                        )
                        / break_count,
                    }
                )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("TOMATO_CHECKPOINT_EVALUATION=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
