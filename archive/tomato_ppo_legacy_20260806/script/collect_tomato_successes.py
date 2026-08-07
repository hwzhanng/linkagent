"""Collect successful PPO trajectories for success-filtered actor warm-starting."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.skills.tomato_pick.contracts import ACTION_DIM, OBSERVATION_DIM


def _parser() -> argparse.ArgumentParser:
    app_launcher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--max-success-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-deterministic", action="store_true")
    app_launcher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    app_launcher = importlib.import_module("isaaclab.app").AppLauncher
    args = _parser().parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.num_envs <= 0 or args.steps <= 0 or args.max_success_episodes <= 0:
        raise ValueError("environment, step, and success limits must be positive")

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
        cfg.enable_sim2real_randomization = True
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
        policy = runner.get_inference_policy(device=wrapped.device)
        raw = wrapped.unwrapped
        observations = wrapped.get_observations()

        # The PPO environment has one global horizon.  Progress labels are not
        # phase deadlines and must not size the trajectory buffer.
        maximum_episode_steps = int(raw.max_episode_length)
        episode_observations = torch.zeros(
            (args.num_envs, maximum_episode_steps, OBSERVATION_DIM),
            device=wrapped.device,
        )
        episode_actions = torch.zeros(
            (args.num_envs, maximum_episode_steps, ACTION_DIM),
            device=wrapped.device,
        )
        episode_phases = torch.zeros(
            (args.num_envs, maximum_episode_steps),
            dtype=torch.int8,
            device=wrapped.device,
        )
        episode_lengths = torch.zeros(
            args.num_envs, dtype=torch.long, device=wrapped.device
        )
        collected_observations = []
        collected_actions = []
        collected_phases = []
        episode_sizes: list[int] = []
        success_events = 0
        drop_events = 0
        completed_episodes = 0
        executed_steps = 0

        with torch.inference_mode():
            for step in range(args.steps):
                active = ~raw._reset_pending
                actions = (
                    policy(observations)
                    if args.policy_deterministic
                    else runner.alg.actor(observations, stochastic_output=True)
                )
                active_ids = active.nonzero(as_tuple=False).squeeze(1)
                if active_ids.numel():
                    positions = episode_lengths[active_ids]
                    if bool((positions >= maximum_episode_steps).any()):
                        raise RuntimeError("active episode exceeded configured phase limits")
                    episode_observations[active_ids, positions] = observations["policy"][
                        active_ids
                    ]
                    episode_actions[active_ids, positions] = actions[active_ids]
                    episode_phases[active_ids, positions] = raw._phase[active_ids].to(
                        torch.int8
                    )
                    episode_lengths[active_ids] += 1

                observations, _, dones, _ = wrapped.step(actions)
                executed_steps = step + 1
                success = raw._last_reward_terms["success"] != 0.0
                drop = raw._last_reward_terms["drop"] != 0.0
                success_events += int(success.sum())
                drop_events += int(drop.sum())
                completed_episodes += int(dones.sum())
                success_ids = (dones & success).nonzero(as_tuple=False).squeeze(1)
                for env_id in success_ids.detach().cpu().tolist():
                    length = int(episode_lengths[env_id])
                    if length <= 0:
                        raise RuntimeError("successful episode contains no active transitions")
                    collected_observations.append(
                        episode_observations[env_id, :length].detach().cpu().clone()
                    )
                    collected_actions.append(
                        episode_actions[env_id, :length].detach().cpu().clone()
                    )
                    collected_phases.append(
                        episode_phases[env_id, :length].detach().cpu().clone()
                    )
                    episode_sizes.append(length)
                    if len(episode_sizes) >= args.max_success_episodes:
                        break
                done_ids = dones.nonzero(as_tuple=False).squeeze(1)
                if done_ids.numel():
                    episode_lengths[done_ids] = 0
                if len(episode_sizes) >= args.max_success_episodes:
                    break

        if not episode_sizes:
            raise RuntimeError("no successful trajectories were collected")
        observations_tensor = torch.cat(collected_observations)
        actions_tensor = torch.cat(collected_actions)
        phases_tensor = torch.cat(collected_phases).long()
        if not bool(torch.isfinite(observations_tensor).all()):
            raise RuntimeError("collected observations contain NaN or Inf")
        if not bool(torch.isfinite(actions_tensor).all()):
            raise RuntimeError("collected actions contain NaN or Inf")
        if not bool(((actions_tensor >= -1.0) & (actions_tensor <= 1.0)).all()):
            raise RuntimeError("collected actions exceed the policy bounds")

        offsets = [0]
        for length in episode_sizes:
            offsets.append(offsets[-1] + length)
        checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        dataset = {
            "schema_version": 1,
            "observations": observations_tensor,
            "actions": actions_tensor,
            "phases": phases_tensor,
            "episode_offsets": torch.tensor(offsets, dtype=torch.long),
            "metadata": {
                "source_checkpoint": str(args.checkpoint.resolve()),
                "source_checkpoint_sha256": checkpoint_sha256,
                "seed": args.seed,
                "sim2real_mode": "full",
                "deterministic": args.policy_deterministic,
                "num_envs": args.num_envs,
                "executed_steps": executed_steps,
                "completed_episodes": completed_episodes,
                "observed_success_events": success_events,
                "observed_drop_events": drop_events,
                "success_episodes_saved": len(episode_sizes),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, args.output)
        phase_counts = torch.bincount(phases_tensor, minlength=4).tolist()
        report = {
            **dataset["metadata"],
            "output": str(args.output.resolve()),
            "transitions_saved": int(len(phases_tensor)),
            "episode_length": {
                "min": min(episode_sizes),
                "max": max(episode_sizes),
                "mean": sum(episode_sizes) / len(episode_sizes),
            },
            "phase_transition_counts": phase_counts,
        }
        print("TOMATO_SUCCESS_DATASET=" + json.dumps(report, sort_keys=True), flush=True)
        return 0
    finally:
        if wrapped is not None:
            wrapped.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
