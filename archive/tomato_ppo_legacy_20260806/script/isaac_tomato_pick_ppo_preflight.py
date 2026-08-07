"""Construct and audit the PPO stack without collecting a training rollout."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import sys
import traceback
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be > 0")

    launcher = AppLauncher(args)
    app = launcher.app
    wrapped = None
    try:
        gym = importlib.import_module("gymnasium")
        torch = importlib.import_module("torch")
        distributions = importlib.import_module("torch.distributions")
        rsl_runner = importlib.import_module("rsl_rl.runners.on_policy_runner")
        rsl_wrapper = importlib.import_module("isaaclab_rl.rsl_rl")
        task = importlib.import_module("simulation.tomato_pick.isaac")
        env_cfg_module = importlib.import_module("simulation.tomato_pick.isaac.env_cfg")
        ppo_cfg_module = importlib.import_module(
            "simulation.tomato_pick.isaac.agents.rsl_rl_ppo_cfg"
        )
        profile_module = importlib.import_module("simulation.tomato_pick.config")

        env_cfg = env_cfg_module.TomatoPickEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device
        env_cfg.seed = args.seed
        raw_env = gym.make(task.TASK_ID, cfg=env_cfg)

        agent_cfg = ppo_cfg_module.TomatoPickPPORunnerCfg()
        agent_cfg.device = args.device
        profile = profile_module.load_ppo_profile()
        trainer_profile = profile["trainer"]
        policy_profile = profile["policy"]
        algorithm_profile = profile["algorithm"]
        # Use the exact compatibility migration used by Isaac Lab's official
        # train.py entry point.  Calling OnPolicyRunner with the raw config
        # would leak deprecated ``stochastic`` fields into RSL-RL 5.x.
        agent_cfg = rsl_wrapper.handle_deprecated_rsl_rl_cfg(
            agent_cfg, metadata.version("rsl-rl-lib")
        )
        wrapped = rsl_wrapper.RslRlVecEnvWrapper(
            raw_env, clip_actions=agent_cfg.clip_actions
        )
        runner = rsl_runner.OnPolicyRunner(
            wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        algorithm = runner.alg
        observations = wrapped.get_observations()

        checks: dict[str, bool] = {
            "ppo_not_started": True,
            "runner_iteration_zero": runner.current_learning_iteration == 0,
            "storage_empty_before_preflight": algorithm.storage.step == 0,
            "optimizer_has_no_state": len(algorithm.optimizer.state) == 0,
            "actor_reads_policy_only": algorithm.actor.obs_groups == ["policy"],
            "critic_reads_privileged_only": algorithm.critic.obs_groups == ["critic"],
            "wrapper_does_not_clip_actions": wrapped.clip_actions is None,
            "fresh_on_policy_rollouts_without_replay": (
                trainer_profile["rollout_source"] == "fresh_current_policy_only"
                and trainer_profile["replay_buffer_enabled"] is False
            ),
            "runner_profile_matches_executable_config": (
                agent_cfg.num_steps_per_env
                == trainer_profile["rollout_steps_per_environment"]
                and agent_cfg.max_iterations == trainer_profile["maximum_iterations"]
                and agent_cfg.save_interval == trainer_profile["save_interval_iterations"]
                and agent_cfg.actor.hidden_dims == policy_profile["hidden_dimensions"]
                and agent_cfg.actor.activation == policy_profile["activation"]
                and agent_cfg.actor.distribution_cfg.init_std
                == policy_profile["initial_action_std"]
                and agent_cfg.actor.distribution_cfg.min_std
                == policy_profile["minimum_action_std"]
                and agent_cfg.actor.distribution_cfg.max_std
                == policy_profile["maximum_action_std"]
                and agent_cfg.algorithm.learning_rate == algorithm_profile["learning_rate"]
                and agent_cfg.algorithm.entropy_coef
                == algorithm_profile["entropy_coefficient"]
                and agent_cfg.algorithm.schedule == algorithm_profile["schedule"]
                and agent_cfg.algorithm.gamma == algorithm_profile["gamma"]
                and agent_cfg.algorithm.lam == algorithm_profile["gae_lambda"]
                and agent_cfg.algorithm.clip_param == algorithm_profile["clip_parameter"]
                and agent_cfg.algorithm.desired_kl == algorithm_profile["desired_kl"]
                and agent_cfg.algorithm.max_grad_norm
                == algorithm_profile["maximum_gradient_norm"]
            ),
            "actor_and_critic_observation_normalization_enabled": (
                agent_cfg.actor.obs_normalization and agent_cfg.critic.obs_normalization
            ),
            "advantage_normalization_is_rollout_wide": (
                not agent_cfg.algorithm.normalize_advantage_per_mini_batch
                and algorithm_profile["advantage_normalization"]
                == "once_over_complete_rollout"
            ),
            "profile_declares_correct_bootstrap_semantics": (
                algorithm_profile["timeout_value_bootstrap"] is True
                and algorithm_profile["true_termination_value_bootstrap"] is False
            ),
            "actor_observation_shape_matches_profile": tuple(
                observations["policy"].shape
            )
            == (args.num_envs, policy_profile["actor_observation_dim"]),
            "critic_observation_shape_matches_environment": tuple(
                observations["critic"].shape
            )
            == (args.num_envs, env_cfg.state_space),
            "all_observations_finite": all(
                bool(torch.isfinite(value).all()) for value in observations.values()
            ),
        }

        parameter_snapshot = {
            name: value.detach().clone()
            for name, value in algorithm.actor.named_parameters()
        }
        actions = algorithm.act(observations)
        saved_log_prob = algorithm.transition.actions_log_prob.clone()
        saved_values = algorithm.transition.values.clone()
        distribution = algorithm.actor.distribution
        custom_log_prob = algorithm.actor.get_output_log_prob(actions)
        latent_mean, latent_std = algorithm.actor.output_distribution_params
        # All continuous dimensions are executed throughout the episode.
        # Rebuild the same joint tanh-Gaussian density explicitly so the
        # preflight catches any mismatch between sampled actions and PPO's
        # stored log-probability.
        bounded_actions = actions.clamp(
            -1.0 + distribution.epsilon, 1.0 - distribution.epsilon
        )
        latent_actions = torch.atanh(bounded_actions)
        per_dimension_reference = distributions.Normal(
            latent_mean, latent_std
        ).log_prob(latent_actions) - distribution._log_tanh_jacobian(latent_actions)
        reference_log_prob = per_dimension_reference.sum(dim=-1)
        deterministic_actions = algorithm.actor(observations)
        self_kl = algorithm.actor.get_kl_divergence(
            algorithm.actor.output_distribution_params,
            algorithm.actor.output_distribution_params,
        )
        ratio = torch.exp(custom_log_prob - saved_log_prob)

        checks.update(
            {
                "sampled_actions_finite": bool(torch.isfinite(actions).all()),
                "sampled_actions_strictly_bounded": bool(
                    (actions > -1.0).all() and (actions < 1.0).all()
                ),
                "deterministic_actions_bounded": bool(
                    (deterministic_actions >= -1.0).all()
                    and (deterministic_actions <= 1.0).all()
                ),
                "joint_log_prob_is_one_scalar_per_env": tuple(custom_log_prob.shape)
                == (args.num_envs,),
                "log_prob_finite": bool(torch.isfinite(custom_log_prob).all()),
                "tanh_jacobian_matches_torch_reference": bool(
                    torch.allclose(custom_log_prob, reference_log_prob, atol=2.0e-5, rtol=2.0e-5)
                ),
                "saved_old_log_prob_detached": not saved_log_prob.requires_grad,
                "saved_distribution_params_detached": all(
                    not value.requires_grad
                    for value in algorithm.transition.distribution_params
                ),
                "unchanged_policy_ratio_is_one": bool(
                    torch.allclose(ratio, torch.ones_like(ratio), atol=1.0e-6)
                ),
                "self_kl_is_zero": bool(
                    torch.allclose(self_kl, torch.zeros_like(self_kl), atol=1.0e-7)
                ),
                "std_is_finite_and_bounded": bool(
                    torch.isfinite(distribution.std).all()
                    and (distribution.std >= distribution.min_std).all()
                    and (distribution.std <= distribution.max_std).all()
                ),
                "value_predictions_finite": bool(torch.isfinite(saved_values).all()),
            }
        )

        # Validate timeout bootstrapping independently of environment timing.
        zeros = torch.zeros(args.num_envs, device=algorithm.device)
        dones = torch.ones(args.num_envs, dtype=torch.long, device=algorithm.device)
        algorithm.process_env_step(
            observations,
            zeros,
            dones,
            {"time_outs": torch.ones(args.num_envs, dtype=torch.bool, device=algorithm.device)},
        )
        expected_timeout_reward = algorithm.gamma * saved_values.squeeze(1)
        checks["truncation_bootstraps_value"] = bool(
            torch.allclose(
                algorithm.storage.rewards[0, :, 0], expected_timeout_reward, atol=1.0e-6
            )
        )
        algorithm.storage.clear()

        algorithm.act(observations)
        algorithm.process_env_step(
            observations,
            zeros,
            dones,
            {"time_outs": torch.zeros(args.num_envs, dtype=torch.bool, device=algorithm.device)},
        )
        checks["true_termination_does_not_bootstrap"] = bool(
            torch.count_nonzero(algorithm.storage.rewards[0]).item() == 0
        )
        algorithm.storage.clear()

        # Exercise RSL-RL's own backward GAE implementation on a deterministic
        # synthetic rollout, then restore the ephemeral critic parameters.
        critic_state = {
            name: value.detach().clone()
            for name, value in algorithm.critic.state_dict().items()
        }
        with torch.no_grad():
            for parameter in algorithm.critic.parameters():
                parameter.zero_()
            storage = algorithm.storage
            storage.rewards.fill_(1.0)
            storage.values.zero_()
            storage.dones.zero_()
            boundary_step = min(10, storage.num_transitions_per_env - 2)
            storage.dones[boundary_step] = 1
            algorithm.compute_returns(observations)
            expected_returns = torch.zeros_like(storage.returns)
            advantage = torch.zeros(
                (args.num_envs, 1), device=algorithm.device
            )
            for step in reversed(range(storage.num_transitions_per_env)):
                mask = 1.0 - storage.dones[step].float()
                advantage = 1.0 + mask * algorithm.gamma * algorithm.lam * advantage
                expected_returns[step] = advantage
            checks["gae_backward_recursion_matches_reference"] = bool(
                torch.allclose(storage.returns, expected_returns, atol=1.0e-6)
            )
            checks["episode_boundary_does_not_leak_advantage"] = bool(
                torch.allclose(
                    storage.returns[boundary_step],
                    torch.ones_like(storage.returns[boundary_step]),
                )
            )
        algorithm.critic.load_state_dict(critic_state)
        algorithm.storage.clear()

        checks["actor_parameters_unchanged"] = all(
            torch.equal(parameter_snapshot[name], value.detach())
            for name, value in algorithm.actor.named_parameters()
        )
        checks["optimizer_still_has_no_state"] = len(algorithm.optimizer.state) == 0
        checks["storage_empty_after_preflight"] = algorithm.storage.step == 0

        report = {
            "schema_version": 1,
            "passed": all(checks.values()),
            "task_id": task.TASK_ID,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "device": str(algorithm.device),
            "checks": checks,
            "distribution": type(distribution).__name__,
            "initial_std": float(distribution.std[0, 0].item()),
            "max_abs_sampled_action": float(actions.abs().max().item()),
            "ppo_started": False,
            "optimizer_steps": 0,
            "rollout_steps_collected": 0,
        }
        print("ISAAC_TOMATO_PPO_PREFLIGHT=" + json.dumps(report, sort_keys=True), flush=True)
        return 0 if report["passed"] else 2
    except BaseException as exc:
        print(
            "ISAAC_TOMATO_PPO_PREFLIGHT_EXCEPTION="
            + json.dumps({"type": type(exc).__name__, "message": str(exc)}),
            flush=True,
        )
        traceback.print_exc()
        return 3
    finally:
        if wrapped is not None:
            wrapped.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
