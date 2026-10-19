"""2026 RoTO 2.0 multimodal_rl PPO, unchanged, with one pressure track.

Robot: LinkAgent's vendor RIGHT L20 only. Task: unbounded cumulative fruit
spin without drop or concentrated load. Physics/contact/actuation are NOT
learned neural dynamics. No independently maintained PPO implementation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_roto2_training(env, cfg, agent_cfg, writer):
    """Minimal environment adapter/glue; delegate PPO entirely to 2026 library."""
    import copy
    import gymnasium as gym
    import numpy as np
    import torch

    from multimodal_rl.models.encoder import Encoder
    from multimodal_rl.models.running_standard_scaler import RunningStandardScaler
    from multimodal_rl.rl.memories import Memory
    from multimodal_rl.rl.policy_value import DeterministicValue, GaussianPolicy
    from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
    from multimodal_rl.rl.trainer import Trainer
    from multimodal_rl.wrappers.frame_stack import FrameStack
    from multimodal_rl.wrappers.isaaclab_wrapper import IsaacLabWrapper

    obs, _ = env.reset()
    gym_obs = {}
    for name, value in obs["policy"].items():
        gym_obs[name] = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(value.shape[-1] * cfg.obs_stack,), dtype=np.float32,
        )
    single_obs_space = gym.spaces.Dict({
        "policy": gym.spaces.Dict(gym_obs)
    })
    action_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(cfg.num_actions,), dtype=np.float32
    )
    env.unwrapped.set_spaces(
        single_obs_space,
        gym.vector.utils.batch_space(single_obs_space, cfg.scene.num_envs),
        action_space,
        gym.vector.utils.batch_space(action_space, cfg.scene.num_envs),
    )
    if cfg.obs_stack > 1:
        env = FrameStack(env, obs_stack=cfg.obs_stack)

    class TaskAwareWrapper(IsaacLabWrapper):
        """Keep custom object/sensor reset consistent across evaluation episodes."""
        def reset_eval_envs(self):
            # Upstream's generic scene.reset() omits our episode-specific
            # fruit repositioning, dropout and Jacobian counters.
            self._unwrapped._reset_idx(self.eval_env_ids)
            self._unwrapped.episode_length_buf[self.eval_env_ids] = 0
            new_obs = self._unwrapped.get_observations()
            if self.obs_stack > 1:
                for key, frame_buffer in self._env.frames.items():
                    for index in range(len(frame_buffer)):
                        frame = frame_buffer[index].clone()
                        frame[self.eval_env_ids] = new_obs["policy"][key][self.eval_env_ids]
                        frame_buffer[index] = frame
                new_obs = self._env.observation(new_obs)
            self._observations = new_obs
            return new_obs, self._info

    wrapped = TaskAwareWrapper(
        env, cfg.num_eval_envs, obs_stack=cfg.obs_stack, debug=cfg.debug
    )
    observation_space = wrapped.observation_space["policy"]
    encoder = Encoder(
        observation_space, wrapped.action_space, cfg, agent_cfg,
        device=wrapped.device,
    )
    policy = GaussianPolicy(
        z_dim=encoder.num_outputs,
        observation_space=observation_space,
        action_space=wrapped.action_space, device=wrapped.device,
        **agent_cfg["policy"],
    )
    value = DeterministicValue(
        z_dim=encoder.num_outputs,
        observation_space=observation_space,
        action_space=wrapped.action_space, device=wrapped.device,
        **agent_cfg["value"],
    )
    preprocessing = RunningStandardScaler(
        size=1, device=wrapped.device, dtype=torch.float32, debug=cfg.debug,
    )
    memory = Memory(
        memory_size=agent_cfg["agent"]["rollouts"],
        num_envs=wrapped.num_train_envs, device=wrapped.device,
        env_cfg=cfg,
    )
    agent_config = copy.deepcopy(PPO_DEFAULT_CONFIG)
    agent_config.update(agent_cfg["agent"])
    agent = PPO(
        encoder, policy, value, preprocessing,
        memory=memory, cfg=agent_config,
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        device=wrapped.device, writer=writer,
        ssl_task=None, dtype=torch.float32,
        debug=cfg.debug,
    )
    trainer = Trainer(
        env=wrapped, agents=agent, agent_cfg=agent_cfg,
        num_timesteps_M=agent_cfg["trainer"]["max_global_timesteps_M"],
        num_eval_envs=cfg.num_eval_envs, ssl_task=None, writer=writer,
    )
    return wrapped, trainer


def main(argv=None):
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=256)
    parser.add_argument("--million-steps", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pressure-variant", choices=("plain", "scalar", "jacobian"),
        default="jacobian",
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Run 20 actual PhysX steps before spending GPU time")
    parser.add_argument("--enable-wandb", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs < 2:
        raise ValueError("At least 2 environments required for train/eval split")
    if args.million_steps <= 0:
        raise ValueError("--million-steps must be positive")

    launcher = AppLauncher(args)
    application = launcher.app
    env = None
    try:
        import gymnasium as gym
        import torch
        import yaml

        from simulation.roto2_pressure.asset import ensure_single_hand_usd
        os.environ["LINKAGENT_ROTATION_USD"] = str(ensure_single_hand_usd())

        # 2026 library installed from an exact source SHA. We deliberately
        # avoid importing RoTO 2.x's original Allegro/Shadow environments,
        # whose robot topology and simulator release differ from LinkAgent.
        import multimodal_rl
        from multimodal_rl.tools.writer import Writer
        from simulation.roto2_pressure.env import FruitRotationEnv
        from simulation.roto2_pressure.env_cfg import FruitRotationEnvCfg

        task_id = "LinkAgent-Fruit-RoTO2-Direct-v0"
        if task_id not in gym.registry:
            gym.register(id=task_id, entry_point=FruitRotationEnv,
                         disable_env_checker=True)
        agent_cfg = yaml.safe_load(
            (ROOT / "simulation/roto2_pressure/agent.yaml").read_text()
        )
        agent_cfg["seed"] = args.seed
        agent_cfg["trainer"]["num_eval_envs"] = min(
            16, max(1, args.num_envs // 8)
        )
        agent_cfg["trainer"]["max_global_timesteps_M"] = args.million_steps
        agent_cfg["experiment"]["wandb"] = bool(args.enable_wandb)
        agent_cfg["experiment"]["experiment_name"] = (
            f"{args.pressure_variant}_seed{args.seed}"
        )
        agent_cfg["log_path"] = str(ROOT)
        cfg = FruitRotationEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed = args.seed
        cfg.pressure_variant = args.pressure_variant
        cfg.num_eval_envs = agent_cfg["trainer"]["num_eval_envs"]
        cfg.obs_list = tuple(agent_cfg["observations"]["obs_list"])
        cfg.obs_stack = agent_cfg["observations"]["obs_stack"]

        env = gym.make(task_id, cfg=cfg)
        if args.smoke:
            policy, _ = env.reset()
            for t in range(20):
                policy, reward, failed, timeout, info = env.step(
                    torch.zeros((args.num_envs, 16), device=env.unwrapped.device)
                )
                inputs = policy["policy"]
                if inputs["tactile"].shape != (args.num_envs, 432):
                    raise RuntimeError("Tactile sensor array shape drift")
                if inputs["prop"].shape != (args.num_envs, 70):
                    raise RuntimeError("Proprioception shape drift")
                if not (torch.isfinite(inputs["prop"]).all()
                        and torch.isfinite(inputs["tactile"]).all()
                        and torch.isfinite(reward).all()):
                    raise RuntimeError(f"Nonfinite physical rollout at {t}")
            print(json.dumps({
                "physx_smoke_passed": True, "steps": 20,
                "num_envs": args.num_envs, "actor_inputs": [70, 432],
                "pressure_variant": args.pressure_variant,
            }, indent=2))
            return 0
        writer = Writer(agent_cfg)
        wrapped, trainer = build_roto2_training(env, cfg, agent_cfg, writer)
        env = wrapped
        trainer.train()
        return 0
    finally:
        if env is not None:
            env.close()
        application.close()


if __name__ == "__main__":
    raise SystemExit(main())
