"""Run the 2026 RoTO 2.0 / multimodal_rl PPO UNMODIFIED on one L20 fruit task.

Prerequisites are frozen by script/install_roto2_pressure.sh. This launcher
does not import or use the former locally implemented PPO classes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    # Imports of Isaac Lab must happen only after AppLauncher begins.
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=256)
    parser.add_argument("--million-steps", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pressure-variant", choices=("plain", "scalar", "jacobian"),
                        default="jacobian")
    parser.add_argument("--smoke", action="store_true",
                        help="Run only 20 actual PhysX policy steps, without training")
    parser.add_argument("--enable-wandb", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs < 2:
        raise ValueError("At least 2 environments are required for train/eval split")
    if args.million_steps <= 0:
        raise ValueError("--million-steps must be positive")
    if sys.platform != "linux":
        raise RuntimeError("Isaac Lab CUDA/PhysX training requires Linux with NVIDIA GPU")

    launcher = AppLauncher(args)
    application = launcher.app
    env = None
    try:
        import gymnasium as gym
        import torch
        import yaml

        # The vendor right hand is taken strictly from LinkAgent, not Allegro.
        from simulation.roto2_pressure.asset import ensure_single_hand_usd
        os.environ["LINKAGENT_ROTATION_USD"] = str(ensure_single_hand_usd())

        import roto
        import multimodal_rl
        roto_scripts = Path(roto.__file__).resolve().parents[1] / "scripts"
        if not (roto_scripts / "common_utils.py").exists():
            raise RuntimeError(
                "RoTO source checkout not installed editable. "
                "Run script/install_roto2_pressure.sh"
            )
        sys.path.insert(0, str(roto_scripts))
        from common_utils import make_env, train_one_seed
        from multimodal_rl.tools.writer import Writer
        from simulation.roto2_pressure.env import FruitRotationEnv
        from simulation.roto2_pressure.env_cfg import FruitRotationEnvCfg

        task_id = "LinkAgent-Fruit-RoTO2-Direct-v0"
        if task_id not in gym.registry:
            gym.register(
                id=task_id, entry_point=FruitRotationEnv,
                disable_env_checker=True,
            )
        cfg_path = REPO / "simulation/roto2_pressure/agent.yaml"
        agent_cfg = yaml.safe_load(cfg_path.read_text())
        agent_cfg["seed"] = args.seed
        agent_cfg["trainer"]["num_eval_envs"] = min(16, max(1, args.num_envs // 8))
        agent_cfg["trainer"]["max_global_timesteps_M"] = args.million_steps
        agent_cfg["experiment"]["wandb"] = bool(args.enable_wandb)
        agent_cfg["experiment"]["experiment_name"] = (
            f"{args.pressure_variant}_seed{args.seed}"
        )
        agent_cfg["log_path"] = str(REPO)
        cfg = FruitRotationEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed = args.seed
        cfg.pressure_variant = args.pressure_variant
        cfg.num_eval_envs = agent_cfg["trainer"]["num_eval_envs"]
        cfg.obs_list = tuple(agent_cfg["observations"]["obs_list"])
        cfg.obs_stack = agent_cfg["observations"]["obs_stack"]

        # Smoke test checks real CUDA PhysX and actual tactile/pressure data.
        # Unlike the upstream PPO, this path needs no external writer.
        if args.smoke:
            env = FruitRotationEnv(cfg)
            observations, _ = env.reset()
            for t in range(20):
                observations, reward, failed, timeout, info = env.step(
                    torch.zeros((args.num_envs, 16), device=env.device)
                )
                policy = observations["policy"]
                if set(policy) != {"prop", "tactile"}:
                    raise RuntimeError(f"Policy modalities mismatch: {list(policy)}")
                if policy["tactile"].shape != (args.num_envs, 432):
                    raise RuntimeError("Tactile tensor shape mismatch")
                if policy["prop"].shape != (args.num_envs, 70):
                    raise RuntimeError("Proprioception tensor shape mismatch")
                if not (torch.isfinite(policy["prop"]).all()
                        and torch.isfinite(policy["tactile"]).all()
                        and torch.isfinite(reward).all()):
                    raise RuntimeError(f"Nonfinite physics observation/reward at step {t}")
            print(json.dumps({
                "physx_smoke_passed": True, "steps": 20, "num_envs": args.num_envs,
                "modalities": {"prop": 70, "tactile": 432},
                "pressure_variant": args.pressure_variant,
                "release_basis": "RoTO2 2026 original multimodal_rl PPO",
            }, indent=2))
            return 0

        # Keep all RoTO2 trainer/agent/encoder/GAE/optimizer/checkpoint code
        # unchanged. Only the robot/task data and tiny pressure Jacobian differ.
        from types import SimpleNamespace
        writer = Writer(agent_cfg)
        facade = SimpleNamespace(
            num_envs=args.num_envs, device=args.device,
            video=False, video_length=0,
            gym_env_id=task_id, task=task_id,
        )
        env = make_env(agent_cfg, cfg, writer, facade)
        train_one_seed(
            facade, env, agent_cfg=agent_cfg, env_cfg=cfg,
            writer=writer, seed=args.seed
        )
        return 0
    finally:
        if env is not None:
            env.close()
        application.close()


if __name__ == "__main__":
    raise SystemExit(main())
