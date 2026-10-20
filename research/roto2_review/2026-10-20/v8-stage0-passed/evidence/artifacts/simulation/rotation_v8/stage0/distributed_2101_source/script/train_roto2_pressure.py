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
import traceback

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
        def step(self, actions):
            obs, reward, terminated, truncated, info = super().step(actions)
            done = (terminated | truncated).reshape(-1)
            if self.obs_stack > 1 and done.any():
                for frames in self._env.frames.values():
                    latest = frames[-1]
                    for index in range(len(frames) - 1):
                        frame = frames[index].clone()
                        frame[done] = latest[done]
                        frames[index] = frame
                obs = self._env.observation(obs)
                self._observations = obs
            return obs, reward, terminated, truncated, info

        def reset_eval_envs(self):
            # Upstream's generic scene.reset() omits our episode-specific
            # fruit repositioning, dropout and Jacobian counters.
            self._unwrapped._reset_idx(self.eval_env_ids)
            self._unwrapped.episode_length_buf[self.eval_env_ids] = 0
            new_obs = self._unwrapped.get_observations(env_ids=self.eval_env_ids)
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
    parser.add_argument("--initial-action-std", type=float, default=None,
                        help="Fresh-run Gaussian exploration std; network/PPO are unchanged")
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None,
                        help="Override rollout length without changing upstream PPO")
    parser.add_argument("--domain-profile", choices=("full", "gentle", "nominal"), default="gentle")
    parser.add_argument("--hand-posture", choices=("vertical", "palm_up"), default="vertical")
    parser.add_argument("--reward-stage", choices=("contact", "rotation", "regrasp", "spin", "continuous", "baseline"), default="rotation")
    parser.add_argument("--goal-direction", choices=("both", "positive", "negative"), default="both")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 20 actual PhysX steps before spending GPU time")
    parser.add_argument("--enable-wandb", action="store_true")
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--probe-output", type=Path, default=None)
    parser.add_argument("--reset-grasps", type=Path, default=None)
    parser.add_argument("--validate-contracts", action="store_true")
    parser.add_argument("--probe-nominal", action="store_true",
                        help="Diagnostic only: 64 mm sphere and deterministic reset")
    parser.add_argument("--search-pad-reset", action="store_true")
    parser.add_argument("--search-redundant-reset", action="store_true",
                        help="Stage-0 multi-contact grasp proposal search; no PPO")
    parser.add_argument("--validate-redundant-reset", action="store_true",
                        help="Stage-0 physically verified whole-finger N-1 release tests; no PPO")
    parser.add_argument("--calibration-rounds", type=int, default=32)
    parser.add_argument("--calibration-horizon", type=int, default=80)
    parser.add_argument("--calibration-spread", type=float, default=.12)
    parser.add_argument("--calibration-release-mode", choices=("full_open", "adaptive"), default="full_open")
    parser.add_argument("--calibration-region", choices=("broad", "palm"), default="broad")
    parser.add_argument("--calibration-replicas", type=int, default=1)
    parser.add_argument("--calibration-replica-layout", choices=("adjacent", "distributed"),
                        default="distributed")
    parser.add_argument("--probe-reset-replay", action="store_true",
                        help="Diagnostic matched reset-state ablation; no PPO")
    parser.add_argument("--calibration-video", type=Path, default=None)
    parser.add_argument("--calibration-density", type=float, default=950.0,
                        help="Matched N-1 conditions use one density in the existing 850-1050 range")
    parser.add_argument("--probe-pad-surfaces", action="store_true")
    parser.add_argument("--probe-free-hand", action="store_true")
    parser.add_argument("--probe-no-self-collision", action="store_true")
    parser.add_argument("--evaluate-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None,
                        help="Restore model/optimizers; --million-steps is an additional budget")
    parser.add_argument("--evaluation-output", type=Path, default=None)
    parser.add_argument("--record-video", type=Path, default=None,
                        help="Render an actual first-episode rollout of --evaluate-checkpoint")
    parser.add_argument("--video-env-index", type=int, default=0)
    parser.add_argument("--video-max-steps", type=int, default=480)
    parser.add_argument("--audit-rotation-output", type=Path, default=None)
    parser.add_argument("--probe-rotation-output", type=Path, default=None)
    parser.add_argument("--trace-policy-output", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.record_video:
        if args.evaluate_checkpoint is None:
            parser.error("--record-video requires --evaluate-checkpoint")
        args.enable_cameras = True
    if args.calibration_video:
        if not args.validate_redundant_reset:
            parser.error("--calibration-video requires --validate-redundant-reset")
        args.enable_cameras = True
    if args.initial_action_std is not None and (args.initial_action_std <= 0 or args.resume_checkpoint):
        parser.error("--initial-action-std must be positive and is for fresh runs only")
    if args.entropy_weight is not None and args.entropy_weight < 0:
        parser.error("--entropy-weight must be non-negative")
    if args.rollout_steps is not None and args.rollout_steps < 2:
        parser.error("--rollout-steps must be >=2")
    if args.num_envs < 2:
        raise ValueError("At least 2 environments required for train/eval split")
    if args.million_steps <= 0:
        raise ValueError("--million-steps must be positive")

    launcher = AppLauncher(args)
    application = launcher.app
    env = None
    exit_code = 0
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
        from simulation.roto2_pressure.reward_geometry import reward_version
        REWARD_VERSION = reward_version(args.reward_stage)

        task_id = "LinkAgent-Fruit-RoTO2-Direct-v0"
        if task_id not in gym.registry:
            gym.register(id=task_id, entry_point=FruitRotationEnv,
                         disable_env_checker=True)
        agent_cfg = yaml.safe_load(
            (ROOT / "simulation/roto2_pressure/agent.yaml").read_text()
        )
        if args.initial_action_std is not None:
            import math
            agent_cfg["policy"]["initial_log_std"] = math.log(args.initial_action_std)
        if args.entropy_weight is not None:
            agent_cfg["agent"]["entropy_loss_scale"] = args.entropy_weight
        if args.rollout_steps is not None:
            agent_cfg["agent"]["rollouts"] = args.rollout_steps
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
        cfg.reward_stage = args.reward_stage
        cfg.goal_direction = args.goal_direction
        cfg.diagnostic_all_finger_contacts = args.validate_redundant_reset or args.probe_reset_replay
        if cfg.diagnostic_all_finger_contacts:
            cfg.scene.lazy_sensor_update = False
        if args.reward_stage in ("regrasp", "continuous"):
            cfg.target_angular_speed_rad_s = (.15, .30)
        elif args.reward_stage == "baseline":
            cfg.target_angular_speed_rad_s = (.30, .50)
            cfg.randomize_reset_grasp = True
        cfg.domain_profile = args.domain_profile
        cfg.hand_posture = args.hand_posture
        if args.hand_posture == "palm_up":
            # Vendor palm faces local +X. -90 degrees about Y puts it up,
            # with fingers extending horizontally toward world -X.
            # Existing reset code transforms fruit position from the hand frame.
            import math
            cfg.robot.init_state.pos = (0.0, 0.0, 0.90)
            # Isaac Lab 3's InitialStateCfg uses XYZW, unlike older releases.
            cfg.robot.init_state.rot = (0.0, -math.sqrt(.5), 0.0, math.sqrt(.5))
            cfg.viewer.eye = (.18, -.36, 1.32)
            cfg.viewer.lookat = (-.12, 0.0, .96)
        if args.domain_profile in ("nominal", "gentle"):
            radii = (.032, .032) if args.domain_profile == "nominal" else (.0315, .0325)
            cfg.events.geometry.params["axis_ranges_m"] = {k: radii for k in "xyz"}
            cfg.reset_joint_noise_rad = 0.0 if args.domain_profile == "nominal" else .004
            cfg.reset_position_noise_m = 0.0 if args.domain_profile == "nominal" else .00025
        if args.reset_grasps:
            cfg.reset_grasps_file = str(args.reset_grasps.resolve())
        if args.validate_redundant_reset or args.probe_reset_replay:
            if not 850 <= args.calibration_density <= 1050:
                raise ValueError("Calibration density must remain inside task randomization range")
            cfg.events.geometry.params["density_range_kg_m3"] = (
                args.calibration_density, args.calibration_density)
        if args.search_redundant_reset or args.validate_redundant_reset or args.probe_reset_replay:
            cfg.randomize_reset_grasp = False
        if args.probe_nominal:
            cfg.domain_profile = "nominal"
            if not args.probe_steps:
                raise ValueError("--probe-nominal requires --probe-steps")
            cfg.events.geometry.params["axis_ranges_m"] = {k: (0.032, 0.032) for k in "xyz"}
            cfg.reset_joint_noise_rad = 0.0
            cfg.reset_position_noise_m = 0.0
        if args.probe_no_self_collision:
            if not args.probe_steps:
                raise ValueError("Self-collision diagnostic requires --probe-steps")
            cfg.robot.spawn.articulation_props.enabled_self_collisions = False
        cfg.num_eval_envs = agent_cfg["trainer"]["num_eval_envs"]
        cfg.obs_list = tuple(agent_cfg["observations"]["obs_list"])
        cfg.obs_stack = agent_cfg["observations"]["obs_stack"]

        env = gym.make(task_id, cfg=cfg, render_mode="rgb_array" if (args.record_video or args.calibration_video) else None)
        if args.search_redundant_reset or args.validate_redundant_reset or args.probe_reset_replay:
            if args.pressure_variant != "plain" or args.hand_posture != "palm_up":
                raise ValueError("Stage-0 calibration requires plain + palm_up")
            if args.probe_output is None:
                raise ValueError("Stage-0 calibration requires --probe-output")
            from simulation.roto2_pressure.calibrate_redundancy import search_redundant_resets, validate_n_minus_one
            if args.probe_reset_replay:
                from simulation.roto2_pressure.probe_reset_replay import probe_reset_replay
                probe_reset_replay(env, args.probe_output, args.calibration_horizon)
            elif args.search_redundant_reset:
                search_redundant_resets(env, args.probe_output, args.calibration_rounds,
                                       args.calibration_horizon, args.calibration_spread,
                                       args.calibration_region, args.calibration_replicas,
                                       args.calibration_replica_layout)
            else:
                validate_n_minus_one(env, args.probe_output, args.calibration_release_mode,
                                     args.calibration_video, args.video_env_index)
            return 0
        if args.search_pad_reset:
            from simulation.roto2_pressure.calibrate_reset import search_pad_resets
            search_pad_resets(env, args.probe_output)
            return 0
        if args.probe_pad_surfaces:
            from simulation.roto2_pressure.validation import pad_surface_probe
            pad_surface_probe(env, args.probe_output)
            return 0
        if args.probe_free_hand:
            from simulation.roto2_pressure.validation import free_hand_probe
            free_hand_probe(env, args.probe_output)
            return 0
        if args.probe_steps:
            from simulation.roto2_pressure.validation import physical_probe
            physical_probe(env, args.probe_steps, args.probe_output)
            return 0
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
        class LocalWriter(Writer):
            """Keep an atomic current-model checkpoint independent of eval selection."""
            def write_checkpoint(self, mean_eval_return, timestep):
                # The upstream loop evaluates at t=0 before a full episode.
                # Exclude that warm-up score from best-checkpoint selection.
                if timestep == 0:
                    return
                super().write_checkpoint(mean_eval_return, timestep)
                destination = Path(self.checkpoint_dir) / "last_agent.pt"
                temporary = destination.with_suffix(".pt.tmp")
                torch.save({name: self._get_module_state(module)
                            for name, module in self.checkpoint_modules.items()}, temporary)
                temporary.replace(destination)
                transitions = timestep + args.num_envs - cfg.num_eval_envs
                (Path(self.log_dir) / "progress.json").write_text(json.dumps({
                    "training_completed": False, "training_transitions": transitions,
                    "cumulative_training_transitions": self.prior_transitions + transitions,
                    "checkpoint": str(destination),
                    "upstream_eval_return": float(mean_eval_return),
                }, indent=2) + "\n")

        writer = LocalWriter(agent_cfg)
        writer.prior_transitions = 0
        if args.resume_checkpoint:
            source_run = args.resume_checkpoint.resolve().parent.parent
            metadata = source_run / ("training_result.json" if args.resume_checkpoint.name == "final_agent.pt"
                                     else "progress.json")
            if metadata.is_file():
                prior = json.loads(metadata.read_text())
                writer.prior_transitions = prior.get("cumulative_training_transitions", prior["training_transitions"])
        import hashlib
        import shutil
        snapshot = Path(writer.log_dir) / "source_snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        for source in (ROOT / "simulation/roto2_pressure").glob("*"):
            if source.is_file():
                shutil.copy2(source, snapshot / source.name)
        shutil.copy2(Path(__file__), snapshot / Path(__file__).name)
        reset_data = Path(cfg.reset_grasps_file).read_bytes() if cfg.reset_grasps_file else b""
        (Path(writer.log_dir) / "run_config.json").write_text(json.dumps({
            "seed": args.seed, "num_envs": args.num_envs, "num_eval_envs": cfg.num_eval_envs,
            "million_steps": args.million_steps, "pressure_variant": args.pressure_variant,
            "physics_dt": cfg.sim.dt, "decimation": cfg.decimation,
            "reward_version": REWARD_VERSION, "reward_stage": cfg.reward_stage,
            "goal_direction": cfg.goal_direction,
            "target_angular_speed_rad_s": cfg.target_angular_speed_rad_s,
            "domain_profile": cfg.domain_profile,
            "hand_posture": cfg.hand_posture,
            "hand_mount_position": cfg.robot.init_state.pos,
            "hand_mount_quaternion_xyzw": cfg.robot.init_state.rot,
            "exploration_std_override": args.initial_action_std,
            "entropy_loss_scale_effective": agent_cfg["agent"]["entropy_loss_scale"],
            "rollout_steps_effective": agent_cfg["agent"]["rollouts"],
            "learning_epochs": agent_cfg["agent"]["learning_epochs"],
            "mini_batches": agent_cfg["agent"]["mini_batches"],
            "reset_joint_noise_rad": cfg.reset_joint_noise_rad,
            "reset_position_noise_m": cfg.reset_position_noise_m,
            "reset_grasps_file": cfg.reset_grasps_file,
            "reset_grasps_sha256": hashlib.sha256(reset_data).hexdigest(),
            "reset_grasp_selection": "uniform_per_reset" if cfg.randomize_reset_grasp else "env_id_modulo",
            "resumed_from": str(args.resume_checkpoint) if args.resume_checkpoint else None,
            "prior_training_transitions": writer.prior_transitions,
            "fruit_axes_m": env.unwrapped._axes_m.cpu().tolist(),
            "fruit_mass_kg": env.unwrapped._fruit.data.body_mass.torch.cpu().tolist(),
            "upstream_ppo": "multimodal_rl@4ac55e854b4c732534d443715a0322d4220b461c",
        }, indent=2) + "\n")
        wrapped, trainer = build_roto2_training(env, cfg, agent_cfg, writer)
        trainer.headless = args.headless
        env = wrapped
        if args.resume_checkpoint is not None:
            if not args.resume_checkpoint.is_file():
                raise FileNotFoundError(args.resume_checkpoint)
            trainer.agent.load(str(args.resume_checkpoint))
            for module in (trainer.agent.encoder, trainer.agent.policy, trainer.agent.value):
                module.train()
        if args.validate_contracts:
            from simulation.roto2_pressure.validation import validate_wrapper
            checks = validate_wrapper(wrapped)
            (Path(writer.log_dir) / "wrapper_validation.json").write_text(json.dumps(checks, indent=2) + "\n")
        if args.evaluate_checkpoint is not None:
            if args.trace_policy_output:
                from simulation.roto2_pressure.rotation_diagnostics import trace_policy
                trace_policy(trainer, wrapped, args.evaluate_checkpoint, args.trace_policy_output)
                return 0
            if args.probe_rotation_output:
                from simulation.roto2_pressure.rotation_diagnostics import probe_rotation_actions
                probe_rotation_actions(trainer, wrapped, args.evaluate_checkpoint, args.probe_rotation_output)
                return 0
            if args.audit_rotation_output:
                from simulation.roto2_pressure.audit import audit_checkpoint
                audit_checkpoint(trainer, wrapped, args.evaluate_checkpoint, args.audit_rotation_output)
                return 0
            if args.record_video:
                from simulation.roto2_pressure.record import record_checkpoint
                record_checkpoint(trainer, wrapped, args.evaluate_checkpoint,
                                  args.record_video, args.seed, args.video_env_index,
                                  args.video_max_steps)
                return 0
            from simulation.roto2_pressure.evaluate import evaluate_checkpoint
            if not args.evaluate_checkpoint.is_file():
                raise FileNotFoundError(args.evaluate_checkpoint)
            output = args.evaluation_output or (
                ROOT / "logs/roto2_pressure_fruit"
                / f"eval_{args.pressure_variant}_seed{args.seed}.json"
            )
            evaluate_checkpoint(trainer, wrapped, args.evaluate_checkpoint, output)
            return 0
        initial_parameters = {
            name: torch.cat([p.detach().flatten().clone() for p in module.parameters()])
            for name, module in (("encoder", trainer.agent.encoder),
                                 ("policy", trainer.agent.policy), ("value", trainer.agent.value))
        }
        torch.save({name: writer._get_module_state(module)
                    for name, module in writer.checkpoint_modules.items()},
                   Path(writer.checkpoint_dir) / "initial_agent.pt")
        import time
        started = time.monotonic()
        best_return, aborted = trainer.train()
        if aborted:
            raise RuntimeError("Upstream PPO aborted (nonfinite update or pruning)")
        changes = {}
        for name, before in initial_parameters.items():
            after = torch.cat([p.detach().flatten() for p in getattr(trainer.agent, name).parameters()])
            if not torch.isfinite(after).all():
                raise RuntimeError(f"Nonfinite learned parameters: {name}")
            changes[name] = float((after - before).norm())
        if trainer.rl_update < 1 or not all(v > 0 for v in changes.values()):
            raise RuntimeError("Training did not update all three model modules")
        final_checkpoint = Path(writer.checkpoint_dir) / "final_agent.pt"
        torch.save({name: writer._get_module_state(module)
                    for name, module in writer.checkpoint_modules.items()}, final_checkpoint)
        result = {"training_completed": True, "ppo_updates": trainer.rl_update,
                  "training_transitions": trainer.timesteps * trainer.num_train_envs,
                  "cumulative_training_transitions": writer.prior_transitions + trainer.timesteps * trainer.num_train_envs,
                  "parameter_change_l2": changes, "finite_parameters": True,
                  "elapsed_seconds": time.monotonic() - started,
                  "final_checkpoint": str(final_checkpoint), "seed": args.seed,
                  "variant": args.pressure_variant,
                  "resumed_from": str(args.resume_checkpoint) if args.resume_checkpoint else None}
        (Path(writer.log_dir) / "training_result.json").write_text(json.dumps(result, indent=2) + "\n")
        if writer.tb_writer:
            writer.tb_writer.flush()
            writer.tb_writer.close()
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        application.close(exit_code=exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
