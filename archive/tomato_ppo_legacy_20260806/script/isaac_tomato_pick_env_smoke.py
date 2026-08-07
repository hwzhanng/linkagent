"""Instantiate and step the frozen tomato PPO environment without training."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import traceback
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.skills.tomato_pick.contracts import (
    ACTION_DIM,
    G20_JOINT_ORDER,
    OBSERVATION_DIM,
)


def main(argv: Sequence[str] | None = None) -> int:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=("zero", "excite"), default="zero")
    parser.add_argument("--sim2real-mode", choices=("full", "nominal"), default="full")
    parser.add_argument("--override-gravity-z", type=float, default=None)
    parser.add_argument("--override-fruit-precontact-distance", type=float, default=None)
    parser.add_argument("--require-reset-ready", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("--num-envs and --steps must be > 0")

    launcher = AppLauncher(args)
    app = launcher.app
    raw_env = None
    try:
        gym = importlib.import_module("gymnasium")
        torch = importlib.import_module("torch")
        task = importlib.import_module("simulation.tomato_pick.isaac")
        cfg_module = importlib.import_module("simulation.tomato_pick.isaac.env_cfg")

        cfg = cfg_module.TomatoPickEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.enable_sim2real_randomization = args.sim2real_mode == "full"
        if args.override_gravity_z is not None:
            cfg.sim.gravity = (0.0, 0.0, args.override_gravity_z)
        if args.override_fruit_precontact_distance is not None:
            distance = args.override_fruit_precontact_distance
            cfg.fruit_precontact_distance_range_m = (distance, distance)

        raw_env = gym.make(task.TASK_ID, cfg=cfg)
        environment = raw_env.unwrapped
        observation, _ = raw_env.reset(seed=args.seed)
        initial_hand_targets = environment._joint_position_targets[
            :, environment._hand_joint_ids
        ].clone()
        checks = {
            "actor_shape_113": tuple(observation["policy"].shape)
            == (args.num_envs, OBSERVATION_DIM),
            "critic_shape_153": tuple(observation["critic"].shape)
            == (args.num_envs, cfg.state_space),
            "action_shape_22": environment.single_action_space.shape == (ACTION_DIM,),
            "actor_finite": bool(torch.isfinite(observation["policy"]).all()),
            "critic_finite": bool(torch.isfinite(observation["critic"]).all()),
            "robot_base_fixed": bool(environment._robot.is_fixed_base),
            "robot_self_collisions_disabled": environment._self_collision_check["enabled"] is False,
            "sixteen_independent_g20_joints": tuple(environment._hand_joint_names)
            == tuple(f"right_hand_{name}" for name in G20_JOINT_ORDER),
            "ten_mimic_followers_software_coupled": len(environment._mimic_follower_ids) == 10,
            "frozen_mass_profile_valid": environment._mass_profile_validation["valid"],
            "nominal_arm_joint_limit_margin_valid": (
                environment._nominal_arm_minimum_margin_fraction
                >= environment.cfg.minimum_arm_joint_limit_margin_fraction
            ),
            "one_tomato_filter_per_fingertip": all(
                sensor._num_filter_shapes == 1
                for sensor in environment._contact_sensors.values()
            ),
            "all_step_observations_finite": True,
            "all_step_rewards_finite": True,
            "done_shapes_valid": True,
            "termination_masks_disjoint": True,
            "internal_joint_state_finite": True,
            "ppo_not_started": True,
        }
        terminal_count = 0
        active_excitation_steps = 0
        maximum_hand_target_delta = 0.0
        for step_index in range(args.steps):
            actions = torch.zeros(
                (args.num_envs, ACTION_DIM), dtype=torch.float32, device=environment.device
            )
            if args.policy == "excite":
                active = ~environment._reset_pending
                active_excitation_steps += int(active.sum())
                direction = 1.0 if (step_index // 4) % 2 == 0 else -1.0
                # Excite all six Cartesian channels and every independent G20
                # joint with small bounded commands.  This verifies wiring;
                # it is deliberately not a scripted picking policy.
                actions[active, :3] = torch.tensor(
                    (0.15, -0.10, 0.10), device=environment.device
                )
                actions[active, 3:6] = torch.tensor(
                    (0.08, -0.06, 0.10), device=environment.device
                )
                actions[active, 6:22] = 0.20 * direction

            observation, reward, terminated, truncated, _ = raw_env.step(actions)
            maximum_hand_target_delta = max(
                maximum_hand_target_delta,
                float(
                    (
                        environment._joint_position_targets[:, environment._hand_joint_ids]
                        - initial_hand_targets
                    )
                    .abs()
                    .max()
                ),
            )
            checks["all_step_observations_finite"] &= bool(
                torch.isfinite(observation["policy"]).all()
                and torch.isfinite(observation["critic"]).all()
            )
            checks["all_step_rewards_finite"] &= bool(torch.isfinite(reward).all())
            checks["done_shapes_valid"] &= tuple(terminated.shape) == (
                args.num_envs,
            ) and tuple(truncated.shape) == (args.num_envs,)
            checks["termination_masks_disjoint"] &= not bool((terminated & truncated).any())
            checks["internal_joint_state_finite"] &= bool(
                torch.isfinite(environment._robot.data.joint_pos.torch).all()
                and torch.isfinite(environment._robot.data.joint_vel.torch).all()
            )
            terminal_count += int((terminated | truncated).sum())

        if args.require_reset_ready:
            checks["validated_pregrasp_reset_completed"] = (
                environment._reset_completed_count >= args.num_envs
            )
        if args.policy == "excite" and active_excitation_steps > 0:
            checks["direct_hand_action_changes_targets"] = maximum_hand_target_delta > 1.0e-5
            checks["cartesian_action_reaches_six_dof_ik"] = bool(
                environment._last_ik_command[:, :6].abs().max() > 0.0
            )

        report = {
            "schema_version": 2,
            "passed": all(checks.values()),
            "task_id": task.TASK_ID,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "seed": args.seed,
            "device": str(environment.device),
            "policy": args.policy,
            "sim2real_mode": args.sim2real_mode,
            "checks": checks,
            "reset_completed_count": environment._reset_completed_count,
            "active_excitation_steps": active_excitation_steps,
            "maximum_hand_target_delta_rad": maximum_hand_target_delta,
            "terminal_count": terminal_count,
            "ppo_started": False,
        }
        print("ISAAC_TOMATO_ENV_SMOKE_REPORT=" + json.dumps(report, sort_keys=True), flush=True)
        return 0 if report["passed"] else 2
    except BaseException as exc:
        print(
            "ISAAC_TOMATO_ENV_SMOKE_EXCEPTION="
            + json.dumps({"type": type(exc).__name__, "message": str(exc)}),
            flush=True,
        )
        traceback.print_exc()
        return 3
    finally:
        if raw_env is not None:
            raw_env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
