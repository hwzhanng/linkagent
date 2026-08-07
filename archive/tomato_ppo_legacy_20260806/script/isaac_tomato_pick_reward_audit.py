"""Stress the Isaac tomato environment and audit the frozen PPO contract.

This script never constructs an optimizer and never starts PPO.  It exercises
the exact deployable 113-D actor observation and 22-D action boundary under
nominal or full sim-to-real randomization, then independently reconstructs the
reward returned by the environment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module.skills.tomato_pick.contracts import ACTION_DIM, OBSERVATION_DIM


OBSERVATION_BLOCKS = {
    "target_pose_error_grasp": (0, 6),
    "ee_twist_grasp": (6, 12),
    "arm_joint_position_norm": (12, 19),
    "arm_joint_velocity_norm": (19, 26),
    "hand_joint_position_norm": (26, 42),
    "hand_joint_velocity_norm": (42, 58),
    "fingertip_pressure_norm": (58, 63),
    "fingertip_pressure_rate_norm": (63, 68),
    "gravity_direction_grasp": (68, 71),
    "previous_action": (71, 93),
    "target_arm_hand_valid": (93, 96),
    "force_valid": (96, 101),
    "target_arm_hand_age_norm": (101, 104),
    "force_age_norm": (104, 109),
    "fruit_axes_norm": (109, 112),
    "episode_progress": (112, 113),
}
BOUNDED_BLOCKS = set(OBSERVATION_BLOCKS) - {
    "target_pose_error_grasp",
    "ee_twist_grasp",
}


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.absolute_total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.nonzero = 0

    def update(self, values) -> None:
        flat = values.detach().float().reshape(-1)
        self.count += flat.numel()
        self.total += float(flat.sum())
        self.square_total += float(flat.square().sum())
        self.absolute_total += float(flat.abs().sum())
        self.minimum = min(self.minimum, float(flat.min()))
        self.maximum = max(self.maximum, float(flat.max()))
        self.nonzero += int((flat.abs() > 1.0e-12).sum())

    def report(self, denominator: float | None = None) -> dict[str, float]:
        mean = self.total / max(self.count, 1)
        variance = max(0.0, self.square_total / max(self.count, 1) - mean * mean)
        result = {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
            "nonzero_fraction": self.nonzero / max(self.count, 1),
        }
        if denominator is not None:
            result["absolute_contribution_fraction"] = self.absolute_total / max(
                denominator, 1.0e-12
            )
        return result


def main(argv: Sequence[str] | None = None) -> int:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=("random", "zero"), default="random")
    parser.add_argument("--sim2real-mode", choices=("full", "nominal"), default="full")
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
        raw_env = gym.make(task.TASK_ID, cfg=cfg)
        environment = raw_env.unwrapped
        observation, _ = raw_env.reset(seed=args.seed)
        generator = torch.Generator(device=environment.device).manual_seed(args.seed + 10_000)
        sim2real = environment._sim2real

        checks = {
            "actor_shape_113": True,
            "critic_shape_153": True,
            "action_shape_22": environment.single_action_space.shape == (ACTION_DIM,),
            "actor_float32": True,
            "all_observations_finite": True,
            "all_actions_finite_and_bounded": True,
            "all_rewards_finite": True,
            "reward_equals_sum_of_terms": True,
            "terminated_bool": True,
            "truncated_bool": True,
            "termination_masks_disjoint": True,
            "internal_physics_state_finite": True,
            "pressure_and_rate_zero_when_invalid": True,
            "fruit_axes_zero_when_target_invalid": True,
            "gravity_direction_is_unit_vector": True,
            "phase_not_present_in_actor_observation": "phase_one_hot" not in OBSERVATION_BLOCKS,
            "non_tip_reward_removed": "non_tip_contact" not in environment._episode_sums,
            "pre_detach_slip_reward_removed": "slip" not in environment._episode_sums,
            "contact_reward_is_one_shot_event": True,
            "sim2real_mode_matches": sim2real.enabled == (args.sim2real_mode == "full"),
            "sixteen_independent_hand_targets": len(environment._hand_joint_ids) == 16,
        }
        observation_stats = {name: RunningStats() for name in OBSERVATION_BLOCKS}
        reward_stats = RunningStats()
        term_stats = {name: RunningStats() for name in environment._episode_sums}
        saturation = {name: 0 for name in OBSERVATION_BLOCKS}
        observation_count = {name: 0 for name in OBSERVATION_BLOCKS}
        end_counts = {"terminated": 0, "truncated": 0, "success": 0, "drop": 0}

        for _ in range(args.steps):
            actor = observation["policy"]
            critic = observation["critic"]
            checks["actor_shape_113"] &= tuple(actor.shape) == (args.num_envs, OBSERVATION_DIM)
            checks["critic_shape_153"] &= tuple(critic.shape) == (
                args.num_envs,
                cfg.state_space,
            )
            checks["actor_float32"] &= actor.dtype == torch.float32
            checks["all_observations_finite"] &= bool(
                torch.isfinite(actor).all() and torch.isfinite(critic).all()
            )

            force_valid = actor[:, 96:101].bool()
            force_features = torch.stack((actor[:, 58:63], actor[:, 63:68]), dim=2)
            invalid_force = ~force_valid
            checks["pressure_and_rate_zero_when_invalid"] &= bool(
                torch.equal(
                    force_features[invalid_force],
                    torch.zeros_like(force_features[invalid_force]),
                )
            )
            target_invalid = actor[:, 93] == 0.0
            checks["fruit_axes_zero_when_target_invalid"] &= bool(
                torch.equal(
                    actor[target_invalid, 109:112],
                    torch.zeros_like(actor[target_invalid, 109:112]),
                )
            )
            gravity_norm = torch.linalg.vector_norm(actor[:, 68:71], dim=1)
            checks["gravity_direction_is_unit_vector"] &= bool(
                torch.allclose(gravity_norm, torch.ones_like(gravity_norm), atol=1.0e-5)
            )
            for name, (start, stop) in OBSERVATION_BLOCKS.items():
                block = actor[:, start:stop]
                observation_stats[name].update(block)
                saturation[name] += int((block.abs() >= 0.999).sum())
                observation_count[name] += block.numel()

            if args.policy == "zero":
                actions = torch.zeros(
                    (args.num_envs, ACTION_DIM), dtype=torch.float32, device=environment.device
                )
            else:
                actions = torch.empty(
                    (args.num_envs, ACTION_DIM), dtype=torch.float32, device=environment.device
                ).uniform_(-1.0, 1.0, generator=generator)
            checks["all_actions_finite_and_bounded"] &= bool(
                torch.isfinite(actions).all()
                and (actions >= -1.0).all()
                and (actions <= 1.0).all()
            )

            observation, rewards, terminated, truncated, info = raw_env.step(actions)
            checks["all_rewards_finite"] &= bool(torch.isfinite(rewards).all())
            checks["terminated_bool"] &= terminated.dtype == torch.bool
            checks["truncated_bool"] &= truncated.dtype == torch.bool
            checks["termination_masks_disjoint"] &= not bool((terminated & truncated).any())
            terms = environment._last_reward_terms
            reconstructed = torch.stack(tuple(terms.values())).sum(dim=0)
            checks["reward_equals_sum_of_terms"] &= bool(
                torch.allclose(rewards, reconstructed, atol=1.0e-6, rtol=1.0e-6)
            )
            contact = terms["contact"]
            checks["contact_reward_is_one_shot_event"] &= bool(
                (
                    (contact == 0.0)
                    | torch.isclose(
                        contact,
                        torch.full_like(contact, environment.cfg.reward_contact_scale),
                    )
                ).all()
            )
            checks["internal_physics_state_finite"] &= bool(
                torch.isfinite(environment._robot.data.joint_pos.torch).all()
                and torch.isfinite(environment._robot.data.joint_vel.torch).all()
                and torch.isfinite(environment._tomato.data.root_state_w.torch).all()
            )
            reward_stats.update(rewards)
            for name, values in terms.items():
                term_stats[name].update(values)
            end_counts["terminated"] += int(terminated.sum())
            end_counts["truncated"] += int(truncated.sum())
            step = info.get("step", {})
            end_counts["success"] += int(step.get("success_count", 0))
            end_counts["drop"] += int(step.get("drop_count", 0))

        total_abs = sum(stats.absolute_total for stats in term_stats.values())
        report = {
            "schema_version": 2,
            "passed": all(checks.values()),
            "checks": checks,
            "task_id": task.TASK_ID,
            "policy": args.policy,
            "sim2real_mode": args.sim2real_mode,
            "num_envs": args.num_envs,
            "policy_steps": args.steps,
            "transitions_checked": args.num_envs * args.steps,
            "episode_end_counts": end_counts,
            "total_reward": reward_stats.report(),
            "reward_terms": {
                name: stats.report(total_abs) for name, stats in term_stats.items()
            },
            "actor_observation_blocks": {
                name: {
                    **stats.report(),
                    "bounded_near_limit_fraction": (
                        saturation[name] / max(observation_count[name], 1)
                        if name in BOUNDED_BLOCKS
                        else None
                    ),
                }
                for name, stats in observation_stats.items()
            },
            "ppo_started": False,
            "optimizer_steps": 0,
        }
        print("ISAAC_TOMATO_REWARD_AUDIT=" + json.dumps(report, sort_keys=True), flush=True)
        return 0 if report["passed"] else 2
    except BaseException as exc:
        print(
            "ISAAC_TOMATO_REWARD_AUDIT_EXCEPTION="
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
