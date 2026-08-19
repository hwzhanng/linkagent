"""Small asymmetric PPO runner for the hand-only PhysX tomato curriculum.

This runner is intentionally separate from the existing sphere trainer.  The
actor consumes the deployable 110D observation while the critic receives the
163D simulator-truth state.  ``--stage 1`` trains the palm-supported
five-finger hold; ``--stage 2`` enables the virtual stem pick trajectory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import signal
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Asymmetric PPO for PhysX tomato hand grasp")
parser.add_argument("--stage", type=int, choices=(0, 1, 2, 3, 4), default=1)
parser.add_argument(
    "--stage4-band",
    type=int,
    choices=(1, 2, 3, 4),
    default=1,
    help="progressive stage-4 randomization: 1=entry, 2=intermediate, 3=full envelope",
)
parser.add_argument(
    "--stage4-progress",
    type=float,
    default=1.0,
    help="fraction in (0,1] between the previous stage-4 band and the selected band",
)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--horizon", type=int, default=64)
parser.add_argument("--updates", type=int, default=4)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--minibatch-size", type=int, default=1024)
parser.add_argument("--learning-rate", type=float, default=3.0e-4)
parser.add_argument("--entropy-coef", type=float, default=2.0e-3)
parser.add_argument("--value-coef", type=float, default=0.5)
parser.add_argument("--clip-eps", type=float, default=0.2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--initial-std", type=float, default=0.45)
parser.add_argument("--coarse-initial-std", type=float, default=None)
parser.add_argument("--residual-initial-std", type=float, default=None)
parser.add_argument("--minimum-std", type=float, default=0.10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval-interval", type=int, default=1)
parser.add_argument("--eval-steps", type=int, default=160)
parser.add_argument("--eval-only", action="store_true")
parser.add_argument("--output-dir", type=Path, default=ROOT / "simulation/ppo_tomato_grasp/artifacts/physx_tomato_stage01")
parser.add_argument("--resume", type=Path, default=None)
parser.add_argument("--bc-initialize", type=Path, default=None)
parser.add_argument("--bc-anchor-coef", type=float, default=0.0)
parser.add_argument("--bc-anchor-updates", type=int, default=50)
parser.add_argument("--actor-freeze-updates", type=int, default=0)
parser.add_argument("--policy-anchor-coef", type=float, default=0.0)
parser.add_argument("--policy-anchor-updates", type=int, default=0)
parser.add_argument("--size-closure-anchor-coef", type=float, default=0.0)
parser.add_argument("--size-closure-anchor-updates", type=int, default=0)
parser.add_argument("--size-closure-gain", type=float, default=1.5)
parser.add_argument("--size-closure-threshold", type=float, default=0.58)
parser.add_argument("--scripted-action-prior", action="store_true")
parser.add_argument("--scripted-residual-scale", type=float, default=0.05)
parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
parser.add_argument("--wandb-project", default="linkbot-tomato-grasp-physx-stage01")
parser.add_argument("--wandb-run-name", default=None)
parser.add_argument(
    "--enveloping",
    action="store_true",
    help="train the independent 14D palm-supported five-finger enclosure task",
)
parser.add_argument("--teacher-data", type=Path, default=None,
                    help="optional 14D teacher dataset used for actor warm start")
parser.add_argument("--teacher-bc-epochs", type=int, default=20)
parser.add_argument("--enveloping-coverage-deg", type=float, default=200.0)
parser.add_argument("--enveloping-hold-steps", type=int, default=10)
parser.add_argument("--enveloping-palm-margin", type=int, default=3)
parser.add_argument("--enveloping-course", choices=("g0", "g1", "final"), default="g1",
                    help="G0 core geometry, G1 moderate randomization, or final envelope")
parser.add_argument("--enveloping-safe-force", type=float, default=16.0,
                    help="training-only per-digit safety prior for the enveloping curriculum")
parser.add_argument("--enveloping-pick-speed", type=float, default=None,
                    help="override the nominal stage-2+ upward pick speed (m/s)")
parser.add_argument("--enveloping-pick-displacement", type=float, default=None,
                    help="override the stem displacement used for stage-2+ pick (m)")
parser.add_argument("--enveloping-stem-limit", type=float, default=None,
                    help="override the fixed stem tensile limit for an easy retention curriculum (N)")
parser.add_argument("--enveloping-stem-duration", type=float, default=None,
                    help="override the fixed critical duration for stem break (s)")
parser.add_argument("--enveloping-distal-init", type=float, default=0.0,
                    help="optional positive initial bias for the four distal residual actions")
parser.add_argument("--enveloping-stem-gravity-compensation", action="store_true",
                    help="carry attached fruit weight in the stem (Clean-v5 physics ablation)")
parser.add_argument("--enveloping-pick-direction-z-sign", type=float, choices=(-1.0, 1.0), default=-1.0,
                    help="plant-anchor motion sign in the hand-fixed frame")
parser.add_argument("--enveloping-tactile-reflex", action="store_true",
                    help="enable bounded post-break distal tactile reflex")
parser.add_argument("--enveloping-postbreak-contact-weight", type=float, default=None)
parser.add_argument("--enveloping-postbreak-support-weight", type=float, default=None)
parser.add_argument("--enveloping-postbreak-missing-weight", type=float, default=None)
parser.add_argument("--enveloping-drop-penalty", type=float, default=None)
parser.add_argument("--enveloping-postbreak-grace-steps", type=int, default=0,
                    help="bounded post-detachment reaction window in policy steps")
parser.add_argument("--enveloping-retention-only", action="store_true",
                    help="train the post-break hold curriculum from a seeded enclosure")
parser.add_argument("--enveloping-adaptive-retention", action="store_true",
                    help="enable the bounded per-digit force/slip retention reflex")
parser.add_argument("--retention-seed-bank", type=Path, default=None,
                    help="optional Kx16 joint-state bank for retention-only resets")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from torch import nn
from torch.distributions import Normal

from simulation.ppo_tomato_grasp.physx_tomato_env import PhysxTomatoGraspEnv, PhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp.stage0_scripted_controller import Stage0ControllerCfg, Stage0ScriptedController
from simulation.ppo_tomato_grasp.physx_enveloping_env import (
    EnvelopingPhysxTomatoGraspEnv,
    EnvelopingPhysxTomatoGraspEnvCfg,
)
from simulation.ppo_tomato_grasp.enveloping_teacher import EnvelopingTeacher


class AsymmetricActorCritic(nn.Module):
    def __init__(self, actor_dim: int, critic_dim: int, action_dim: int, minimum_std: float, initial_std: float, coarse_initial_std: float | None = None, residual_initial_std: float | None = None):
        super().__init__()
        self.actor_body = nn.Sequential(nn.Linear(actor_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU())
        self.critic_body = nn.Sequential(nn.Linear(critic_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU())
        self.actor_mean = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)
        initial = torch.full((action_dim,), initial_std)
        if coarse_initial_std is not None:
            initial[:5] = coarse_initial_std
        if residual_initial_std is not None:
            initial[5:] = residual_initial_std
        if bool((initial <= 0.0).any()):
            raise ValueError("all initial action standard deviations must be positive")
        self.log_std = nn.Parameter(initial.log())
        self.minimum_log_std = math.log(minimum_std)

    def distribution(self, observation: torch.Tensor) -> Normal:
        return Normal(self.actor_mean(self.actor_body(observation)), self.log_std.clamp(self.minimum_log_std, 1.0).exp())

    def critic(self, state: torch.Tensor) -> torch.Tensor:
        return self.value(self.critic_body(state)).squeeze(-1)


def squashed_log_prob(distribution: Normal, raw: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return (distribution.log_prob(raw) - torch.log(1.0 - action.square() + 1.0e-6)).sum(-1)


def save_checkpoint(path: Path, policy: AsymmetricActorCritic, optimizer, update: int, total_steps: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "version": "physx_enveloping_or_tomato_stage01_v1",
        "stage": args_cli.stage,
        "actor_observation_dim": 110,
        "critic_state_dim": 163,
        "action_dim": int(policy.actor_mean.out_features),
        "task": "palm_supported_enveloping" if args_cli.enveloping else "legacy_tomato",
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "total_steps": total_steps,
        "metrics": metrics,
        "run_config": vars(args_cli),
    }, temporary)
    temporary.replace(path)


def _scripted_prior(env: PhysxTomatoGraspEnv, controller: Stage0ScriptedController) -> torch.Tensor:
    return controller.command(
        env.contact_forces,
        env.palm_force,
        # In stage 1 the enclosure itself is the retention phase.  The stage-0
        # harvesting gate enables this stronger missing-contact recovery after
        # stem break; waiting for a nonexistent break in stage 1 caused a
        # five-contact transient followed by a drop.
        retention_mode=controller.enclosure_latched,
        center_error_m=env.tomato.data.root_pos_w - env.grasp_center,
        object_radius_m=env.tomato_axes.mean(dim=-1),
    )


@torch.inference_mode()
def evaluate(env: PhysxTomatoGraspEnv, policy: AsymmetricActorCritic, steps: int, controller: Stage0ScriptedController | EnvelopingTeacher | None = None) -> dict[str, float]:
    observation, _ = env.reset()
    actor = observation["policy"]
    returns = torch.zeros(env.num_envs, device=actor.device)
    success = torch.zeros(env.num_envs, dtype=torch.bool, device=actor.device)
    strict_success = torch.zeros_like(success)
    five_digit = torch.zeros_like(success)
    overloaded = torch.zeros_like(success)
    dropped = torch.zeros_like(success)
    alive = torch.ones_like(success)
    first_termination_code = torch.zeros(env.num_envs, dtype=torch.long, device=actor.device)
    pick_started = torch.zeros_like(success)
    stem_broken = torch.zeros_like(success)
    post_break_hold = torch.zeros_like(success)
    strict_hold_max = torch.zeros(env.num_envs, device=actor.device)
    strict_coverage_max = torch.zeros(env.num_envs, device=actor.device)
    strict_distal_all_max = torch.zeros(env.num_envs, device=actor.device)
    strict_first_palm_max = torch.zeros(env.num_envs, device=actor.device)
    strict_first_three_max = torch.zeros(env.num_envs, device=actor.device)
    strict_proximal_only_max = torch.zeros(env.num_envs, device=actor.device)
    strict_opposition_pass_max = torch.zeros(env.num_envs, device=actor.device)
    strict_hemisphere_pass_max = torch.zeros(env.num_envs, device=actor.device)
    strict_load_share_pass_max = torch.zeros(env.num_envs, device=actor.device)
    strict_force_safe_max = torch.zeros(env.num_envs, device=actor.device)
    action_sum = torch.zeros(env.cfg.action_space, device=actor.device)
    action_abs_sum = torch.zeros_like(action_sum)
    action_samples = torch.zeros((), device=actor.device)
    if controller is not None:
        controller.reset()
    max_force = torch.zeros(env.num_envs, device=actor.device)
    for _ in range(steps):
        policy_action = torch.tanh(policy.distribution(actor).mean)
        action_sum += (policy_action * alive.unsqueeze(-1)).sum(dim=0)
        action_abs_sum += (policy_action.abs() * alive.unsqueeze(-1)).sum(dim=0)
        action_samples += alive.sum()
        action = policy_action
        if controller is not None and isinstance(controller, Stage0ScriptedController):
            action = (_scripted_prior(env, controller) + args_cli.scripted_residual_scale * policy_action).clamp(-1.0, 1.0)
        observation, reward, terminated, truncated, info = env.step(action)
        returns += reward * alive.float()
        termination_code = env.last_termination_code
        success |= alive & (env.success | (termination_code == 1))
        strict_success |= alive & (termination_code == 1) if args_cli.enveloping else torch.zeros_like(success)
        five_digit |= alive & (env.contact_forces >= env.cfg.contact_threshold_n).all(dim=-1) & (
            env.palm_force >= env.cfg.palm_support_force_n
        )
        overloaded |= alive & (env.overloaded | (termination_code == 3))
        dropped |= alive & (env.dropped | (termination_code == 2))
        pick_started |= alive & env.pick_started
        stem_broken |= alive & env.stem_broke
        post_break_hold |= alive & env.stem_broke & (
            env.stable_steps >= env.cfg.success_hold_steps
        )
        if args_cli.enveloping and isinstance(info, dict):
            log = info.get("log", {})
            strict_hold_max = torch.maximum(
                strict_hold_max,
                torch.full_like(strict_hold_max, float(log.get("strict_hold_max", 0.0))),
            )
            strict_coverage_max = torch.maximum(
                strict_coverage_max,
                torch.full_like(strict_coverage_max, float(log.get("strict_coverage_max_deg_proxy", 0.0))),
            )
            strict_distal_all_max = torch.maximum(
                strict_distal_all_max,
                torch.full_like(strict_distal_all_max, float(log.get("strict_distal_all_rate", 0.0))),
            )
            strict_first_palm_max = torch.maximum(
                strict_first_palm_max,
                torch.full_like(strict_first_palm_max, float(log.get("strict_palm_first_rate", 0.0))),
            )
            strict_first_three_max = torch.maximum(
                strict_first_three_max,
                torch.full_like(strict_first_three_max, float(log.get("strict_first_three_rate", 0.0))),
            )
            strict_proximal_only_max = torch.maximum(
                strict_proximal_only_max,
                torch.full_like(strict_proximal_only_max, float(log.get("strict_proximal_only_rate", 0.0))),
            )
            for buffer, key in (
                (strict_opposition_pass_max, "strict_opposition_pass_rate"),
                (strict_hemisphere_pass_max, "strict_hemisphere_pass_rate"),
                (strict_load_share_pass_max, "strict_load_share_pass_rate"),
                (strict_force_safe_max, "strict_force_safe_rate"),
            ):
                buffer[:] = torch.maximum(buffer, torch.full_like(buffer, float(log.get(key, 0.0))))
        max_force = torch.maximum(
            max_force,
            torch.where(alive, env.contact_forces.max(dim=-1).values, max_force),
        )
        newly_done = alive & (terminated | truncated)
        first_termination_code[newly_done] = termination_code[newly_done]
        alive &= ~newly_done
        actor = observation["policy"]
        if not bool(alive.any()):
            break
    metrics = {
        "eval/return_mean": float(returns.mean()),
        "eval/success_rate": float(success.float().mean()),
        "eval/max_digit_force_n": float(max_force.mean()),
        "eval/overload_rate": float(overloaded.float().mean()),
        "eval/drop_rate": float(dropped.float().mean()),
        "eval/five_digit_enclosure_rate": float(five_digit.float().mean()),
        "eval/failure_drop_rate": float((first_termination_code == 2).float().mean()),
        "eval/failure_overload_rate": float((first_termination_code == 3).float().mean()),
        "eval/failure_damage_rate": float((first_termination_code == 4).float().mean()),
        "eval/failure_timeout_rate": float((first_termination_code == 5).float().mean()),
        "eval/pick_triggered_rate": float(pick_started.float().mean()),
        "eval/stem_break_rate": float(stem_broken.float().mean()),
        "eval/post_break_hold_rate": float(post_break_hold.float().mean()),
    }
    if args_cli.enveloping:
        metrics.update({
            "eval/strict_envelop_success_rate": float(strict_success.float().mean()),
            "eval/strict_hold_max_mean": float(strict_hold_max.mean()),
            "eval/strict_coverage_max_deg_proxy": float(strict_coverage_max.max()),
            "eval/strict_distal_all_rate_max": float(strict_distal_all_max.max()),
            "eval/strict_palm_first_rate_max": float(strict_first_palm_max.max()),
            "eval/strict_first_three_rate_max": float(strict_first_three_max.max()),
            "eval/strict_proximal_only_rate_max": float(strict_proximal_only_max.max()),
            "eval/strict_opposition_pass_rate_max": float(strict_opposition_pass_max.max()),
            "eval/strict_hemisphere_pass_rate_max": float(strict_hemisphere_pass_max.max()),
            "eval/strict_load_share_pass_rate_max": float(strict_load_share_pass_max.max()),
            "eval/strict_force_safe_rate_max": float(strict_force_safe_max.max()),
        })
    denominator = action_samples.clamp_min(1.0)
    for index in range(env.cfg.action_space):
        metrics[f"eval/action_mean_{index}"] = float(action_sum[index] / denominator)
        metrics[f"eval/action_abs_mean_{index}"] = float(action_abs_sum[index] / denominator)
    env.reset()
    return metrics


def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.horizon <= 0 or args_cli.updates <= 0:
        raise ValueError("num_envs, horizon and updates must be positive")
    if args_cli.stage != 4 and args_cli.stage4_band != 1:
        raise ValueError("--stage4-band is only meaningful with --stage 4")
    if not 0.0 < args_cli.stage4_progress <= 1.0:
        raise ValueError("--stage4-progress must lie in (0, 1]")
    if args_cli.stage != 4 and args_cli.stage4_progress != 1.0:
        raise ValueError("--stage4-progress is only meaningful with --stage 4")
    if args_cli.stage4_band == 1 and args_cli.stage4_progress != 1.0:
        raise ValueError("stage-4 band 1 has no earlier band to interpolate from")
    torch.manual_seed(args_cli.seed)
    cfg = EnvelopingPhysxTomatoGraspEnvCfg() if args_cli.enveloping else PhysxTomatoGraspEnvCfg()
    cfg.seed = args_cli.seed
    # Environment dynamics use 2 for every stem-enabled curriculum. The CLI
    # stage additionally distinguishes easy-stem (2) from normal-stem (3).
    cfg.training_stage = min(args_cli.stage, 2)
    # Curriculum entry: begin stage 0/1 with large tomatoes for which the
    # imported thumb and four-finger geometry has verified contact reach.
    # Stage 2 (and later stage-1 continuation runs) broaden back to 25--35 mm.
    if args_cli.stage <= 1:
        cfg.events["tomato_geometry"].params["axis_range_m"] = (0.032, 0.035)
        cfg.events["tomato_geometry"].params["isotropic_axes"] = True
        cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (850.0, 850.0)
        cfg.position_jitter = (0.0005, 0.0005, 0.00025)
        cfg.friction_range = (0.55, 0.55)
        cfg.restitution_range = (0.0, 0.0)
        cfg.stiffness_range_n_m = (18000.0, 18000.0)
        cfg.tactile_gain_range = (1.0, 1.0)
        cfg.tactile_bias_range_n = (0.0, 0.0)
        cfg.sensor_delay_steps_max = 0
        cfg.sensor_dropout_probability = 0.0
        cfg.sensor_stale_probability = 0.0
    elif args_cli.stage == 2:
        # Stage-2 entry curriculum: retain the verified stage-1 sphere and
        # narrow nominal material/sensor priors, then add only the easy stem
        # and fixed slow pick. Shape/material randomization belongs to later
        # stages and must not be conflated with learning the break/hold chain.
        cfg.events["tomato_geometry"].params["axis_range_m"] = (0.032, 0.035)
        cfg.events["tomato_geometry"].params["isotropic_axes"] = True
        cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (850.0, 850.0)
        cfg.position_jitter = (0.0005, 0.0005, 0.00025)
        cfg.friction_range = (0.65, 0.65)
        cfg.restitution_range = (0.0, 0.0)
        cfg.stiffness_range_n_m = (18000.0, 18000.0)
        cfg.tactile_gain_range = (1.0, 1.0)
        cfg.tactile_bias_range_n = (0.0, 0.0)
        cfg.sensor_delay_steps_max = 0
        cfg.sensor_dropout_probability = 0.0
        cfg.sensor_stale_probability = 0.0
        cfg.pick_speed_m_s = 0.010
        cfg.pick_speed_range_m_s = (0.010, 0.010)
        cfg.stem_tensile_limit_n = (0.20, 0.20)
        cfg.stem_critical_duration_s = (0.08, 0.08)
    elif args_cli.stage == 3:
        # Stage-3 entry: restore a realistic first-prior stem band and all
        # four modeled load components while keeping object/sensor variation
        # narrow. Broad domain randomization is reserved for stage 4.
        cfg.events["tomato_geometry"].params["axis_range_m"] = (0.032, 0.035)
        cfg.events["tomato_geometry"].params["isotropic_axes"] = True
        cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (800.0, 950.0)
        cfg.position_jitter = (0.0005, 0.0005, 0.00025)
        cfg.friction_range = (0.55, 0.75)
        cfg.restitution_range = (0.0, 0.02)
        cfg.stiffness_range_n_m = (14000.0, 22000.0)
        cfg.tactile_gain_range = (1.0, 1.0)
        cfg.tactile_bias_range_n = (0.0, 0.0)
        cfg.sensor_delay_steps_max = 0
        cfg.sensor_dropout_probability = 0.0
        cfg.sensor_stale_probability = 0.0
        cfg.pick_speed_m_s = 0.015
        cfg.pick_speed_range_m_s = (0.015, 0.015)
        cfg.pick_twist_rate_rad_s = 0.10
        cfg.pick_twist_angle_rad = 0.15
        cfg.stem_tensile_limit_n = (0.50, 0.90)
        cfg.stem_critical_duration_s = (0.15, 0.30)
    elif args_cli.stage == 4:
        # Three explicit bands prevent a hidden curriculum jump.  Band 3 is
        # the full envelope from the execution plan; bands 1/2 are retained
        # in checkpoints and reports rather than being called final results.
        stage4_bands = {
            1: {
                # Large-fruit entry: 70--80 mm diameter (semi-axis 35--40 mm).
                "axes": (0.035, 0.040), "density": (750.0, 1000.0),
                "jitter": (0.0010, 0.0010, 0.0005), "friction": (0.45, 0.85),
                "restitution": (0.0, 0.04), "stiffness": (10000.0, 26000.0),
                "gain": (0.90, 1.10), "bias": (-0.02, 0.02), "delay": 1,
                "dropout": 0.005, "stale": 0.005,
                "response": (0.96, 1.04), "actuator_bias": (-0.002, 0.002),
                "deadband": (0.0, 0.0015), "backlash": (0.0, 0.0020),
                "pick_speed": (0.014, 0.016), "pick_tilt_deg": 1.5,
                "stem": (0.45, 0.95), "stem_duration": (0.12, 0.32),
            },
            2: {
                # Intermediate large-fruit band: 74--90 mm diameter.
                "axes": (0.037, 0.045), "density": (720.0, 1030.0),
                "jitter": (0.0020, 0.0020, 0.0010), "friction": (0.35, 0.90),
                "restitution": (0.0, 0.05), "stiffness": (6500.0, 28000.0),
                "gain": (0.86, 1.14), "bias": (-0.025, 0.025), "delay": 2,
                "dropout": 0.010, "stale": 0.010,
                "response": (0.88, 1.06), "actuator_bias": (-0.006, 0.006),
                "deadband": (0.0, 0.004), "backlash": (0.0, 0.006),
                "pick_speed": (0.012, 0.018), "pick_tilt_deg": 3.0,
                "stem": (0.40, 1.00), "stem_duration": (0.10, 0.35),
            },
            3: {
                # Broad large-fruit band before the final 76--100 mm envelope.
                "axes": (0.038, 0.048), "density": (700.0, 1050.0),
                "jitter": (0.0040, 0.0040, 0.0020), "friction": (0.25, 0.90),
                "restitution": (0.0, 0.06), "stiffness": (2500.0, 28000.0),
                "gain": (0.82, 1.18), "bias": (-0.03, 0.03), "delay": 3,
                "dropout": 0.020, "stale": 0.020,
                "response": (0.72, 1.08), "actuator_bias": (-0.012, 0.012),
                "deadband": (0.0, 0.010), "backlash": (0.0, 0.015),
                "pick_speed": (0.010, 0.020), "pick_tilt_deg": 5.0,
                "stem": (0.35, 1.05), "stem_duration": (0.08, 0.38),
            },
            # Primary delivery band: retain the verified large-fruit contact
            # geometry (32--35 mm semi-axes) while enabling the complete
            # non-size randomization envelope. Small-fruit stress testing is
            # intentionally kept out of this main training distribution.
            4: {
                "axes": (0.038, 0.050), "density": (700.0, 1050.0),
                "jitter": (0.0040, 0.0040, 0.0020), "friction": (0.25, 0.90),
                "restitution": (0.0, 0.06), "stiffness": (2500.0, 28000.0),
                "gain": (0.82, 1.18), "bias": (-0.03, 0.03), "delay": 3,
                "dropout": 0.020, "stale": 0.020,
                "response": (0.72, 1.08), "actuator_bias": (-0.012, 0.012),
                "deadband": (0.0, 0.010), "backlash": (0.0, 0.015),
                "pick_speed": (0.010, 0.020), "pick_tilt_deg": 5.0,
                "stem": (0.35, 1.05), "stem_duration": (0.08, 0.38),
            },
        }
        stage4 = stage4_bands[args_cli.stage4_band]
        if args_cli.stage4_band > 1 and args_cli.stage4_progress < 1.0:
            previous = stage4_bands[args_cli.stage4_band - 1]
            fraction = args_cli.stage4_progress

            def blend(left, right):
                if isinstance(right, tuple):
                    return tuple(blend(lvalue, rvalue) for lvalue, rvalue in zip(left, right, strict=True))
                return float(left) + fraction * (float(right) - float(left))

            stage4 = {
                key: (int(round(blend(previous[key], value))) if key == "delay" else blend(previous[key], value))
                for key, value in stage4.items()
            }
        cfg.events["tomato_geometry"].params["axis_range_m"] = stage4["axes"]
        # Enter the large-fruit curriculum with an isotropic 35--40 mm
        # semi-axis band.  Ellipsoidal aspect-ratio variation is introduced
        # from band 2 onward, after the palm-support/enclosure timing is
        # learned at the correct physical scale.
        cfg.events["tomato_geometry"].params["isotropic_axes"] = args_cli.stage4_band > 1
        cfg.events["tomato_geometry"].params["density_range_kg_m3"] = stage4["density"]
        cfg.position_jitter = stage4["jitter"]
        cfg.friction_range = stage4["friction"]
        cfg.restitution_range = stage4["restitution"]
        cfg.stiffness_range_n_m = stage4["stiffness"]
        cfg.tactile_gain_range = stage4["gain"]
        cfg.tactile_bias_range_n = stage4["bias"]
        cfg.sensor_delay_steps_max = stage4["delay"]
        cfg.sensor_dropout_probability = stage4["dropout"]
        cfg.sensor_stale_probability = stage4["stale"]
        cfg.actuator_response_scale_range = stage4["response"]
        cfg.actuator_bias_range_rad = stage4["actuator_bias"]
        cfg.actuator_deadband_range_rad = stage4["deadband"]
        cfg.actuator_backlash_range_rad = stage4["backlash"]
        cfg.pick_speed_range_m_s = stage4["pick_speed"]
        cfg.pick_direction_tilt_max_rad = math.radians(stage4["pick_tilt_deg"])
        cfg.pick_speed_m_s = 0.015
        cfg.pick_twist_rate_rad_s = 0.10
        cfg.pick_twist_angle_rad = 0.15
        cfg.stem_tensile_limit_n = stage4["stem"]
        cfg.stem_critical_duration_s = stage4["stem_duration"]
    if args_cli.enveloping:
        # The legacy stage switch above is intentionally left untouched for
        # backward compatibility.  Override only the task-specific G0/G1
        # distribution here so the two training namespaces cannot drift.
        cfg.training_stage = 1 if args_cli.stage <= 1 else 2
        cfg.strict_enveloping_enabled = True
        cfg.strict_enveloping_require_stem_break = cfg.training_stage >= 2
        cfg.strict_angular_coverage_min_deg = args_cli.enveloping_coverage_deg
        cfg.strict_hold_steps = args_cli.enveloping_hold_steps
        cfg.strict_palm_first_margin_steps = args_cli.enveloping_palm_margin
        course = args_cli.enveloping_course
        # Stage-4 bands are the explicit domain-randomization curriculum.
        # Do not let the legacy G0/G1/final course aliases overwrite the
        # band geometry and perturbation ranges that were just selected
        # above; doing so silently turns band 1 into a much larger jump.
        # The aliases remain useful for stages 1--3 and for backwards
        # compatible one-shot runs.
        apply_course_geometry = args_cli.stage < 4
        if apply_course_geometry and course == "g0":
            cfg.events["tomato_geometry"].params["axis_range_m"] = (0.040, 0.040)
            cfg.events["tomato_geometry"].params["isotropic_axes"] = True
        elif apply_course_geometry and course == "g1":
            cfg.events["tomato_geometry"].params["axis_range_m"] = (0.040, 0.048)
            cfg.events["tomato_geometry"].params["isotropic_axes"] = False
        elif apply_course_geometry:
            cfg.events["tomato_geometry"].params["axis_range_m"] = (0.038, 0.050)
            cfg.events["tomato_geometry"].params["isotropic_axes"] = False
        cfg.events["tomato_geometry"].params["accepted_mass_kg"] = (0.02, 1.20)
        if apply_course_geometry and course == "g0":
            cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (850.0, 850.0)
            cfg.position_jitter = (0.0005, 0.0005, 0.00025)
            cfg.friction_range = (0.70, 0.70)
            cfg.restitution_range = (0.0, 0.0)
            cfg.stiffness_range_n_m = (18000.0, 18000.0)
            cfg.tactile_gain_range = (1.0, 1.0)
            cfg.tactile_bias_range_n = (0.0, 0.0)
            cfg.sensor_delay_steps_max = 0
            cfg.sensor_dropout_probability = 0.0
            cfg.sensor_stale_probability = 0.0
        elif apply_course_geometry and course == "g1":
            cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (780.0, 1050.0)
            cfg.position_jitter = (0.0015, 0.0015, 0.00075)
            cfg.friction_range = (0.55, 0.85)
            cfg.restitution_range = (0.0, 0.03)
            cfg.stiffness_range_n_m = (9000.0, 24000.0)
            cfg.tactile_gain_range = (0.96, 1.04)
            cfg.tactile_bias_range_n = (-0.008, 0.008)
            cfg.sensor_delay_steps_max = 1
            cfg.sensor_dropout_probability = 0.003
            cfg.sensor_stale_probability = 0.003
        elif apply_course_geometry:
            cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (700.0, 1050.0)
            cfg.position_jitter = (0.0040, 0.0040, 0.0020)
            cfg.friction_range = (0.25, 0.90)
            cfg.restitution_range = (0.0, 0.06)
            cfg.stiffness_range_n_m = (2500.0, 28000.0)
            cfg.tactile_gain_range = (0.82, 1.18)
            cfg.tactile_bias_range_n = (-0.03, 0.03)
            cfg.sensor_delay_steps_max = 3
            cfg.sensor_dropout_probability = 0.02
            cfg.sensor_stale_probability = 0.02
        cfg.command_rate_limit_rad_s = 0.35
        cfg.episode_length_s = 12.0
        # Large-fruit palm support carries a substantially higher normal load
        # than the archived small-fruit pinch curriculum.  These are only
        # bootstrap training ceilings; the independent verifier keeps its
        # lower per-digit pressure limits for final evaluation.
        cfg.safe_force_n = args_cli.enveloping_safe_force
        cfg.force_ceiling_n = (
            args_cli.enveloping_safe_force,
            0.8125 * args_cli.enveloping_safe_force,
            0.8125 * args_cli.enveloping_safe_force,
            0.6875 * args_cli.enveloping_safe_force,
            0.6875 * args_cli.enveloping_safe_force,
        )
        cfg.severe_overload_scale = 1.40
        cfg.stem_gravity_compensation = args_cli.enveloping_stem_gravity_compensation
        cfg.pick_direction_z_sign = args_cli.enveloping_pick_direction_z_sign
        cfg.tactile_reflex_enabled = args_cli.enveloping_tactile_reflex
        cfg.retention_only = args_cli.enveloping_retention_only
        cfg.adaptive_retention_reflex_enabled = args_cli.enveloping_adaptive_retention
        cfg.post_break_grace_steps = max(0, int(args_cli.enveloping_postbreak_grace_steps))
        if args_cli.retention_seed_bank is not None:
            seed_bank = torch.load(
                args_cli.retention_seed_bank.expanduser().resolve(),
                map_location="cpu",
                weights_only=False,
            )
            seed_positions = seed_bank.get("joint_pos") if isinstance(seed_bank, dict) else None
            if not isinstance(seed_positions, torch.Tensor):
                raise ValueError("retention seed bank must contain a joint_pos tensor")
            cfg.retention_seed_joint_positions = seed_positions
        if args_cli.enveloping_postbreak_contact_weight is not None:
            cfg.post_break_contact_reward_weight = args_cli.enveloping_postbreak_contact_weight
        if args_cli.enveloping_postbreak_support_weight is not None:
            cfg.post_break_support_reward_weight = args_cli.enveloping_postbreak_support_weight
        if args_cli.enveloping_postbreak_missing_weight is not None:
            cfg.post_break_missing_digit_penalty_weight = args_cli.enveloping_postbreak_missing_weight
        if args_cli.enveloping_drop_penalty is not None:
            cfg.drop_penalty_weight = args_cli.enveloping_drop_penalty
        if args_cli.enveloping_pick_speed is not None:
            if args_cli.enveloping_pick_speed <= 0.0:
                raise ValueError("--enveloping-pick-speed must be positive")
            cfg.pick_speed_m_s = float(args_cli.enveloping_pick_speed)
            cfg.pick_speed_range_m_s = (float(args_cli.enveloping_pick_speed), float(args_cli.enveloping_pick_speed))
        if args_cli.enveloping_pick_displacement is not None:
            if args_cli.enveloping_pick_displacement <= 0.0:
                raise ValueError("--enveloping-pick-displacement must be positive")
            cfg.pick_displacement_m = float(args_cli.enveloping_pick_displacement)
        if args_cli.enveloping_stem_limit is not None:
            if args_cli.enveloping_stem_limit <= 0.0:
                raise ValueError("--enveloping-stem-limit must be positive")
            cfg.stem_tensile_limit_n = (float(args_cli.enveloping_stem_limit), float(args_cli.enveloping_stem_limit))
        if args_cli.enveloping_stem_duration is not None:
            if args_cli.enveloping_stem_duration <= 0.0:
                raise ValueError("--enveloping-stem-duration must be positive")
            cfg.stem_critical_duration_s = (float(args_cli.enveloping_stem_duration), float(args_cli.enveloping_stem_duration))
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    env = (EnvelopingPhysxTomatoGraspEnv(cfg, render_mode=None)
           if args_cli.enveloping else PhysxTomatoGraspEnv(cfg, render_mode=None))
    print("[tomato-ppo] env constructed", flush=True)
    observation, _ = env.reset()
    print("[tomato-ppo] env reset", flush=True)
    actor_observation = observation["policy"]
    critic_state = env._get_states()
    print(f"[tomato-ppo] state ready actor={tuple(actor_observation.shape)} critic={tuple(critic_state.shape)}", flush=True)
    device = actor_observation.device
    policy = AsymmetricActorCritic(actor_observation.shape[-1], critic_state.shape[-1], cfg.action_space, args_cli.minimum_std, args_cli.initial_std, args_cli.coarse_initial_std, args_cli.residual_initial_std).to(device)
    controller = None
    if args_cli.enveloping and args_cli.scripted_action_prior:
        raise ValueError("--scripted-action-prior belongs to the archived 10D teacher; use --teacher-data for 14D")
    if args_cli.scripted_action_prior:
        if args_cli.bc_initialize is not None:
            raise ValueError("scripted action prior uses a zero residual actor and is incompatible with --bc-initialize")
        if not 0.0 < args_cli.scripted_residual_scale <= 0.25:
            raise ValueError("--scripted-residual-scale must be in (0, 0.25]")
        nn.init.zeros_(policy.actor_mean.weight)
        nn.init.zeros_(policy.actor_mean.bias)
        controller = Stage0ScriptedController(
            args_cli.num_envs,
            device,
            Stage0ControllerCfg(
                contact_threshold_n=cfg.contact_threshold_n,
                palm_support_force_n=cfg.palm_support_force_n,
                target_force_n=cfg.target_force_n,
                prelatch_seek_increment=0.003,
                seek_increment=0.001,
                retention_seek_increment=0.002,
                missing_hysteresis_steps=4,
                trim_limit=0.8,
            ),
        )
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.learning_rate)
    start_update = 0
    total_steps = 0
    if args_cli.bc_initialize is not None:
        if args_cli.enveloping:
            raise ValueError("legacy --bc-initialize is 10D; use --teacher-data for the enveloping task")
        bc = torch.load(args_cli.bc_initialize.expanduser().resolve(), map_location=device, weights_only=False)
        if (bc.get("actor_observation_dim"), bc.get("action_dim"), bc.get("converged")) != (110, 10, True):
            raise ValueError("BC checkpoint is not a converged 110D/10D initializer")
        actor_state = bc.get("actor_state_dict")
        if not isinstance(actor_state, dict):
            raise ValueError("BC checkpoint is missing actor_state_dict")
        missing, unexpected = policy.load_state_dict(actor_state, strict=False)
        if unexpected or any(name.startswith(("actor_body", "actor_mean")) for name in missing):
            raise ValueError(f"BC actor state mismatch: missing={missing}, unexpected={unexpected}")
        print(f"[tomato-ppo] initialized actor from BC: {args_cli.bc_initialize}", flush=True)
    bc_reference = None
    if args_cli.bc_anchor_coef > 0.0:
        if args_cli.bc_initialize is None or args_cli.bc_anchor_updates <= 0:
            raise ValueError("BC anchoring requires --bc-initialize and positive --bc-anchor-updates")
        bc_reference = copy.deepcopy(policy).eval()
        for parameter in bc_reference.parameters():
            parameter.requires_grad_(False)
    if args_cli.teacher_data is not None:
        if not args_cli.enveloping:
            raise ValueError("--teacher-data is only valid with --enveloping")
        if args_cli.teacher_bc_epochs <= 0:
            raise ValueError("--teacher-bc-epochs must be positive")
        teacher_data = torch.load(args_cli.teacher_data.expanduser().resolve(), map_location=device, weights_only=False)
        teacher_obs = teacher_data.get("observation")
        teacher_action = teacher_data.get("action")
        if not isinstance(teacher_obs, torch.Tensor) or not isinstance(teacher_action, torch.Tensor):
            raise ValueError("teacher dataset must contain observation and action tensors")
        if tuple(teacher_obs.shape[1:]) != (110,) or tuple(teacher_action.shape[1:]) != (14,):
            raise ValueError("teacher dataset does not match the 110D/14D contract")
        teacher_obs = teacher_obs.to(device=device, dtype=torch.float32)
        teacher_action = teacher_action.to(device=device, dtype=torch.float32).clamp(-0.98, 0.98)
        teacher_batch = min(2048, teacher_obs.shape[0])
        teacher_optimizer = torch.optim.Adam(
            list(policy.actor_body.parameters()) + list(policy.actor_mean.parameters()),
            lr=max(args_cli.learning_rate, 3.0e-4),
        )
        policy.train()
        for _ in range(args_cli.teacher_bc_epochs):
            permutation = torch.randperm(teacher_obs.shape[0], device=device)
            for start in range(0, teacher_obs.shape[0], teacher_batch):
                indices = permutation[start : start + teacher_batch]
                predicted = torch.tanh(policy.actor_mean(policy.actor_body(teacher_obs[indices])))
                loss = (predicted - teacher_action[indices]).square().mean()
                teacher_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                teacher_optimizer.step()
        print(f"[tomato-ppo] initialized 14D actor from teacher dataset: {args_cli.teacher_data}", flush=True)
    if args_cli.resume is not None:
        checkpoint = torch.load(args_cli.resume.expanduser().resolve(), map_location=device, weights_only=False)
        expected_action_dim = 14 if args_cli.enveloping else 10
        if (checkpoint.get("actor_observation_dim"), checkpoint.get("critic_state_dim"), checkpoint.get("action_dim")) != (110, 163, expected_action_dim):
            raise ValueError(f"checkpoint does not match the expected 110D/163D/{expected_action_dim}D contract")
        policy.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Resume model and Adam moments, but honor the learning rate selected
        # for the new curriculum/fine-tuning run.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args_cli.learning_rate
        if args_cli.enveloping:
            # A curriculum resume must be able to lower exploration; loading
            # the previous log_std unchanged can reintroduce violent closure
            # noise precisely when randomization is widened.
            policy.log_std.data.fill_(math.log(args_cli.initial_std))
        start_update = int(checkpoint.get("update", 0))
        total_steps = int(checkpoint.get("total_steps", 0))
    if args_cli.enveloping and args_cli.enveloping_distal_init != 0.0:
        if not -1.0 <= args_cli.enveloping_distal_init <= 1.0:
            raise ValueError("--enveloping-distal-init must lie in [-1, 1]")
        # This is a curriculum prior, not a fixed pose: the actor still sees
        # tactile/joint feedback and PPO can move each residual independently.
        # A small positive bias prevents the new distal channels from being
        # permanently silent while the four-finger participation signal is
        # still sparse.
        policy.actor_mean.bias.data[10:] += float(args_cli.enveloping_distal_init)
    policy_reference = None
    if args_cli.policy_anchor_coef > 0.0:
        if args_cli.resume is None or args_cli.policy_anchor_updates <= 0:
            raise ValueError("policy anchoring requires --resume and positive --policy-anchor-updates")
        policy_reference = copy.deepcopy(policy).eval()
        for parameter in policy_reference.parameters():
            parameter.requires_grad_(False)
    size_closure_reference = None
    if args_cli.size_closure_anchor_coef > 0.0:
        if args_cli.resume is None or args_cli.size_closure_anchor_updates <= 0:
            raise ValueError("size-closure anchoring requires --resume and positive anchor updates")
        if args_cli.size_closure_gain <= 0.0 or not 0.0 <= args_cli.size_closure_threshold <= 1.0:
            raise ValueError("invalid size-closure gain or threshold")
        size_closure_reference = copy.deepcopy(policy).eval()
        for parameter in size_closure_reference.parameters():
            parameter.requires_grad_(False)

    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    run = None
    if args_cli.wandb_mode != "disabled":
        try:
            import wandb
            run = wandb.init(project=args_cli.wandb_project, name=args_cli.wandb_run_name, mode=args_cli.wandb_mode, dir=str(output_dir), config={"stage": args_cli.stage, "task": "palm_supported_enveloping" if args_cli.enveloping else "legacy_tomato", "actor_dim": 110, "critic_dim": 163, "action_dim": int(cfg.action_space), "num_envs": args_cli.num_envs, "horizon": args_cli.horizon, "device": str(device)})
            print(f"[tomato-ppo] wandb_url={run.url}", flush=True)
        except Exception as exc:
            print(f"[tomato-ppo] W&B unavailable: {type(exc).__name__}: {exc}; local JSONL continues", flush=True)

    stop = {"value": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("value", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("value", True))
    started = time.time()
    latest_metrics: dict[str, float] = {}
    best_score = (-1, -1.0, float("-inf"))
    try:
        if args_cli.eval_only:
            latest_metrics = {
                "update": start_update,
                "total_steps": total_steps,
                "stage": args_cli.stage,
                **evaluate(env, policy, args_cli.eval_steps, controller),
            }
            (output_dir / "evaluation.json").write_text(
                json.dumps(latest_metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(latest_metrics, flush=True)
            return
        for update in range(start_update + 1, start_update + args_cli.updates + 1):
            obs_buf = torch.zeros((args_cli.horizon, args_cli.num_envs, 110), device=device)
            state_buf = torch.zeros((args_cli.horizon, args_cli.num_envs, 163), device=device)
            action_buf = torch.zeros((args_cli.horizon, args_cli.num_envs, cfg.action_space), device=device)
            logp_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), device=device)
            reward_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), device=device)
            done_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), device=device)
            value_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), device=device)
            for step in range(args_cli.horizon):
                obs_buf[step], state_buf[step] = actor_observation, critic_state
                with torch.no_grad():
                    distribution = policy.distribution(actor_observation)
                    raw = distribution.rsample()
                    policy_action = torch.tanh(raw)
                    logp_buf[step] = squashed_log_prob(distribution, raw, policy_action)
                    value_buf[step] = policy.critic(critic_state)
                action_buf[step] = policy_action
                action = policy_action
                if controller is not None:
                    action = (
                        _scripted_prior(env, controller)
                        + args_cli.scripted_residual_scale * policy_action
                    ).clamp(-1.0, 1.0)
                observation, reward, terminated, truncated, _ = env.step(action)
                reward_buf[step] = reward
                done_buf[step] = (terminated | truncated).float()
                actor_observation = observation["policy"]
                critic_state = env._get_states()
                if controller is not None:
                    done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                    if done_ids.numel() > 0:
                        controller.reset(done_ids)
                total_steps += args_cli.num_envs
            with torch.no_grad():
                next_value = policy.critic(critic_state)
                advantage = torch.zeros_like(reward_buf)
                gae = torch.zeros(args_cli.num_envs, device=device)
                for step in reversed(range(args_cli.horizon)):
                    nonterminal = 1.0 - done_buf[step]
                    next_state_value = next_value if step == args_cli.horizon - 1 else value_buf[step + 1]
                    delta = reward_buf[step] + args_cli.gamma * next_state_value * nonterminal - value_buf[step]
                    gae = delta + args_cli.gamma * args_cli.gae_lambda * nonterminal * gae
                    advantage[step] = gae
                returns = advantage + value_buf
            flat_obs = obs_buf.reshape(-1, 110)
            flat_state = state_buf.reshape(-1, 163)
            flat_action = action_buf.reshape(-1, cfg.action_space)
            flat_old_logp = logp_buf.reshape(-1)
            flat_advantage = advantage.reshape(-1)
            flat_advantage = (flat_advantage - flat_advantage.mean()) / (flat_advantage.std(unbiased=False) + 1.0e-8)
            flat_returns = returns.reshape(-1)
            batch_size = flat_obs.shape[0]
            minibatch = min(args_cli.minibatch_size, batch_size)
            actor_loss_sum = critic_loss_sum = entropy_sum = 0.0
            optimization_steps = 0
            for _ in range(args_cli.epochs):
                permutation = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch):
                    indices = permutation[start : start + minibatch]
                    distribution = policy.distribution(flat_obs[indices])
                    bounded = flat_action[indices].clamp(-0.999999, 0.999999)
                    raw = torch.atanh(bounded)
                    ratio = (squashed_log_prob(distribution, raw, bounded) - flat_old_logp[indices]).exp()
                    clipped = ratio.clamp(1.0 - args_cli.clip_eps, 1.0 + args_cli.clip_eps)
                    actor_loss = -torch.minimum(ratio * flat_advantage[indices], clipped * flat_advantage[indices]).mean()
                    critic_loss = 0.5 * (policy.critic(flat_state[indices]) - flat_returns[indices]).square().mean()
                    entropy = distribution.entropy().sum(-1).mean()
                    actor_frozen = (update - start_update) <= args_cli.actor_freeze_updates
                    loss = args_cli.value_coef * critic_loss
                    if not actor_frozen:
                        loss = loss + actor_loss - args_cli.entropy_coef * entropy
                    if bc_reference is not None and not actor_frozen:
                        anchor_fraction = max(
                            0.0,
                            1.0 - (update - start_update - 1) / float(args_cli.bc_anchor_updates),
                        )
                        with torch.no_grad():
                            reference_mean = torch.tanh(
                                bc_reference.actor_mean(bc_reference.actor_body(flat_obs[indices]))
                            )
                        current_mean = torch.tanh(distribution.mean)
                        loss = loss + (
                            args_cli.bc_anchor_coef
                            * anchor_fraction
                            * (current_mean - reference_mean).square().mean()
                        )
                    if policy_reference is not None and not actor_frozen:
                        anchor_fraction = max(
                            0.0,
                            1.0 - (update - start_update - 1) / float(args_cli.policy_anchor_updates),
                        )
                        with torch.no_grad():
                            reference_mean = torch.tanh(
                                policy_reference.actor_mean(policy_reference.actor_body(flat_obs[indices]))
                            )
                        current_mean = torch.tanh(distribution.mean)
                        loss = loss + (
                            args_cli.policy_anchor_coef
                            * anchor_fraction
                            * (current_mean - reference_mean).square().mean()
                        )
                    if size_closure_reference is not None and not actor_frozen:
                        closure_fraction = max(
                            0.0,
                            1.0 - (update - start_update - 1) / float(args_cli.size_closure_anchor_updates),
                        )
                        with torch.no_grad():
                            reference_mean = torch.tanh(
                                size_closure_reference.actor_mean(
                                    size_closure_reference.actor_body(flat_obs[indices])
                                )
                            )
                            # Actor dimensions 3:6 are the deployable,
                            # normalized semi-axis estimate. Smaller observed
                            # fruit receives only a global-flexion offset; no
                            # simulator-only force or target pose is exposed.
                            observed_size = flat_obs[indices, 3:6].mean(dim=-1)
                            closure_offset = args_cli.size_closure_gain * (
                                args_cli.size_closure_threshold - observed_size
                            ).clamp_min(0.0)
                            closure_target = (
                                reference_mean[:, 0] + closure_offset
                            ).clamp(-1.0, 1.0)
                        current_global_flexion = torch.tanh(distribution.mean[:, 0])
                        loss = loss + (
                            args_cli.size_closure_anchor_coef
                            * closure_fraction
                            * (current_global_flexion - closure_target).square().mean()
                        )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    actor_loss_sum += float(actor_loss.detach())
                    critic_loss_sum += float(critic_loss.detach())
                    entropy_sum += float(entropy.detach())
                    optimization_steps += 1
            latest_metrics = {
                "update": update,
                "total_steps": total_steps,
                "stage": args_cli.stage,
                "train/reward_mean": float(reward_buf.mean()),
                "train/return_mean": float(reward_buf.sum(0).mean()),
                "train/actor_loss": actor_loss_sum / max(1, optimization_steps),
                "train/critic_loss": critic_loss_sum / max(1, optimization_steps),
                "train/entropy": entropy_sum / max(1, optimization_steps),
                "train/std_mean": float(policy.log_std.clamp_min(policy.minimum_log_std).exp().mean()),
                "train/coarse_std_mean": float(policy.log_std[:5].clamp_min(policy.minimum_log_std).exp().mean()),
                "train/residual_std_mean": float(policy.log_std[5:].clamp_min(policy.minimum_log_std).exp().mean()),
                "train/actor_frozen": float((update - start_update) <= args_cli.actor_freeze_updates),
                "elapsed_seconds": time.time() - started,
            }
            if args_cli.eval_interval > 0 and update % args_cli.eval_interval == 0:
                latest_metrics.update(evaluate(env, policy, args_cli.eval_steps, controller))
                observation, _ = env.reset()
                if controller is not None:
                    controller.reset()
                actor_observation = observation["policy"]
                critic_state = env._get_states()
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(latest_metrics, sort_keys=True) + "\n")
            if run is not None:
                run.log(latest_metrics, step=update)
            save_checkpoint(output_dir / "latest.pt", policy, optimizer, update, total_steps, latest_metrics)
            evaluated_success = latest_metrics.get("eval/success_rate")
            if evaluated_success is not None:
                evaluated_overload = float(latest_metrics.get("eval/overload_rate", 1.0))
                # Stage 4's declared gate is two-dimensional. Prefer a safe
                # checkpoint even if an unsafe one has marginally higher raw
                # success; retain every evaluated checkpoint for audit.
                safe = int(args_cli.stage < 4 or evaluated_overload < 0.02)
                score = (safe, float(evaluated_success), -evaluated_overload)
                save_checkpoint(
                    output_dir / f"eval_update_{update}.pt",
                    policy,
                    optimizer,
                    update,
                    total_steps,
                    latest_metrics,
                )
                if score > best_score:
                    best_score = score
                    save_checkpoint(output_dir / "best.pt", policy, optimizer, update, total_steps, latest_metrics)
            if update == start_update + 1 or update % 10 == 0:
                print(latest_metrics, flush=True)
            if stop["value"]:
                break
    finally:
        if run is not None:
            run.finish()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
