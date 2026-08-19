"""Independent randomized evaluation for the 14D palm-supported policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=20260820)
parser.add_argument("--coverage-deg", type=float, default=200.0)
parser.add_argument("--hold-steps", type=int, default=10)
parser.add_argument("--axis-low", type=float, default=0.038)
parser.add_argument("--axis-high", type=float, default=0.050)
parser.add_argument("--safe-force", type=float, default=12.0)
parser.add_argument("--stage", type=int, choices=(1, 2), default=1,
                    help="1=grasp-only verifier; 2=also require stem break")
parser.add_argument("--stem-gravity-compensation", action="store_true",
                    help="carry the attached fruit weight in the stem (Clean-v5 baseline)")
parser.add_argument("--pick-direction-z-sign", type=float, choices=(-1.0, 1.0), default=-1.0,
                    help="plant-anchor motion sign in the hand-fixed frame")
parser.add_argument("--stem-tensile-limit", type=float, default=None,
                    help="override the sampled stem tensile limit (N)")
parser.add_argument("--stem-critical-duration", type=float, default=None,
                    help="override stem damage critical duration (s)")
parser.add_argument("--tactile-reflex", action="store_true",
                    help="enable bounded post-break distal tactile reflex")
parser.add_argument("--tactile-reflex-gain", type=float, default=None,
                    help="override tactile reflex gain (rad/N)")
parser.add_argument("--tactile-reflex-max", type=float, default=None,
                    help="override tactile reflex maximum correction (rad)")
parser.add_argument("--adaptive-retention-reflex", action="store_true",
                    help="enable low-pass per-digit force/slip retention reflex")
parser.add_argument("--adaptive-retention-force-gain", type=float, default=None,
                    help="adaptive retention force gain (rad/N)")
parser.add_argument("--adaptive-retention-slip-gain", type=float, default=None,
                    help="adaptive retention slip gain (rad/(m/s))")
parser.add_argument("--adaptive-retention-max", type=float, default=None,
                    help="adaptive retention maximum correction (rad)")
parser.add_argument("--pick-speed", type=float, default=0.02,
                    help="fixed plant-anchor pick speed (m/s)")
parser.add_argument("--pick-displacement", type=float, default=0.035,
                    help="fixed plant-anchor pick displacement (m)")
parser.add_argument("--nominal-physics", action="store_true",
                    help="disable evaluator material/sensor/actuator randomization")
parser.add_argument("--post-break-target-latch", action="store_true",
                    help="blend post-break commands toward the last stable target")
parser.add_argument("--post-break-command-mix", type=float, default=0.15)
parser.add_argument("--post-break-grace-steps", type=int, default=0,
                    help="bounded post-detachment reaction window in policy steps")
parser.add_argument("--effort-limit", type=float, default=None)
parser.add_argument("--actuator-stiffness", type=float, default=None)
parser.add_argument("--actuator-damping", type=float, default=None)
parser.add_argument("--retention-only", action="store_true",
                    help="start each episode from a seeded post-break enclosure")
parser.add_argument("--retention-seed-bank", type=Path, default=None,
                    help="optional Kx16 joint-state bank for retention-only resets")
parser.add_argument("--disable-joint-projection", action="store_true",
                    help="diagnostic: do not overwrite measured DOF pose when it crosses the safe clamp")
parser.add_argument("--disable-postbreak-joint-projection", action="store_true",
                    help="diagnostic: keep pre-pick projection but let post-break contacts settle naturally")
parser.add_argument("--fixed-action", type=str, default=None,
                    help="comma-separated 14D action for controller-free physics probing")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch
from torch import nn
from torch.distributions import Normal

from simulation.ppo_tomato_grasp.physx_enveloping_env import (
    EnvelopingPhysxTomatoGraspEnv,
    EnvelopingPhysxTomatoGraspEnvCfg,
)


class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(110, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.actor_mean = nn.Linear(128, 14)
        self.log_std = nn.Parameter(torch.zeros(14))

    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.actor_mean(self.actor_body(obs)), self.log_std.exp())


def make_cfg() -> EnvelopingPhysxTomatoGraspEnvCfg:
    cfg = EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.training_stage = args_cli.stage
    cfg.strict_enveloping_enabled = True
    cfg.strict_angular_coverage_min_deg = args_cli.coverage_deg
    cfg.strict_hold_steps = args_cli.hold_steps
    cfg.strict_enveloping_require_stem_break = args_cli.stage >= 2
    cfg.stem_gravity_compensation = args_cli.stem_gravity_compensation
    cfg.pick_direction_z_sign = args_cli.pick_direction_z_sign
    cfg.pick_speed_m_s = args_cli.pick_speed
    cfg.pick_speed_range_m_s = (args_cli.pick_speed, args_cli.pick_speed)
    cfg.pick_displacement_m = args_cli.pick_displacement
    cfg.tactile_reflex_enabled = args_cli.tactile_reflex
    cfg.adaptive_retention_reflex_enabled = args_cli.adaptive_retention_reflex
    cfg.post_break_target_latch = args_cli.post_break_target_latch
    cfg.post_break_command_mix = args_cli.post_break_command_mix
    cfg.post_break_grace_steps = max(0, int(args_cli.post_break_grace_steps))
    cfg.retention_only = args_cli.retention_only
    if args_cli.retention_seed_bank is not None:
        bank = torch.load(args_cli.retention_seed_bank.expanduser().resolve(), map_location="cpu", weights_only=False)
        seed_positions = bank.get("joint_pos") if isinstance(bank, dict) else None
        if not isinstance(seed_positions, torch.Tensor):
            raise ValueError("retention seed bank must contain a joint_pos tensor")
        cfg.retention_seed_joint_positions = seed_positions
    cfg.joint_projection_enabled = not args_cli.disable_joint_projection
    cfg.joint_projection_after_break = not args_cli.disable_postbreak_joint_projection
    if args_cli.tactile_reflex_gain is not None:
        cfg.tactile_reflex_gain_rad_per_n = args_cli.tactile_reflex_gain
    if args_cli.tactile_reflex_max is not None:
        cfg.tactile_reflex_max_correction_rad = args_cli.tactile_reflex_max
    if args_cli.adaptive_retention_force_gain is not None:
        cfg.adaptive_retention_force_gain_rad_per_n = args_cli.adaptive_retention_force_gain
    if args_cli.adaptive_retention_slip_gain is not None:
        cfg.adaptive_retention_slip_gain_rad_per_m_s = args_cli.adaptive_retention_slip_gain
    if args_cli.adaptive_retention_max is not None:
        cfg.adaptive_retention_max_correction_rad = args_cli.adaptive_retention_max
    if args_cli.stem_tensile_limit is not None:
        cfg.stem_tensile_limit_n = (args_cli.stem_tensile_limit, args_cli.stem_tensile_limit)
    if args_cli.stem_critical_duration is not None:
        cfg.stem_critical_duration_s = (args_cli.stem_critical_duration, args_cli.stem_critical_duration)
    cfg.events["tomato_geometry"].params["axis_range_m"] = (args_cli.axis_low, args_cli.axis_high)
    cfg.events["tomato_geometry"].params["isotropic_axes"] = False
    cfg.events["tomato_geometry"].params["accepted_mass_kg"] = (0.02, 1.20)
    cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (700.0, 1050.0)
    cfg.position_jitter = (0.004, 0.004, 0.002)
    cfg.friction_range = (0.25, 0.90)
    cfg.restitution_range = (0.0, 0.06)
    cfg.stiffness_range_n_m = (2500.0, 28000.0)
    cfg.tactile_gain_range = (0.82, 1.18)
    cfg.tactile_bias_range_n = (-0.03, 0.03)
    cfg.sensor_delay_steps_max = 3
    cfg.sensor_dropout_probability = 0.02
    cfg.sensor_stale_probability = 0.02
    cfg.actuator_response_scale_range = (0.72, 1.08)
    cfg.actuator_bias_range_rad = (-0.012, 0.012)
    cfg.actuator_deadband_range_rad = (0.0, 0.010)
    cfg.actuator_backlash_range_rad = (0.0, 0.015)
    # Evaluation uses a separate, explicitly reported force prior.  It is
    # not silently substituted for real-hardware safety limits.
    cfg.safe_force_n = args_cli.safe_force
    cfg.force_ceiling_n = (args_cli.safe_force, 0.85 * args_cli.safe_force,
                           0.85 * args_cli.safe_force, 0.75 * args_cli.safe_force,
                           0.75 * args_cli.safe_force)
    cfg.severe_overload_scale = 1.40
    cfg.episode_length_s = 12.0
    actuator = cfg.robot_cfg.actuators["left_hand"]
    if args_cli.effort_limit is not None:
        actuator.effort_limit_sim = args_cli.effort_limit
    if args_cli.actuator_stiffness is not None:
        actuator.stiffness = args_cli.actuator_stiffness
    if args_cli.actuator_damping is not None:
        actuator.damping = args_cli.actuator_damping
    if args_cli.nominal_physics:
        cfg.position_jitter = (0.0, 0.0, 0.0)
        cfg.friction_range = (0.70, 0.70)
        cfg.restitution_range = (0.0, 0.0)
        cfg.stiffness_range_n_m = (18000.0, 18000.0)
        cfg.tactile_gain_range = (1.0, 1.0)
        cfg.tactile_bias_range_n = (0.0, 0.0)
        cfg.sensor_delay_steps_max = 0
        cfg.sensor_dropout_probability = 0.0
        cfg.sensor_stale_probability = 0.0
        cfg.actuator_response_scale_range = (1.0, 1.0)
        cfg.actuator_bias_range_rad = (0.0, 0.0)
        cfg.actuator_deadband_range_rad = (0.0, 0.0)
        cfg.actuator_backlash_range_rad = (0.0, 0.0)
    return cfg


@torch.inference_mode()
def main() -> None:
    if args_cli.num_envs <= 0 or args_cli.steps <= 0:
        raise ValueError("num-envs and steps must be positive")
    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=args_cli.device, weights_only=False)
    if (checkpoint.get("actor_observation_dim"), checkpoint.get("action_dim")) != (110, 14):
        raise ValueError("checkpoint does not match the 110D/14D contract")
    device = torch.device(args_cli.device)
    actor = Actor().to(device)
    actor.load_state_dict(checkpoint["model"], strict=False)
    actor.eval()
    fixed_action = None
    if args_cli.fixed_action is not None:
        values = [float(v.strip()) for v in args_cli.fixed_action.split(",") if v.strip()]
        if len(values) != 14:
            raise ValueError("--fixed-action requires exactly 14 comma-separated values")
        fixed_action = torch.tensor(values, device=device).clamp(-1.0, 1.0).expand(args_cli.num_envs, -1)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    env = EnvelopingPhysxTomatoGraspEnv(make_cfg())
    obs, _ = env.reset(seed=args_cli.seed)
    completed = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    success = torch.zeros_like(completed)
    drop = torch.zeros_like(completed)
    overload = torch.zeros_like(completed)
    damage = torch.zeros_like(completed)
    timeout = torch.zeros_like(completed)
    drop_before_pick = torch.zeros_like(completed)
    drop_after_pick = torch.zeros_like(completed)
    drop_after_break = torch.zeros_like(completed)
    pick_started = torch.zeros_like(completed)
    stem_broken = torch.zeros_like(completed)
    post_break_hold = torch.zeros_like(completed)
    five_digit = torch.zeros_like(completed)
    palm_first_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    distal_all_seen = torch.zeros_like(palm_first_seen)
    max_force = torch.zeros((env.num_envs, 5), device=device)
    max_palm = torch.zeros(env.num_envs, device=device)
    max_coverage = torch.zeros(env.num_envs, device=device)
    strict_hold_max = torch.zeros(env.num_envs, device=device)
    measured_geometry_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    measured_geometry_rate_max = 0.0
    try:
        for _ in range(args_cli.steps):
            action = fixed_action if fixed_action is not None else torch.tanh(actor.actor_mean(actor.actor_body(obs["policy"])))
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            code = env.last_termination_code
            completed += done.long()
            success += (done & (code == 1)).long()
            drop += (done & (code == 2)).long()
            overload += (done & (code == 3)).long()
            damage += (done & (code == 4)).long()
            timeout += (done & (code == 5)).long()
            drop_before_pick += (done & env.last_drop_before_pick).long()
            drop_after_pick += (done & env.last_drop_after_pick).long()
            drop_after_break += (done & env.last_drop_after_break).long()
            pick_started |= env.pick_started
            stem_broken |= env.stem_broke
            post_break_hold |= env.stem_broke & (env.stable_steps >= env.cfg.success_hold_steps)
            five_digit += (done & (env.contact_forces >= env.cfg.contact_threshold_n).all(-1)).long()
            max_force = torch.maximum(max_force, env.contact_forces)
            max_palm = torch.maximum(max_palm, env.palm_force)
            if isinstance(info, dict):
                log = info.get("log", {})
                palm_first_seen |= torch.full_like(palm_first_seen, float(log.get("strict_palm_first_rate", 0.0)) > 0.0)
                distal_all_seen |= torch.full_like(distal_all_seen, float(log.get("strict_distal_all_rate", 0.0)) > 0.0)
                max_coverage = torch.maximum(max_coverage, torch.full_like(max_coverage, float(log.get("strict_coverage_max_deg_proxy", 0.0))))
                strict_hold_max = torch.maximum(strict_hold_max, torch.full_like(strict_hold_max, float(log.get("strict_hold_max", 0.0))))
                measured_rate = float(log.get("strict_contact_geometry_measured_rate", 0.0))
                measured_geometry_rate_max = max(measured_geometry_rate_max, measured_rate)
                if measured_rate > 0.0:
                    # The scalar is an aggregate over the active batch.  It
                    # is sufficient to distinguish real PhysX contact-point
                    # data from the kinematic tip fallback in this report;
                    # per-episode strict success remains the authoritative
                    # metric below.
                    measured_geometry_seen |= env.strict_contact_geometry_measured.any(-1)
            if not bool((~done).any()):
                break
        denominator = completed.sum().clamp_min(1).float()
        report = {
            "version": "enveloping_random_eval_v2",
            "checkpoint": str(checkpoint_path),
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "axis_range_m": [args_cli.axis_low, args_cli.axis_high],
            "nominal_physics": bool(args_cli.nominal_physics),
            "coverage_gate_deg": args_cli.coverage_deg,
            "hold_gate_steps": args_cli.hold_steps,
            "safe_force_prior_n": args_cli.safe_force,
            "post_break_grace_steps": int(args_cli.post_break_grace_steps),
            "completed_episodes": int(completed.sum()),
            "strict_envelop_success_rate": float(success.sum() / denominator),
            "drop_rate": float(drop.sum() / denominator),
            "overload_rate": float(overload.sum() / denominator),
            "damage_rate": float(damage.sum() / denominator),
            "timeout_rate": float(timeout.sum() / denominator),
            "drop_before_pick_rate": float(drop_before_pick.sum() / denominator),
            "drop_after_pick_rate": float(drop_after_pick.sum() / denominator),
            "drop_after_break_rate": float(drop_after_break.sum() / denominator),
            "pick_triggered_rate": float(pick_started.float().mean()),
            "stem_break_rate": float(stem_broken.float().mean()),
            "post_break_hold_rate": float(post_break_hold.float().mean()),
            "five_digit_current_on_done_rate": float(five_digit.sum() / denominator),
            "palm_first_observed_rate": float(palm_first_seen.float().mean()),
            "distal_all_observed_rate": float(distal_all_seen.float().mean()),
            "coverage_max_deg_proxy": float(max_coverage.max()),
            "strict_hold_max": float(strict_hold_max.max()),
            "contact_geometry_measured_rate_max": measured_geometry_rate_max,
            "contact_geometry_measured_episode_seen_rate": float(measured_geometry_seen.float().mean()),
            "digit_peak_force_n_mean": [float(v) for v in max_force.mean(0)],
            "palm_peak_force_n_mean": float(max_palm.mean()),
            "hardware_command_allowed": False,
            "contact_geometry_proxy": measured_geometry_rate_max <= 0.0,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
