"""Run one independent randomized shard of the Stage-0 PhysX gate.

Each process constructs fresh USD ellipsoids, so size, density, mass and the
per-reset parameters are genuinely resampled between shards.  The parent gate
runner concatenates these first-episode-only JSONL records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--max-steps", type=int, default=180)
parser.add_argument("--settle-steps", type=int, default=0)
parser.add_argument("--residual-span-fraction", type=float, default=0.12)
parser.add_argument("--axis-low-m", type=float, default=0.032)
parser.add_argument("--axis-high-m", type=float, default=0.035)
parser.add_argument("--zero-position-jitter", action="store_true")
parser.add_argument("--linear-damping", type=float, default=None)
parser.add_argument("--angular-damping", type=float, default=None)
parser.add_argument("--pregrasp-stem-scale", type=float, default=None)
parser.add_argument("--friction", type=float, default=0.55)
parser.add_argument("--retention-seek", type=float, default=0.002)
parser.add_argument("--retention-all-digit-increment", type=float, default=0.0)
parser.add_argument("--position-jitter-xy-m", type=float, default=0.002)
parser.add_argument("--prelatch-seek", type=float, default=0.003)
parser.add_argument("--palm-rim-height-m", type=float, default=0.015)
parser.add_argument("--hold-seek", type=float, default=0.001)
parser.add_argument("--missing-hysteresis", type=int, default=4)
parser.add_argument("--trim-limit", type=float, default=0.8)
parser.add_argument("--pick-twist-rate", type=float, default=0.0)
parser.add_argument("--pick-twist-angle", type=float, default=0.0)
parser.add_argument("--pick-speed", type=float, default=0.02)
parser.add_argument("--stem-tensile-limit", type=float, default=0.70)
parser.add_argument("--stem-critical-duration", type=float, default=0.20)
parser.add_argument("--balance-increment", type=float, default=0.0)
parser.add_argument("--force-feedback-gain", type=float, default=0.0)
parser.add_argument("--force-feedback-deadband-n", type=float, default=0.08)
parser.add_argument("--force-feedback-max-increment", type=float, default=0.015)
parser.add_argument("--one-sided-force-feedback", action="store_true")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--capture-env", type=int, default=None)
parser.add_argument("--trajectory-output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np

from simulation.ppo_tomato_grasp.physx_tomato_env import (
    PhysxTomatoGraspEnv,
    PhysxTomatoGraspEnvCfg,
)
from simulation.ppo_tomato_grasp.stage0_scripted_controller import (
    Stage0ControllerCfg,
    Stage0ScriptedController,
)


def main() -> None:
    if args.num_envs <= 0 or args.max_steps <= 0:
        raise ValueError("--num-envs and --max-steps must be positive")
    if (args.capture_env is None) != (args.trajectory_output is None):
        raise ValueError("--capture-env and --trajectory-output must be used together")
    if args.capture_env is not None and not 0 <= args.capture_env < args.num_envs:
        raise ValueError("--capture-env must select one configured environment")
    cfg = PhysxTomatoGraspEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.training_stage = 2
    cfg.residual_span_fraction = args.residual_span_fraction
    cfg.palm_cup_rim_height_m = args.palm_rim_height_m
    cfg.pick_twist_rate_rad_s = args.pick_twist_rate
    cfg.pick_twist_angle_rad = args.pick_twist_angle
    cfg.pick_speed_m_s = args.pick_speed
    if args.linear_damping is not None:
        cfg.tomato_cfg.spawn.rigid_props.linear_damping = args.linear_damping
    if args.angular_damping is not None:
        cfg.tomato_cfg.spawn.rigid_props.angular_damping = args.angular_damping
    if args.pregrasp_stem_scale is not None:
        cfg.pregrasp_stem_wrench_scale = args.pregrasp_stem_scale
    # Stage 0 proves the nominal physical/control chain.  Delay, dropout and
    # stale-frame fault injection belong to the later sim2real curriculum;
    # retaining them here confounds grasp feasibility with sensor-fault
    # robustness.  Contact sensors and their filtering remain fully active.
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.tactile_gain_range = (1.0, 1.0)
    cfg.tactile_bias_range_n = (0.0, 0.0)
    cfg.friction_range = (args.friction, args.friction)
    cfg.restitution_range = (0.0, 0.0)
    cfg.stiffness_range_n_m = (18000.0, 18000.0)
    cfg.stem_tensile_limit_n = (
        args.stem_tensile_limit,
        args.stem_tensile_limit,
    )
    cfg.stem_critical_duration_s = (
        args.stem_critical_duration,
        args.stem_critical_duration,
    )
    cfg.position_jitter = (
        args.position_jitter_xy_m,
        args.position_jitter_xy_m,
        min(0.001, 0.5 * args.position_jitter_xy_m),
    )
    if args.zero_position_jitter:
        cfg.position_jitter = (0.0, 0.0, 0.0)
    if not 0.025 <= args.axis_low_m <= args.axis_high_m <= 0.035:
        raise ValueError("axis range must lie inside [0.025, 0.035]")
    cfg.events["tomato_geometry"].params["axis_range_m"] = (
        args.axis_low_m,
        args.axis_high_m,
    )
    cfg.events["tomato_geometry"].params["isotropic_axes"] = True
    env = PhysxTomatoGraspEnv(cfg)
    try:
        env.reset(seed=args.seed)
        controller = Stage0ScriptedController(
            args.num_envs,
            env.device,
            Stage0ControllerCfg(
                contact_threshold_n=cfg.contact_threshold_n,
                palm_support_force_n=cfg.palm_support_force_n,
                target_force_n=cfg.target_force_n,
                prelatch_seek_increment=args.prelatch_seek,
                seek_increment=args.hold_seek,
                retention_seek_increment=args.retention_seek,
                retention_all_digit_increment=args.retention_all_digit_increment,
                missing_hysteresis_steps=args.missing_hysteresis,
                trim_limit=args.trim_limit,
                retention_balance_increment=args.balance_increment,
                force_feedback_gain_per_n=args.force_feedback_gain,
                force_feedback_deadband_n=args.force_feedback_deadband_n,
                force_feedback_max_increment=args.force_feedback_max_increment,
                one_sided_force_feedback=args.one_sided_force_feedback,
            ),
        )
        # Let the randomized free body establish a quiet palm contact before
        # any enclosure motion.  These steps intentionally command the same
        # valid open-manifold pose used by reset; they are not part of the
        # tactile search or pick budget.
        open_action = controller.open_action
        for _ in range(args.settle_steps):
            env.step(open_action)
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
        done_step = torch.full(
            (args.num_envs,), -1, dtype=torch.long, device=env.device
        )
        done_code = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
        break_seen = torch.zeros_like(alive)
        all_five_seen = torch.zeros_like(alive)
        palm_seen = torch.zeros_like(alive)
        max_hold = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
        peak_digit = torch.zeros((args.num_envs, 5), device=env.device)
        peak_palm = torch.zeros(args.num_envs, device=env.device)
        peak_damage = torch.zeros(args.num_envs, device=env.device)
        max_prepick_steps = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
        max_relative_speed = torch.zeros(args.num_envs, device=env.device)
        pick_started_seen = torch.zeros_like(alive)
        last_center_error = torch.zeros((args.num_envs, 3), device=env.device)
        trajectory: dict[str, list[np.ndarray | float | int | bool]] = {
            "joint_pos_rad": [], "tomato_pos_m": [], "tomato_quat_wxyz": [],
            "action": [], "digit_force_n": [], "palm_force_n": [],
            "stem_damage": [], "stem_broke": [], "pick_started": [],
            "post_break_hold_steps": [],
        }
        initial_center_error = (
            env.tomato.data.root_pos_w - env.grasp_center
        ).clone()
        break_digit_force = torch.full(
            (args.num_envs, 5), float("nan"), device=env.device
        )
        break_palm_force = torch.full(
            (args.num_envs,), float("nan"), device=env.device
        )
        break_center_error = torch.full(
            (args.num_envs, 3), float("nan"), device=env.device
        )
        postbreak_min_digit_force = torch.full(
            (args.num_envs, 5), float("inf"), device=env.device
        )
        postbreak_min_palm_force = torch.full(
            (args.num_envs,), float("inf"), device=env.device
        )
        for step in range(args.max_steps):
            center_error = env.tomato.data.root_pos_w - env.grasp_center
            last_center_error = torch.where(
                alive.unsqueeze(-1), center_error, last_center_error
            )
            action = controller.command(
                env.contact_forces,
                env.palm_force,
                retention_mode=env.stem_broke,
                center_error_m=center_error,
                object_radius_m=env.tomato_axes.mean(dim=-1),
            )
            if args.capture_env is not None and int(done_step[args.capture_env]) < 0:
                index = args.capture_env
                trajectory["joint_pos_rad"].append(
                    env.robot.data.joint_pos[index, env.active_joint_ids].detach().cpu().numpy().copy()
                )
                trajectory["tomato_pos_m"].append(
                    env.tomato.data.root_pos_w[index].detach().cpu().numpy().copy()
                )
                trajectory["tomato_quat_wxyz"].append(
                    env.tomato.data.root_quat_w[index].detach().cpu().numpy().copy()
                )
                trajectory["action"].append(action[index].detach().cpu().numpy().copy())
                trajectory["digit_force_n"].append(
                    env.contact_forces[index].detach().cpu().numpy().copy()
                )
                trajectory["palm_force_n"].append(float(env.palm_force[index]))
                trajectory["stem_damage"].append(float(env.stem_damage[index]))
                trajectory["stem_broke"].append(bool(env.stem_broke[index]))
                trajectory["pick_started"].append(bool(env.pick_started[index]))
                trajectory["post_break_hold_steps"].append(int(env.stable_steps[index]))
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            sample = alive.unsqueeze(-1)
            active_contact = env.contact_forces >= cfg.contact_threshold_n
            peak_digit = torch.where(
                sample, torch.maximum(peak_digit, env.contact_forces), peak_digit
            )
            peak_palm = torch.where(
                alive, torch.maximum(peak_palm, env.palm_force), peak_palm
            )
            peak_damage = torch.where(
                alive, torch.maximum(peak_damage, env.damage_score), peak_damage
            )
            max_prepick_steps = torch.where(
                alive,
                torch.maximum(max_prepick_steps, env.prepick_stable_steps),
                max_prepick_steps,
            )
            max_relative_speed = torch.where(
                alive,
                torch.maximum(max_relative_speed, env._relative_speed()),
                max_relative_speed,
            )
            pick_started_seen |= alive & env.pick_started
            all_five_seen |= alive & active_contact.all(dim=-1)
            palm_seen |= alive & (env.palm_force >= cfg.palm_support_force_n)
            broke_now = alive & ~break_seen & (
                env.stem_broke | env.last_break_event
            )
            break_digit_force = torch.where(
                broke_now.unsqueeze(-1), env.contact_forces, break_digit_force
            )
            break_palm_force = torch.where(
                broke_now, env.palm_force, break_palm_force
            )
            break_center_error = torch.where(
                broke_now.unsqueeze(-1), center_error, break_center_error
            )
            break_seen |= alive & (env.stem_broke | env.last_break_event)
            postbreak_active = alive & break_seen
            postbreak_min_digit_force = torch.where(
                postbreak_active.unsqueeze(-1),
                torch.minimum(postbreak_min_digit_force, env.contact_forces),
                postbreak_min_digit_force,
            )
            postbreak_min_palm_force = torch.where(
                postbreak_active,
                torch.minimum(postbreak_min_palm_force, env.palm_force),
                postbreak_min_palm_force,
            )
            max_hold = torch.where(
                alive, torch.maximum(max_hold, env.stable_steps), max_hold
            )
            first_done = alive & done
            done_step = torch.where(
                first_done, torch.full_like(done_step, step + 1), done_step
            )
            done_code = torch.where(
                first_done, env.last_termination_code, done_code
            )
            alive &= ~done
            if not bool(alive.any()):
                break

        # Gate-local timeout code 6 means the environment never emitted a
        # terminal result inside the scripted budget.
        done_step = torch.where(
            alive, torch.full_like(done_step, args.max_steps), done_step
        )
        done_code = torch.where(alive, torch.full_like(done_code, 6), done_code)
        axes = env.tomato_axes.detach().cpu()
        masses = env.object_mass.detach().cpu()
        friction = env.object_friction.detach().cpu()
        records = []
        for index in range(args.num_envs):
            records.append(
                {
                    "seed": args.seed,
                    "env": index,
                    "success": int(done_code[index]) == 1,
                    "termination_code": int(done_code[index]),
                    "done_step": int(done_step[index]),
                    "stem_broke": bool(break_seen[index]),
                    "post_break_hold_steps": int(max_hold[index]),
                    "five_digit_contact_seen": bool(all_five_seen[index]),
                    "palm_support_seen": bool(palm_seen[index]),
                    "axes_m": axes[index].tolist(),
                    "mass_kg": float(masses[index]),
                    "friction": float(friction[index]),
                    "peak_digit_force_n": peak_digit[index].detach().cpu().tolist(),
                    "peak_palm_force_n": float(peak_palm[index]),
                    "peak_damage_score": float(peak_damage[index]),
                    "max_prepick_stable_steps": int(max_prepick_steps[index]),
                    "max_relative_speed_m_s": float(max_relative_speed[index]),
                    "pick_started": bool(pick_started_seen[index]),
                    "initial_center_error_m": initial_center_error[index].detach().cpu().tolist(),
                    "break_digit_force_n": break_digit_force[index].detach().cpu().tolist(),
                    "break_palm_force_n": float(break_palm_force[index]),
                    "break_center_error_m": break_center_error[index].detach().cpu().tolist(),
                    "postbreak_min_digit_force_n": postbreak_min_digit_force[index].detach().cpu().tolist(),
                    "postbreak_min_palm_force_n": float(postbreak_min_palm_force[index]),
                    "last_center_error_m": last_center_error[index].detach().cpu().tolist(),
                }
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if args.trajectory_output is not None:
            args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.trajectory_output,
                **{key: np.asarray(values) for key, values in trajectory.items()},
                seed=np.asarray(args.seed),
                env_index=np.asarray(args.capture_env),
                termination_code=np.asarray(int(done_code[args.capture_env])),
                done_step=np.asarray(int(done_step[args.capture_env])),
            )
        summary = {
            "seed": args.seed,
            "scenes": len(records),
            "successes": sum(record["success"] for record in records),
            "output": str(args.output),
        }
        print("STAGE0_SHARD=" + json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
