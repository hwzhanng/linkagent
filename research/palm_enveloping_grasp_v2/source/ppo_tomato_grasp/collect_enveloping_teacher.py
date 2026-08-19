"""Collect staged 14D enclosure-teacher trajectories in headless PhysX."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--seed", type=int, default=20260816)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--axis-low", type=float, default=0.035)
parser.add_argument("--axis-high", type=float, default=0.050)
parser.add_argument("--isotropic", action="store_true", help="use an equal-axis entry fruit band")
parser.add_argument("--single-branch", type=int, choices=(0, 1, 2), default=None,
                    help="repeat one weak-prior branch for BC warm start; omit for multi-branch data")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import torch

from simulation.ppo_tomato_grasp.enveloping_teacher import EnvelopingTeacher, EnvelopingTeacherCfg
from simulation.ppo_tomato_grasp.physx_enveloping_env import (
    EnvelopingPhysxTomatoGraspEnv,
    EnvelopingPhysxTomatoGraspEnvCfg,
)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    cfg = EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.training_stage = 1
    cfg.episode_length_s = 15.0
    cfg.command_rate_limit_rad_s = 0.25
    # Teacher collection is a geometry/sequence bootstrap, not a safety
    # benchmark.  Keep the forces in the trace and label this relaxed ceiling
    # explicitly so it cannot be mistaken for final low-pressure training.
    cfg.safe_force_n = 20.0
    cfg.force_ceiling_n = (20.0, 20.0, 20.0, 20.0, 20.0)
    cfg.severe_overload_scale = 2.0
    cfg.events["tomato_geometry"].params["axis_range_m"] = (args_cli.axis_low, args_cli.axis_high)
    cfg.events["tomato_geometry"].params["isotropic_axes"] = args_cli.isotropic
    cfg.events["tomato_geometry"].params["accepted_mass_kg"] = (0.02, 1.20)
    cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (700.0, 1050.0)
    cfg.position_jitter = (0.002, 0.002, 0.001)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.tactile_gain_range = (1.0, 1.0)
    cfg.tactile_bias_range_n = (0.0, 0.0)
    cfg.friction_range = (0.70, 0.70)
    cfg.restitution_range = (0.0, 0.0)
    cfg.stiffness_range_n_m = (18000.0, 18000.0)
    env = EnvelopingPhysxTomatoGraspEnv(cfg)
    teacher = EnvelopingTeacher(args_cli.num_envs, env.device)
    if args_cli.single_branch is not None:
        # A single-branch dataset is only a BC initializer.  The default
        # collector remains multi-branch so PPO still sees a weak, non-fixed
        # geometry prior during later curriculum stages.
        branch = int(args_cli.single_branch)
        teacher.root_prior[:] = teacher.root_prior[branch]
        teacher.residual_prior[:] = teacher.residual_prior[branch]
    observation, _ = env.reset(seed=args_cli.seed)
    actor_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    force_rows: list[torch.Tensor] = []
    palm_rows: list[torch.Tensor] = []
    joint_rows: list[torch.Tensor] = []
    tip_rows: list[torch.Tensor] = []
    center_rows: list[torch.Tensor] = []
    phase_rows: list[torch.Tensor] = []
    strict_rows: list[torch.Tensor] = []
    strict_event_rows: list[torch.Tensor] = []
    strict_hold_rows: list[torch.Tensor] = []
    strict_coverage_rows: list[torch.Tensor] = []
    strict_first_palm_rows: list[torch.Tensor] = []
    strict_first_three_rows: list[torch.Tensor] = []
    strict_distal_all_rows: list[torch.Tensor] = []
    strict_proximal_only_rows: list[torch.Tensor] = []
    alive = torch.ones(args_cli.num_envs, dtype=torch.bool, device=env.device)
    for _ in range(args_cli.steps):
        action = teacher.command(env.contact_forces, env.palm_force)
        actor_rows.append(observation["policy"].detach().cpu())
        action_rows.append(action.detach().cpu())
        force_rows.append(env.contact_forces.detach().cpu())
        palm_rows.append(env.palm_force.detach().cpu())
        joint_rows.append(env.robot.data.joint_pos[:, env.active_joint_ids].detach().cpu())
        tip_rows.append(env._tip_positions().detach().cpu())
        center_rows.append(env.tomato.data.root_pos_w.detach().cpu())
        phase_rows.append(teacher.phase.detach().cpu())
        strict_rows.append(env.strict_success_latched.detach().cpu())
        observation, _, terminated, truncated, info = env.step(action)
        log = info.get("log", {}) if isinstance(info, dict) else {}
        # These values are emitted before DirectRLEnv resets completed envs.
        # Store them as audit traces instead of relying only on the latch
        # sampled before step(), which is necessarily false on a fresh reset.
        strict_event_rows.append(torch.full(
            (args_cli.num_envs,),
            float(log.get("strict_new_success_rate", 0.0)),
            device="cpu",
        ))
        strict_hold_rows.append(torch.full(
            (args_cli.num_envs,),
            float(log.get("strict_hold_max", 0.0)),
            device="cpu",
        ))
        strict_coverage_rows.append(torch.full(
            (args_cli.num_envs,),
            float(log.get("strict_coverage_max_deg_proxy", 0.0)),
            device="cpu",
        ))
        for rows, key in (
            (strict_first_palm_rows, "strict_palm_first_rate"),
            (strict_first_three_rows, "strict_first_three_rate"),
            (strict_distal_all_rows, "strict_distal_all_rate"),
            (strict_proximal_only_rows, "strict_proximal_only_rate"),
        ):
            rows.append(torch.full(
                (args_cli.num_envs,),
                float(log.get(key, 0.0)),
                device="cpu",
            ))
        done = terminated | truncated
        alive &= ~done
        if bool(done.any()):
            teacher.reset(done.nonzero(as_tuple=False).flatten())
        if not bool(alive.any()):
            break

    data = {
        "version": "enveloping_teacher_dataset_v1",
        "seed": args_cli.seed,
        "observation_dim": 110,
        "action_dim": 14,
        "teacher": "weak_geometry_prior_plus_palm_gate_plus_per_digit_tactile_trim",
        "observation": torch.cat(actor_rows),
        "action": torch.cat(action_rows),
        "digit_force_n": torch.cat(force_rows),
        "palm_force_n": torch.cat(palm_rows),
        "joint_pos_rad": torch.cat(joint_rows),
        "tip_positions_w": torch.cat(tip_rows),
        "fruit_center_w": torch.cat(center_rows),
        "teacher_phase": torch.cat(phase_rows),
        "strict_success_latched": torch.cat(strict_rows),
        "strict_success_event_rate": torch.cat(strict_event_rows),
        "strict_hold_max": torch.cat(strict_hold_rows),
        "strict_coverage_max_deg_proxy": torch.cat(strict_coverage_rows),
        "strict_palm_first_rate": torch.cat(strict_first_palm_rows),
        "strict_first_three_rate": torch.cat(strict_first_three_rows),
        "strict_distal_all_rate": torch.cat(strict_distal_all_rows),
        "strict_proximal_only_rate": torch.cat(strict_proximal_only_rows),
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args_cli.output.with_suffix(args_cli.output.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(args_cli.output)
    summary = {
        "samples": int(data["observation"].shape[0]),
        "teacher_phase_counts": torch.bincount(data["teacher_phase"].flatten(), minlength=4).tolist(),
        "strict_success_sample_rate": float(data["strict_success_latched"].float().mean()),
        "strict_success_event_rate": float(data["strict_success_event_rate"].float().mean()),
        "strict_hold_max": float(data["strict_hold_max"].max()),
        "strict_coverage_max_deg_proxy": float(data["strict_coverage_max_deg_proxy"].max()),
        "strict_palm_first_rate_max": float(data["strict_palm_first_rate"].max()),
        "strict_first_three_rate_max": float(data["strict_first_three_rate"].max()),
        "strict_distal_all_rate_max": float(data["strict_distal_all_rate"].max()),
        "strict_proximal_only_rate_max": float(data["strict_proximal_only_rate"].max()),
        "all_five_force_sample_rate": float((data["digit_force_n"] >= cfg.contact_threshold_n).all(dim=-1).float().mean()),
        "output": str(args_cli.output.resolve()),
        "hardware_command_allowed": False,
        "teacher_safety_relaxed": True,
    }
    args_cli.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
