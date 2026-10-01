"""Direct 16-joint morphology search with the tomato moved out of collision.

This separates hand kinematic reachability from the 14-D action manifold.
Candidates are simulation-only and must pass later dynamic collision checks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
DEFAULT_SCENARIO = Path(__file__).resolve().parent / "scenarios/v3_large90_nominal.json"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=7)
parser.add_argument("--elite-count", type=int, default=24)
parser.add_argument("--seed", type=int, default=20260824)
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument(
    "--target-shell-offset-m", type=float, default=0.0,
    help="signed approach-ray endpoint offset; negative values deliberately cross the nominal shell",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp.enveloping_verifier import _contact_geometry_metrics
from simulation.ppo_tomato_grasp_v3.direct_joint_action import V3_DIRECT_LOWER_RAD, V3_DIRECT_UPPER_RAD
from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v3.env import V3EnvelopingPhysxTomatoGraspEnv, V3EnvelopingPhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp_v3.mimic import apply_urdf_mimic_targets
from simulation.ppo_tomato_grasp_v3.direct_joint_action import joint_targets_to_v3_direct_action
from simulation.ppo_tomato_grasp_v3.provenance import atomic_json_write, sha256_file
from simulation.ppo_tomato_grasp_v3.scenario import apply_contact_collision_mode, load_scenario


def _cfg(scenario: dict) -> V3EnvelopingPhysxTomatoGraspEnvCfg:
    cfg = V3EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg = apply_profile(cfg, scenario["evaluation"]["profile"], num_envs=args.num_envs, seed=args.seed)
    obj, geometry = scenario["object"], scenario["contact_geometry"]
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.axis_range_m = (float(obj["semi_axes_m"][0]), float(obj["semi_axes_m"][0]))
    cfg.isotropic_axes = True
    cfg.accepted_mass_kg = tuple(float(v) for v in obj["accepted_mass_kg"])
    cfg.density_range_kg_m3 = (float(obj["density_kg_m3"]), float(obj["density_kg_m3"]))
    cfg.events["tomato_geometry"].params.update(
        axis_range_m=cfg.axis_range_m,
        isotropic_axes=True,
        accepted_mass_kg=cfg.accepted_mass_kg,
        density_range_kg_m3=cfg.density_range_kg_m3,
    )
    apply_contact_collision_mode(cfg, geometry)
    cfg.fingertip_collision_proxy_radius_m = float(geometry["fingertip_proxy_radius_m"])
    cfg.v3_urdf_terminal_mimic_enabled = True
    cfg.v3_action_contract = "direct_flexion14_v1"
    cfg.v3_urdf_mimic_servo_rate_rad_s = float(scenario["physics"]["urdf_mimic_servo_rate_rad_s"])
    return cfg


def _initial_bank(n: int, lower: torch.Tensor, upper: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    bank = lower + torch.rand((n, 16), generator=generator, device=lower.device) * (upper - lower)
    # Straight-finger flexion uses joint1 + joint2; retain the calibrated
    # narrow ab/adduction axes but bias both flexion layers into enclosure.
    bank[:, 1:5] = lower[1:5] + torch.empty((n, 4), device=lower.device).uniform_(0.18, 0.92, generator=generator) * (upper[1:5] - lower[1:5])
    bank[:, 12:16] = lower[12:16] + torch.empty((n, 4), device=lower.device).uniform_(0.10, 0.98, generator=generator) * (upper[12:16] - lower[12:16])
    return bank


@torch.inference_mode()
def main() -> None:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    scenario_path = args.scenario.expanduser().resolve()
    scenario, scenario_hash = load_scenario(scenario_path)
    env = V3EnvelopingPhysxTomatoGraspEnv(_cfg(scenario))
    env.reset(seed=args.seed)
    device = env.device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    active_lower = torch.tensor(V3_DIRECT_LOWER_RAD, device=device)
    active_upper = torch.tensor(V3_DIRECT_UPPER_RAD, device=device)
    if args.settle_steps < 0:
        raise ValueError("settle steps must be non-negative")
    reset_joint = torch.tensor(
        (0.6752, 0.02745098, 0.02745098, 0.02745098, 0.02745098,
         -0.9420, -0.13254902, -0.04180392, 0.04588235, 0.16619608,
         1.3248, 0.02529412, 0.02117647, 0.02117647, 0.02117647, 0.02117647),
        device=device,
    ).view(1, 16)
    open_action = joint_targets_to_v3_direct_action(reset_joint).expand(args.num_envs, -1)
    for _ in range(args.settle_steps):
        env.step(open_action)
    center = env.tomato.data.root_pos_w.detach().clone()
    # Move the object away so candidate joints are scored purely by forward
    # kinematics relative to the saved nominal center.
    transforms = env.tomato.root_physx_view.get_transforms().clone()
    transforms[:, 2] += 1.0
    env.tomato.root_physx_view.set_transforms(transforms, env.tomato._ALL_INDICES)
    velocities = env.tomato.root_physx_view.get_velocities().clone()
    velocities.zero_()
    env.tomato.root_physx_view.set_velocities(velocities, env.tomato._ALL_INDICES)
    bank = _initial_bank(args.num_envs, active_lower, active_upper, generator)
    all_rows: list[dict] = []
    elite = None
    for iteration in range(args.iterations):
        if iteration > 0:
            assert elite is not None
            parent = elite[torch.randint(elite.shape[0], (args.num_envs,), generator=generator, device=device)]
            sigma = 0.18 * (0.58 ** (iteration - 1))
            bank = parent + sigma * (active_upper - active_lower) * torch.randn(
                (args.num_envs, 16), generator=generator, device=device
            )
            bank = torch.maximum(active_lower, torch.minimum(active_upper, bank))
        full = env.robot.root_physx_view.get_dof_positions().clone().to(device)
        full[:, env.active_joint_ids] = bank
        limits = env.robot.data.soft_joint_pos_limits
        full = apply_urdf_mimic_targets(full, env.robot.joint_names, limits[..., 0], limits[..., 1])
        zeros = torch.zeros_like(full)
        env.robot.root_physx_view.set_dof_positions(full, env.robot._ALL_INDICES)
        env.robot.root_physx_view.set_dof_velocities(zeros, env.robot._ALL_INDICES)
        env.robot.root_physx_view.set_dof_position_targets(full, env.robot._ALL_INDICES)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        tips = env.robot.data.body_pos_w[:, env.tip_ids]
        distance = torch.linalg.vector_norm(tips - center.unsqueeze(1), dim=-1)
        active = torch.ones((args.num_envs, 5), dtype=torch.bool, device=device)
        normal = torch.zeros((args.num_envs, 3), device=device)
        normal[:, 2] = 1.0
        coverage, opposition, hemisphere = _contact_geometry_metrics(
            tips.unsqueeze(0), active.unsqueeze(0), center.unsqueeze(0), normal.unsqueeze(0)
        )
        coverage, opposition, hemisphere = coverage[0], opposition[0], hemisphere[0]
        shell = float(scenario["object"]["semi_axes_m"][0]) + float(
            scenario["contact_geometry"]["physical_tip_radius_m"]
        ) + float(args.target_shell_offset_m)
        radial_error = (distance - shell).abs()
        worst_error = radial_error.max(-1).values
        score = (
            -1200.0 * radial_error.mean(-1)
            - 800.0 * worst_error
            + 0.06 * coverage
            + 6.0 * (-opposition).clamp(-1.0, 1.0)
            + 4.0 * (1.0 - hemisphere).clamp(0.0, 1.0)
        )
        order = torch.argsort(score, descending=True)
        elite = bank[order[: args.elite_count]].clone()
        for rank, idx in enumerate(order[: min(12, args.num_envs)].tolist()):
            all_rows.append({
                "iteration": iteration,
                "rank": rank,
                "score": float(score[idx].cpu()),
                "active_joint_position_rad": [float(v) for v in bank[idx].cpu()],
                "tip_distance_m": [float(v) for v in distance[idx].cpu()],
                "radial_error_m": [float(v) for v in radial_error[idx].cpu()],
                "coverage_deg": float(coverage[idx].cpu()),
                "opposition_cos": float(opposition[idx].cpu()),
                "hemisphere_score": float(hemisphere[idx].cpu()),
                "tip_position_world_m": [[float(v) for v in xyz] for xyz in tips[idx].cpu()],
            })
    final_rows = sorted(all_rows, key=lambda row: row["score"], reverse=True)[:32]
    best = final_rows[0]
    report = {
        "version": "v3_direct_joint_geometry_cem_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_command_allowed": False,
        "interpretation": "collision-free direct-joint geometry only; no action-manifold or dynamic success claim",
        "scenario": str(scenario_path),
        "scenario_sha256": sha256_file(scenario_path),
        "scenario_canonical_sha256": scenario_hash,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "elite_count": args.elite_count,
        "settle_steps": args.settle_steps,
        "target_shell_offset_m": args.target_shell_offset_m,
        "nominal_center_world_m_env0": [float(v) for v in center[0].cpu()],
        "target_tip_center_radius_m": float(scenario["object"]["semi_axes_m"][0]) + float(
            scenario["contact_geometry"]["physical_tip_radius_m"]
        ) + float(args.target_shell_offset_m),
        "best_candidate": best,
        "ranked_candidates": final_rows,
    }
    atomic_json_write(output, report)
    print(json.dumps({"output": str(output), "best_candidate": best}, indent=2), flush=True)
    env.close()
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
