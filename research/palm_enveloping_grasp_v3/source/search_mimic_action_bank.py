"""Kinematic-first action search on the restored vendor-URDF morphology.

The tomato is held kinematic during this P2 reachability screen.  Results may
seed the dynamic event teacher, but cannot pass a grasp gate by themselves.
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
parser.add_argument("--num-envs", type=int, default=96)
parser.add_argument("--approach-steps", type=int, default=240)
parser.add_argument("--hold-steps", type=int, default=80)
parser.add_argument("--seed", type=int, default=20260820)
parser.add_argument("--center-action", type=str)
parser.add_argument("--noise-scale", type=float, default=0.55)
parser.add_argument("--vary-dims", type=str, help="comma-separated action dimensions to vary")
parser.add_argument(
    "--dynamic-fruit", action="store_true",
    help="leave tomato dynamic and gravity-loaded; upgrades the run from a kinematic reach screen to a P2/P3 dynamic candidate search",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v3.env import (
    V3EnvelopingPhysxTomatoGraspEnv,
    V3EnvelopingPhysxTomatoGraspEnvCfg,
)
from simulation.ppo_tomato_grasp_v3.mimic import URDF_MIMIC_SPECS
from simulation.ppo_tomato_grasp_v3.direct_joint_action import joint_targets_to_v3_direct_action
from simulation.ppo_tomato_grasp.enveloping_verifier import _contact_geometry_metrics
from simulation.ppo_tomato_grasp_v3.provenance import atomic_json_write, sha256_file
from simulation.ppo_tomato_grasp_v3.scenario import apply_contact_collision_mode, canonical_hash, load_scenario


def _parse_action(raw: str | None, fallback: list[float]) -> torch.Tensor:
    values = fallback if raw is None else [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(values) != 14 or any(abs(float(v)) > 1.0 for v in values):
        raise ValueError("center action must contain 14 values in [-1,1]")
    return torch.tensor(values, dtype=torch.float32)


def _make_cfg(scenario: dict) -> V3EnvelopingPhysxTomatoGraspEnvCfg:
    cfg = V3EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.strict_enveloping_enabled = True
    cfg.strict_enveloping_require_stem_break = False
    cfg = apply_profile(
        cfg,
        scenario["evaluation"]["profile"],
        num_envs=args.num_envs,
        seed=args.seed,
    )
    task, obj = scenario["task"], scenario["object"]
    geometry, strict = scenario["contact_geometry"], scenario["strict"]
    safety, physics = scenario["safety"], scenario["physics"]
    cfg.position_jitter = tuple(float(v) for v in task["initial_position_jitter_m"])
    cfg.axis_range_m = (float(obj["semi_axes_m"][0]), float(obj["semi_axes_m"][0]))
    cfg.isotropic_axes = True
    cfg.accepted_mass_kg = tuple(float(v) for v in obj["accepted_mass_kg"])
    cfg.density_range_kg_m3 = (float(obj["density_kg_m3"]), float(obj["density_kg_m3"]))
    geometry_event = cfg.events["tomato_geometry"].params
    geometry_event.update(
        axis_range_m=cfg.axis_range_m,
        isotropic_axes=True,
        accepted_mass_kg=cfg.accepted_mass_kg,
        density_range_kg_m3=cfg.density_range_kg_m3,
    )
    cfg.tomato_cfg.spawn.rigid_props.kinematic_enabled = not args.dynamic_fruit
    cfg.tomato_cfg.spawn.rigid_props.disable_gravity = not args.dynamic_fruit
    cfg.tomato_contact_offset_m = float(obj["contact_offset_m"])
    cfg.tomato_rest_offset_m = float(obj["rest_offset_m"])
    cfg.tomato_reset_clearance_m = float(obj["reset_clearance_m"])
    cfg.friction_range = (float(obj["friction"]), float(obj["friction"]))
    cfg.restitution_range = (float(obj["restitution"]), float(obj["restitution"]))
    cfg.tomato_cfg.spawn.physics_material.compliant_contact_stiffness = float(obj["contact_stiffness_n_m"])
    apply_contact_collision_mode(cfg, geometry)
    cfg.fingertip_collision_proxy_radius_m = float(geometry["fingertip_proxy_radius_m"])
    cfg.fingertip_proxy_contact_offset_m = float(geometry["proxy_contact_offset_m"])
    cfg.fingertip_proxy_rest_offset_m = float(geometry["proxy_rest_offset_m"])
    cfg.virtual_distal_contact_enabled = bool(geometry["virtual_distal_contact_for_shaping"])
    cfg.palm_cup_inner_half_extent_m = tuple(float(v) for v in geometry["palm_cup_inner_half_extent_m"])
    cfg.palm_cup_rim_height_m = float(geometry["palm_cup_rim_height_m"])
    cfg.strict_angular_coverage_min_deg = float(strict["angular_coverage_min_deg"])
    cfg.strict_thumb_opposition_cos_max = float(strict["thumb_opposition_cos_max"])
    cfg.strict_same_hemisphere_score_max = float(strict["same_hemisphere_score_max"])
    cfg.strict_palm_load_share_min = float(strict["palm_load_share"][0])
    cfg.strict_palm_load_share_max = float(strict["palm_load_share"][1])
    cfg.strict_palm_first_margin_steps = int(strict["palm_first_margin_steps"])
    cfg.strict_hold_steps = int(strict["diagnostic_hold_steps"])
    cfg.safe_force_n = float(safety["safe_force_n"])
    cfg.force_ceiling_n = tuple(float(v) for v in safety["force_ceiling_n"])
    cfg.command_rate_limit_rad_s = float(safety["command_rate_limit_rad_s"])
    cfg.v3_urdf_terminal_mimic_enabled = True
    cfg.v3_urdf_mimic_servo_rate_rad_s = float(physics["urdf_mimic_servo_rate_rad_s"])
    cfg.v3_action_contract = "direct_flexion14_v1"
    cfg.v3_hand_disable_gravity = bool(physics["hand_disable_gravity"])
    cfg.v3_use_tomato_filtered_contact_force = bool(physics["use_tomato_filtered_contact_force"])
    cfg.v3_contact_history_substeps = int(physics["contact_history_substeps"])
    for sensor_name in (
        "v3_thumb_pad_contact", "v3_index_pad_contact", "v3_middle_pad_contact",
        "v3_ring_pad_contact", "v3_little_pad_contact",
    ):
        getattr(cfg.scene, sensor_name).history_length = cfg.v3_contact_history_substeps
    cfg.v3_joint_drive_effort_limit_nm = float(physics["joint_drive_effort_limit_nm"])
    cfg.v3_joint_drive_stiffness_nm_rad = float(physics["joint_drive_stiffness_nm_rad"])
    cfg.v3_joint_drive_damping_nm_s_rad = float(physics["joint_drive_damping_nm_s_rad"])
    cfg.v3_mimic_drive_effort_limit_nm = float(physics["mimic_drive_effort_limit_nm"])
    cfg.v3_mimic_drive_stiffness_nm_rad = float(physics["mimic_drive_stiffness_nm_rad"])
    cfg.v3_mimic_drive_damping_nm_s_rad = float(physics["mimic_drive_damping_nm_s_rad"])
    cfg.v3_mimic_drive_armature_kg_m2 = float(physics["mimic_drive_armature_kg_m2"])
    cfg.robot_cfg.spawn.rigid_props.disable_gravity = cfg.v3_hand_disable_gravity
    actuator = cfg.robot_cfg.actuators["left_hand"]
    actuator.effort_limit_sim = cfg.v3_joint_drive_effort_limit_nm
    actuator.stiffness = cfg.v3_joint_drive_stiffness_nm_rad
    actuator.damping = cfg.v3_joint_drive_damping_nm_s_rad
    mimic_actuator = cfg.robot_cfg.actuators["terminal_mimic"]
    mimic_actuator.effort_limit_sim = cfg.v3_mimic_drive_effort_limit_nm
    mimic_actuator.stiffness = cfg.v3_mimic_drive_stiffness_nm_rad
    mimic_actuator.damping = cfg.v3_mimic_drive_damping_nm_s_rad
    mimic_actuator.armature = cfg.v3_mimic_drive_armature_kg_m2
    cfg.episode_length_s = max(24.0, (args.approach_steps + args.hold_steps + 20) * cfg.policy_dt_s)
    return cfg


def _candidate_bank(center: torch.Tensor, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(args.seed)
    center = center.to(device)
    noise = torch.randn((args.num_envs, 14), generator=generator, device=device)
    vary_mask = torch.ones(14, dtype=torch.bool, device=device)
    if args.vary_dims:
        vary_mask.zero_()
        for raw in args.vary_dims.split(","):
            index = int(raw.strip())
            if not 0 <= index < 14:
                raise ValueError("vary dimensions must lie in [0,13]")
            vary_mask[index] = True
        noise *= vary_mask
    actions = (center.unsqueeze(0) + float(args.noise_scale) * noise).clamp(-1.0, 1.0)
    # Preserve a small global bank so the search can escape the historical
    # candidate.  Bias independent distal coordinates toward true curl.
    global_count = max(args.num_envs // 3, 1)
    global_samples = 2.0 * torch.rand((global_count, 14), generator=generator, device=device) - 1.0
    global_samples[:, :4] = torch.empty((global_count, 4), device=device).uniform_(-1.0, 1.0, generator=generator)
    global_samples[:, 4:12] = torch.empty((global_count, 8), device=device).uniform_(-0.2, 1.0, generator=generator)
    actions[:global_count] = torch.where(vary_mask.unsqueeze(0), global_samples, center.unsqueeze(0))
    actions[0] = center
    if args.num_envs > 1 and not args.vary_dims:
        positive_distal = center.clone()
        positive_distal[10:14] = 0.85
        actions[1] = positive_distal
    return actions


@torch.inference_mode()
def main() -> None:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    scenario_path = args.scenario.expanduser().resolve()
    scenario, scenario_hash = load_scenario(scenario_path)
    cfg = _make_cfg(scenario)
    env = V3EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)
    center = _parse_action(args.center_action, scenario["controller"]["target_action"])
    actions = _candidate_bank(center, env.device)
    reset_joint = torch.tensor(
        (0.6752, 0.02745098, 0.02745098, 0.02745098, 0.02745098,
         -0.9420, -0.13254902, -0.04180392, 0.04588235, 0.16619608,
         1.3248, 0.02529412, 0.02117647, 0.02117647, 0.02117647, 0.02117647),
        device=env.device,
    ).view(1, 16)
    open_action = joint_targets_to_v3_direct_action(reset_joint).expand_as(actions).clone()
    hold_count = torch.zeros((args.num_envs, 5), device=env.device)
    all_five_hold = torch.zeros(args.num_envs, device=env.device)
    max_simultaneous = torch.zeros(args.num_envs, device=env.device)
    max_coverage = torch.zeros(args.num_envs, device=env.device)
    min_opposition = torch.ones(args.num_envs, device=env.device)
    min_hemisphere = torch.ones(args.num_envs, device=env.device)
    max_force = torch.zeros((args.num_envs, 5), device=env.device)
    max_palm = torch.zeros(args.num_envs, device=env.device)
    min_tip_distance = torch.full((args.num_envs, 5), float("inf"), device=env.device)
    overload = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
    final_tip_distance = torch.zeros((args.num_envs, 5), device=env.device)
    for step in range(args.approach_steps + args.hold_steps):
        fraction = min(float(step + 1) / max(float(args.approach_steps), 1.0), 1.0)
        env.step(open_action + fraction * (actions - open_action))
        physical = env._v2_measured_link4_force
        active = physical >= env.cfg.contact_threshold_n
        simultaneous = active.sum(-1).float()
        max_simultaneous = torch.maximum(max_simultaneous, simultaneous)
        max_force = torch.maximum(max_force, physical)
        max_palm = torch.maximum(max_palm, env.palm_force)
        max_coverage = torch.maximum(max_coverage, env._v3_physical_coverage_current)
        eligible = simultaneous >= 3
        min_opposition = torch.where(
            eligible & active[:, 0],
            torch.minimum(min_opposition, env._v3_physical_opposition_current),
            min_opposition,
        )
        min_hemisphere = torch.where(
            eligible,
            torch.minimum(min_hemisphere, env._v3_physical_hemisphere_current),
            min_hemisphere,
        )
        tips = env._tip_positions()
        distances = torch.linalg.vector_norm(
            tips - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1
        )
        final_tip_distance = distances
        min_tip_distance = torch.minimum(min_tip_distance, distances)
        overload |= env.overloaded
        if step >= args.approach_steps:
            hold_count += active.float()
            all_five_hold += active.all(-1).float()
    occupancy = hold_count / max(float(args.hold_steps), 1.0)
    weakest = occupancy.min(-1).values
    distal_all = env.strict_distal_participation.all(-1)
    coverage_progress = (max_coverage / float(scenario["strict"]["angular_coverage_min_deg"])).clamp(0.0, 1.2)
    all_active = torch.ones((args.num_envs, 5), dtype=torch.bool, device=env.device)
    normal = torch.zeros((args.num_envs, 3), device=env.device)
    normal[:, 2] = 1.0
    tips = env._tip_positions()
    geometric_coverage, geometric_opposition, geometric_hemisphere = _contact_geometry_metrics(
        tips.unsqueeze(0), all_active.unsqueeze(0), env.tomato.data.root_pos_w.unsqueeze(0), normal.unsqueeze(0)
    )
    geometric_coverage = geometric_coverage[0]
    geometric_opposition = geometric_opposition[0]
    geometric_hemisphere = geometric_hemisphere[0]
    target_shell = float(scenario["object"]["semi_axes_m"][0]) + float(
        scenario["contact_geometry"]["physical_tip_radius_m"]
    )
    final_radial_error = (final_tip_distance - target_shell).abs()
    score = (
        55.0 * weakest
        + 25.0 * (all_five_hold / max(float(args.hold_steps), 1.0))
        + 4.0 * max_simultaneous
        + 8.0 * coverage_progress
        + 3.0 * (-min_opposition).clamp(0.0, 1.0)
        + 2.0 * (1.0 - min_hemisphere).clamp(0.0, 1.0)
        + 4.0 * distal_all.float()
        - 30.0 * overload.float()
        - 0.5 * max_force.sum(-1)
        - 500.0 * final_radial_error.mean(-1)
        - 250.0 * final_radial_error.max(-1).values
        + 0.025 * geometric_coverage
        + 2.0 * (-geometric_opposition).clamp(0.0, 1.0)
        + 1.0 * (1.0 - geometric_hemisphere).clamp(0.0, 1.0)
    )
    full_pos = env.robot.root_physx_view.get_dof_positions().to(env.device)
    name_to_id = {name: idx for idx, name in enumerate(env.robot.joint_names)}
    mimic_error = torch.zeros((args.num_envs, 5), device=env.device)
    for column, (child, (parent, multiplier, offset)) in enumerate(URDF_MIMIC_SPECS.items()):
        expected = float(multiplier) * full_pos[:, name_to_id[parent]] + float(offset)
        limits = env.robot.data.soft_joint_pos_limits[:, name_to_id[child]]
        expected = expected.clamp(limits[:, 0], limits[:, 1])
        mimic_error[:, column] = (full_pos[:, name_to_id[child]] - expected).abs()
    order = torch.argsort(score, descending=True)

    def row(index: int) -> dict:
        return {
            "candidate_id": index,
            "score": float(score[index].cpu()),
            "action": [float(v) for v in actions[index].cpu()],
            "hold_contact_occupancy": [float(v) for v in occupancy[index].cpu()],
            "all_five_hold_occupancy": float((all_five_hold[index] / max(float(args.hold_steps), 1.0)).cpu()),
            "max_simultaneous_physical_digits": int(max_simultaneous[index].cpu()),
            "max_coverage_deg": float(max_coverage[index].cpu()),
            "min_opposition_cos": float(min_opposition[index].cpu()),
            "min_hemisphere_score": float(min_hemisphere[index].cpu()),
            "min_tip_distance_m": [float(v) for v in min_tip_distance[index].cpu()],
            "final_tip_distance_m": [float(v) for v in final_tip_distance[index].cpu()],
            "final_radial_error_m": [float(v) for v in final_radial_error[index].cpu()],
            "geometric_coverage_deg": float(geometric_coverage[index].cpu()),
            "geometric_opposition_cos": float(geometric_opposition[index].cpu()),
            "geometric_hemisphere_score": float(geometric_hemisphere[index].cpu()),
            "peak_digit_force_n": [float(v) for v in max_force[index].cpu()],
            "peak_palm_force_n": float(max_palm[index].cpu()),
            "distal_all_participated": bool(distal_all[index].cpu()),
            "overload_seen": bool(overload[index].cpu()),
            "final_active_joint_position_rad": [
                float(v) for v in full_pos[index, env.active_joint_ids].cpu()
            ],
            "mimic_abs_error_rad": [float(v) for v in mimic_error[index].cpu()],
        }

    report = {
        "version": "v3_dynamic_action_search_v1" if args.dynamic_fruit else "v3_mimic_kinematic_action_search_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_command_allowed": False,
        "interpretation": (
            "dynamic gravity-loaded P2/P3 candidate search; candidates still require independent event-teacher validation"
            if args.dynamic_fruit else
            "fixed-fruit P2 morphology screen only; not dynamic grasp evidence"
        ),
        "dynamic_fruit": bool(args.dynamic_fruit),
        "scenario": str(scenario_path),
        "scenario_sha256": sha256_file(scenario_path),
        "scenario_canonical_sha256": scenario_hash,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "approach_steps": args.approach_steps,
        "hold_steps": args.hold_steps,
        "center_action": [float(v) for v in center],
        "noise_scale": args.noise_scale,
        "vary_dims": None if not args.vary_dims else [int(v.strip()) for v in args.vary_dims.split(",")],
        "all_five_seen_count": int((max_simultaneous == 5).sum().cpu()),
        "all_five_hold_positive_count": int((all_five_hold > 0).sum().cpu()),
        "distal_all_participated_count": int(distal_all.sum().cpu()),
        "maximum_mimic_abs_error_rad": float(mimic_error.max().cpu()),
        "ranked_candidates": [row(int(index)) for index in order[: min(24, args.num_envs)].tolist()],
        "best_thumb_candidates": [
            row(int(index))
            for index in torch.argsort(min_tip_distance[:, 0])[: min(24, args.num_envs)].tolist()
        ],
        "best_final_thumb_candidates": [
            row(int(index))
            for index in torch.argsort(final_tip_distance[:, 0])[: min(24, args.num_envs)].tolist()
        ],
    }
    atomic_json_write(output, report)
    print(json.dumps({
        "output": str(output),
        "all_five_seen_count": report["all_five_seen_count"],
        "distal_all_participated_count": report["distal_all_participated_count"],
        "maximum_mimic_abs_error_rad": report["maximum_mimic_abs_error_rad"],
        "best": report["ranked_candidates"][0],
    }, indent=2), flush=True)
    env.close()
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
