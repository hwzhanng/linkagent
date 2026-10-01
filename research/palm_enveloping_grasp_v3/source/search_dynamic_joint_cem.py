"""Iterative dynamic CEM over the 14-D direct action contract.

Every candidate is scored by a fruit-present closed-loop rollout (the real
actuator pipeline, gravity-loaded tomato), so fitness reflects the achieved
load-deflected envelope instead of an ideal teleported FK snapshot.
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
DEFAULT_SCENARIO = Path(__file__).resolve().parent / "scenarios/v3_large100_nominal.json"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=96)
parser.add_argument("--generations", type=int, default=6)
parser.add_argument("--approach-steps", type=int, default=240)
parser.add_argument("--hold-steps", type=int, default=80)
parser.add_argument("--elite-count", type=int, default=16)
parser.add_argument("--seed", type=int, default=20260821)
parser.add_argument(
    "--warm-start", type=Path,
    help="JSON with ranked candidates whose 'action' arrays seed generation 0",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp.enveloping_verifier import _contact_geometry_metrics
from simulation.ppo_tomato_grasp_v3.env import V3EnvelopingPhysxTomatoGraspEnv, V3EnvelopingPhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp_v3.provenance import atomic_json_write, sha256_file
from simulation.ppo_tomato_grasp_v3.scenario import apply_contact_collision_mode, load_scenario


def _make_cfg(scenario: dict) -> V3EnvelopingPhysxTomatoGraspEnvCfg:
    cfg = V3EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.training_stage = 1
    cfg.strict_enveloping_enabled = True
    cfg.strict_enveloping_require_stem_break = False
    from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
    cfg = apply_profile(cfg, scenario["evaluation"]["profile"], num_envs=args.num_envs, seed=args.seed)
    task, obj = scenario["task"], scenario["object"]
    geometry, strict = scenario["contact_geometry"], scenario["strict"]
    safety, controller = scenario["safety"], scenario["controller"]
    physics = scenario["physics"]
    cfg.position_jitter = tuple(float(v) for v in task["initial_position_jitter_m"])
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
    for key in (
        "angular_coverage_min_deg", "thumb_opposition_cos_max",
        "same_hemisphere_score_max",
    ):
        cfg_attr = f"strict_{key}"
        if hasattr(cfg, cfg_attr):
            setattr(cfg, cfg_attr, float(strict[key]))
    cfg.strict_palm_load_share_min = float(strict["palm_load_share"][0])
    cfg.strict_palm_load_share_max = float(strict["palm_load_share"][1])
    cfg.safe_force_n = float(safety["safe_force_n"])
    cfg.force_ceiling_n = tuple(float(v) for v in safety["force_ceiling_n"])
    cfg.command_rate_limit_rad_s = float(safety["command_rate_limit_rad_s"])
    cfg.v3_action_contract = str(controller["action_contract"])
    cfg.v3_urdf_terminal_mimic_enabled = bool(physics["urdf_terminal_mimic_enabled"])
    cfg.v3_direct_servo_zero_velocity = bool(physics["direct_servo_zero_velocity"])
    cfg.position_servo_enabled = bool(physics["direct_position_servo_enabled"])
    cfg.v3_filter_fingertip_proxy_support_collision = bool(
        physics["filter_fingertip_proxy_support_collision"]
    )
    cfg.v3_disable_palm_cup_rim_collision = bool(physics["disable_palm_cup_rim_collision"])
    cfg.v3_joint_drive_effort_limit_nm = float(physics["joint_drive_effort_limit_nm"])
    cfg.v3_joint_drive_stiffness_nm_rad = float(physics["joint_drive_stiffness_nm_rad"])
    cfg.v3_joint_drive_damping_nm_s_rad = float(physics["joint_drive_damping_nm_s_rad"])
    cfg.v3_mimic_drive_effort_limit_nm = float(physics["mimic_drive_effort_limit_nm"])
    cfg.v3_mimic_drive_stiffness_nm_rad = float(physics["mimic_drive_stiffness_nm_rad"])
    cfg.v3_mimic_drive_damping_nm_s_rad = float(physics["mimic_drive_damping_nm_s_rad"])
    cfg.v3_mimic_drive_armature_kg_m2 = float(physics["mimic_drive_armature_kg_m2"])
    cfg.v3_hand_disable_gravity = bool(physics["hand_disable_gravity"])
    cfg.episode_length_s = max(24.0, (args.approach_steps + args.hold_steps + 20) * cfg.policy_dt_s)
    return cfg


def main() -> None:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    scenario_path = args.scenario.expanduser().resolve()
    scenario, scenario_hash = load_scenario(scenario_path)
    env = V3EnvelopingPhysxTomatoGraspEnv(_make_cfg(scenario))
    env.reset(seed=args.seed)
    device = env.device
    generator = torch.Generator(device=device).manual_seed(args.seed)

    reset_joint = torch.tensor(
        (0.6752, 0.02745098, 0.02745098, 0.02745098, 0.02745098,
         -0.9420, -0.13254902, -0.04180392, 0.04588235, 0.16619608,
         1.3248, 0.02529412, 0.02117647, 0.02117647, 0.02117647, 0.02117647),
        device=device,
    ).view(1, 16)
    open_action_single = joint_targets_to_v3_direct_action(reset_joint)[0]

    warm: list[list[float]] = []
    if args.warm_start is not None:
        warm_doc = json.loads(args.warm_start.expanduser().resolve().read_text(encoding="utf-8"))
        for row in warm_doc.get("ranked_candidates", [])[: args.elite_count]:
            action = row.get("action")
            if action is not None and len(action) == 14:
                warm.append([float(v) for v in action])

    lower = torch.full((14,), -1.0, device=device)
    upper = torch.ones((14,), device=device)
    sigma = 0.35

    all_rows: list[dict] = []
    best_global: dict | None = None
    for generation in range(args.generations):
        if generation == 0 and warm:
            actions = torch.tensor(warm, device=device)
            pad = args.num_envs - actions.shape[0]
            noise = sigma * torch.randn((pad, 14), generator=generator, device=device)
            base = actions[torch.randint(actions.shape[0], (pad,), generator=generator, device=device)]
            actions = torch.cat((actions, (base + noise).clamp(-1.0, 1.0)), dim=0)
        elif generation == 0:
            actions = (lower + torch.rand((args.num_envs, 14), generator=generator, device=device) * (upper - lower))
        else:
            parent = elite[torch.randint(elite.shape[0], (args.num_envs,), generator=generator, device=device)]
            actions = (parent + sigma * (0.6 ** (generation - 1)) * torch.randn(
                (args.num_envs, 14), generator=generator, device=device
            )).clamp(-1.0, 1.0)
            # Index flexion channels (4=joint1, 8=joint2) get an extra closure
            # bias: the index finger is the shortest and needs deeper curl to
            # reach the shell under load deflection.
            index_bias = 0.15 * (0.9 ** (generation - 1))
            actions[:, 4] = (actions[:, 4] + index_bias).clamp(-1.0, 1.0)
            actions[:, 8] = (actions[:, 8] + index_bias).clamp(-1.0, 1.0)

        open_action = open_action_single.unsqueeze(0).expand_as(actions).clone()
        hold_count = torch.zeros((args.num_envs, 5), device=device)
        all_five_hold = torch.zeros(args.num_envs, device=device)
        max_simultaneous = torch.zeros(args.num_envs, device=device)
        max_coverage = torch.zeros(args.num_envs, device=device)
        min_opposition = torch.ones(args.num_envs, device=device)
        min_hemisphere = torch.ones(args.num_envs, device=device)
        max_force = torch.zeros((args.num_envs, 5), device=device)
        overload = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
        final_tip_distance = torch.zeros((args.num_envs, 5), device=device)
        final_fruit_center = torch.zeros((args.num_envs, 3), device=device)
        band_lower = torch.tensor([0.3, 0.4, 0.4, 0.4, 0.3], device=device)
        band_upper = torch.tensor([4.0, 5.0, 5.0, 6.0, 4.0], device=device)

        for step in range(args.approach_steps + args.hold_steps):
            fraction = min(float(step + 1) / max(float(args.approach_steps), 1.0), 1.0)
            env.step(open_action + fraction * (actions - open_action))
            physical = env._v2_measured_link4_force
            active = physical >= env.cfg.contact_threshold_n
            in_band = (physical >= band_lower) & (physical <= band_upper)
            simultaneous = active.sum(-1).float()
            max_simultaneous = torch.maximum(max_simultaneous, simultaneous)
            max_force = torch.maximum(max_force, physical)
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
            final_fruit_center = env.tomato.data.root_pos_w.clone()
            overload |= env.overloaded
            if step >= args.approach_steps:
                hold_count += in_band.float()
                all_five_hold += in_band.all(-1).float()

        occupancy = hold_count / max(float(args.hold_steps), 1.0)
        weakest = occupancy.min(-1).values
        distal_all = env.strict_distal_participation.all(-1)
        coverage_progress = (max_coverage / float(scenario["strict"]["angular_coverage_min_deg"])).clamp(0.0, 1.2)
        target_shell = float(scenario["object"]["semi_axes_m"][0]) + float(
            scenario["contact_geometry"]["physical_tip_radius_m"]
        )
        radial_gap = (final_tip_distance - target_shell)
        index_gap = radial_gap[:, 1].clamp(min=0.0)
        score = (
            55.0 * weakest
            + 40.0 * (all_five_hold / max(float(args.hold_steps), 1.0))
            + 4.0 * max_simultaneous
            + 8.0 * coverage_progress
            + 3.0 * (-min_opposition).clamp(0.0, 1.0)
            + 2.0 * (1.0 - min_hemisphere).clamp(0.0, 1.0)
            + 4.0 * distal_all.float()
            - 30.0 * overload.float()
            - 12.0 * radial_gap.clamp(min=0.0).mean(-1) * 100.0
            - 30.0 * radial_gap.max(-1).values.clamp(min=0.0)
            - 60.0 * index_gap
            + 15.0 * hold_count[:, 1]
        )
        order = torch.argsort(score, descending=True)
        elite = actions[order[: args.elite_count]].clone()

        full_pos = env.robot.root_physx_view.get_dof_positions().to(device)
        name_to_id = {name: idx for idx, name in enumerate(env.robot.joint_names)}
        mimic_error = torch.zeros((args.num_envs, 5), device=device)
        from simulation.ppo_tomato_grasp_v3.mimic import URDF_MIMIC_SPECS
        for column, (child, (parent, multiplier, offset)) in enumerate(URDF_MIMIC_SPECS.items()):
            expected = float(multiplier) * full_pos[:, name_to_id[parent]] + float(offset)
            limits = env.robot.data.soft_joint_pos_limits[:, name_to_id[child]]
            expected = expected.clamp(limits[:, 0], limits[:, 1])
            mimic_error[:, column] = (full_pos[:, name_to_id[child]] - expected).abs()

        def row(index: int) -> dict:
            return {
                "generation": generation,
                "candidate_id": int(index),
                "score": float(score[index].cpu()),
                "action": [float(v) for v in actions[index].cpu()],
                "hold_contact_occupancy": [float(v) for v in occupancy[index].cpu()],
                "all_five_hold_occupancy": float((all_five_hold[index] / max(float(args.hold_steps), 1.0)).cpu()),
                "max_simultaneous_physical_digits": int(max_simultaneous[index].cpu()),
                "max_coverage_deg": float(max_coverage[index].cpu()),
                "min_opposition_cos": float(min_opposition[index].cpu()),
                "min_hemisphere_score": float(min_hemisphere[index].cpu()),
                "final_tip_distance_m": [float(v) for v in final_tip_distance[index].cpu()],
                "peak_digit_force_n": [float(v) for v in max_force[index].cpu()],
                "distal_all_participated": bool(distal_all[index].cpu()),
                "overload_seen": bool(overload[index].cpu()),
                "mimic_abs_error_rad": [float(v) for v in mimic_error[index].cpu()],
                "final_fruit_center_m": [float(v) for v in final_fruit_center[index].cpu()],
            }

        top = [row(int(index)) for index in order[:12].tolist()]
        all_rows.extend(top)
        gen_best = top[0]
        print(json.dumps({
            "generation": generation,
            "best_score": gen_best["score"],
            "best_simultaneous": gen_best["max_simultaneous_physical_digits"],
            "best_all_five_occupancy": gen_best["all_five_hold_occupancy"],
        }), flush=True)
        if best_global is None or gen_best["score"] > best_global["score"]:
            best_global = gen_best

    report = {
        "version": "v3_dynamic_joint_cem_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_command_allowed": False,
        "interpretation": "fruit-present rollout CEM; scores reflect load-deflected dynamic envelopes",
        "scenario": str(scenario_path),
        "scenario_sha256": sha256_file(scenario_path),
        "scenario_canonical_sha256": scenario_hash,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "generations": args.generations,
        "warm_start": str(args.warm_start) if args.warm_start else None,
        "all_five_seen_count": sum(1 for r in all_rows if r["max_simultaneous_physical_digits"] == 5),
        "ranked_candidates": sorted(all_rows, key=lambda r: r["score"], reverse=True)[:32],
    }
    atomic_json_write(output, report)
    print(json.dumps({"output": str(output), "best": report["ranked_candidates"][0]}, indent=2), flush=True)
    env.close()
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    from simulation.ppo_tomato_grasp_v3.direct_joint_action import joint_targets_to_v3_direct_action
    main()
