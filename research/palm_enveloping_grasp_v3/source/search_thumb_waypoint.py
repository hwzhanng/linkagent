"""Search a lift-swing-settle thumb waypoint around the palm collision."""

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
parser.add_argument("--num-envs", type=int, default=192)
parser.add_argument("--phase-steps", type=int, default=120)
parser.add_argument("--hold-steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=20260826)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp.hand_manifold import LEFT_SAFE_LOWER_RAD, LEFT_SAFE_UPPER_RAD
from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v3.direct_joint_action import joint_targets_to_v3_direct_action, v3_direct_action_to_joint_targets
from simulation.ppo_tomato_grasp_v3.env import V3EnvelopingPhysxTomatoGraspEnv, V3EnvelopingPhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp_v3.provenance import atomic_json_write, sha256_file
from simulation.ppo_tomato_grasp_v3.scenario import apply_contact_collision_mode, load_scenario


TARGET_JOINT = (
    0.5874857902526855, 0.9678264260292053, 0.46631214022636414, 0.47739705443382263,
    0.7955846786499023, -0.047879092395305634, -0.12768957018852234,
    -0.03742750734090805, 0.04571472480893135, 0.17167335748672485,
    0.03123142570257187, 0.5707497000694275, 0.38167804479599,
    0.1510673463344574, 0.060747042298316956, 0.4736952483654022,
)
RESET_JOINT = (
    0.6752, 0.02745098, 0.02745098, 0.02745098, 0.02745098,
    -0.9420, -0.13254902, -0.04180392, 0.04588235, 0.16619608,
    1.3248, 0.02529412, 0.02117647, 0.02117647, 0.02117647, 0.02117647,
)


def make_cfg(scenario: dict) -> V3EnvelopingPhysxTomatoGraspEnvCfg:
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
        axis_range_m=cfg.axis_range_m, isotropic_axes=True,
        accepted_mass_kg=cfg.accepted_mass_kg, density_range_kg_m3=cfg.density_range_kg_m3,
    )
    apply_contact_collision_mode(cfg, geometry)
    cfg.fingertip_collision_proxy_radius_m = float(geometry["fingertip_proxy_radius_m"])
    cfg.fingertip_proxy_contact_offset_m = float(geometry["proxy_contact_offset_m"])
    cfg.fingertip_proxy_rest_offset_m = float(geometry["proxy_rest_offset_m"])
    cfg.v3_urdf_terminal_mimic_enabled = True
    cfg.v3_urdf_mimic_servo_rate_rad_s = float(scenario["physics"]["urdf_mimic_servo_rate_rad_s"])
    cfg.v3_action_contract = "direct_flexion14_v1"
    cfg.v3_direct_servo_zero_velocity = True
    cfg.episode_length_s = 40.0
    return cfg


@torch.inference_mode()
def main() -> None:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    scenario_path = args.scenario.expanduser().resolve()
    scenario, scenario_hash = load_scenario(scenario_path)
    env = V3EnvelopingPhysxTomatoGraspEnv(make_cfg(scenario))
    env.reset(seed=args.seed)
    device = env.device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    reset_joint = torch.tensor(RESET_JOINT, device=device).view(1, 16)
    target_joint = torch.tensor(TARGET_JOINT, device=device).view(1, 16)
    open_action = joint_targets_to_v3_direct_action(reset_joint).expand(args.num_envs, -1).clone()
    target_action = joint_targets_to_v3_direct_action(target_joint).expand(args.num_envs, -1).clone()
    lower = torch.tensor(LEFT_SAFE_LOWER_RAD, device=device)
    upper = torch.tensor(LEFT_SAFE_UPPER_RAD, device=device)
    waypoint_joint = reset_joint.expand(args.num_envs, -1).clone()
    # Only the three non-swing thumb axes define the lift waypoint.
    for joint_id in (0, 10, 11):
        waypoint_joint[:, joint_id] = torch.empty(args.num_envs, device=device).uniform_(
            float(lower[joint_id]), float(upper[joint_id]), generator=generator
        )
    waypoint_action = joint_targets_to_v3_direct_action(waypoint_joint)
    action = open_action.clone()
    min_distance = torch.full((args.num_envs,), float("inf"), device=device)
    max_thumb_force = torch.zeros(args.num_envs, device=device)
    max_palm_force = torch.zeros(args.num_envs, device=device)
    max_speed = torch.zeros(args.num_envs, device=device)
    total = 3 * args.phase_steps + args.hold_steps
    lift_channels = [0, 2, 3]
    for step in range(total):
        phase = min(step // args.phase_steps, 3)
        fraction = float((step % args.phase_steps) + 1) / float(args.phase_steps)
        if phase == 0:
            action[:, lift_channels] = open_action[:, lift_channels] + fraction * (
                waypoint_action[:, lift_channels] - open_action[:, lift_channels]
            )
        elif phase == 1:
            action[:, lift_channels] = waypoint_action[:, lift_channels]
            action[:, 1] = open_action[:, 1] + fraction * (target_action[:, 1] - open_action[:, 1])
        else:
            action[:, 1] = target_action[:, 1]
            action[:, lift_channels] = waypoint_action[:, lift_channels] + fraction * (
                target_action[:, lift_channels] - waypoint_action[:, lift_channels]
            ) if phase == 2 else target_action[:, lift_channels]
        env.step(action)
        distance = torch.linalg.vector_norm(env._tip_positions()[:, 0] - env.tomato.data.root_pos_w, dim=-1)
        min_distance = torch.minimum(min_distance, distance)
        max_thumb_force = torch.maximum(max_thumb_force, env._v2_measured_link4_force[:, 0])
        max_palm_force = torch.maximum(max_palm_force, env.palm_force)
        max_speed = torch.maximum(max_speed, torch.linalg.vector_norm(env.tomato.data.root_lin_vel_w, dim=-1))
    actual = env.robot.data.joint_pos[:, env.active_joint_ids]
    target_q = v3_direct_action_to_joint_targets(target_action)
    thumb_ids = [0, 5, 10, 11]
    error = actual[:, thumb_ids] - target_q[:, thumb_ids]
    final_distance = torch.linalg.vector_norm(env._tip_positions()[:, 0] - env.tomato.data.root_pos_w, dim=-1)
    rmse = torch.sqrt(error.square().mean(-1))
    score = rmse + 0.8 * (final_distance - 0.051).abs() + 4.0 * env.dropped.float() + 2.0 * env.overloaded.float()
    order = torch.argsort(score)
    rows = []
    for idx in order[: min(32, args.num_envs)].tolist():
        rows.append({
            "environment_id": idx,
            "score": float(score[idx].cpu()),
            "waypoint_thumb_joint_position_rad": [float(v) for v in waypoint_joint[idx, thumb_ids].cpu()],
            "min_thumb_tip_distance_m": float(min_distance[idx].cpu()),
            "final_thumb_tip_distance_m": float(final_distance[idx].cpu()),
            "peak_thumb_force_n": float(max_thumb_force[idx].cpu()),
            "peak_palm_force_n": float(max_palm_force[idx].cpu()),
            "max_tomato_speed_m_s": float(max_speed[idx].cpu()),
            "final_thumb_joint_position_rad": [float(v) for v in actual[idx, thumb_ids].cpu()],
            "thumb_joint_error_rad": [float(v) for v in error[idx].cpu()],
            "thumb_joint_rmse_rad": float(rmse[idx].cpu()),
            "dropped": bool(env.dropped[idx].cpu()),
            "overloaded": bool(env.overloaded[idx].cpu()),
        })
    report = {
        "version": "v3_thumb_lift_swing_settle_search_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_command_allowed": False,
        "scenario": str(scenario_path),
        "scenario_sha256": sha256_file(scenario_path),
        "scenario_canonical_sha256": scenario_hash,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "phase_steps": args.phase_steps,
        "hold_steps": args.hold_steps,
        "best_waypoint": rows[0],
        "ranked_waypoints": rows,
        "interpretation": "dynamic thumb-only waypoint screen; straight fingers remain open",
    }
    atomic_json_write(output, report)
    print(json.dumps({"output": str(output), "best_waypoint": rows[0]}, indent=2), flush=True)
    env.close()
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
