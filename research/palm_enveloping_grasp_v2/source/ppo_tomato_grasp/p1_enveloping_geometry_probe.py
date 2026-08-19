"""P1 PhysX feasibility probe for large-fruit palm-supported enclosure.

One invocation evaluates one isotropic tomato semi-axis with many parallel
candidate hand actions.  It is deliberately a feasibility probe, not PPO:
the hand is driven through an open -> proximal-preposition -> full-curl
sequence, while contact topology, palm support, joint participation and
termination are recorded.  The probe never controls real hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--axis-m", type=float, required=True, help="isotropic tomato semi-axis in metres")
parser.add_argument(
    "--axis-high-m",
    type=float,
    default=None,
    help="upper semi-axis for an anisotropic range; omitted means isotropic",
)
parser.add_argument(
    "--anisotropic",
    action="store_true",
    help="sample three independent axes in [axis-m, axis-high-m]",
)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=20260816)
parser.add_argument("--rest-x", type=float, default=0.040)
parser.add_argument("--rest-y", type=float, default=0.000)
parser.add_argument("--xy-jitter", type=float, default=0.0)
parser.add_argument("--candidate-seed", type=int, default=47600)
parser.add_argument(
    "--no-palm-rim",
    action="store_true",
    help="disable the shallow palm-cup edge pads for a no-capture-wall control",
)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from simulation.ppo_tomato_grasp.physx_tomato_env import (
    PhysxTomatoGraspEnv,
    PhysxTomatoGraspEnvCfg,
)


FINGER_NAMES = ("index", "middle", "ring", "little")
ROOT_IDS = (1, 2, 3, 4)
DISTAL_IDS = (12, 13, 14, 15)


def _candidate_actions(num_envs: int, device: torch.device, seed: int) -> torch.Tensor:
    """Sample around the previously reachable manifold, retaining diversity."""

    generator = torch.Generator(device=device).manual_seed(seed)
    center = torch.tensor(
        (0.30, 0.00, -0.03, -0.79, 1.00, 0.00, -0.79, 0.54, -0.89, -0.02),
        device=device,
    )
    scale = torch.tensor(
        (0.18, 0.45, 0.35, 0.25, 0.06, 0.25, 0.22, 0.22, 0.20, 0.25),
        device=device,
    )
    candidates = (center + scale * torch.randn((num_envs, 10), device=device, generator=generator)).clamp(-1.0, 1.0)
    candidates[0] = center
    # Keep the opposition coordinate in the physically useful branch and
    # include explicit distal-emphasis candidates through curl_shape.
    candidates[:, 4] = candidates[:, 4].clamp(0.70, 1.0)
    candidates[:, 1] = candidates[:, 1].clamp(-0.25, 1.0)
    if num_envs > 1:
        candidates[1, 1] = 0.85
    if num_envs > 2:
        candidates[2, 1] = 1.0
    return candidates


def _angular_coverage_deg(tips: torch.Tensor, center: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    """Proxy coverage from fingertip directions; marked as proxy in output."""

    values = torch.zeros(tips.shape[0], device=tips.device)
    for row in range(tips.shape[0]):
        indices = active[row].nonzero(as_tuple=False).flatten()
        if indices.numel() < 3:
            continue
        points = tips[row, indices] - center[row]
        angles = torch.atan2(points[:, 1], points[:, 0])
        angles = torch.sort(angles).values
        gaps = torch.diff(angles)
        wrap = (angles[0] + 2.0 * math.pi) - angles[-1]
        gaps = torch.cat((gaps, wrap.reshape(1)))
        values[row] = (2.0 * math.pi - gaps.max()) * (180.0 / math.pi)
    return values


def _sensor_geometry(env: PhysxTomatoGraspEnv) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("thumb_contact", "index_contact", "middle_contact", "ring_contact", "little_contact", "palm_contact"):
        sensor = env.scene.sensors.get(name)
        if sensor is None:
            result[name] = {"available": False}
            continue
        positions = getattr(sensor.data, "contact_pos_w", None)
        normals = getattr(sensor.data, "contact_normal_w", None)
        result[name] = {
            "available": positions is not None,
            "position_shape": list(positions.shape) if positions is not None else None,
            "normal_available": normals is not None,
            "normal_shape": list(normals.shape) if normals is not None else None,
        }
    return result


def _config() -> PhysxTomatoGraspEnvCfg:
    cfg = PhysxTomatoGraspEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.training_stage = 1
    cfg.tomato_rest_xy = (args_cli.rest_x, args_cli.rest_y)
    cfg.position_jitter = (args_cli.xy_jitter, args_cli.xy_jitter, 0.0)
    axis_high = args_cli.axis_high_m if args_cli.axis_high_m is not None else args_cli.axis_m
    cfg.events["tomato_geometry"].params["axis_range_m"] = (args_cli.axis_m, axis_high)
    cfg.events["tomato_geometry"].params["isotropic_axes"] = not args_cli.anisotropic
    # Large fruit geometry is being tested; do not reject a physically
    # plausible 80--100 mm diameter object merely because the old small-fruit
    # mass prior capped it at 180 g.
    cfg.events["tomato_geometry"].params["accepted_mass_kg"] = (0.02, 1.20)
    cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (700.0, 1050.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.tactile_gain_range = (1.0, 1.0)
    cfg.tactile_bias_range_n = (0.0, 0.0)
    cfg.friction_range = (0.70, 0.70)
    cfg.restitution_range = (0.0, 0.0)
    cfg.stiffness_range_n_m = (18000.0, 18000.0)
    cfg.stem_tensile_limit_n = (0.70, 0.70)
    cfg.stem_critical_duration_s = (0.20, 0.20)
    cfg.pick_speed_range_m_s = (0.015, 0.015)
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)
    if args_cli.no_palm_rim:
        cfg.palm_cup_rim_height_m = 0.0
    return cfg


def main() -> int:
    axis_high = args_cli.axis_high_m if args_cli.axis_high_m is not None else args_cli.axis_m
    if not 0.035 <= args_cli.axis_m <= 0.060 or not 0.035 <= axis_high <= 0.060:
        raise ValueError("--axis-m/--axis-high-m must be in [0.035, 0.060] for the P1 large-fruit sweep")
    if axis_high < args_cli.axis_m:
        raise ValueError("--axis-high-m must be >= --axis-m")
    if args_cli.anisotropic and args_cli.axis_high_m is None:
        raise ValueError("--anisotropic requires --axis-high-m")
    if args_cli.num_envs < 4:
        raise ValueError("--num-envs must be at least 4")

    cfg = _config()
    env = PhysxTomatoGraspEnv(cfg)
    device = env.device
    candidate = _candidate_actions(args_cli.num_envs, device, args_cli.candidate_seed)
    _, _ = env.reset()
    initial_joint = env.robot.data.joint_pos[:, env.active_joint_ids].detach().clone()
    max_joint_delta = torch.zeros_like(initial_joint)
    max_contact = torch.zeros((args_cli.num_envs, 5), device=device)
    max_simultaneous = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    max_all_five_palm_streak = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    all_five_palm_streak = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    max_coverage = torch.zeros(args_cli.num_envs, device=device)
    palm_peak = torch.zeros(args_cli.num_envs, device=device)
    first_palm = torch.full((args_cli.num_envs,), -1, dtype=torch.long, device=device)
    first_three = torch.full((args_cli.num_envs,), -1, dtype=torch.long, device=device)
    first_five = torch.full((args_cli.num_envs,), -1, dtype=torch.long, device=device)
    first_done = torch.full((args_cli.num_envs,), -1, dtype=torch.long, device=device)
    termination_code = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    alive = torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)

    open_action = torch.tensor((-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0), device=device)
    proximal = candidate.clone()
    proximal[:, 1] = 0.0
    proximal[:, 5:] = 0.0
    phases = (
        (open_action.expand(args_cli.num_envs, -1), 12),
        (proximal, 40),
        (candidate, 90),
        (candidate, 50),
    )
    previous_target = open_action.expand(args_cli.num_envs, -1).clone()
    step_counter = 0
    for target, phase_steps in phases:
        for phase_step in range(phase_steps):
            fraction = min(1.0, float(phase_step + 1) / max(1, min(25, phase_steps // 2)))
            action = previous_target + (target - previous_target) * fraction
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            sample = alive
            step_counter += 1

            joint_now = env.robot.data.joint_pos[:, env.active_joint_ids]
            max_joint_delta = torch.where(
                sample.unsqueeze(-1),
                torch.maximum(max_joint_delta, (joint_now - initial_joint).abs()),
                max_joint_delta,
            )
            max_contact = torch.where(
                sample.unsqueeze(-1), torch.maximum(max_contact, env.contact_forces), max_contact
            )
            palm_peak = torch.where(sample, torch.maximum(palm_peak, env.palm_force), palm_peak)
            active = env.contact_forces >= cfg.contact_threshold_n
            simultaneous = active.sum(dim=-1)
            max_simultaneous = torch.where(
                sample, torch.maximum(max_simultaneous, simultaneous), max_simultaneous
            )
            palm_ok = env.palm_force >= cfg.palm_support_force_n
            first_palm = torch.where(sample & palm_ok & (first_palm < 0), torch.full_like(first_palm, step_counter), first_palm)
            first_three = torch.where(sample & (simultaneous >= 3) & (first_three < 0), torch.full_like(first_three, step_counter), first_three)
            first_five = torch.where(sample & (simultaneous == 5) & (first_five < 0), torch.full_like(first_five, step_counter), first_five)
            all_five_palm_streak = torch.where(sample & (simultaneous == 5) & palm_ok, all_five_palm_streak + 1, torch.zeros_like(all_five_palm_streak))
            max_all_five_palm_streak = torch.maximum(max_all_five_palm_streak, all_five_palm_streak)
            coverage = _angular_coverage_deg(env._tip_positions(), env.tomato.data.root_pos_w, active)
            max_coverage = torch.where(sample, torch.maximum(max_coverage, coverage), max_coverage)
            first_done = torch.where(sample & done & (first_done < 0), torch.full_like(first_done, step_counter), first_done)
            termination_code = torch.where(sample & done, env.last_termination_code, termination_code)
            alive &= ~done
        previous_target = target.clone()

    root_delta = max_joint_delta[:, list(ROOT_IDS)]
    distal_delta = max_joint_delta[:, list(DISTAL_IDS)]
    distal_participation = distal_delta > 0.03
    proximal_only = (root_delta.mean(dim=-1) > 0.15) & (distal_delta.mean(dim=-1) < 0.03) & (max_simultaneous < 5)
    palm_first_margin = torch.where((first_palm >= 0) & (first_three >= 0), first_three - first_palm, torch.full_like(first_palm, -1))
    strict_candidate = (
        (max_simultaneous == 5)
        & (max_all_five_palm_streak >= 10)
        & (first_palm >= 0)
        & (first_three >= 0)
        & (palm_first_margin >= 3)
        & (max_coverage >= 120.0)
        & distal_participation.all(dim=-1)
        & ~proximal_only
    )
    results = []
    for index in range(args_cli.num_envs):
        results.append({
            "env": index,
            "action": candidate[index].detach().cpu().tolist(),
            "max_simultaneous_digits": int(max_simultaneous[index]),
            "max_all_five_palm_streak": int(max_all_five_palm_streak[index]),
            "max_angular_coverage_deg_proxy": float(max_coverage[index]),
            "palm_peak_n": float(palm_peak[index]),
            "peak_digit_force_n": max_contact[index].detach().cpu().tolist(),
            "first_palm_support_step": int(first_palm[index]),
            "first_three_digit_step": int(first_three[index]),
            "first_all_five_step": int(first_five[index]),
            "palm_first_margin_steps": int(palm_first_margin[index]),
            "root_joint_delta_rad": root_delta[index].detach().cpu().tolist(),
            "distal_joint_delta_rad": distal_delta[index].detach().cpu().tolist(),
            "distal_joint_participation": distal_participation[index].detach().cpu().tolist(),
            "proximal_only_closure": bool(proximal_only[index]),
            "first_termination_step": int(first_done[index]),
            "termination_code": int(termination_code[index]),
            "strict_candidate": bool(strict_candidate[index]),
        })
    results.sort(key=lambda row: (
        row["strict_candidate"],
        row["max_all_five_palm_streak"],
        row["max_angular_coverage_deg_proxy"],
        row["max_simultaneous_digits"],
    ), reverse=True)
    report = {
        "schema_version": 1,
        "audit": "p1_physx_large_fruit_geometry_probe",
        "axis_m": args_cli.axis_m,
        "axis_high_m": axis_high,
        "geometry_mode": "anisotropic_range" if args_cli.anisotropic else "isotropic",
        "diameter_m": 2.0 * args_cli.axis_m,
        "diameter_mm": 2000.0 * args_cli.axis_m,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "candidate_seed": args_cli.candidate_seed,
        "rest_xy_m": [args_cli.rest_x, args_cli.rest_y],
        "xy_jitter_m": args_cli.xy_jitter,
        "collision_variant": "flat_support_no_rim" if args_cli.no_palm_rim else "palm_support_with_rim_prior",
        "contact_geometry_source": "fingertip_body_position_proxy_for_P1; ContactSensor contact_pos_w availability recorded separately",
        "sensor_geometry": _sensor_geometry(env),
        "candidate_count": len(results),
        "strict_candidate_count": int(sum(row["strict_candidate"] for row in results)),
        "max_simultaneous_digits": int(max(row["max_simultaneous_digits"] for row in results)),
        "max_angular_coverage_deg_proxy": float(max(row["max_angular_coverage_deg_proxy"] for row in results)),
        "max_all_five_palm_streak": int(max(row["max_all_five_palm_streak"] for row in results)),
        "all_four_distal_participation_rate": float(sum(all(row["distal_joint_participation"]) for row in results) / len(results)),
        "proximal_only_closure_rate": float(sum(row["proximal_only_closure"] for row in results) / len(results)),
        "top_candidates": results[: min(10, len(results))],
        "hardware_command_allowed": False,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args_cli.output.with_suffix(args_cli.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args_cli.output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    # Isaac Sim 4.5 can block inside ``SimulationApp.close()`` while Kit
    # extension worker threads unwind.  The report is atomically written
    # above, so terminate this short-lived audit process explicitly instead
    # of keeping a GPU context around and blocking the next sweep point.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
