"""Independent reset-safe v3 evaluator with strict sub-gate traces."""

from __future__ import annotations

import argparse
from copy import deepcopy
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
parser.add_argument("--override", type=Path, help="JSON scenario fragment; merged, validated, and hashed")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--seed", type=int)
parser.add_argument("--num-envs", type=int)
parser.add_argument("--steps", type=int)
parser.add_argument("--residual-scale", type=float, help="audited override; included in the scenario hash")
parser.add_argument("--trace-env", type=int, default=0)
parser.add_argument("--no-trace", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch
from torch import nn

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.contract import V2_ACTION_DIM, V2_ACTOR_OBSERVATION_DIM
from simulation.ppo_tomato_grasp_v2.residual_controller import ResidualBaseController, ResidualBaseControllerCfg
from simulation.ppo_tomato_grasp_v3.env import V3EnvelopingPhysxTomatoGraspEnv, V3EnvelopingPhysxTomatoGraspEnvCfg
from simulation.ppo_tomato_grasp_v3.event_teacher import EventDrivenEnclosureTeacher, EventDrivenTeacherCfg
from simulation.ppo_tomato_grasp_v3.direct_joint_teacher import DirectJointEventTeacher, DirectJointTeacherCfg
from simulation.ppo_tomato_grasp_v3.provenance import atomic_json_write, sha256_file
from simulation.ppo_tomato_grasp_v3.scenario import apply_contact_collision_mode, canonical_hash, derive_run_scenario, load_scenario, merge_scenario_override, validate_scenario
from simulation.ppo_tomato_grasp_v3.strict_telemetry import GATE_ORDER, decode_failed_gates, failure_histogram


class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(V2_ACTOR_OBSERVATION_DIM, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),
        )
        self.mean = nn.Linear(128, V2_ACTION_DIM)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean(self.body(observation)))


def make_cfg(scenario: dict, *, seed: int, num_envs: int, device: str):
    cfg = V3EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = device
    cfg.training_stage = 1
    cfg.strict_enveloping_enabled = True
    cfg.strict_enveloping_require_stem_break = False
    cfg = apply_profile(cfg, scenario["evaluation"]["profile"], num_envs=num_envs, seed=seed)
    task, obj = scenario["task"], scenario["object"]
    geometry, strict = scenario["contact_geometry"], scenario["strict"]
    safety, controller = scenario["safety"], scenario["controller"]
    cfg.position_jitter = tuple(float(v) for v in task["initial_position_jitter_m"])
    cfg.axis_range_m = (float(obj["semi_axes_m"][0]), float(obj["semi_axes_m"][0]))
    cfg.isotropic_axes = True
    cfg.accepted_mass_kg = tuple(float(v) for v in obj["accepted_mass_kg"])
    cfg.density_range_kg_m3 = (float(obj["density_kg_m3"]), float(obj["density_kg_m3"]))
    geometry_event = cfg.events["tomato_geometry"].params
    geometry_event["axis_range_m"] = cfg.axis_range_m
    geometry_event["isotropic_axes"] = True
    geometry_event["accepted_mass_kg"] = cfg.accepted_mass_kg
    geometry_event["density_range_kg_m3"] = cfg.density_range_kg_m3
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
    cfg.strict_require_measured_distal_contact = bool(strict["require_measured_distal_contact"])
    cfg.v3_require_current_physical_five_digit_contact = bool(strict["require_current_physical_five_digit_contact"])
    cfg.v3_sequential_contact_gate = bool(strict.get("sequential_contact_allowed", False))
    cfg.safe_force_n = float(safety["safe_force_n"])
    cfg.force_ceiling_n = tuple(float(v) for v in safety["force_ceiling_n"])
    cfg.command_rate_limit_rad_s = float(safety["command_rate_limit_rad_s"])
    cfg.distal_action_filter_alpha = float(controller["distal_filter_alpha"])
    cfg.v3_urdf_terminal_mimic_enabled = bool(scenario["physics"]["urdf_terminal_mimic_enabled"])
    cfg.v3_urdf_mimic_servo_rate_rad_s = float(scenario["physics"]["urdf_mimic_servo_rate_rad_s"])
    cfg.v3_action_contract = str(controller["action_contract"])
    cfg.v3_direct_servo_zero_velocity = bool(scenario["physics"]["direct_servo_zero_velocity"])
    cfg.position_servo_enabled = bool(scenario["physics"]["direct_position_servo_enabled"])
    cfg.v3_filter_fingertip_proxy_support_collision = bool(
        scenario["physics"]["filter_fingertip_proxy_support_collision"]
    )
    cfg.v3_disable_palm_cup_rim_collision = bool(
        scenario["physics"]["disable_palm_cup_rim_collision"]
    )
    cfg.v3_joint_drive_effort_limit_nm = float(scenario["physics"]["joint_drive_effort_limit_nm"])
    cfg.v3_joint_drive_stiffness_nm_rad = float(scenario["physics"]["joint_drive_stiffness_nm_rad"])
    cfg.v3_joint_drive_damping_nm_s_rad = float(scenario["physics"]["joint_drive_damping_nm_s_rad"])
    cfg.v3_mimic_drive_effort_limit_nm = float(scenario["physics"]["mimic_drive_effort_limit_nm"])
    cfg.v3_mimic_drive_stiffness_nm_rad = float(scenario["physics"]["mimic_drive_stiffness_nm_rad"])
    cfg.v3_mimic_drive_damping_nm_s_rad = float(scenario["physics"]["mimic_drive_damping_nm_s_rad"])
    cfg.v3_mimic_drive_armature_kg_m2 = float(scenario["physics"]["mimic_drive_armature_kg_m2"])
    cfg.v3_hand_disable_gravity = bool(scenario["physics"]["hand_disable_gravity"])
    cfg.v3_use_tomato_filtered_contact_force = bool(
        scenario["physics"]["use_tomato_filtered_contact_force"]
    )
    cfg.v3_contact_history_substeps = int(scenario["physics"]["contact_history_substeps"])
    for sensor_name in (
        "v3_thumb_pad_contact", "v3_index_pad_contact", "v3_middle_pad_contact",
        "v3_ring_pad_contact", "v3_little_pad_contact",
    ):
        getattr(cfg.scene, sensor_name).history_length = cfg.v3_contact_history_substeps
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
    required_steps = int(controller["approach_steps"]) + 4 * int(controller["per_digit_steps"]) + int(controller["hold_steps"]) + 20
    cfg.episode_length_s = max(24.0, required_steps * float(cfg.policy_dt_s))
    return cfg


def make_controller(scenario: dict, num_envs: int, device: torch.device):
    c = scenario["controller"]
    if c["kind"] == "direct_joint_event_v1":
        direct = c["direct_teacher"]
        return DirectJointEventTeacher(
            num_envs,
            device,
            tuple(float(v) for v in c["target_action"]),
            DirectJointTeacherCfg(
                coarse_steps=int(direct["coarse_steps"]),
                distal_steps=int(direct["distal_steps"]),
                force_lower_n=tuple(float(v) for v in direct["force_lower_n"]),
                force_upper_n=tuple(float(v) for v in direct["force_upper_n"]),
                retreat_action_step=float(direct["retreat_action_step"]),
                hold_closure_margin=float(direct["hold_closure_margin"]),
                palm_support_threshold_n=float(direct["palm_support_threshold_n"]),
                precontact_fraction=float(direct["precontact_fraction"]),
                contact_latch_steps=int(direct["contact_latch_steps"]),
                lost_contact_grace_steps=int(direct["lost_contact_grace_steps"]),
                reacquire_action_step=float(direct["reacquire_action_step"]),
                max_closure_extension=float(direct["max_closure_extension"]),
                contact_detect_n=tuple(float(v) for v in direct["contact_detect_n"]),
                contact_latch_start_step=int(direct["contact_latch_start_step"]),
                search_probe_steps=int(direct["search_probe_steps"]),
                high_force_dwell_steps=int(direct["high_force_dwell_steps"]),
                digit_start_steps=(
                    None
                    if direct.get("digit_start_steps") is None
                    else tuple(int(v) for v in direct["digit_start_steps"])
                ),
            ),
        )
    base_cfg = ResidualBaseControllerCfg(
        approach_steps=int(c["approach_steps"]),
        per_digit_steps=int(c["per_digit_steps"]),
        approach_smoothing=float(c["approach_smoothing"]),
        thumb_delay_steps=int(c["thumb_delay_steps"]),
        thumb_ramp_steps=int(c["thumb_ramp_steps"]),
        thumb_first_steps=int(c["thumb_first_steps"]),
        thumb_first_global_scale=float(c["thumb_first_global_scale"]),
        simultaneous_reach=False,
        lower_force_n=tuple(float(v) for v in c["straight_digit_force_lower_n"]),
        upper_force_n=tuple(float(v) for v in c["straight_digit_force_upper_n"]),
        insertion_step_fraction_per_digit=tuple(float(v) for v in c["insertion_step_per_digit"]),
        retreat_fraction_per_digit=tuple(float(v) for v in c["retreat_step_per_digit"]),
        thumb_lower_force_n=float(c["thumb_force_band_n"][0]),
        thumb_upper_force_n=float(c["thumb_force_band_n"][1]),
        thumb_hold_target_offset=tuple(float(v) for v in c["thumb_hold_target_offset"]),
        thumb_search_amplitude=tuple(float(v) for v in c["thumb_search_amplitude"]),
        thumb_search_period_steps=int(c["thumb_search_period_steps"]),
        thumb_first_contact_gate_steps=int(c["thumb_contact_gate_steps"]),
        thumb_contact_latch_force_n=float(c["thumb_contact_latch_force_n"]),
        thumb_contact_latch_steps=int(c["thumb_contact_latch_steps"]),
        thumb_contact_latch_freeze_coarse=bool(c["thumb_contact_latch_freeze_coarse"]),
    )
    target = tuple(float(v) for v in c["target_action"])
    if c["kind"] == "candidate123_residual_base":
        return ResidualBaseController(num_envs, device, target, base_cfg)
    if c["kind"] == "event_latched_v1":
        e = c["event_teacher"]
        return EventDrivenEnclosureTeacher(
            num_envs, device, target, base_cfg,
            EventDrivenTeacherCfg(
                contact_threshold_n=float(e["contact_threshold_n"]),
                force_lower_n=tuple(float(v) for v in e["force_lower_n"]),
                force_upper_n=tuple(float(v) for v in e["force_upper_n"]),
                reacquire_step=float(e["reacquire_step"]),
                retreat_step=float(e["retreat_step"]),
                trust_region=float(e["trust_region"]),
                thumb_reacquire_direction=tuple(float(v) for v in e["thumb_reacquire_direction"]),
                load_share_max=float(e["load_share_max"]),
                search_period_steps=int(e["search_period_steps"]),
                fruit_speed_hold_m_s=float(e["fruit_speed_hold_m_s"]),
                high_force_dwell_steps=int(e.get("high_force_dwell_steps", 3)),
                force_prediction_gain=float(e.get("force_prediction_gain", 1.0)),
                balance_target_n=tuple(float(v) for v in e.get("balance_target_n", e["force_lower_n"])),
                digit_latch_start_step=int(e.get("digit_latch_start_step", 0)),
                palm_latch_threshold_n=float(e.get("palm_latch_threshold_n", 0.30)),
            ),
        )
    raise ValueError(f"unknown controller kind {c['kind']!r}")


def load_actor(checkpoint_path: Path, device: torch.device) -> Actor:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (checkpoint.get("actor_observation_dim"), checkpoint.get("action_dim")) != (
        V2_ACTOR_OBSERVATION_DIM, V2_ACTION_DIM
    ):
        raise ValueError("checkpoint does not match the v2/v3 actor contract")
    model = checkpoint["model"]
    actor = Actor().to(device)
    actor.load_state_dict({
        "body.0.weight": model["actor_body.0.weight"], "body.0.bias": model["actor_body.0.bias"],
        "body.2.weight": model["actor_body.2.weight"], "body.2.bias": model["actor_body.2.bias"],
        "body.4.weight": model["actor_body.4.weight"], "body.4.bias": model["actor_body.4.bias"],
        "mean.weight": model["actor_mean.weight"], "mean.bias": model["actor_mean.bias"],
    })
    actor.eval()
    return actor


def residual_mask(name: str, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(V2_ACTION_DIM, device=device)
    if name == "all":
        mask[:] = 1.0
    elif name == "thumb_distal":
        mask[5] = 1.0
        mask[10:] = 1.0
    elif name == "distal":
        mask[10:] = 1.0
    elif name == "coverage":
        mask[2:] = 1.0
    else:
        raise ValueError(f"unknown residual mask {name!r}")
    return mask


def phase_name(step: int, controller: dict) -> str:
    if step < int(controller["thumb_first_steps"]):
        return "thumb_preopposition"
    if step < int(controller["approach_steps"]):
        return "coarse_approach"
    offset = step - int(controller["approach_steps"])
    digit_window = int(controller["per_digit_steps"])
    if offset < 4 * digit_window:
        return ("index_insert", "middle_insert", "ring_insert", "little_insert")[offset // digit_window]
    return "joint_hold"


def trace_row(env, action: torch.Tensor, index: int, controller: dict) -> dict:
    step = int(env.episode_length_buf[index].item())
    tomato_pos = env.tomato.data.root_pos_w[index]
    tomato_vel = env.tomato.data.root_lin_vel_w[index]
    tip_distance = torch.linalg.vector_norm(
        env._tip_positions()[index] - tomato_pos.unsqueeze(0), dim=-1
    )
    row = {
        "episode_step_before_action": step,
        "phase": phase_name(step, controller),
        "action": [float(v) for v in action[index].detach().cpu().tolist()],
        "physical_digit_force_n": [float(v) for v in env._v2_measured_link4_force[index].detach().cpu().tolist()],
        "unfiltered_digit_net_force_n": [
            float(v) for v in env._v3_unfiltered_link4_force[index].detach().cpu().tolist()
        ],
        "merged_digit_force_n": [float(v) for v in env.contact_forces[index].detach().cpu().tolist()],
        "palm_force_n": float(env.palm_force[index].detach().cpu()),
        "physical_coverage_deg": float(env._v3_physical_coverage_current[index].detach().cpu()),
        "physical_opposition_cos": float(env._v3_physical_opposition_current[index].detach().cpu()),
        "physical_hemisphere_score": float(env._v3_physical_hemisphere_current[index].detach().cpu()),
        "physical_palm_load_share": float(env._v3_physical_palm_load_share[index].detach().cpu()),
        "strict_streak": int(env._v3_all_conditions_streak[index].detach().cpu()),
        "strict_gates": {name: bool(env._v3_gate_current[name][index].detach().cpu()) for name in GATE_ORDER},
        "tomato_position_m": [float(v) for v in tomato_pos.detach().cpu().tolist()],
        "tomato_linear_velocity_m_s": [float(v) for v in tomato_vel.detach().cpu().tolist()],
        "tip_distance_m": [float(v) for v in tip_distance.detach().cpu().tolist()],
        "active_joint_position_rad": [float(v) for v in env.robot.data.joint_pos[index, env.active_joint_ids].detach().cpu().tolist()],
        "active_joint_velocity_rad_s": [float(v) for v in env.robot.data.joint_vel[index, env.active_joint_ids].detach().cpu().tolist()],
        "commanded_active_joint_target_rad": [float(v) for v in env.prev_targets[index].detach().cpu().tolist()],
        "actuator_active_joint_target_rad": [float(v) for v in env.robot.data.joint_pos_target[index, env.active_joint_ids].detach().cpu().tolist()],
        "applied_active_joint_torque_nm": [float(v) for v in env.robot.data.applied_torque[index, env.active_joint_ids].detach().cpu().tolist()],
        "all_joint_names": list(env.robot.joint_names),
        "all_joint_position_rad": [float(v) for v in env.robot.data.joint_pos[index].detach().cpu().tolist()],
        "all_joint_velocity_rad_s": [float(v) for v in env.robot.data.joint_vel[index].detach().cpu().tolist()],
    }
    if len(getattr(env, "_v3_pad_body_ids", ())) == 5:
        row["fixed_pad_position_m"] = [
            [float(v) for v in xyz]
            for xyz in env.robot.data.body_pos_w[
                index, env._v3_pad_body_ids
            ].detach().cpu().tolist()
        ]
    active_controller = getattr(env, "_v3_evaluation_controller", None)
    if isinstance(active_controller, DirectJointEventTeacher):
        row["teacher_contact_latched"] = [
            bool(v) for v in active_controller.contact_latched[index].detach().cpu().tolist()
        ]
        row["teacher_contact_age_steps"] = [
            int(v) for v in active_controller.contact_age[index].detach().cpu().tolist()
        ]
        row["teacher_lost_contact_age_steps"] = [
            int(v) for v in active_controller.lost_contact_age[index].detach().cpu().tolist()
        ]
        row["teacher_search_sign"] = [
            float(v) for v in active_controller.search_sign[index].detach().cpu().tolist()
        ]
        row["teacher_search_active"] = [
            bool(v) for v in active_controller.search_active[index].detach().cpu().tolist()
        ]
        row["teacher_high_force_age_steps"] = [
            int(v) for v in active_controller.high_force_age[index].detach().cpu().tolist()
        ]
        row["teacher_probe_tip_distance_m"] = [
            float(v) for v in active_controller.probe_tip_distance[index].detach().cpu().tolist()
        ]
    return row


@torch.inference_mode()
def main() -> None:
    scenario, scenario_hash = load_scenario(args_cli.scenario)
    override_provenance = None
    if args_cli.override is not None:
        override_path = args_cli.override.expanduser().resolve()
        override = json.loads(override_path.read_text(encoding="utf-8"))
        scenario, scenario_hash = merge_scenario_override(scenario, override)
        override_provenance = {"path": str(override_path), "sha256": sha256_file(override_path)}
    if args_cli.residual_scale is not None:
        if not 0.0 <= args_cli.residual_scale <= 0.20:
            raise ValueError("--residual-scale must be in the audited range [0,0.20]")
        scenario = deepcopy(scenario)
        scenario["controller"]["residual_scale"] = float(args_cli.residual_scale)
        scenario = validate_scenario(scenario)
        scenario_hash = canonical_hash(scenario)
    defaults = scenario["evaluation"]
    seed = int(defaults["seed"] if args_cli.seed is None else args_cli.seed)
    num_envs = int(defaults["num_envs"] if args_cli.num_envs is None else args_cli.num_envs)
    steps = int(defaults["steps"] if args_cli.steps is None else args_cli.steps)
    if not 0 <= args_cli.trace_env < num_envs:
        raise ValueError("--trace-env must identify an existing environment")
    run_scenario, run_hash = derive_run_scenario(
        scenario, seed=seed, num_envs=num_envs, steps=steps, purpose="p1_v3_strict_evaluation"
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args_cli.device)
    checkpoint_path = (
        args_cli.checkpoint.expanduser().resolve()
        if args_cli.checkpoint is not None
        else (PROJECT / scenario["evaluation"]["checkpoint"]).resolve()
    )
    actor = load_actor(checkpoint_path, device) if float(scenario["controller"]["residual_scale"]) > 0.0 else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env = V3EnvelopingPhysxTomatoGraspEnv(make_cfg(scenario, seed=seed, num_envs=num_envs, device=args_cli.device))
    observation, _ = env.reset(seed=seed)
    controller = make_controller(scenario, num_envs, device)
    # Trace-only handle; the environment does not use or own the controller.
    env._v3_evaluation_controller = controller
    actor_mask = residual_mask(scenario["controller"]["residual_mask"], device)
    completed = 0
    records: list[dict] = []
    trace: list[dict] = []
    for _ in range(steps):
        if isinstance(controller, DirectJointEventTeacher):
            base_action = controller.command(
                env.episode_length_buf,
                env._v2_measured_link4_force,
                env.palm_force,
                torch.linalg.vector_norm(env.tomato.data.root_lin_vel_w, dim=-1),
                torch.linalg.vector_norm(
                    env._tip_positions() - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1
                ),
                env.robot.data.joint_pos[:, env.active_joint_ids],
            )
        elif isinstance(controller, EventDrivenEnclosureTeacher):
            base_action = controller.command(
                env.episode_length_buf,
                env._v2_measured_link4_force,
                env.palm_force,
                torch.linalg.vector_norm(env.tomato.data.root_lin_vel_w, dim=-1),
            )
        else:
            base_action = controller.command(env.episode_length_buf, env._v2_measured_link4_force)
        if actor is None:
            action = base_action
        else:
            residual = actor(observation["policy"])
            scale = float(scenario["controller"]["residual_scale"])
            phase_gate = (env.episode_length_buf >= int(scenario["controller"]["approach_steps"])).float()
            action = (
                base_action + scale * residual * actor_mask.unsqueeze(0) * phase_gate.unsqueeze(-1)
            ).clamp(-1.0, 1.0)
        if not args_cli.no_trace:
            trace.append(trace_row(env, action, args_cli.trace_env, scenario["controller"]))
        observation, _reward, terminated, truncated, _info = env.step(action)
        done = terminated | truncated
        if bool(done.any()):
            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            snapshot = env.consume_terminal_snapshots(done)
            for local_index in range(done_ids.numel()):
                break_code = int(snapshot["v3_first_break_code"][local_index].cpu())
                terminal_code = int(snapshot["v3_terminal_failure_code"][local_index].cpu())
                record = {
                    "environment_id": int(done_ids[local_index].cpu()),
                    "termination_code": int(snapshot["termination_code"][local_index].cpu()),
                    "strict_success": bool(snapshot["v3_strict_success"][local_index].cpu()),
                    "dropped": bool(snapshot["dropped"][local_index].cpu()),
                    "overloaded": bool(snapshot["overloaded"][local_index].cpu()),
                    "all_conditions_max_streak": int(snapshot["v3_all_conditions_max_streak"][local_index].cpu()),
                    "first_three_physical_step": int(snapshot["v3_first_three_physical"][local_index].cpu()),
                    "first_five_physical_step": int(snapshot["v3_first_five_physical"][local_index].cpu()),
                    "first_strict_step": int(snapshot["v3_first_strict_step"][local_index].cpu()),
                    "first_break_step": int(snapshot["v3_first_break_step"][local_index].cpu()),
                    "first_break_code": break_code,
                    "first_break_reasons": decode_failed_gates(break_code),
                    "first_five_failure_code": int(snapshot["v3_first_five_failure_code"][local_index].cpu()),
                    "first_five_failure_reasons": decode_failed_gates(int(snapshot["v3_first_five_failure_code"][local_index].cpu())),
                    "best_gate_pass_count": int(snapshot["v3_best_gate_pass_count"][local_index].cpu()),
                    "best_gate_step": int(snapshot["v3_best_gate_step"][local_index].cpu()),
                    "best_gate_failure_code": int(snapshot["v3_best_gate_failure_code"][local_index].cpu()),
                    "best_gate_failure_reasons": decode_failed_gates(int(snapshot["v3_best_gate_failure_code"][local_index].cpu())),
                    "terminal_failure_code": terminal_code,
                    "terminal_failure_reasons": decode_failed_gates(terminal_code),
                    "physical_coverage_max_deg": float(snapshot["v3_physical_coverage_max"][local_index].cpu()),
                    "physical_opposition_min_cos": float(snapshot["v3_physical_opposition_min"][local_index].cpu()),
                    "physical_hemisphere_min_score": float(snapshot["v3_physical_hemisphere_min"][local_index].cpu()),
                    "terminal_physical_palm_load_share": float(snapshot["v3_physical_palm_load_share"][local_index].cpu()),
                    "terminal_physical_five": bool(snapshot["v3_physical_five_current"][local_index].cpu()),
                    "max_simultaneous_physical_digits": int(snapshot["v3_max_simultaneous_physical"][local_index].cpu()),
                    "first_physical_digit_step": [int(v) for v in snapshot["v3_first_physical_digit_step"][local_index].cpu().tolist()],
                    "physical_digit_max_streak": [int(v) for v in snapshot["v3_physical_digit_max_streak"][local_index].cpu().tolist()],
                    "peak_physical_digit_force_n": [float(v) for v in snapshot["v3_physical_peak_force"][local_index].cpu().tolist()],
                    "peak_merged_digit_force_n": [float(v) for v in snapshot["peak_force"][local_index].cpu().tolist()],
                    "peak_palm_force_n": float(snapshot["peak_palm_force"][local_index].cpu()),
                    "gate_max_streak": {
                        name: int(snapshot[f"v3_{name}_max_streak"][local_index].cpu()) for name in GATE_ORDER
                    },
                    "leave_one_out_max_streak": {
                        name: int(snapshot[f"v3_without_{name}_max_streak"][local_index].cpu()) for name in GATE_ORDER
                    },
                }
                records.append(record)
            completed += int(done.sum().item())
            controller.reset(done_ids)
        if completed >= num_envs:
            break
    if completed < num_envs:
        raise RuntimeError(f"evaluation ended with only {completed}/{num_envs} completed episodes")
    first_break_codes = [record["first_break_code"] for record in records if record["first_break_step"] >= 0]
    no_strict_window_count = sum(record["first_strict_step"] < 0 for record in records)
    terminal_codes = [record["terminal_failure_code"] for record in records]
    report = {
        "version": "v3_independent_strict_eval_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": run_scenario,
        "scenario_sha256": scenario_hash,
        "scenario_override": override_provenance,
        "run_sha256": run_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_sha256": sha256_file(Path(__file__)),
        "hardware_command_allowed": False,
        "resolved_thresholds": {
            "physical_contact_threshold_n": float(env.cfg.contact_threshold_n),
            "palm_support_force_n": float(env.cfg.palm_support_force_n),
            "coverage_min_deg": float(env.cfg.strict_angular_coverage_min_deg),
            "thumb_opposition_cos_max": float(env.cfg.strict_thumb_opposition_cos_max),
            "same_hemisphere_score_max": float(env.cfg.strict_same_hemisphere_score_max),
            "palm_load_share": [float(env.cfg.strict_palm_load_share_min), float(env.cfg.strict_palm_load_share_max)],
            "strict_hold_steps": int(env.cfg.strict_hold_steps),
        },
        "completed_episodes": completed,
        "success_rate": sum(record["strict_success"] for record in records) / completed,
        "drop_rate": sum(record["dropped"] for record in records) / completed,
        "overload_rate": sum(record["overloaded"] for record in records) / completed,
        "all_conditions_max_streak_mean": sum(record["all_conditions_max_streak"] for record in records) / completed,
        "all_conditions_max_streak_max": max(record["all_conditions_max_streak"] for record in records),
        "physical_coverage_max_mean_deg": sum(record["physical_coverage_max_deg"] for record in records) / completed,
        "first_break_histogram": failure_histogram(first_break_codes),
        "first_break_not_applicable_no_strict_window": no_strict_window_count,
        "terminal_failure_histogram": failure_histogram(terminal_codes),
        "episode_records": records,
        "trace_env": None if args_cli.no_trace else args_cli.trace_env,
        "trace_semantics": "state_before_action; terminal frame is represented by episode_records",
        "trace": trace,
    }
    output = args_cli.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v3 evaluation: {output}")
    atomic_json_write(output, report)
    print(json.dumps({key: report[key] for key in (
        "version", "scenario_sha256", "run_sha256", "completed_episodes", "success_rate",
        "drop_rate", "overload_rate", "all_conditions_max_streak_mean",
        "all_conditions_max_streak_max", "physical_coverage_max_mean_deg",
        "first_break_histogram", "terminal_failure_histogram",
    )}, indent=2))
    env.close()
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
