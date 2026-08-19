"""Inspect the resolved open pose and physical fingertip proxy geometry.

This is an offline PhysX diagnostic only.  It records the reset pose, one
bounded open-action step, tip-to-fruit distances, raw and merged force
channels, and the local proxy transform.  It never sends hardware commands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--profile", default="v1_nominal_reliable")
parser.add_argument("--physical-fingertip-proxy", action="store_true")
parser.add_argument("--fingertip-proxy-radius", type=float, default=0.006)
parser.add_argument("--disable-virtual-distal-contact", action="store_true")
parser.add_argument("--hold-steps", type=int, default=180)
parser.add_argument("--seed", type=int, default=20260820)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

from simulation.ppo_tomato_grasp_v2.config_profiles import apply_profile
from simulation.ppo_tomato_grasp_v2.env import (
    V2EnvelopingPhysxTomatoGraspEnv,
    V2EnvelopingPhysxTomatoGraspEnvCfg,
)
from simulation.ppo_tomato_grasp.enveloping_action_manifold import enveloping_action_to_joint_targets


def _json_values(values: torch.Tensor):
    """Convert any rank tensor to JSON-safe nested Python floats."""
    raw = values.detach().cpu().tolist()
    if isinstance(raw, list):
        def convert(item):
            if isinstance(item, list):
                return [convert(child) for child in item]
            return float(item)
        return convert(raw)
    return float(raw)


@torch.inference_mode()
def main() -> None:
    cfg = V2EnvelopingPhysxTomatoGraspEnvCfg()
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.training_stage = 1
    cfg.episode_length_s = 8.0
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)
    cfg.fingertip_collision_proxy_physical = bool(args.physical_fingertip_proxy)
    cfg.fingertip_collision_proxy_radius_m = float(args.fingertip_proxy_radius)
    if args.disable_virtual_distal_contact:
        cfg.virtual_distal_contact_enabled = False
    cfg = apply_profile(cfg, args.profile, num_envs=1, seed=args.seed)
    cfg.position_jitter = (0.0, 0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    cfg.actuator_response_scale_range = (1.0, 1.0)
    cfg.actuator_bias_range_rad = (0.0, 0.0)
    cfg.actuator_deadband_range_rad = (0.0, 0.0)
    cfg.actuator_backlash_range_rad = (0.0, 0.0)

    env = V2EnvelopingPhysxTomatoGraspEnv(cfg)
    env.reset(seed=args.seed)
    tomato = env.tomato.data.root_pos_w.detach().clone()
    tips_reset = env._tip_positions().detach().clone()
    q_reset = env.robot.data.joint_pos[:, env.active_joint_ids].detach().clone()
    d_reset = torch.linalg.vector_norm(tips_reset[:, 1:] - tomato.unsqueeze(1), dim=-1)
    open_action = torch.zeros((1, 14), device=env.device)
    open_action[:, :10] = torch.tensor(
        (-1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        device=env.device,
    )
    # Record the first step and continue holding the open action.  This
    # distinguishes an initial projection impulse from a slow contact/settling
    # failure that appears only after the object or hand moves.
    env.step(open_action)
    max_contact = env.contact_forces.detach().clone()
    max_measured_link4 = env._v2_measured_link4_force.detach().clone()
    max_link3 = env._v2_link3_force.detach().clone()
    max_palm = env.palm_force.detach().clone()
    step_trace = []
    for step in range(max(int(args.hold_steps) - 1, 0)):
        env.step(open_action)
        max_contact = torch.maximum(max_contact, env.contact_forces)
        max_measured_link4 = torch.maximum(max_measured_link4, env._v2_measured_link4_force)
        max_link3 = torch.maximum(max_link3, env._v2_link3_force)
        max_palm = torch.maximum(max_palm, env.palm_force)
        if step < 20 or bool(env.overloaded.any()) or bool(env.dropped.any()):
            step_trace.append({
                "step": step + 1,
                "contact_forces_n": _json_values(env.contact_forces),
                "link3_force_n": _json_values(env._v2_link3_force),
                "measured_link4_force_n": _json_values(env._v2_measured_link4_force),
                "palm_force_n": _json_values(env.palm_force),
                "tomato_pos_w": _json_values(env.tomato.data.root_pos_w),
                "termination_code": [int(v) for v in env.last_termination_code.detach().cpu().tolist()],
                "overloaded": [bool(v) for v in env.overloaded.detach().cpu().tolist()],
                "dropped": [bool(v) for v in env.dropped.detach().cpu().tolist()],
            })
    tips_step = env._tip_positions().detach().clone()
    q_step = env.robot.data.joint_pos[:, env.active_joint_ids].detach().clone()
    d_step = torch.linalg.vector_norm(tips_step[:, 1:] - env.tomato.data.root_pos_w.unsqueeze(1), dim=-1)
    proxy_transforms = []
    for path in getattr(env, "_v2_fingertip_proxy_paths", ()):
        prim = env.sim.stage.GetPrimAtPath(path)
        xform = prim.GetAttribute("xformOp:translate:v2")
        proxy_transforms.append({
            "path": path,
            "valid": bool(prim.IsValid()),
            "local_translate": [float(v) for v in xform.Get()] if xform and xform.HasAuthoredValue() else [0.0, 0.0, 0.0],
        })
    report = {
        "version": "v2_open_pose_contact_diagnostic_v1",
        "profile": args.profile,
        "seed": args.seed,
        "physical_fingertip_proxy": bool(args.physical_fingertip_proxy),
        "fingertip_proxy_radius_m": float(cfg.fingertip_collision_proxy_radius_m),
        "virtual_distal_contact_enabled": bool(cfg.virtual_distal_contact_enabled),
        "tomato_root_pos_w": _json_values(tomato),
        "q_reset_rad": _json_values(q_reset),
        "q_after_one_open_step_rad": _json_values(q_step),
        "action_after_one_open_step": _json_values(env.actions),
        "mapped_target_after_one_open_step_rad": _json_values(enveloping_action_to_joint_targets(env.actions)),
        "prev_targets_after_one_open_step_rad": _json_values(env.prev_targets),
        "tip_positions_reset_w": _json_values(tips_reset),
        "tip_positions_after_one_open_step_w": _json_values(tips_step),
        "tip_to_tomato_distances_reset_m": _json_values(d_reset),
        "tip_to_tomato_distances_after_one_open_step_m": _json_values(d_step),
        "contact_forces_after_step_n": _json_values(env.contact_forces),
        "measured_link4_force_after_step_n": _json_values(env._v2_measured_link4_force),
        "merged_link4_force_after_step_n": _json_values(env._v2_link4_force),
        "link3_force_after_step_n": _json_values(env._v2_link3_force),
        "palm_force_after_step_n": [float(v) for v in env.palm_force.detach().cpu().tolist()],
        "measured_palm_force_after_step_n": [float(v) for v in env._v2_measured_palm_force.detach().cpu().tolist()],
        "hold_steps": int(args.hold_steps),
        "max_contact_forces_during_hold_n": _json_values(max_contact),
        "max_link3_forces_during_hold_n": _json_values(max_link3),
        "max_measured_link4_forces_during_hold_n": _json_values(max_measured_link4),
        "max_palm_force_during_hold_n": _json_values(max_palm),
        "step_trace": step_trace,
        "termination_code": [int(v) for v in env.last_termination_code.detach().cpu().tolist()],
        "dropped": [bool(v) for v in env.dropped.detach().cpu().tolist()],
        "overloaded": [bool(v) for v in env.overloaded.detach().cpu().tolist()],
        "proxy_transforms": proxy_transforms,
        "hardware_command_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
