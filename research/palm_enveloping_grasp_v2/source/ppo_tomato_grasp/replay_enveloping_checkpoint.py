"""Headless randomized rollout recorder for an enveloping-grasp checkpoint.

The recorder deliberately does not open a GUI.  It evaluates a checkpoint in
the same independent PhysX randomization used by the verifier and stores a
compact torch artifact containing joint motion, object pose, tactile forces,
contact geometry, actions, and termination codes.  The artifact is suitable
for later plotting or offline video rendering without replaying the policy in
the Isaac Sim window.
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
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=16)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--seed", type=int, default=20260820)
parser.add_argument("--coverage-deg", type=float, default=200.0)
parser.add_argument("--hold-steps", type=int, default=10)
parser.add_argument("--axis-low", type=float, default=0.038)
parser.add_argument("--axis-high", type=float, default=0.050)
parser.add_argument("--safe-force", type=float, default=12.0)
parser.add_argument("--stage", type=int, choices=(1, 2), default=2)
parser.add_argument("--nominal-physics", action="store_true",
                    help="disable material, sensor and actuator randomization")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch
from torch import nn

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
    cfg.safe_force_n = args_cli.safe_force
    cfg.force_ceiling_n = (args_cli.safe_force, 0.85 * args_cli.safe_force,
                           0.85 * args_cli.safe_force, 0.75 * args_cli.safe_force,
                           0.75 * args_cli.safe_force)
    cfg.severe_overload_scale = 1.40
    cfg.episode_length_s = 12.0
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
    if args_cli.num_envs <= 0 or args_cli.episodes <= 0 or args_cli.steps <= 0:
        raise ValueError("num-envs, episodes, and steps must be positive")
    device = torch.device(args_cli.device)
    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (checkpoint.get("actor_observation_dim"), checkpoint.get("action_dim")) != (110, 14):
        raise ValueError("checkpoint does not match the 110D/14D contract")
    actor = Actor().to(device)
    actor.load_state_dict({k: v for k, v in checkpoint["model"].items()
                           if k in actor.state_dict()}, strict=False)
    actor.eval()
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    env = EnvelopingPhysxTomatoGraspEnv(make_cfg())
    obs, _ = env.reset(seed=args_cli.seed)
    records: list[dict[str, torch.Tensor]] = []
    episode_id = torch.arange(env.num_envs, device=device, dtype=torch.int64)
    completed = 0
    try:
        for step in range(args_cli.steps):
            # Capture the state immediately before the command. Isaac Lab
            # resets completed environments inside ``step``; recording before
            # the call therefore preserves the actual final grasp frame.
            geometry, measured = env._geometry_contact_points()
            policy_action = torch.tanh(actor.actor_mean(actor.actor_body(obs["policy"])))
            pre = {
                "env_id": torch.arange(env.num_envs, device=device, dtype=torch.int64),
                "episode_id": episode_id.clone(),
                "step": torch.full((env.num_envs,), step, device=device, dtype=torch.int32),
                "action": policy_action.detach(),
                "joint_pos": env.robot.data.joint_pos[:, env.active_joint_ids].detach(),
                "tomato_pos_w": env.tomato.data.root_pos_w.detach(),
                "tomato_quat_w": env.tomato.data.root_quat_w.detach(),
                "tomato_axes_m": env.tomato_axes.detach(),
                "contact_forces_n": env.contact_forces.detach(),
                "palm_force_n": env.palm_force.detach(),
                "contact_points_w": geometry.detach(),
                "contact_points_measured": measured.detach(),
                # Capture the state before ``step``.  DirectRLEnv may reset
                # completed environments inside that call, so reading these
                # fields afterwards loses the terminal stem/drop state.
                "pick_started": env.pick_started.detach().clone(),
                "pick_progress_m": env.pick_progress_m.detach().clone(),
                "stem_broke": env.stem_broke.detach().clone(),
                "dropped": env.dropped.detach().clone(),
            }
            obs, _, terminated, truncated, _ = env.step(policy_action)
            done = terminated | truncated
            terminal_code = env.last_termination_code.detach().clone()
            # The state snapshot is pre-action, but a break/drop can happen
            # during this action.  Preserve those event bits explicitly so a
            # terminal row remains auditable even if the environment reset
            # its episode buffers before returning.
            pre["stem_broke"] = pre["stem_broke"] | env.last_drop_after_break.detach()
            pre["dropped"] = pre["dropped"] | (terminal_code == 2)
            if bool(done.any()):
                env_ids = torch.nonzero(done, as_tuple=False).flatten()
                take = min(args_cli.episodes - completed, int(env_ids.numel()))
                if take > 0:
                    completed += take
                pre["done"] = done.detach()
                pre["done_code"] = terminal_code
                pre["success"] = (terminal_code == 1)
                if bool(done.any()):
                    episode_id = episode_id + done.to(torch.int64)
            else:
                pre["done"] = done.detach()
                pre["done_code"] = terminal_code
                pre["success"] = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            # One row per environment per step, including nonterminal frames,
            # forms a directly plottable randomized trajectory tensor.
            records.append({key: value.detach().cpu() for key, value in pre.items()})
            if completed >= args_cli.episodes:
                break
        if not records:
            raise RuntimeError("no completed episodes were recorded")
        keys = records[0].keys()
        artifact = {key: torch.cat([item[key] for item in records], dim=0) for key in keys}
        artifact["metadata"] = {
            "version": "enveloping_random_replay_v2",
            "checkpoint": str(checkpoint_path),
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "requested_episodes": args_cli.episodes,
            "recorded_rows": int(artifact["success"].shape[0]),
            "steps_limit": args_cli.steps,
            "axis_range_m": [args_cli.axis_low, args_cli.axis_high],
            "coverage_gate_deg": args_cli.coverage_deg,
            "hold_gate_steps": args_cli.hold_steps,
            "nominal_physics": args_cli.nominal_physics,
            "hardware_command_allowed": False,
            "contact_geometry_note": "PhysX manifold average when finite contact points exist; explicit per-digit fingertip fallback otherwise.",
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args_cli.output)
        summary = {
            **artifact["metadata"],
            "success_rate": float(artifact["success"].float().mean()),
            "drop_rate": float(artifact["dropped"].float().mean()),
            "stem_break_rate": float(artifact["stem_broke"].float().mean()),
            "measured_geometry_rate": float(artifact["contact_points_measured"].float().mean()),
        }
        summary_path = args_cli.output.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
