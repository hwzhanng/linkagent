"""Capture representative first-episode trajectories for a stage-1 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=37100)
parser.add_argument("--stage", type=int, choices=(1, 2, 4), default=1)
parser.add_argument("--stage4-band", type=int, choices=(4,), default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch import nn

from simulation.ppo_tomato_grasp.physx_tomato_env import PhysxTomatoGraspEnv, PhysxTomatoGraspEnvCfg


class ReplayActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(110, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.actor_mean = nn.Linear(128, 10)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor_mean(self.actor_body(observation)))


def main() -> None:
    checkpoint_path = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    actor = ReplayActor().to(args.device)
    actor.load_state_dict({
        key: value for key, value in payload["model"].items()
        if key.startswith(("actor_body.", "actor_mean."))
    })
    actor.eval()
    cfg = PhysxTomatoGraspEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.training_stage = min(args.stage, 2)
    cfg.events["tomato_geometry"].params["axis_range_m"] = (0.032, 0.035)
    cfg.events["tomato_geometry"].params["isotropic_axes"] = True
    cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (850.0, 850.0)
    cfg.position_jitter = (0.0005, 0.0005, 0.00025)
    cfg.friction_range = ((0.65, 0.65) if args.stage == 2 else (0.55, 0.55))
    cfg.restitution_range = (0.0, 0.0)
    cfg.stiffness_range_n_m = (18000.0, 18000.0)
    cfg.tactile_gain_range = (1.0, 1.0)
    cfg.tactile_bias_range_n = (0.0, 0.0)
    cfg.sensor_delay_steps_max = 0
    cfg.sensor_dropout_probability = 0.0
    cfg.sensor_stale_probability = 0.0
    if args.stage == 2:
        cfg.pick_speed_m_s = 0.010
        cfg.stem_tensile_limit_n = (0.20, 0.20)
        cfg.stem_critical_duration_s = (0.08, 0.08)
    elif args.stage == 4:
        cfg.events["tomato_geometry"].params["isotropic_axes"] = False
        cfg.events["tomato_geometry"].params["density_range_kg_m3"] = (700.0, 1050.0)
        cfg.position_jitter = (0.0040, 0.0040, 0.0020)
        cfg.friction_range = (0.25, 0.90)
        cfg.restitution_range = (0.0, 0.06)
        cfg.stiffness_range_n_m = (2500.0, 28000.0)
        cfg.tactile_gain_range = (0.82, 1.18)
        cfg.tactile_bias_range_n = (-0.03, 0.03)
        cfg.sensor_delay_steps_max = 3
        cfg.sensor_dropout_probability = 0.020
        cfg.sensor_stale_probability = 0.020
        cfg.actuator_response_scale_range = (0.72, 1.08)
        cfg.actuator_bias_range_rad = (-0.012, 0.012)
        cfg.actuator_deadband_range_rad = (0.0, 0.010)
        cfg.actuator_backlash_range_rad = (0.0, 0.015)
        cfg.pick_speed_range_m_s = (0.010, 0.020)
        cfg.pick_direction_tilt_max_rad = 0.0872664626
        cfg.pick_speed_m_s = 0.015
        cfg.pick_twist_rate_rad_s = 0.10
        cfg.pick_twist_angle_rad = 0.15
        cfg.stem_tensile_limit_n = (0.35, 1.05)
        cfg.stem_critical_duration_s = (0.08, 0.38)
    env = PhysxTomatoGraspEnv(cfg)
    try:
        observation, _ = env.reset(seed=args.seed)
        policy_obs = observation["policy"]
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
        success = torch.zeros_like(alive)
        dropped = torch.zeros_like(alive)
        overloaded = torch.zeros_like(alive)
        frames: dict[str, list[np.ndarray]] = {
            name: [] for name in (
                "joint_pos_rad", "tomato_pos_m", "tomato_quat_wxyz", "action",
                "digit_force_n", "palm_force_n", "stable_steps", "alive",
                "pick_started", "stem_broke", "stem_damage",
            )
        }
        for _ in range(args.steps):
            with torch.inference_mode():
                action = actor(policy_obs)
            frames["joint_pos_rad"].append(env.robot.data.joint_pos[:, env.active_joint_ids].detach().cpu().numpy())
            frames["tomato_pos_m"].append((env.tomato.data.root_pos_w - env.scene.env_origins).detach().cpu().numpy())
            frames["tomato_quat_wxyz"].append(env.tomato.data.root_quat_w.detach().cpu().numpy())
            frames["action"].append(action.detach().cpu().numpy())
            frames["digit_force_n"].append(env.contact_forces.detach().cpu().numpy())
            frames["palm_force_n"].append(env.palm_force.detach().cpu().numpy())
            frames["stable_steps"].append(env.stable_steps.detach().cpu().numpy())
            frames["alive"].append(alive.detach().cpu().numpy())
            frames["pick_started"].append(env.pick_started.detach().cpu().numpy())
            frames["stem_broke"].append(env.stem_broke.detach().cpu().numpy())
            frames["stem_damage"].append(env.stem_damage.detach().cpu().numpy())
            observation, _, terminated, truncated, _ = env.step(action)
            code = env.last_termination_code
            success |= alive & (env.success | (code == 1))
            dropped |= alive & (env.dropped | (code == 2))
            overloaded |= alive & (env.overloaded | (code == 3))
            alive &= ~(terminated | truncated)
            policy_obs = observation["policy"]
            if not bool(alive.any()):
                break
        stacked = {key: np.stack(value) for key, value in frames.items()}
        success_ids = success.nonzero(as_tuple=False).squeeze(-1)
        failure_ids = (~success).nonzero(as_tuple=False).squeeze(-1)
        if success_ids.numel() == 0 or failure_ids.numel() == 0:
            raise RuntimeError("capture batch must contain both success and failure")
        selected = {"success": int(success_ids[0]), "failure": int(failure_ids[0])}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for label, env_id in selected.items():
            output = args.output_dir / f"{label}_seed{args.seed}_env{env_id}.npz"
            np.savez_compressed(
                output,
                **{key: value[:, env_id] for key, value in stacked.items()},
                seed=np.asarray(args.seed), env_index=np.asarray(env_id),
                success=np.asarray(bool(success[env_id])),
                dropped=np.asarray(bool(dropped[env_id])),
                overloaded=np.asarray(bool(overloaded[env_id])),
            )
            files[label] = {
                "path": str(output.resolve()),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "env_index": env_id,
            }
        summary = {
            "seed": args.seed,
            "environments": args.num_envs,
            "first_episode_successes": int(success.sum()),
            "first_episode_drops": int(dropped.sum()),
            "first_episode_overloads": int(overloaded.sum()),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "files": files,
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("STAGE1_REPLAY=" + json.dumps(summary), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
