"""Search for and visualize the nearest-to-success PhysX sphere grasp.

The PPO trainer stores scalar metrics, not per-step trajectories.  This tool
therefore replays the selected checkpoint, searches several deterministic
episodes, saves the closest trajectory, and optionally opens an Isaac Sim
viewport that plays the recorded poses.  The replay is explicitly a
reconstructed sample, not an exact historical training episode.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Replay nearest PhysX sphere grasp")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--search-episodes", type=int, default=12)
parser.add_argument("--steps", type=int, default=160)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output-dir", type=Path, default=None)
parser.add_argument("--replay-fps", type=float, default=30.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from torch import nn

from simulation.ppo_tomato_grasp.physx_sphere_env import PhysxSphereGraspEnv, PhysxSphereGraspEnvCfg


class ReplayActor(nn.Module):
    """Actor-only copy of the frozen 110D/10D PPO network."""

    def __init__(self) -> None:
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(110, 256), nn.LayerNorm(256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(256, 10)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(self.actor_body(obs))


def _load_actor(checkpoint: Path, device: torch.device) -> ReplayActor:
    data = torch.load(checkpoint.expanduser().resolve(), map_location=device, weights_only=False)
    expected = (110, 10)
    if (data.get("actor_observation_dim"), data.get("action_dim")) != expected:
        raise ValueError(f"checkpoint contract mismatch: expected {expected}")
    actor = ReplayActor().to(device)
    actor_state = {k.removeprefix("actor_body."): v for k, v in data["model"].items() if k.startswith("actor_body.")}
    actor.load_state_dict({"actor_body." + k: v for k, v in actor_state.items()}, strict=False)
    actor.actor_mean.load_state_dict({k.removeprefix("actor_mean."): v for k, v in data["model"].items() if k.startswith("actor_mean.")})
    actor.eval()
    return actor


def _capture_frame(env: PhysxSphereGraspEnv, action: torch.Tensor) -> dict[str, torch.Tensor]:
    ids = torch.as_tensor(env.active_joint_ids, device=env.device)
    return {
        "joint_pos": env.robot.data.joint_pos[:, ids].detach().cpu().clone(),
        "sphere_pos": env.sphere.data.root_pos_w.detach().cpu().clone(),
        "sphere_quat": env.sphere.data.root_quat_w.detach().cpu().clone(),
        "contact_forces": env.contact_forces.detach().cpu().clone(),
        "stable_steps": env.stable_steps.detach().cpu().clone(),
        "secure": env.secure_latched.detach().cpu().clone(),
        "success": env.success.detach().cpu().clone(),
        "damage": env.damage_score.detach().cpu().clone(),
        "action": action.detach().cpu().clone(),
    }


def _score(frames: list[dict[str, torch.Tensor]], env_index: int) -> tuple[float, dict[str, float]]:
    stable = torch.stack([f["stable_steps"][env_index] for f in frames]).numpy().astype(float)
    secure = torch.stack([f["secure"][env_index] for f in frames]).numpy().astype(float)
    success = torch.stack([f["success"][env_index] for f in frames]).numpy().astype(float)
    force = torch.stack([f["contact_forces"][env_index] for f in frames]).numpy()
    pos = torch.stack([f["sphere_pos"][env_index] for f in frames]).numpy()
    center = pos[0]
    distance = np.linalg.norm(pos - center[None, :], axis=-1)
    contact_coverage = (force >= 0.04).sum(axis=-1)
    max_stable = float(stable.max())
    max_secure = float(secure.max())
    max_coverage = float(contact_coverage.max())
    min_distance = float(distance.min())
    peak_force = float(force.max())
    # Stable retention dominates; contact coverage and low object drift break ties.
    score = 100.0 * max_stable + 30.0 * max_secure + 8.0 * max_coverage
    score += 4.0 * math.exp(-min_distance / 0.035) - 0.5 * peak_force
    summary = {
        "score": score,
        "max_stable_steps": max_stable,
        "max_secure": max_secure,
        "max_contact_digits": max_coverage,
        "min_center_drift_m": min_distance,
        "peak_contact_force_n": peak_force,
        "success_observed": float(success.max()),
        "best_frame": int(np.argmax(100.0 * stable + 30.0 * secure + 8.0 * contact_coverage)),
    }
    return score, summary


def _set_pose(env: PhysxSphereGraspEnv, joint_pos: torch.Tensor, sphere_pos: torch.Tensor, sphere_quat: torch.Tensor) -> None:
    # Directly write one captured state before each render tick.  This is a
    # visual replay of measured simulated poses; it intentionally does not
    # claim to re-run the contact solver from the saved trajectory.
    robot_ids = torch.tensor([0], device=env.robot.root_physx_view.get_dof_positions().device)
    dofs = env.robot.root_physx_view.get_dof_positions().clone()
    dofs[robot_ids[:, None], torch.as_tensor(env.active_joint_ids, device=dofs.device)] = joint_pos.to(dofs.device)
    env.robot.root_physx_view.set_dof_positions(dofs, indices=robot_ids)
    velocities = env.robot.root_physx_view.get_dof_velocities().clone()
    velocities[robot_ids] = 0.0
    env.robot.root_physx_view.set_dof_velocities(velocities, indices=robot_ids)
    transforms = env.sphere.root_physx_view.get_transforms().clone()
    transforms[robot_ids, :3] = sphere_pos.to(transforms.device)
    # Isaac Sim's PhysX transform view uses xyzw, while IsaacLab root_quat_w is wxyz.
    transforms[robot_ids, 3:7] = sphere_quat[:, [1, 2, 3, 0]].to(transforms.device)
    env.sphere.root_physx_view.set_transforms(transforms, indices=robot_ids)
    env.sphere.root_physx_view.set_velocities(env.sphere.root_physx_view.get_velocities() * 0.0, indices=robot_ids)


def _frame_viewport() -> None:
    if args_cli.headless:
        return
    try:
        import omni.kit.commands
        import omni.kit.viewport.utility
        from pxr import Usd

        viewport = omni.kit.viewport.utility.get_active_viewport()
        if viewport is None:
            return
        prims = ["/World/envs/env_0/Robot", "/World/envs/env_0/Sphere"]
        omni.kit.commands.execute(
            "FramePrimsCommand",
            prim_to_move=viewport.camera_path,
            prims_to_frame=prims,
            time_code=Usd.TimeCode.Default(),
            aspect_ratio=max(viewport.resolution[0] / max(viewport.resolution[1], 1), 1e-3),
            zoom=0.65,
        )
    except Exception as exc:
        print(f"[replay] viewport framing unavailable: {type(exc).__name__}: {exc}", flush=True)


def main() -> None:
    if args_cli.search_episodes <= 0 or args_cli.steps <= 0:
        raise ValueError("search-episodes and steps must be positive")
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    cfg = PhysxSphereGraspEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device
    env = PhysxSphereGraspEnv(cfg, render_mode=None)
    device = env.device
    actor = _load_actor(args_cli.checkpoint, device)
    best_score = -float("inf")
    best_frames: list[dict[str, torch.Tensor]] | None = None
    best_summary: dict[str, float] | None = None
    print(f"[replay] searching {args_cli.search_episodes} episodes x {args_cli.steps} steps from {args_cli.checkpoint}", flush=True)
    with torch.inference_mode():
        for episode in range(args_cli.search_episodes):
            obs, _ = env.reset()
            frames = []
            for _ in range(args_cli.steps):
                action = torch.tanh(actor(obs["policy"]))
                frames.append(_capture_frame(env, action))
                obs, *_ = env.step(action)
            frames.append(_capture_frame(env, torch.zeros((1, 10), device=device)))
            score, summary = _score(frames, 0)
            summary["episode"] = float(episode)
            print(f"[replay] episode={episode} score={score:.3f} stable={summary['max_stable_steps']:.0f} "
                  f"digits={summary['max_contact_digits']:.0f} drift={summary['min_center_drift_m']:.4f}m", flush=True)
            if score > best_score:
                best_score, best_frames, best_summary = score, frames, summary
    assert best_frames is not None and best_summary is not None
    output_dir = args_cli.output_dir or (args_cli.checkpoint.parent / "nearest_replay")
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {key: torch.stack([frame[key][0] for frame in best_frames]).numpy() for key in best_frames[0]}
    np.savez_compressed(output_dir / "nearest_trajectory.npz", **arrays)
    metadata = {
        "kind": "reconstructed_nearest_replay",
        "historical_trajectory_available": False,
        "checkpoint": str(args_cli.checkpoint.expanduser().resolve()),
        "checkpoint_update": int(torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False).get("update", -1)),
        "search_episodes": args_cli.search_episodes,
        "steps": args_cli.steps,
        **best_summary,
    }
    (output_dir / "nearest_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[replay] selected episode={best_summary['episode']:.0f} summary={json.dumps(best_summary)}", flush=True)
    print(f"[replay] trajectory={output_dir / 'nearest_trajectory.npz'}", flush=True)
    if args_cli.headless:
        env.close()
        return

    _frame_viewport()
    print("[replay] viewport replay started; close Isaac Sim or Ctrl-C to exit", flush=True)
    frame_period = 1.0 / max(args_cli.replay_fps, 1.0)
    try:
        # Hold each recorded policy frame for one wall-clock policy interval.
        for frame in best_frames:
            start = time.monotonic()
            joint = frame["joint_pos"]
            pos = frame["sphere_pos"]
            quat = frame["sphere_quat"]
            while time.monotonic() - start < frame_period and simulation_app.is_running():
                _set_pose(env, joint, pos, quat)
                simulation_app.update()
        # Leave the nearest state on screen for inspection.
        final = best_frames[min(int(best_summary["best_frame"]), len(best_frames) - 1)]
        while simulation_app.is_running():
            _set_pose(env, final["joint_pos"], final["sphere_pos"], final["sphere_quat"])
            simulation_app.update()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
