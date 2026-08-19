"""Open the recorded final tomato grasp replay in the Isaac Sim viewport.

This is an inspection-only viewer.  It does not retrain, overwrite a
checkpoint, or claim to reproduce the original rollout's contact solver
history: recorded robot and tomato poses are applied to the loaded PhysX hand
asset at the display rate so the viewer can inspect the learned motion.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="View a recorded tomato grasp in Isaac Sim")
parser.add_argument(
    "--replay",
    type=Path,
    default=ROOT / "artifacts/stage4_main_band4_replays/success_seed47650_env0.npz",
)
parser.add_argument("--fps", type=float, default=20.0)
parser.add_argument("--hold-seconds", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=47650)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from simulation.ppo_tomato_grasp.physx_tomato_env import (
    PhysxTomatoGraspEnv,
    PhysxTomatoGraspEnvCfg,
)


def _configure_viewer() -> PhysxTomatoGraspEnvCfg:
    """Use the final large-fruit/non-size-randomized delivery band."""

    cfg = PhysxTomatoGraspEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device
    cfg.training_stage = 2
    cfg.events["tomato_geometry"].params["axis_range_m"] = (0.032, 0.035)
    cfg.events["tomato_geometry"].params["isotropic_axes"] = False
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
    cfg.pick_speed_range_m_s = (0.010, 0.020)
    cfg.pick_direction_tilt_max_rad = np.deg2rad(5.0)
    cfg.pick_speed_m_s = 0.015
    cfg.stem_tensile_limit_n = (0.35, 1.05)
    cfg.stem_critical_duration_s = (0.08, 0.38)
    return cfg


def _frame_viewport() -> None:
    """Frame the one hand and tomato in the active viewport."""

    if args_cli.headless:
        return
    try:
        import omni.kit.commands
        import omni.kit.viewport.utility
        from pxr import Usd

        viewport = omni.kit.viewport.utility.get_active_viewport()
        if viewport is None:
            return
        # Keep the manually chosen close-up camera.  FramePrimsCommand tends
        # to zoom out to include the whole 1.5 m environment cell, which is
        # exactly the failure mode this viewer is meant to avoid.
        _ = viewport
        _ = Usd
    except Exception as exc:  # pragma: no cover - depends on the GUI build
        print(f"[viewer] automatic viewport framing unavailable: {type(exc).__name__}: {exc}", flush=True)


def _set_pose(env: PhysxTomatoGraspEnv, frame: dict[str, np.ndarray]) -> None:
    """Apply one recorded frame to the first environment for inspection."""

    device = env.device
    one = torch.tensor([0], dtype=torch.long, device=device)
    dof_ids = torch.as_tensor(env.active_joint_ids, dtype=torch.long, device=device)
    dofs = env.robot.root_physx_view.get_dof_positions().clone()
    dofs[one[:, None], dof_ids] = torch.as_tensor(frame["joint_pos_rad"], device=device)
    env.robot.root_physx_view.set_dof_positions(dofs, indices=one)
    velocities = env.robot.root_physx_view.get_dof_velocities().clone()
    velocities[one] = 0.0
    env.robot.root_physx_view.set_dof_velocities(velocities, indices=one)

    transforms = env.tomato.root_physx_view.get_transforms().clone()
    transforms[one, :3] = torch.as_tensor(frame["tomato_pos_m"], device=device)
    # PhysX tensor transforms use xyzw; the recording stores IsaacLab wxyz.
    quat = torch.as_tensor(frame["tomato_quat_wxyz"], device=device)
    transforms[one, 3:7] = quat[[1, 2, 3, 0]]
    env.tomato.root_physx_view.set_transforms(transforms, indices=one)
    env.tomato.root_physx_view.set_velocities(
        env.tomato.root_physx_view.get_velocities() * 0.0, indices=one
    )


def main() -> None:
    replay_path = args_cli.replay.expanduser().resolve()
    if not replay_path.is_file():
        raise FileNotFoundError(replay_path)
    replay = np.load(replay_path)
    required = {"joint_pos_rad", "tomato_pos_m", "tomato_quat_wxyz"}
    missing = required.difference(replay.files)
    if missing:
        raise ValueError(f"replay is missing fields: {sorted(missing)}")
    frames = [
        {key: np.asarray(replay[key][idx]) for key in replay.files if replay[key].ndim > 0}
        for idx in range(len(replay["joint_pos_rad"]))
    ]
    print(f"[viewer] loading {replay_path}", flush=True)
    print(f"[viewer] frames={len(frames)} fps={args_cli.fps:g}; close Isaac Sim to exit", flush=True)

    torch.manual_seed(args_cli.seed)
    cfg = _configure_viewer()
    env = PhysxTomatoGraspEnv(cfg, render_mode=None)
    try:
        env.reset()
        # The default training dome light (1800 lux) is intentionally bright
        # for batch learning but washes the imported URDF materials out in a
        # desktop viewport.  Reduce it for inspection and add a gentle key.
        try:
            from pxr import Gf, UsdGeom, UsdLux

            stage = env.sim.stage
            light = stage.GetPrimAtPath("/World/Light")
            if light.IsValid() and light.IsA(UsdLux.DomeLight):
                UsdLux.DomeLight(light).GetIntensityAttr().Set(260.0)
                UsdLux.DomeLight(light).GetColorAttr().Set(Gf.Vec3f(0.72, 0.78, 0.86))
            key = UsdLux.DistantLight.Define(stage, "/World/ReplayKey")
            key.GetIntensityAttr().Set(500.0)
            key.GetColorAttr().Set(Gf.Vec3f(1.0, 0.86, 0.72))
            UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-38.0, -28.0, 34.0))
        except Exception as exc:  # pragma: no cover - renderer-dependent
            print(f"[viewer] lighting adjustment unavailable: {type(exc).__name__}: {exc}", flush=True)

        # A close side/top view makes the palm support and all five fingertips
        # occupy most of the viewport rather than the complete environment.
        env.sim.set_camera_view([0.25, -0.30, 0.255], [0.038, 0.0, 0.195])
        simulation_app.update()
        _frame_viewport()
        period = 1.0 / max(args_cli.fps, 1.0)
        for index, frame in enumerate(frames):
            started = time.monotonic()
            while simulation_app.is_running() and time.monotonic() - started < period:
                _set_pose(env, frame)
                simulation_app.update()
            if index % max(1, int(args_cli.fps)) == 0:
                if "digit_force_n" in frame:
                    force = np.asarray(frame["digit_force_n"])
                    print(
                        f"[viewer] frame={index:03d}/{len(frames)-1:03d} "
                        f"contact_max={float(force.max()):.3f} N",
                        flush=True,
                    )
        final = frames[-1]
        print("[viewer] playback finished; final grasp pose is held in the viewport", flush=True)
        hold = float(args_cli.hold_seconds)
        deadline = time.monotonic() + hold if hold > 0.0 else None
        while simulation_app.is_running() and (deadline is None or time.monotonic() < deadline):
            _set_pose(env, final)
            simulation_app.update()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
