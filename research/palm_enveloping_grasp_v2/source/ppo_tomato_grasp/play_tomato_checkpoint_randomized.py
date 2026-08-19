"""Play the final PPO checkpoint live in randomized PhysX tomato episodes.

Unlike the NPZ viewer, this script never reads a saved trajectory.  Each
episode is produced by the environment reset and the action comes directly
from ``best.pt``.  Isaac Sim remains interactive so the user can watch the
hand close, make contact, pick, hold, or fail under a new randomized draw.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=ROOT / "artifacts/stage4_main_band4_train_256x120/best.pt",
)
parser.add_argument("--seed", type=int, default=92001)
parser.add_argument("--fps", type=float, default=20.0)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--stochastic", action="store_true", help="sample the policy instead of using its mean")
parser.add_argument("--max-episodes", type=int, default=0, help="0 means keep playing until the window closes")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from torch import nn
from torch.distributions import Normal

from simulation.ppo_tomato_grasp.physx_tomato_env import PhysxTomatoGraspEnv, PhysxTomatoGraspEnvCfg


class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(110, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.actor_mean = nn.Linear(128, 10)
        self.log_std = nn.Parameter(torch.zeros(10))

    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.actor_mean(self.actor_body(obs)), self.log_std.exp())


def _cfg() -> PhysxTomatoGraspEnvCfg:
    cfg = PhysxTomatoGraspEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.training_stage = 2
    # Final delivery band: large fruit geometry with complete non-size
    # randomization, matching the independently evaluated checkpoint.
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
    cfg.pick_direction_tilt_max_rad = math.radians(5.0)
    cfg.pick_speed_m_s = 0.015
    cfg.stem_tensile_limit_n = (0.35, 1.05)
    cfg.stem_critical_duration_s = (0.08, 0.38)
    # Render only once per 20 Hz policy step; this keeps the UI responsive.
    cfg.sim.render_interval = cfg.decimation
    return cfg


def _lighting_and_camera(env: PhysxTomatoGraspEnv) -> None:
    try:
        from pxr import Gf, UsdGeom, UsdLux

        stage = env.sim.stage
        light = stage.GetPrimAtPath("/World/Light")
        if light.IsValid() and light.IsA(UsdLux.DomeLight):
            dome = UsdLux.DomeLight(light)
            dome.GetIntensityAttr().Set(260.0)
            dome.GetColorAttr().Set(Gf.Vec3f(0.72, 0.78, 0.86))
        key = UsdLux.DistantLight.Define(stage, "/World/CheckpointPlayKey")
        key.GetIntensityAttr().Set(500.0)
        key.GetColorAttr().Set(Gf.Vec3f(1.0, 0.86, 0.72))
        UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-38.0, -28.0, 34.0))
    except Exception as exc:
        print(f"[play] lighting adjustment unavailable: {type(exc).__name__}: {exc}", flush=True)
    env.sim.set_camera_view([0.25, -0.30, 0.255], [0.038, 0.0, 0.195])


def main() -> None:
    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = torch.device(args_cli.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (checkpoint.get("actor_observation_dim"), checkpoint.get("action_dim")) != (110, 10):
        raise ValueError("checkpoint does not match the 110D actor / 10D action contract")
    actor = Actor().to(device)
    actor.load_state_dict(checkpoint["model"], strict=False)
    actor.eval()

    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    env = PhysxTomatoGraspEnv(_cfg(), render_mode="human")
    try:
        obs, _ = env.reset()
        _lighting_and_camera(env)
        simulation_app.update()
        print(f"[play] checkpoint={checkpoint_path}", flush=True)
        print("[play] LIVE randomized PhysX playback; no saved trajectory is used", flush=True)
        episode = 0
        step = 0
        period = 1.0 / max(args_cli.fps, 1.0)
        while simulation_app.is_running():
            started = time.monotonic()
            with torch.inference_mode():
                distribution = actor.distribution(obs["policy"])
                action = distribution.sample() if args_cli.stochastic else distribution.mean
                action = torch.tanh(action)
                obs, reward, terminated, truncated, _ = env.step(action)
            step += 1
            if step % 10 == 0:
                pos = env.tomato.data.root_pos_w[0].detach().cpu().tolist()
                force = env.contact_forces[0].detach().cpu().tolist()
                print(
                    f"[play] episode={episode} step={step} tomato=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                    f"digits={[round(v, 2) for v in force]} palm={float(env.palm_force[0]):.2f}N",
                    flush=True,
                )
            if bool((terminated | truncated).any()):
                code = int(env.last_termination_code[0].item())
                label = {1: "SUCCESS", 2: "DROP", 3: "OVERLOAD", 4: "DAMAGE", 5: "TIMEOUT"}.get(code, "RESET")
                print(f"[play] episode={episode} result={label} (code={code}); new randomized episode", flush=True)
                episode += 1
                step = 0
                if args_cli.max_episodes > 0 and episode >= args_cli.max_episodes:
                    break
            remaining = period - (time.monotonic() - started)
            if remaining > 0.0:
                time.sleep(min(remaining, 0.04))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
