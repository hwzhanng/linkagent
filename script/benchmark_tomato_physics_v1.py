"""Measure Physics-v1 simulator throughput without PPO optimizer overhead.

The benchmark intentionally drives the public Gym environment with a fixed
action stream.  CUDA is synchronized only at block boundaries so the timing
does not add a device round-trip to every policy step.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _synchronize(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(argv: Sequence[str] | None = None) -> int:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--measure-steps", type=int, default=32)
    parser.add_argument("--block-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action-scale", type=float, default=0.45)
    parser.add_argument("--contact-max-data", type=int)
    parser.add_argument("--disable-friction-tracking", action="store_true")
    parser.add_argument("--disable-contact-points", action="store_true")
    parser.add_argument("--disable-contact-pose", action="store_true")
    parser.add_argument("--detailed-step-diagnostics", action="store_true")
    parser.add_argument("--disable-self-collisions", action="store_true")
    parser.add_argument("--solver-position-iterations", type=int)
    parser.add_argument("--solver-velocity-iterations", type=int)
    parser.add_argument("--replicate-physics", action="store_true")
    parser.add_argument("--clone-in-fabric", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs < 1:
        raise ValueError("--num-envs must be positive")
    if args.warmup_steps < 0 or args.measure_steps < 1 or args.block_steps < 1:
        raise ValueError("step counts must be non-negative and measurement positive")

    launch_started = time.perf_counter()
    launcher = AppLauncher(args)
    app = launcher.app
    raw_env = None
    try:
        gym = importlib.import_module("gymnasium")
        torch = importlib.import_module("torch")
        task = importlib.import_module("simulation.tomato_pick.isaac.physics_v1")
        cfg_module = importlib.import_module(
            "simulation.tomato_pick.isaac.physics_v1.env_cfg"
        )

        cfg = cfg_module.TomatoPickPhysicsV1EnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        cfg.detailed_step_diagnostics = args.detailed_step_diagnostics
        if args.disable_self_collisions:
            cfg.robot.spawn.articulation_props.enabled_self_collisions = False
        if args.solver_position_iterations is not None:
            cfg.robot.spawn.articulation_props.solver_position_iteration_count = (
                args.solver_position_iterations
            )
        if args.solver_velocity_iterations is not None:
            cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = (
                args.solver_velocity_iterations
            )
        if args.replicate_physics:
            cfg.scene.replicate_physics = True
        if args.clone_in_fabric:
            cfg.scene.clone_in_fabric = True
        for sensor in cfg.fingertip_contacts.values():
            if args.contact_max_data is not None:
                sensor.max_contact_data_count_per_prim = args.contact_max_data
            if args.disable_friction_tracking:
                sensor.track_friction_forces = False
            if args.disable_contact_points:
                sensor.track_contact_points = False
            if args.disable_contact_pose:
                sensor.track_pose = False
        raw_env = gym.make(task.TASK_ID, cfg=cfg)
        env = raw_env.unwrapped
        raw_env.reset(seed=args.seed)
        _synchronize(torch)
        initialized_at = time.perf_counter()

        generator = torch.Generator(device=env.device).manual_seed(args.seed + 1)

        def action() -> object:
            return args.action_scale * (
                2.0
                * torch.rand(
                    (args.num_envs, 11),
                    generator=generator,
                    device=env.device,
                )
                - 1.0
            )

        for _ in range(args.warmup_steps):
            raw_env.step(action())
        _synchronize(torch)

        blocks: list[dict[str, float | int]] = []
        measured = 0
        total_terminated = 0
        total_truncated = 0
        total_success = 0
        total_drop = 0
        total_detach = 0
        measurement_started = time.perf_counter()
        while measured < args.measure_steps:
            block_size = min(args.block_steps, args.measure_steps - measured)
            _synchronize(torch)
            block_started = time.perf_counter()
            for _ in range(block_size):
                _obs, _reward, terminated, truncated, info = raw_env.step(action())
                total_terminated += int(terminated.sum().item())
                total_truncated += int(truncated.sum().item())
                step_info = info.get("step", {})
                total_success += int(step_info.get("success_count", 0))
                total_drop += int(step_info.get("drop_count", 0))
                total_detach += int(step_info.get("detach_count", 0))
            _synchronize(torch)
            elapsed = time.perf_counter() - block_started
            blocks.append(
                {
                    "start_step": measured,
                    "steps": block_size,
                    "seconds": elapsed,
                    "seconds_per_policy_step": elapsed / block_size,
                }
            )
            measured += block_size
        measurement_seconds = time.perf_counter() - measurement_started
        transitions = args.num_envs * args.measure_steps
        seconds_per_policy_step = measurement_seconds / args.measure_steps
        report = {
            "schema_version": 1,
            "benchmark": "tomato_pick_physics_v1_fixed_action_rollout",
            "project_root": str(PROJECT_ROOT),
            "num_envs": args.num_envs,
            "seed": args.seed,
            "physics_dt_s": float(cfg.sim.dt),
            "decimation": int(cfg.decimation),
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "block_steps": args.block_steps,
            "action_scale": args.action_scale,
            "articulation": {
                "enabled_self_collisions": bool(
                    cfg.robot.spawn.articulation_props.enabled_self_collisions
                ),
                "solver_position_iteration_count": int(
                    cfg.robot.spawn.articulation_props.solver_position_iteration_count
                ),
                "solver_velocity_iteration_count": int(
                    cfg.robot.spawn.articulation_props.solver_velocity_iteration_count
                ),
            },
            "scene": {
                "replicate_physics": bool(cfg.scene.replicate_physics),
                "clone_in_fabric": bool(cfg.scene.clone_in_fabric),
            },
            "initialization_seconds": initialized_at - launch_started,
            "measurement_seconds": measurement_seconds,
            "seconds_per_policy_step": seconds_per_policy_step,
            "estimated_seconds_per_32_step_rollout": 32.0
            * seconds_per_policy_step,
            "transitions": transitions,
            "transitions_per_second": transitions / measurement_seconds,
            "terminated_count": total_terminated,
            "truncated_count": total_truncated,
            "success_count": total_success,
            "drop_count": total_drop,
            "detach_count": total_detach,
            "finite": all(
                math.isfinite(float(block[key]))
                for block in blocks
                for key in ("seconds", "seconds_per_policy_step")
            ),
            "contact_sensors": {
                name: {
                    "track_pose": bool(sensor.track_pose),
                    "track_contact_points": bool(sensor.track_contact_points),
                    "track_friction_forces": bool(sensor.track_friction_forces),
                    "max_contact_data_count_per_prim": int(
                        sensor.max_contact_data_count_per_prim
                    ),
                }
                for name, sensor in cfg.fingertip_contacts.items()
            },
            "blocks": blocks,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("PHYSICS_V1_BENCHMARK=" + json.dumps(report, sort_keys=True))
        return 0 if report["finite"] else 1
    finally:
        if raw_env is not None:
            raw_env.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
