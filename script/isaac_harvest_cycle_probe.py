"""Probe the shared physical scene and fixed top RGB-D camera used for skill acceptance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physics-cpu", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> int:
    args = _parser().parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        import gymnasium as gym
        import torch

        from module.robotics import LinearMoveCommand
        from proto import ArmSide, Pose
        from simulation.harvest_cycle import HarvestCycleIsaacEnvCfg
        from simulation.harvest_cycle.robotics import IsaacHarvestRoboticsBundle

        cfg = HarvestCycleIsaacEnvCfg()
        cfg.scene.num_envs = 1
        cfg.seed = 42
        if args.physics_cpu:
            cfg.sim.device = "cpu"
        env = gym.make("LinkAgent-HarvestCycle-Acceptance-v1", cfg=cfg)
        try:
            env.reset(seed=42)
            base = env.unwrapped
            zeros = torch.zeros((1, 11), device=base.device)
            for _ in range(4):
                env.step(zeros)
            bundle = IsaacHarvestRoboticsBundle(base)
            initial_arm = bundle.arm.get_state(ArmSide.RIGHT)
            motion_target = Pose(
                (
                    initial_arm.ee_pose.position_m[0],
                    initial_arm.ee_pose.position_m[1] + 0.02,
                    initial_arm.ee_pose.position_m[2],
                ),
                initial_arm.ee_pose.quaternion_wxyz,
                "world",
            )
            lease = bundle.leases.acquire(
                owner_id="isaac-probe", side=ArmSide.RIGHT, ttl_s=2.0
            )
            try:
                motion = bundle.arm.move_linear(
                    lease,
                    LinearMoveCommand(
                        command_id="isaac-probe-movel",
                        side=ArmSide.RIGHT,
                        target_pose=motion_target,
                        speed=0.08,
                        acceleration=0.10,
                        timeout_s=1.5,
                    ),
                )
            finally:
                bundle.leases.release(lease)
            moved_arm = bundle.arm.get_state(ArmSide.RIGHT)
            motion_error_m = float(
                torch.linalg.vector_norm(
                    torch.tensor(moved_arm.ee_pose.position_m, device=base.device)
                    - torch.tensor(motion_target.position_m, device=base.device)
                ).item()
            )
            rgb, depth, intrinsics = base.capture_harvest_rgbd()
            finite_depth = depth[torch.isfinite(depth)]
            observations = base._get_observations()
            result = {
                "schema_version": 1,
                "backend": "isaac_lab_cuda_physx",
                "robot_asset": str(base.cfg.robot.spawn.usd_path),
                "num_envs": base.num_envs,
                "device": str(base.device),
                "actor_observation_shape": list(observations["policy"].shape),
                "critic_observation_shape": list(observations["critic"].shape),
                "action_shape": [base.num_envs, 11],
                "robotics_bundle_protocol": bool(
                    isinstance(bundle, __import__("module.robotics", fromlist=["RoboticsBundle"]).RoboticsBundle)
                ),
                "movel_state": motion.state.value,
                "movel_position_error_m": motion_error_m,
                "simulated_substeps": bundle.sequence,
                "ee_position_world_m": [
                    float(value)
                    for value in base._robot.data.body_pos_w.torch[0, base._ee_body_id]
                ],
                "fruit_position_world_m": [
                    float(value) for value in base._tomato.data.root_pos_w.torch[0]
                ],
                "grasp_basis_world": [
                    [float(value) for value in row]
                    for row in base._grasp_basis_w[0]
                ],
                "ee_quaternion_wxyz": list(initial_arm.ee_pose.quaternion_wxyz),
                "rgb_shape": list(rgb.shape),
                "rgb_mean": float(rgb.float().mean().item()),
                "depth_shape": list(depth.shape),
                "finite_depth_fraction": float(torch.isfinite(depth).float().mean().item()),
                "finite_depth_min_m": float(finite_depth.min().item()),
                "finite_depth_max_m": float(finite_depth.max().item()),
                "intrinsics": [[float(value) for value in row] for row in intrinsics[0]],
            }
            if args.output is not None:
                destination = args.output.expanduser().resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print("HARVEST_CYCLE_ISAAC_PROBE=" + json.dumps(result, sort_keys=True))
            bundle.close()
        finally:
            env.close()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
