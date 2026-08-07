"""Run a real CUDA PhysX and formal LinkBot USD smoke inside Isaac Lab.

This script intentionally imports Isaac Lab only after ``AppLauncher`` starts.
It is not part of the dependency-free unit-test suite and never connects to CAN,
the vendor SDK, a model API, or a physical robot.
"""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import re
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_USD = (
    REPOSITORY_ROOT
    / "botSDK/urdf/02_project_derived/isaac_assets/"
    "linkbot_bimanual_sdk_training_mass1kg/"
    "linkbot_bimanual_sdk_training_mass1kg.usda"
)
EXPECTED_ARM_JOINTS = tuple(
    f"{side}{index}_Joint"
    for index in range(1, 8)
    for side in ("R", "L")
)
EXPECTED_FINGERTIP_LINKS = tuple(
    f"{side}_hand_{finger}_link{5 if finger == 'thumb' else 4}"
    for side in ("left", "right")
    for finger in ("thumb", "index", "middle", "ring", "little")
)


def _app_launcher_class() -> type:
    """Resolve AppLauncher only inside the explicitly selected Isaac runtime."""

    return importlib.import_module("isaaclab.app").AppLauncher


def _parser(app_launcher_class: type) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-usd", type=Path, default=DEFAULT_ROBOT_USD)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--condition-g20-mass",
        action="store_true",
        help="Apply the task-local one-kilogram G20 mass profile before PhysX starts.",
    )
    app_launcher_class.add_app_launcher_args(parser)
    return parser


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _count_authored_mimic_relations(robot_usd: Path) -> int:
    payload = robot_usd.parent / "payloads/Physics/physics.usda"
    content = payload.read_text(encoding="utf-8")
    return len(re.findall(r"\brel\s+newton:mimicJoint\b", content))


def main(argv: Sequence[str] | None = None) -> int:
    app_launcher_class = _app_launcher_class()
    args = _parser(app_launcher_class).parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    robot_usd = args.robot_usd.expanduser().resolve()
    if not robot_usd.is_file():
        raise FileNotFoundError(f"robot USD does not exist: {robot_usd}")

    app_launcher = app_launcher_class(args)
    simulation_app = app_launcher.app
    try:
        torch = importlib.import_module("torch")
        sim_utils = importlib.import_module("isaaclab.sim")
        assets = importlib.import_module("isaaclab.assets")
        Articulation = assets.Articulation
        ArticulationCfg = assets.ArticulationCfg

        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0")
        )
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/Robot",
                spawn=sim_utils.UsdFileCfg(usd_path=str(robot_usd)),
                actuators={},
            )
        )
        mass_conditioning = None
        if args.condition_g20_mass:
            condition_mass = importlib.import_module(
                "simulation.tomato_pick.isaac.assets"
            ).condition_linkbot_mass_properties
            mass_conditioning = condition_mass("/World/Robot")
        sim.reset()
        maximum_abs_joint_position = 0.0
        maximum_abs_joint_velocity = 0.0
        for _ in range(args.steps):
            sim.step(render=False)
            robot.update(sim.get_physics_dt())
            maximum_abs_joint_position = max(
                maximum_abs_joint_position,
                float(torch.nan_to_num(robot.data.joint_pos.torch).abs().max().item()),
            )
            maximum_abs_joint_velocity = max(
                maximum_abs_joint_velocity,
                float(torch.nan_to_num(robot.data.joint_vel.torch).abs().max().item()),
            )

        body_names = set(robot.body_names)
        joint_names = set(robot.joint_names)
        body_pose_w = robot.data.body_pose_w.torch
        key_body_poses_w = {
            name: body_pose_w[0, robot.body_names.index(name)].tolist()
            for name in ("R8_Link", "L8_Link", *EXPECTED_FINGERTIP_LINKS)
            if name in body_names
        }
        checks = {
            "cuda_available": bool(torch.cuda.is_available()),
            "body_count_71": robot.num_bodies == 71,
            "independent_joint_count_56": robot.num_joints == 56,
            "fixed_base": bool(robot.is_fixed_base),
            "all_arm_joints_present": set(EXPECTED_ARM_JOINTS).issubset(joint_names),
            "all_ten_fingertips_present": set(EXPECTED_FINGERTIP_LINKS).issubset(body_names),
            "ten_mimic_relations_authored": _count_authored_mimic_relations(robot_usd) == 10,
            "joint_positions_finite": bool(torch.isfinite(robot.data.joint_pos.torch).all()),
        }
        report = {
            "schema_version": 1,
            "passed": all(checks.values()),
            "steps": args.steps,
            "physics_dt_s": sim.get_physics_dt(),
            "device": "cuda:0",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "isaacsim": _version("isaacsim"),
            "isaaclab": _version("isaaclab"),
            "isaaclab_physx": _version("isaaclab-physx"),
            "robot_usd": str(robot_usd),
            "num_bodies": robot.num_bodies,
            "num_independent_joints": robot.num_joints,
            "authored_mimic_relations": _count_authored_mimic_relations(robot_usd),
            "mass_conditioning": mass_conditioning,
            "maximum_abs_joint_position_rad": maximum_abs_joint_position,
            "maximum_abs_joint_velocity_rad_s": maximum_abs_joint_velocity,
            "key_body_poses_w": key_body_poses_w,
            "checks": checks,
        }
        print(
            "ISAAC_RUNTIME_SMOKE_REPORT="
            + json.dumps(report, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        return 0 if report["passed"] else 2
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
