#!/usr/bin/env python3
"""Load the generated LinkBot USD in PhysX and run a bounded articulation smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_USD = str(
    Path(__file__).resolve().parents[3]
    / "urdf/02_project_derived/isaac_assets/linkbot_bimanual_sdk_description/"
    "linkbot_bimanual_sdk_description.usda"
)

EXPECTED_HAND_VELOCITY_LIMITS = {
    "thumb_joint0": 0.5,
    "thumb_joint1": 0.4,
    "thumb_joint2": 1.0685,
    "thumb_joint3": 1.6148,
    "thumb_joint4": 1.7924,
    "index_joint0": 0.2,
    "index_joint1": 3.1523,
    "index_joint2": 2.4061,
    "index_joint3": 2.2617,
    "middle_joint0": 0.2,
    "middle_joint1": 3.1158,
    "middle_joint2": 2.3382,
    "middle_joint3": 2.1979,
    "ring_joint0": 0.2,
    "ring_joint1": 3.0806,
    "ring_joint2": 2.5159,
    "ring_joint3": 2.3649,
    "little_joint0": 0.2,
    "little_joint1": 3.0013,
    "little_joint2": 2.4225,
    "little_joint3": 2.2772,
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", default=DEFAULT_USD, help="Generated LinkBot USD.")
parser.add_argument("--hold-steps", type=int, default=120)
parser.add_argument("--motion-steps", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg


def collect_mimic_relations(usd_path: Path) -> list[dict[str, object]]:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {usd_path}")
    relations: list[dict[str, object]] = []
    for prim in stage.Traverse():
        offset_attribute = prim.GetAttribute("newton:mimicCoef0")
        mimic_relationship = prim.GetRelationship("newton:mimicJoint")
        if not offset_attribute.IsValid() or not mimic_relationship.IsValid():
            continue
        targets = mimic_relationship.GetTargets()
        if len(targets) != 1:
            continue
        offset_usd = float(offset_attribute.Get())
        offset_rad = math.radians(offset_usd) if prim.IsA(UsdPhysics.RevoluteJoint) else offset_usd
        relations.append(
            {
                "follower": prim.GetName(),
                "leader": targets[0].name,
                "offset_rad": offset_rad,
                "offset_usd": offset_usd,
                "offset_usd_unit": "degrees" if prim.IsA(UsdPhysics.RevoluteJoint) else "distance",
                "multiplier": float(prim.GetAttribute("newton:mimicCoef1").Get()),
            }
        )
    return relations


def tensor_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().flatten().tolist()]


def main() -> bool:
    usd_path = Path(args.usd).expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)

    mimic_relations = collect_mimic_relations(usd_path)
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device, render_interval=30)
    sim = sim_utils.SimulationContext(sim_cfg)

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        actuators={
            "all_imported_drives": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
            )
        },
    )
    robot = Articulation(robot_cfg)
    sim.reset()
    robot.reset()
    sim_dt = sim.get_physics_dt()

    joint_names = robot.joint_names
    joint_name_to_index = {name: index for index, name in enumerate(joint_names)}
    present_sides = [
        side
        for side in ("left", "right")
        if f"{side}_hand_thumb_joint0" in joint_name_to_index
    ]
    present_arms = [
        side
        for side, prefix in (("left", "L"), ("right", "R"))
        if f"{prefix}1_Joint" in joint_name_to_index
    ]
    joint_velocity_limits = robot.data.joint_vel_limits.torch[0]
    hand_velocity_limits: dict[str, dict[str, dict[str, float]]] = {}
    hand_velocity_errors: dict[str, list[float]] = {}
    for side in present_sides:
        hand_velocity_limits[side] = {}
        hand_velocity_errors[side] = []
        for suffix, expected in EXPECTED_HAND_VELOCITY_LIMITS.items():
            name = f"{side}_hand_{suffix}"
            actual = float(joint_velocity_limits[joint_name_to_index[name]].item())
            error = abs(actual - expected)
            hand_velocity_errors[side].append(error)
            hand_velocity_limits[side][name] = {
                "expected_rad_s": expected,
                "actual_rad_s": actual,
                "absolute_error_rad_s": error,
            }
    target = robot.data.joint_pos.torch.clone()
    robot.set_joint_position_target(target)

    initial_root_pose = robot.data.root_pose_w.torch.clone()
    initial_joint_pos = robot.data.joint_pos.torch.clone()
    max_abs_velocity = 0.0

    for _ in range(args.hold_steps):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim_dt)
        velocity = robot.data.joint_vel.torch
        max_abs_velocity = max(max_abs_velocity, float(torch.max(torch.abs(velocity)).item()))

    hold_joint_pos = robot.data.joint_pos.torch.clone()
    arm_motion_deltas = {
        name: delta
        for name, delta in (("L1_Joint", 0.05), ("R1_Joint", -0.05))
        if name in joint_name_to_index
    }
    arm_targets: dict[str, float] = {}
    for name, delta in arm_motion_deltas.items():
        index = joint_name_to_index[name]
        target[:, index] += delta
        arm_targets[name] = float(target[0, index].item())

    for _ in range(args.motion_steps):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim_dt)
        velocity = robot.data.joint_vel.torch
        max_abs_velocity = max(max_abs_velocity, float(torch.max(torch.abs(velocity)).item()))

    final_root_pose = robot.data.root_pose_w.torch.clone()
    final_joint_pos = robot.data.joint_pos.torch.clone()
    final_joint_vel = robot.data.joint_vel.torch.clone()

    root_translation_drift = float(
        torch.linalg.vector_norm(final_root_pose[:, :3] - initial_root_pose[:, :3], dim=1).max().item()
    )
    root_quaternion_drift = float(
        torch.linalg.vector_norm(final_root_pose[:, 3:7] - initial_root_pose[:, 3:7], dim=1).max().item()
    )

    arm_motion_results: dict[str, dict[str, float]] = {}
    arm_errors: list[float] = []
    for name, command in arm_targets.items():
        index = joint_name_to_index[name]
        actual = float(final_joint_pos[0, index].item())
        error = abs(actual - command)
        arm_errors.append(error)
        arm_motion_results[name] = {
            "target_rad": command,
            "actual_rad": actual,
            "absolute_error_rad": error,
        }

    mimic_results: list[dict[str, object]] = []
    mimic_max_residual = 0.0
    mimic_runtime_count = 0
    for relation in mimic_relations:
        follower = str(relation["follower"])
        leader = str(relation["leader"])
        if follower not in joint_name_to_index or leader not in joint_name_to_index:
            mimic_results.append({**relation, "runtime_status": "joint_eliminated_from_dof_view"})
            continue
        follower_value = float(final_joint_pos[0, joint_name_to_index[follower]].item())
        leader_value = float(final_joint_pos[0, joint_name_to_index[leader]].item())
        expected = float(relation["offset_rad"]) + float(relation["multiplier"]) * leader_value
        residual = follower_value - expected
        mimic_max_residual = max(mimic_max_residual, abs(residual))
        mimic_runtime_count += 1
        mimic_results.append(
            {
                **relation,
                "runtime_status": "measured",
                "leader_rad": leader_value,
                "follower_rad": follower_value,
                "expected_follower_rad": expected,
                "residual_rad": residual,
            }
        )

    finite = bool(
        torch.isfinite(final_root_pose).all()
        and torch.isfinite(final_joint_pos).all()
        and torch.isfinite(final_joint_vel).all()
    )
    checks = {
        "physx_articulation_created": robot.num_instances == 1,
        "fixed_base": bool(robot.is_fixed_base),
        "expected_revolute_dofs": robot.num_joints == 28 * len(present_sides),
        "expected_rigid_bodies": robot.num_bodies == 1 + 35 * len(present_sides),
        "expected_mimic_relations_authored": (
            len(mimic_relations) == 5 * len(present_sides)
        ),
        "mimic_runtime_constraint": (
            mimic_runtime_count == 0 or mimic_max_residual < 5.0e-3
        ),
        "finite_state_after_steps": finite,
        "fixed_root_translation": root_translation_drift < 1.0e-6,
        "fixed_root_rotation": root_quaternion_drift < 1.0e-6,
        "small_arm_motion_converged": max(arm_errors) < 1.0e-2,
    }
    for side in present_sides:
        checks[f"{side}_hand_velocity_limits"] = (
            max(hand_velocity_errors[side]) < 1.0e-4
        )

    report = {
        "schema_version": 1,
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "usd": usd_path.name,
        "device": args.device,
        "physics_dt_s": sim_dt,
        "hold_steps": args.hold_steps,
        "motion_steps": args.motion_steps,
        "simulated_duration_s": (args.hold_steps + args.motion_steps) * sim_dt,
        "articulation": {
            "instances": robot.num_instances,
            "fixed_base": bool(robot.is_fixed_base),
            "joints": robot.num_joints,
            "bodies": robot.num_bodies,
            "joint_names": joint_names,
            "present_arms": present_arms,
            "present_hands": present_sides,
        },
        "state": {
            "initial_root_pose": tensor_list(initial_root_pose),
            "final_root_pose": tensor_list(final_root_pose),
            "root_translation_drift_m": root_translation_drift,
            "root_quaternion_component_drift": root_quaternion_drift,
            "initial_max_abs_joint_position_rad": float(torch.max(torch.abs(initial_joint_pos)).item()),
            "hold_max_abs_joint_position_rad": float(torch.max(torch.abs(hold_joint_pos)).item()),
            "final_max_abs_joint_velocity_rad_s": float(torch.max(torch.abs(final_joint_vel)).item()),
            "max_abs_joint_velocity_during_test_rad_s": max_abs_velocity,
        },
        "arm_motion": arm_motion_results,
        "joint_velocity_limits": hand_velocity_limits,
        "mimic": {
            "authored_relations": len(mimic_relations),
            "runtime_measured_relations": mimic_runtime_count,
            "max_absolute_residual_rad": mimic_max_residual,
            "relations": mimic_results,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "scope_note": (
            "This is a bounded simulation smoke test. Left-hand motion limits are measured; "
            "right-hand limits use the left values only as a simulation prior pending hardware validation. "
            "Drive gains remain provisional for dynamics-sensitive Sim2Real training."
        ),
    }

    report_dir = usd_path.parent / "tests"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "physx_smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"PHYSX_SMOKE_REPORT={report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return bool(report["passed"])


if __name__ == "__main__":
    test_passed = False
    test_error: BaseException | None = None
    try:
        test_passed = main()
    except BaseException as exc:
        test_error = exc
    finally:
        try:
            simulation_app.close()
        except SystemExit as exc:
            if test_error is None and exc.code not in (None, 0):
                test_error = exc
    if test_error is not None:
        raise test_error
    raise SystemExit(0 if test_passed else 1)
