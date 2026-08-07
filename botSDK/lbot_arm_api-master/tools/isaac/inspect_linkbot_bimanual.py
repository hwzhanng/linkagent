#!/usr/bin/env python3
"""Interactive LinkBot dual-arm/dual-hand connection and dynamics inspection.

The script never connects to real hardware.  It loads the canonical refined
USD, audits the joint graph and rigid-body mass/inertia tensors, then cycles
through bounded arm and hand position targets.  With ``--viz kit`` the cycle
continues until the Isaac Sim window is closed.  With a positive ``--duration``
it writes a finite automated report and exits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


ASSET_ROOT = (
    Path(__file__).resolve().parents[3]
    / "urdf/02_project_derived/isaac_assets/linkbot_bimanual_sdk_description"
)
DEFAULT_USD = str(ASSET_ROOT / "linkbot_bimanual_sdk_description.usda")
DEFAULT_REPORT = str(ASSET_ROOT / "tests/interactive_inspection_report.json")
DEFAULT_MASS_CSV = str(ASSET_ROOT / "tests/body_mass_inertia.csv")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", default=DEFAULT_USD)
parser.add_argument("--report", default=DEFAULT_REPORT)
parser.add_argument("--mass-csv", default=DEFAULT_MASS_CSV)
parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Seconds to run; 0 keeps the GUI alive until its window is closed.",
)
parser.add_argument(
    "--motion-scale",
    type=float,
    default=1.0,
    help="Scale the already bounded visual test amplitudes (0.1..1.0).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.motion_scale = max(0.1, min(1.0, float(args.motion_scale)))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg


ARM_PATTERN = r"^[LR][1-7]_Joint$"
HAND_INDEPENDENT_PATTERN = (
    r"^(left|right)_hand_(thumb_joint[0-3]|"
    r"(index|middle|ring|little)_joint[0-2])$"
)
HAND_FOLLOWER_PATTERN = (
    r"^(left|right)_hand_(thumb_joint4|"
    r"(index|middle|ring|little)_joint3)$"
)
FLEXION_SUFFIXES = {
    "thumb_joint2",
    "thumb_joint3",
    "index_joint1",
    "index_joint2",
    "middle_joint1",
    "middle_joint2",
    "ring_joint1",
    "ring_joint2",
    "little_joint1",
    "little_joint2",
}
PHASE_NAMES = {
    "hold": "静止连接检查",
    "arms": "双臂小幅关节运动",
    "left_hand": "左手屈伸",
    "right_hand": "右手屈伸",
    "both_hands": "双手同步屈伸",
    "return": "平滑返回初态",
}


def tensor_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().flatten().tolist()]


def audit_usd_connections(usd_path: Path) -> dict:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"无法打开 USD：{usd_path}")
    stage.Load()
    rigid_bodies: list[str] = []
    joints: list[dict] = []
    articulation_roots: list[str] = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
            body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
            joints.append(
                {
                    "name": prim.GetName(),
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "body0": body0,
                    "body1": body1,
                    "connected": len(body1) == 1,
                }
            )
    by_name = {entry["name"]: entry for entry in joints}
    mounts = {
        name: by_name.get(name)
        for name in ("left_hand_mount_joint", "right_hand_mount_joint")
    }
    return {
        "rigid_body_count": len(rigid_bodies),
        "joint_count": len(joints),
        "articulation_roots": articulation_roots,
        "all_joints_have_child_body": all(item["connected"] for item in joints),
        "mount_joints": mounts,
        "mounts_present_and_connected": all(
            item is not None and item["connected"] for item in mounts.values()
        ),
        "joints": joints,
    }


def body_group(name: str) -> str:
    if name.startswith("left_hand_"):
        return "left_hand"
    if name.startswith("right_hand_"):
        return "right_hand"
    if name.startswith("L"):
        return "left_arm"
    if name.startswith("R"):
        return "right_arm"
    return "base"


def audit_mass_inertia(
    robot: Articulation, csv_path: Path,
) -> dict:
    masses = robot.data.body_mass.torch[0].detach().cpu()
    inertias = robot.data.body_inertia.torch[0].detach().cpu().reshape(-1, 3, 3)
    rows: list[dict] = []
    group_mass: dict[str, float] = {}
    all_positive_definite = True
    all_triangle_valid = True
    for index, name in enumerate(robot.body_names):
        inertia = 0.5 * (inertias[index] + inertias[index].T)
        eigenvalues = torch.linalg.eigvalsh(inertia)
        eigen = [float(value) for value in eigenvalues.tolist()]
        positive = eigen[0] > 0.0
        triangle = (
            eigen[0] + eigen[1] >= eigen[2] - 1.0e-8
            and eigen[0] + eigen[2] >= eigen[1] - 1.0e-8
            and eigen[1] + eigen[2] >= eigen[0] - 1.0e-8
        )
        all_positive_definite &= positive
        all_triangle_valid &= triangle
        group = body_group(name)
        mass = float(masses[index].item())
        group_mass[group] = group_mass.get(group, 0.0) + mass
        rows.append(
            {
                "index": index,
                "name": name,
                "group": group,
                "mass_kg": mass,
                "inertia_world_kg_m2": tensor_list(inertia),
                "principal_inertia_kg_m2": eigen,
                "positive_definite": positive,
                "triangle_inequality": triangle,
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "index",
                "name",
                "group",
                "mass_kg",
                "principal_i1_kg_m2",
                "principal_i2_kg_m2",
                "principal_i3_kg_m2",
                "positive_definite",
                "triangle_inequality",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["index"],
                    row["name"],
                    row["group"],
                    f"{row['mass_kg']:.9g}",
                    *(f"{value:.9g}" for value in row["principal_inertia_kg_m2"]),
                    row["positive_definite"],
                    row["triangle_inequality"],
                ]
            )
    return {
        "body_count": len(rows),
        "total_mass_kg": float(masses.sum().item()),
        "group_mass_kg": group_mass,
        "all_masses_positive": bool(torch.all(masses > 0.0).item()),
        "all_inertias_positive_definite": all_positive_definite,
        "all_principal_inertias_satisfy_triangle_inequality": all_triangle_valid,
        "warning": (
            "Hand masses/inertias are vendor URDF values and remain much lower "
            "than the complete physical hand; they are valid numerically, not "
            "yet high-fidelity Sim2Real values."
        ),
        "bodies": rows,
    }


def phase_at(cycle_time: float) -> tuple[str, float]:
    if cycle_time < 3.0:
        return "hold", cycle_time / 3.0
    if cycle_time < 9.0:
        return "arms", (cycle_time - 3.0) / 6.0
    if cycle_time < 15.0:
        return "left_hand", (cycle_time - 9.0) / 6.0
    if cycle_time < 21.0:
        return "right_hand", (cycle_time - 15.0) / 6.0
    if cycle_time < 27.0:
        return "both_hands", (cycle_time - 21.0) / 6.0
    return "return", (cycle_time - 27.0) / 3.0


def main() -> bool:
    usd_path = Path(args.usd).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    csv_path = Path(args.mass_csv).expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(usd_path)

    connection = audit_usd_connections(usd_path)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            render_interval=2,
            device=args.device,
        )
    )
    sim_utils.DistantLightCfg(
        intensity=1800.0, color=(0.80, 0.84, 0.92)
    ).func(
        "/World/InspectionLight",
        sim_utils.DistantLightCfg(
            intensity=1800.0, color=(0.80, 0.84, 0.92)),
    )
    sim.set_camera_view(
        eye=[2.2, -2.7, 0.70],
        target=[0.0, 0.0, -0.59],
    )

    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/LinkBot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                ),
            ),
            actuators={
                # Preserve the arm drives already proven by the canonical USD
                # smoke test instead of inventing a second arm controller.
                "arm_imported_drives": ImplicitActuatorCfg(
                    joint_names_expr=[ARM_PATTERN],
                    stiffness=None,
                    damping=None,
                ),
                # The vendor hand inertias are extremely small.  These gains
                # are deliberately inertia-scaled and only make the joints
                # move visibly without writing parameters back to the asset.
                "hand_inspection_proxy": ImplicitActuatorCfg(
                    joint_names_expr=[HAND_INDEPENDENT_PATTERN],
                    effort_limit_sim=0.25,
                    stiffness=0.12,
                    damping=0.012,
                ),
                "hand_mimic_followers": ImplicitActuatorCfg(
                    joint_names_expr=[HAND_FOLLOWER_PATTERN],
                    stiffness=0.0,
                    damping=0.0,
                ),
            },
        )
    )
    sim.reset()
    robot.reset()
    sim_dt = sim.get_physics_dt()
    base_target = robot.data.joint_pos.torch.clone()
    limits = robot.data.joint_pos_limits.torch[0]
    joint_index = {name: index for index, name in enumerate(robot.joint_names)}
    arm_ids = [
        index for name, index in joint_index.items()
        if name.startswith("L") or name.startswith("R")
    ]
    left_hand_ids = [
        index for name, index in joint_index.items()
        if name.startswith("left_hand_")
        and name.removeprefix("left_hand_") in FLEXION_SUFFIXES
    ]
    right_hand_ids = [
        index for name, index in joint_index.items()
        if name.startswith("right_hand_")
        and name.removeprefix("right_hand_") in FLEXION_SUFFIXES
    ]
    commanded_ids = arm_ids + left_hand_ids + right_hand_ids

    mass_inertia = audit_mass_inertia(robot, csv_path)
    print(
        "[AUDIT] 连接："
        f"{connection['rigid_body_count']} 刚体 / "
        f"{connection['joint_count']} 关节 / "
        f"双手安装={connection['mounts_present_and_connected']}"
    )
    print(
        "[AUDIT] 质量："
        f"总计 {mass_inertia['total_mass_kg']:.6f} kg；"
        f"左手 {mass_inertia['group_mass_kg'].get('left_hand', 0.0):.6f} kg；"
        f"右手 {mass_inertia['group_mass_kg'].get('right_hand', 0.0):.6f} kg"
    )
    print(
        "[NOTICE] 手指 Kp=0.12、Kd=0.012、effort≤0.25 仅用于可视化运动检查；"
        "不是实机辨识参数，也没有写回正式 USD。"
    )

    initial_root = robot.data.root_pose_w.torch.clone()
    max_joint_delta = torch.zeros(
        len(robot.joint_names), device=robot.data.joint_pos.torch.device)
    max_abs_velocity = 0.0
    finite = True
    last_phase = ""
    started_wall = time.monotonic()
    simulated_time = 0.0
    cycle_s = 30.0

    while simulation_app.is_running():
        if args.duration > 0.0 and simulated_time >= args.duration:
            break
        cycle_time = simulated_time % cycle_s
        phase, progress = phase_at(cycle_time)
        if phase != last_phase:
            print(f"[PHASE] {PHASE_NAMES[phase]}")
            last_phase = phase

        target = base_target.clone()
        if phase == "arms":
            wave = math.sin(progress * 2.0 * math.pi)
            for offset, joint_id in enumerate(arm_ids):
                sign = 1.0 if offset % 2 == 0 else -1.0
                target[:, joint_id] += (
                    sign * 0.075 * args.motion_scale * wave)
        elif phase in ("left_hand", "right_hand", "both_hands"):
            closure = (
                0.5 - 0.5 * math.cos(progress * 2.0 * math.pi)
            ) * 0.34 * args.motion_scale
            ids: list[int] = []
            if phase in ("left_hand", "both_hands"):
                ids.extend(left_hand_ids)
            if phase in ("right_hand", "both_hands"):
                ids.extend(right_hand_ids)
            for joint_id in ids:
                target[:, joint_id] += closure
        elif phase == "return":
            # Targets are already the initial state; the drive performs the
            # smooth physical return instead of teleporting joints.
            pass

        low = limits[:, 0] + 1.0e-4
        high = limits[:, 1] - 1.0e-4
        target[0] = torch.maximum(low, torch.minimum(high, target[0]))
        robot.set_joint_position_target_index(target=target)
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(sim_dt)

        position = robot.data.joint_pos.torch[0]
        velocity = robot.data.joint_vel.torch
        max_joint_delta = torch.maximum(
            max_joint_delta, torch.abs(position - base_target[0]))
        max_abs_velocity = max(
            max_abs_velocity,
            float(torch.max(torch.abs(velocity)).item()),
        )
        step_finite = bool(
            torch.isfinite(position).all()
            and torch.isfinite(velocity).all()
        )
        finite &= step_finite
        simulated_time += sim_dt
        if not step_finite:
            print("[ERROR] 检测到非有限关节状态，已提前停止测试。")
            break

    final_root = robot.data.root_pose_w.torch.clone()
    root_translation_drift = float(
        torch.linalg.vector_norm(
            final_root[:, :3] - initial_root[:, :3], dim=1
        ).max().item()
    )
    group_motion = {
        "arms_max_delta_rad": max(
            (float(max_joint_delta[index].item()) for index in arm_ids),
            default=0.0,
        ),
        "left_hand_max_delta_rad": max(
            (
                float(max_joint_delta[index].item())
                for index in left_hand_ids
            ),
            default=0.0,
        ),
        "right_hand_max_delta_rad": max(
            (
                float(max_joint_delta[index].item())
                for index in right_hand_ids
            ),
            default=0.0,
        ),
    }
    finite_run = args.duration <= 0.0 or simulated_time + sim_dt >= args.duration
    checks = {
        "one_articulation_root": len(connection["articulation_roots"]) == 1,
        "expected_71_rigid_bodies": connection["rigid_body_count"] == 71,
        "expected_71_usd_joints_including_fixed_base": (
            connection["joint_count"] == 71
        ),
        "all_joints_have_child_body": connection[
            "all_joints_have_child_body"
        ],
        "both_hand_mounts_connected": connection[
            "mounts_present_and_connected"
        ],
        "expected_56_revolute_dofs": robot.num_joints == 56,
        "mass_count_matches_bodies": (
            mass_inertia["body_count"] == robot.num_bodies == 71
        ),
        "all_masses_positive": mass_inertia["all_masses_positive"],
        "all_inertias_positive_definite": mass_inertia[
            "all_inertias_positive_definite"
        ],
        "all_inertia_triangle_inequalities": mass_inertia[
            "all_principal_inertias_satisfy_triangle_inequality"
        ],
        "finite_dynamic_state": finite,
        "fixed_root_translation": root_translation_drift < 1.0e-6,
    }
    if args.duration >= 9.0:
        checks["arm_motion_observed"] = (
            group_motion["arms_max_delta_rad"] > 0.01)
    if args.duration >= 15.0:
        checks["left_hand_motion_observed"] = (
            group_motion["left_hand_max_delta_rad"] > 0.01)
    if args.duration >= 21.0:
        checks["right_hand_motion_observed"] = (
            group_motion["right_hand_max_delta_rad"] > 0.01)

    report = {
        "schema_version": 1,
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "usd": str(usd_path),
        "device": args.device,
        "visualizer": str(getattr(args, "visualizer", "")),
        "duration_requested_s": args.duration,
        "simulated_duration_s": simulated_time,
        "wall_duration_s": time.monotonic() - started_wall,
        "physics_dt_s": sim_dt,
        "connection_audit": connection,
        "mass_inertia_audit": mass_inertia,
        "motion_test": {
            "cycle_s": cycle_s,
            "phase_order": list(PHASE_NAMES.values()),
            "motion_scale": args.motion_scale,
            "commanded_arm_joints": [
                robot.joint_names[index] for index in arm_ids
            ],
            "commanded_left_hand_joints": [
                robot.joint_names[index] for index in left_hand_ids
            ],
            "commanded_right_hand_joints": [
                robot.joint_names[index] for index in right_hand_ids
            ],
            "group_motion": group_motion,
            "max_abs_joint_velocity_rad_s": max_abs_velocity,
            "root_translation_drift_m": root_translation_drift,
            "runtime_hand_drive_proxy": {
                "stiffness": 0.12,
                "damping": 0.012,
                "effort_limit": 0.25,
                "written_back_to_usd": False,
                "identified_hardware_truth": False,
            },
        },
        "checks": checks,
        "passed": bool(finite_run and all(checks.values())),
        "scope": (
            "Connection topology, bounded joint motion, and numerical "
            "mass/inertia validity. This does not validate the vendor hand "
            "mass distribution or identify real actuator dynamics."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[REPORT] {report_path}")
    print(f"[MASS_CSV] {csv_path}")
    print(f"[RESULT] passed={report['passed']} checks={checks}")
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
