#!/usr/bin/env python3
"""Static left-hand URDF audit for the frozen 16-joint policy contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.ppo_tomato_grasp.hand_manifold import (
    LEFT_ACTIVE_JOINT_ORDER,
    LEFT_SAFE_LOWER_RAD,
    LEFT_SAFE_UPPER_RAD,
    LEFT_SAFE_VELOCITY_RAD_S,
)


def audit_urdf(
    path: Path,
    *,
    velocity_scale: float = 0.5,
    absolute_velocity_cap_rad_s: float = 1.0,
) -> dict[str, object]:
    if not 0.0 < velocity_scale <= 1.0:
        raise ValueError("velocity_scale must lie in (0, 1]")
    if absolute_velocity_cap_rad_s <= 0.0:
        raise ValueError("absolute_velocity_cap_rad_s must be positive")
    root = ET.parse(path).getroot()
    joints = root.findall("joint")
    names = [joint.get("name", "") for joint in joints]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    by_name = {joint.get("name", ""): joint for joint in joints}
    active_rows = []
    missing = []
    type_issues = []
    limit_issues = []
    velocity_issues = []
    axis_issues = []
    for index, name in enumerate(LEFT_ACTIVE_JOINT_ORDER):
        joint = by_name.get(name)
        if joint is None:
            missing.append(name)
            continue
        joint_type = joint.get("type")
        if joint_type not in ("revolute", "continuous"):
            type_issues.append({"joint": name, "type": joint_type})
        limit = joint.find("limit")
        if limit is None:
            limit_issues.append({"joint": name, "reason": "missing limit"})
            continue
        try:
            urdf_lower = float(limit.get("lower", "nan"))
            urdf_upper = float(limit.get("upper", "nan"))
            urdf_velocity = float(limit.get("velocity", "nan"))
        except ValueError:
            limit_issues.append({"joint": name, "reason": "nonnumeric limit"})
            continue
        safe_lower = LEFT_SAFE_LOWER_RAD[index]
        safe_upper = LEFT_SAFE_UPPER_RAD[index]
        uncapped_velocity = velocity_scale * LEFT_SAFE_VELOCITY_RAD_S[index]
        requested_velocity = min(uncapped_velocity, absolute_velocity_cap_rad_s)
        if not (
            math.isfinite(urdf_lower)
            and math.isfinite(urdf_upper)
            and urdf_lower <= safe_lower <= safe_upper <= urdf_upper
        ):
            limit_issues.append({
                "joint": name,
                "urdf": [urdf_lower, urdf_upper],
                "measured_safe": [safe_lower, safe_upper],
            })
        if not math.isfinite(urdf_velocity) or requested_velocity > urdf_velocity + 1.0e-9:
            velocity_issues.append({
                "joint": name,
                "urdf_velocity_rad_s": urdf_velocity,
                "requested_policy_velocity_rad_s": requested_velocity,
            })
        axis = joint.find("axis")
        try:
            xyz = [float(value) for value in (axis.get("xyz") if axis is not None else "").split()]
        except ValueError:
            xyz = []
        norm = math.sqrt(sum(value * value for value in xyz)) if len(xyz) == 3 else float("nan")
        if not math.isfinite(norm) or abs(norm - 1.0) > 1.0e-4:
            axis_issues.append({"joint": name, "axis": xyz})
        active_rows.append({
            "joint": name,
            "type": joint_type,
            "axis": xyz,
            "urdf_limit_rad": [urdf_lower, urdf_upper],
            "measured_safe_rad": [safe_lower, safe_upper],
            "urdf_velocity_rad_s": urdf_velocity,
            "requested_policy_velocity_rad_s": requested_velocity,
            "uncapped_policy_velocity_rad_s": uncapped_velocity,
        })

    left_revolute = [
        joint for joint in joints
        if joint.get("name", "").startswith("left_hand_")
        and joint.get("type") in ("revolute", "continuous")
    ]
    uncontrolled = []
    for joint in left_revolute:
        name = joint.get("name", "")
        if name not in LEFT_ACTIVE_JOINT_ORDER:
            mimic = joint.find("mimic")
            uncontrolled.append({
                "joint": name,
                "has_mimic": mimic is not None,
                "mimic_joint": mimic.get("joint") if mimic is not None else None,
            })
    uncontrolled_without_mimic = [row for row in uncontrolled if not row["has_mimic"]]
    structural_passed = not (duplicates or missing or type_issues or limit_issues or axis_issues)
    physx_ready = structural_passed and not velocity_issues and not uncontrolled_without_mimic
    return {
        "schema_version": 1,
        "status": "left_hand_urdf_static_physx_ready" if physx_ready else "left_hand_urdf_static_audit_blocked",
        "urdf": str(path.resolve()),
        "active_joint_count_found": len(active_rows),
        "left_hand_revolute_joint_count": len(left_revolute),
        "structural_contract_passed": structural_passed,
        "physx_adapter_ready": physx_ready,
        "duplicates": duplicates,
        "missing_active_joints": missing,
        "type_issues": type_issues,
        "limit_issues": limit_issues,
        "velocity_issues": velocity_issues,
        "axis_issues": axis_issues,
        "uncontrolled_revolute_joints": uncontrolled,
        "uncontrolled_without_mimic": uncontrolled_without_mimic,
        "active_joints": active_rows,
        "required_actions": [
            "define and validate PhysX coupling for passive distal revolute joints lacking URDF mimic tags",
        ] if not physx_ready else [],
        "isaac_or_cuda_used": False,
        "hardware_command_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf", type=Path)
    parser.add_argument("--velocity-scale", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_urdf(args.urdf, velocity_scale=args.velocity_scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
