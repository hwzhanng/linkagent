"""Static URDF/USD audit for the Physics-v2 offline lineage.

The audit intentionally uses the Python standard library.  It can run without
Isaac Sim, does not compose a stage, and never connects to robot hardware.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / (
    "botSDK/urdf/02_project_derived/ros2_packages/"
    "linkbot_bimanual_sdk_description/urdf/linkbot_bimanual_sdk_description.urdf"
)
PHYSICS_V1_USD = ROOT / (
    "botSDK/urdf/02_project_derived/isaac_assets/"
    "linkbot_bimanual_sdk_training_mass1kg/"
    "linkbot_bimanual_sdk_training_mass1kg_physics_v1.usda"
)
PHYSICS_V1_PROFILE = ROOT / "simulation/config/tomato_pick_physics_v1.yaml"
SIMULATION_BASELINE = ROOT / (
    "botSDK/urdf/02_project_derived/ros2_packages/"
    "linkbot_bimanual_sdk_description/config/simulation_baseline.yaml"
)
LEFT_HAND_CALIBRATION = ROOT / (
    "botSDK/urdf/02_project_derived/ros2_packages/"
    "linkbot_bimanual_sdk_description/config/left_g20_motion_calibration_20260726.yaml"
)

FINGERS = ("thumb", "index", "middle", "ring", "little")
EXPECTED_TACTILE_CONTACT_LINKS = {
    "thumb": "right_hand_thumb_link4",
    "index": "right_hand_index_link3",
    "middle": "right_hand_middle_link3",
    "ring": "right_hand_ring_link3",
    "little": "right_hand_little_link3",
}
EXPECTED_TACTILE_FRAME_LINKS = {
    "thumb": "right_hand_thumb_link5",
    "index": "right_hand_index_link4",
    "middle": "right_hand_middle_link4",
    "ring": "right_hand_ring_link4",
    "little": "right_hand_little_link4",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _floats(value: str | None, expected: int) -> tuple[float, ...]:
    if value is None:
        return (0.0,) * expected
    result = tuple(float(item) for item in value.split())
    if len(result) != expected or not all(math.isfinite(item) for item in result):
        raise ValueError(f"expected {expected} finite values, got {value!r}")
    return result


def _positive_definite_inertia(values: dict[str, float]) -> bool:
    ixx, iyy, izz = values["ixx"], values["iyy"], values["izz"]
    ixy, ixz, iyz = values["ixy"], values["ixz"], values["iyz"]
    minor2 = ixx * iyy - ixy * ixy
    determinant = (
        ixx * (iyy * izz - iyz * iyz)
        - ixy * (ixy * izz - iyz * ixz)
        + ixz * (ixy * iyz - iyy * ixz)
    )
    return ixx > 0.0 and minor2 > 0.0 and determinant > 0.0


def _geometry_kind(element: ET.Element) -> str:
    geometry = element.find("geometry")
    if geometry is None or not list(geometry):
        return "missing"
    return list(geometry)[0].tag


def _parse_usda_metadata(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    block = re.search(r"customData\s*=\s*\{(?P<body>.*?)\n\s*\}", text, re.DOTALL)
    if block is None:
        return metadata
    pattern = re.compile(r'^\s*(int|string)\s+(\w+)\s*=\s*(.+?)\s*$', re.MULTILINE)
    for kind, key, raw in pattern.findall(block.group("body")):
        metadata[key] = int(raw) if kind == "int" else raw.strip().strip('"')
    return metadata


def audit() -> dict[str, Any]:
    profile = json.loads(PHYSICS_V1_PROFILE.read_text(encoding="utf-8"))
    usd_text = PHYSICS_V1_USD.read_text(encoding="utf-8")
    metadata = _parse_usda_metadata(usd_text)
    root = ET.parse(URDF).getroot()

    links: dict[str, dict[str, Any]] = {}
    for element in root.findall("link"):
        name = element.attrib["name"]
        inertial = element.find("inertial")
        mass_element = inertial.find("mass") if inertial is not None else None
        inertia_element = inertial.find("inertia") if inertial is not None else None
        mass = float(mass_element.attrib["value"]) if mass_element is not None else None
        inertia = (
            {key: float(inertia_element.attrib[key]) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
            if inertia_element is not None
            else None
        )
        links[name] = {
            "mass_kg": mass,
            "inertia_kg_m2": inertia,
            "inertia_positive_definite": (
                _positive_definite_inertia(inertia) if inertia is not None else False
            ),
            "collision_count": len(element.findall("collision")),
            "collision_geometry": Counter(
                _geometry_kind(item) for item in element.findall("collision")
            ),
            "visual_count": len(element.findall("visual")),
        }

    joints: dict[str, dict[str, Any]] = {}
    for element in root.findall("joint"):
        name = element.attrib["name"]
        limit = element.find("limit")
        mimic = element.find("mimic")
        origin = element.find("origin")
        joints[name] = {
            "type": element.attrib["type"],
            "parent": element.find("parent").attrib["link"],
            "child": element.find("child").attrib["link"],
            "axis": _floats(
                element.find("axis").attrib.get("xyz") if element.find("axis") is not None else None,
                3,
            ),
            "origin_xyz_m": _floats(origin.attrib.get("xyz") if origin is not None else None, 3),
            "origin_rpy_rad": _floats(origin.attrib.get("rpy") if origin is not None else None, 3),
            "limit": (
                {key: float(value) for key, value in limit.attrib.items()}
                if limit is not None
                else None
            ),
            "mimic": (
                {
                    "leader": mimic.attrib["joint"],
                    "multiplier": float(mimic.attrib.get("multiplier", "1")),
                    "offset_rad": float(mimic.attrib.get("offset", "0")),
                }
                if mimic is not None
                else None
            ),
        }

    side_reports: dict[str, Any] = {}
    for side in ("right", "left"):
        prefix = f"{side}_hand_"
        side_links = {name: item for name, item in links.items() if name.startswith(prefix)}
        side_joints = {name: item for name, item in joints.items() if name.startswith(prefix)}
        revolute = {name: item for name, item in side_joints.items() if item["type"] == "revolute"}
        mimics = {name: item for name, item in revolute.items() if item["mimic"] is not None}
        independent = {name: item for name, item in revolute.items() if item["mimic"] is None}
        side_reports[side] = {
            "link_count": len(side_links),
            "revolute_joint_count": len(revolute),
            "fixed_joint_count": sum(item["type"] == "fixed" for item in side_joints.values()),
            "independent_joint_count": len(independent),
            "passive_mimic_count": len(mimics),
            "total_mass_kg": sum(item["mass_kg"] or 0.0 for item in side_links.values()),
            "collision_prim_count": sum(item["collision_count"] for item in side_links.values()),
            "visual_prim_count": sum(item["visual_count"] for item in side_links.values()),
            "mimics": {name: item["mimic"] for name, item in sorted(mimics.items())},
        }

    left_suffixes = {name.removeprefix("left_hand_") for name in joints if name.startswith("left_hand_")}
    right_suffixes = {name.removeprefix("right_hand_") for name in joints if name.startswith("right_hand_")}
    paired_hand_joints = sorted(left_suffixes & right_suffixes)
    mirror_type_match = 0
    mirror_limit_compatible = 0
    mirror_mimic_match = 0
    for suffix in paired_hand_joints:
        left, right = joints[f"left_hand_{suffix}"], joints[f"right_hand_{suffix}"]
        mirror_type_match += left["type"] == right["type"]
        left_limit, right_limit = left["limit"], right["limit"]
        same_limit = left_limit == right_limit
        sign_mirrored_limit = (
            left_limit is not None
            and right_limit is not None
            and math.isclose(left_limit["lower"], -right_limit["upper"], abs_tol=1.0e-9)
            and math.isclose(left_limit["upper"], -right_limit["lower"], abs_tol=1.0e-9)
            and left_limit["effort"] == right_limit["effort"]
            and left_limit["velocity"] == right_limit["velocity"]
        )
        mirror_limit_compatible += same_limit or sign_mirrored_limit
        left_mimic, right_mimic = left["mimic"], right["mimic"]
        normalized_left = (
            None
            if left_mimic is None
            else {**left_mimic, "leader": left_mimic["leader"].removeprefix("left_hand_")}
        )
        normalized_right = (
            None
            if right_mimic is None
            else {**right_mimic, "leader": right_mimic["leader"].removeprefix("right_hand_")}
        )
        mirror_mimic_match += normalized_left == normalized_right

    arm_joints = {name: item for name, item in joints.items() if re.fullmatch(r"[LR][1-7]_Joint", name)}
    arm_pair_checks: dict[str, Any] = {}
    for index in range(1, 8):
        left, right = arm_joints[f"L{index}_Joint"], arm_joints[f"R{index}_Joint"]
        left_limit, right_limit = left["limit"], right["limit"]
        same_limits = left_limit["lower"] == right_limit["lower"] and left_limit["upper"] == right_limit["upper"]
        sign_mirrored_limits = (
            math.isclose(left_limit["lower"], -right_limit["upper"], abs_tol=1.0e-9)
            and math.isclose(left_limit["upper"], -right_limit["lower"], abs_tol=1.0e-9)
        )
        arm_pair_checks[f"J{index}"] = {
            "same_axis": left["axis"] == right["axis"],
            "opposite_axis": left["axis"] == tuple(-value for value in right["axis"]),
            "same_limits": same_limits,
            "sign_mirrored_limits": sign_mirrored_limits,
        }

    v1_actual_hash = _sha256(PHYSICS_V1_USD)
    v1_declared_hash = profile["robot_asset"]["derived_usd_sha256"]
    masses = [item["mass_kg"] for item in links.values() if item["mass_kg"] is not None]
    tiny_mass_links = sorted(
        name for name, item in links.items() if item["mass_kg"] is not None and item["mass_kg"] < 1.0e-4
    )
    invalid_inertias = sorted(
        name for name, item in links.items() if not item["inertia_positive_definite"]
    )
    missing_collision = sorted(name for name, item in links.items() if item["collision_count"] == 0)
    tactile_links_exist = all(
        name in links for name in (*EXPECTED_TACTILE_CONTACT_LINKS.values(), *EXPECTED_TACTILE_FRAME_LINKS.values())
    )
    metadata_checks = {
        "rigid_body_count": metadata.get("physicsV1ExpectedHandContactBodyCount") == 56,
        "enabled_self_collision_pairs": metadata.get("physicsV1EnabledPairsPerRobot") == 59,
        "filtered_self_collision_pairs": metadata.get("physicsV1FilteredPairsPerRobot") == 2426,
        "outside_solve_contact_disabled": metadata.get("physicsV1OutsideSolveContactDisabledBodyCount") == 59,
        "stored_proxy_filters": metadata.get("physicsV1StoredActiveHandFilteredPairCount") == 7,
        "passive_mimics": metadata.get("physicsV1PassiveMimicCount") == 5,
    }
    checks = {
        "urdf_has_71_links_and_70_joints": len(links) == 71 and len(joints) == 70,
        "urdf_has_14_arm_revolute_joints": len(arm_joints) == 14,
        "each_hand_has_16_independent_and_5_passive_revolute_joints": all(
            side_reports[side]["independent_joint_count"] == 16
            and side_reports[side]["passive_mimic_count"] == 5
            for side in ("right", "left")
        ),
        "all_link_masses_positive_and_finite": bool(masses)
        and all(value > 0.0 and math.isfinite(value) for value in masses),
        "all_inertias_positive_definite": not invalid_inertias,
        "each_hand_mass_is_exactly_1kg_in_urdf": all(
            math.isclose(side_reports[side]["total_mass_kg"], 1.0, abs_tol=1.0e-9)
            for side in ("right", "left")
        ),
        "each_hand_link_has_collision_and_visual": all(
            side_reports[side]["collision_prim_count"] == 27
            and side_reports[side]["visual_prim_count"] == 27
            for side in ("right", "left")
        ),
        "left_right_hand_joint_topology_matches": (
            left_suffixes == right_suffixes
            and mirror_type_match == len(paired_hand_joints)
            and mirror_limit_compatible == len(paired_hand_joints)
            and mirror_mimic_match == len(paired_hand_joints)
        ),
        "tactile_contact_and_frame_links_exist": tactile_links_exist,
        "physics_v1_layer_metadata_is_internally_count_consistent": all(metadata_checks.values()),
        "physics_v1_solve_contact_disable_count_matches_metadata": (
            usd_text.count("physxRigidBody:solveContact = 0")
            == metadata.get("physicsV1OutsideSolveContactDisabledBodyCount")
        ),
        "physics_v1_declared_asset_hash_matches_actual": v1_declared_hash == v1_actual_hash,
    }

    findings = [
        {
            "id": "physics_v1_declared_asset_hash_mismatch",
            "severity": "high",
            "status": "confirmed_parent_issue",
            "evidence": {"declared_sha256": v1_declared_hash, "actual_sha256": v1_actual_hash},
            "disposition": "Do not rewrite Physics-v1; bind Physics-v2 to the actual checked-in parent hash.",
        },
        {
            "id": "hand_mass_evidence_conflict",
            "severity": "medium",
            "status": "hardware_pending",
            "evidence": {
                "urdf_mass_kg_per_hand": side_reports["right"]["total_mass_kg"],
                "physics_v1_vendor_prior_kg": profile["hand"]["vendor_mass_kg"],
                "legacy_simulation_baseline_claim_kg": 1.0,
            },
            "disposition": "Retain 1 kg geometry-derived asset mass for offline work and randomize conservatively; do not claim calibrated inertia.",
        },
        {
            "id": "distal_only_contact_proxy",
            "severity": "medium",
            "status": "intentional_scope_limit",
            "evidence": {
                "contact_solving_body_count": 12,
                "robot_rigid_body_count": 71,
                "out_of_scope_body_count": metadata.get("physicsV1OutsideSolveContactDisabledBodyCount"),
            },
            "disposition": "Keep the scalable hand cage for initial PPO; add explicit coverage tests and never interpret it as full-arm collision fidelity.",
        },
        {
            "id": "right_task_left_hardware_transfer",
            "severity": "high",
            "status": "hardware_transfer_blocked",
            "evidence": {
                "training_side": "right",
                "direct_motion_calibration_side": "left",
                "right_calibration_basis": "inherited_from_left_by_prior_user_authorization",
            },
            "disposition": "Offline Physics-v2 remains right-sided; left deployment requires an explicit task variant and frame/sensor audit.",
        },
        {
            "id": "frozen_runtime_not_present",
            "severity": "high",
            "status": "integrated_gate_blocker",
            "evidence": {
                "frozen_runtime": "Isaac Lab 3.0.0-beta2.patch1 / Python 3.12",
                "detected_runtime": "Isaac Lab 2.3.2 / Python 3.10",
            },
            "disposition": "Continue pure/static work; do not run or reinterpret integrated Physics-v1 qualification under the different API runtime.",
        },
    ]

    structural_keys = [key for key in checks if key != "physics_v1_declared_asset_hash_matches_actual"]
    return {
        "schema_version": 1,
        "audit_id": "tomato_pick_physics_v2_robot_asset_step2",
        "passed_structural_gate": all(checks[key] for key in structural_keys),
        "physics_v1_identity_gate": checks["physics_v1_declared_asset_hash_matches_actual"],
        "ready_for_offline_physics_development": all(checks[key] for key in structural_keys),
        "ready_for_integrated_training": False,
        "ready_for_hardware_transfer": False,
        "inputs": {
            "urdf": str(URDF.relative_to(ROOT)),
            "urdf_sha256": _sha256(URDF),
            "physics_v1_usd": str(PHYSICS_V1_USD.relative_to(ROOT)),
            "physics_v1_usd_sha256": v1_actual_hash,
            "physics_v1_profile": str(PHYSICS_V1_PROFILE.relative_to(ROOT)),
            "simulation_baseline": str(SIMULATION_BASELINE.relative_to(ROOT)),
            "left_hand_calibration": str(LEFT_HAND_CALIBRATION.relative_to(ROOT)),
        },
        "checks": checks,
        "usd_metadata_checks": metadata_checks,
        "summary": {
            "link_count": len(links),
            "joint_count": len(joints),
            "joint_types": dict(Counter(item["type"] for item in joints.values())),
            "total_robot_mass_kg": sum(masses),
            "minimum_link_mass_kg": min(masses),
            "tiny_mass_link_count": len(tiny_mass_links),
            "missing_collision_links": missing_collision,
            "invalid_inertia_links": invalid_inertias,
            "hands": side_reports,
            "arm_mirror": arm_pair_checks,
            "tactile_contact_links": EXPECTED_TACTILE_CONTACT_LINKS,
            "tactile_frame_links": EXPECTED_TACTILE_FRAME_LINKS,
            "physics_v1_layer_metadata": metadata,
        },
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed_structural_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
