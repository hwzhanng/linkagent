#!/usr/bin/env python3
"""Convert the LinkBot dual-arm/dual-hand URDF into a standalone Isaac asset tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


DERIVED_ROOT = Path(__file__).resolve().parents[3] / "urdf/02_project_derived"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_URDF = str(
    DERIVED_ROOT
    / "ros2_packages/linkbot_bimanual_sdk_description/urdf/"
    "linkbot_bimanual_sdk_description.urdf"
)
DEFAULT_OUTPUT = str(DERIVED_ROOT / "isaac_assets")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", default=DEFAULT_URDF, help="Source URDF path.")
parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Root directory for generated Isaac assets.")
parser.add_argument(
    "--collision-approximation",
    default="convexDecomposition",
    choices=("convexHull", "convexDecomposition"),
    help="Mesh collision approximation authored as a USD override.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


MEASURED_HAND_VELOCITY_LIMITS_RAD_S = {
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


def portable_path(path: Path) -> str:
    """Prefer repository-relative provenance for canonical in-repo assets."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_collision_approximation(stage: Usd.Stage, approximation: str) -> int:
    """Author the requested approximation over explicit URDF collision meshes."""
    token = {
        "convexHull": UsdPhysics.Tokens.convexHull,
        "convexDecomposition": UsdPhysics.Tokens.convexDecomposition,
    }[approximation]
    count = 0
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
        if not mesh_collision:
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set(token)
        count += 1
    stage.GetRootLayer().Save()
    return count


def find_named_prim(stage: Usd.Stage, name: str) -> Usd.Prim | None:
    """Find one prim by name in an overlay or composed USD stage."""
    matches = [prim for prim in stage.TraverseAll() if prim.GetName() == name]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Expected one prim named {name}, found {len(matches)}")
    return matches[0]


def author_hand_velocity_limits(
    physics_stage: Usd.Stage, physx_stage: Usd.Stage
) -> list[dict[str, object]]:
    """Restore measured/inherited hand limits for every side present in the asset."""
    authored: list[dict[str, object]] = []
    for side in ("left", "right"):
        probe_name = f"{side}_hand_thumb_joint0"
        if find_named_prim(physics_stage, probe_name) is None:
            continue
        for suffix, velocity_rad_s in MEASURED_HAND_VELOCITY_LIMITS_RAD_S.items():
            joint_name = f"{side}_hand_{suffix}"
            for stage_name, stage in (
                ("physics", physics_stage),
                ("physx", physx_stage),
            ):
                prim = find_named_prim(stage, joint_name)
                if prim is None or not prim.IsValid():
                    raise RuntimeError(
                        f"Missing {stage_name} hand joint {joint_name}"
                    )
                attribute = prim.GetAttribute("urdf:limit:velocity")
                if not attribute.IsValid():
                    raise RuntimeError(
                        f"Missing {stage_name} velocity attribute for "
                        f"{joint_name}"
                    )
                attribute.Set(float(velocity_rad_s))
            physx_prim = find_named_prim(physx_stage, joint_name)
            if physx_prim is None:
                raise RuntimeError(f"Missing PhysX hand joint {joint_name}")
            physx_attribute = physx_prim.GetAttribute(
                "physxJoint:maxJointVelocity"
            )
            if not physx_attribute.IsValid():
                raise RuntimeError(
                    f"Missing PhysX velocity attribute for {joint_name}"
                )
            physx_attribute.Set(float(math.degrees(velocity_rad_s)))
            authored.append(
                {
                    "side": side,
                    "joint": joint_name,
                    "velocity_limit_rad_s": velocity_rad_s,
                    "physx_velocity_limit_deg_s": math.degrees(
                        velocity_rad_s
                    ),
                    "provenance": (
                        "measured_connected_left_g20_20260726"
                        if side == "left"
                        else "inherited_same_model_same_batch_from_left"
                    ),
                }
            )
    physics_stage.GetRootLayer().Save()
    physx_stage.GetRootLayer().Save()
    return authored


def normalize_revolute_mimic_offsets(stage: Usd.Stage) -> list[dict[str, object]]:
    """Convert URDF mimic offsets from radians to the USD revolute-joint degree unit."""
    corrections: list[dict[str, object]] = []
    # The generated physics layer is an overlay layer. TraverseAll is required
    # because opening it in isolation leaves its overlay ancestors undefined.
    for prim in stage.TraverseAll():
        attribute = prim.GetAttribute("newton:mimicCoef0")
        if not prim.IsA(UsdPhysics.RevoluteJoint) or not attribute.IsValid():
            continue
        offset_rad = float(attribute.Get())
        offset_degrees = math.degrees(offset_rad)
        attribute.Set(offset_degrees)
        corrections.append(
            {
                "joint": prim.GetName(),
                "source_offset_rad": offset_rad,
                "usd_offset_degrees": offset_degrees,
            }
        )
    stage.GetRootLayer().Save()
    return corrections


def author_mimic_safe_open_state(
    stage: Usd.Stage, leader_position_rad: float = 0.1
) -> list[dict[str, object]]:
    """Author a limit-feasible open pose for every mimic leader/follower pair."""
    authored: list[dict[str, object]] = []
    for follower in stage.TraverseAll():
        offset_attribute = follower.GetAttribute("newton:mimicCoef0")
        multiplier_attribute = follower.GetAttribute("newton:mimicCoef1")
        mimic_relationship = follower.GetRelationship("newton:mimicJoint")
        if (
            not follower.IsA(UsdPhysics.RevoluteJoint)
            or not offset_attribute.IsValid()
            or not multiplier_attribute.IsValid()
            or not mimic_relationship.IsValid()
        ):
            continue
        targets = mimic_relationship.GetTargets()
        if len(targets) != 1:
            continue
        leader = stage.GetPrimAtPath(targets[0])
        if not leader.IsValid() or not leader.IsA(UsdPhysics.RevoluteJoint):
            continue

        offset_degrees = float(offset_attribute.Get())
        multiplier = float(multiplier_attribute.Get())
        leader_degrees = math.degrees(leader_position_rad)
        follower_degrees = offset_degrees + multiplier * leader_degrees

        for joint, position_degrees in (
            (leader, leader_degrees),
            (follower, follower_degrees),
        ):
            joint.CreateAttribute(
                "state:angular:physics:position", Sdf.ValueTypeNames.Float
            ).Set(position_degrees)
            joint.CreateAttribute(
                "drive:angular:physics:targetPosition", Sdf.ValueTypeNames.Float
            ).Set(position_degrees)

        authored.append(
            {
                "leader": leader.GetName(),
                "follower": follower.GetName(),
                "leader_position_rad": leader_position_rad,
                "follower_position_rad": math.radians(follower_degrees),
            }
        )
    stage.GetRootLayer().Save()
    return authored


def inspect_stage(stage: Usd.Stage) -> dict[str, object]:
    result: dict[str, object] = {
        "default_prim": str(stage.GetDefaultPrim().GetPath()),
        "revolute_joints": 0,
        "fixed_joints": 0,
        "rigid_bodies": 0,
        "articulation_roots": [],
        "collision_meshes": 0,
        "collision_approximations": {},
        "mimic_attributes": [],
    }
    articulation_roots: list[str] = []
    mimic_attributes: list[dict[str, object]] = []
    approximations: dict[str, int] = {}

    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            result["revolute_joints"] += 1
        if prim.IsA(UsdPhysics.FixedJoint):
            result["fixed_joints"] += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            result["rigid_bodies"] += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            result["collision_meshes"] += 1
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            approximation_name = str(approximation or "")
            approximations[approximation_name] = approximations.get(approximation_name, 0) + 1
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if "mimic" in name.lower():
                mimic_attributes.append(
                    {
                        "prim": str(prim.GetPath()),
                        "attribute": name,
                        "value": str(attribute.Get()),
                    }
                )

    result["articulation_roots"] = articulation_roots
    result["collision_approximations"] = approximations
    result["mimic_attributes"] = mimic_attributes
    return result


def main() -> None:
    urdf_path = Path(args.urdf).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    package_dir = urdf_path.parent.parent
    package_name = package_dir.name

    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    output_root.mkdir(parents=True, exist_ok=True)

    arm_pattern = r"^[LR][1-7]_Joint$"
    converter_cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(output_root),
        fix_base=True,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        make_instanceable=True,
        collision_from_visuals=False,
        collision_type="Convex Decomposition",
        self_collision=False,
        robot_type="Manipulator",
        ros_package_paths=[{"name": package_name, "path": str(package_dir)}],
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type={arm_pattern: "force"},
            target_type={arm_pattern: "position"},
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={arm_pattern: 2000.0},
                damping={arm_pattern: 20.0},
            ),
        ),
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
    )

    print(json.dumps(converter_cfg.to_dict(), ensure_ascii=False, indent=2))
    converter = UrdfConverter(converter_cfg)
    usd_path = Path(converter.usd_path).resolve()

    physics_path = usd_path.parent / "payloads" / "Physics" / "physics.usda"
    physics_stage = Usd.Stage.Open(str(physics_path))
    if physics_stage is None:
        raise RuntimeError(f"Unable to open generated physics layer: {physics_path}")
    physx_path = usd_path.parent / "payloads" / "Physics" / "physx.usda"
    physx_stage = Usd.Stage.Open(str(physx_path))
    if physx_stage is None:
        raise RuntimeError(f"Unable to open generated PhysX layer: {physx_path}")
    hand_velocity_limits = author_hand_velocity_limits(
        physics_stage, physx_stage
    )
    hand_sides = sorted(
        {str(item["side"]) for item in hand_velocity_limits}
    )
    mimic_offset_corrections = normalize_revolute_mimic_offsets(physics_stage)
    mimic_safe_open_state = author_mimic_safe_open_state(physics_stage)

    instances_path = usd_path.parent / "payloads" / "instances.usda"
    instances_stage = Usd.Stage.Open(str(instances_path))
    if instances_stage is None:
        raise RuntimeError(f"Unable to open generated instance layer: {instances_path}")
    instances_stage.Load()
    collision_override_count = apply_collision_approximation(
        instances_stage, args.collision_approximation
    )

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Unable to open generated USD: {usd_path}")
    stage.Load()
    inspection = inspect_stage(stage)
    asset_files = {
        path.relative_to(usd_path.parent).as_posix(): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(usd_path.parent.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc"}
    }

    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_urdf": portable_path(urdf_path),
        "source_urdf_sha256": sha256(urdf_path),
        "generated_usd": portable_path(usd_path),
        "generated_usd_sha256": sha256(usd_path),
        "isaac_asset_root": portable_path(usd_path.parent),
        "asset_files": asset_files,
        "conversion": {
            "fix_base": True,
            "merge_fixed_joints": False,
            "self_collision": False,
            "collision_from_visuals": False,
            "collision_approximation_override": args.collision_approximation,
            "collision_mesh_override_count": collision_override_count,
            "mimic_offset_unit_corrections": mimic_offset_corrections,
            "mimic_safe_open_state": mimic_safe_open_state,
            "arm_drive": {
                "joint_pattern": arm_pattern,
                "drive_type": "force",
                "target_type": "position",
                "stiffness_nm_per_rad": 2000.0,
                "damping_nm_s_per_rad": 20.0,
                "provenance": "vendor_simulation_starting_point_not_hardware_identification",
            },
            "hand_drive": {
                "overridden": False,
                "reason": "vendor damping, friction, torque and drive gains are not identified yet",
                "velocity_limits": {
                    "present_sides": hand_sides,
                    "left": (
                        "measured_connected_g20_20260726"
                        if "left" in hand_sides
                        else "not_present"
                    ),
                    "right": (
                        "inherited_from_same_model_same_batch_left_by_user_authorization"
                        if "right" in hand_sides
                        else "not_present"
                    ),
                    "authored_joints": hand_velocity_limits,
                },
                "continuous_hardware_speed_cap_raw": 204,
                "continuous_hardware_speed_cap_percent": 80.0,
            },
            "tactile": {
                "normal_and_tangential_force_unit": "N",
                "per_finger_full_scale_n": 20.0,
                "matrix_interpretation": "spatial_distribution_weights_0_255",
            },
        },
        "stage_inspection": inspection,
    }
    metadata_path = usd_path.parent / "linkbot_asset_manifest.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + os.linesep, encoding="utf-8")

    print(f"GENERATED_USD={usd_path}")
    print(f"ASSET_MANIFEST={metadata_path}")
    print(json.dumps(inspection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    conversion_error: BaseException | None = None
    try:
        main()
    except BaseException as exc:
        conversion_error = exc
    finally:
        try:
            simulation_app.close()
        except SystemExit as exc:
            if conversion_error is None and exc.code not in (None, 0):
                conversion_error = exc
    if conversion_error is not None:
        raise conversion_error
