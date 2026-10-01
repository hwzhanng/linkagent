"""Read-only USD audit of fingertip origins, bounds, collision shapes and joint axes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET = PROJECT / "simulation/ppo_tomato_grasp/artifacts/physx_assets/left_hand_urdf/left_hand_l20_physx.usd"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from pxr import Usd, UsdGeom, UsdPhysics

from simulation.ppo_tomato_grasp_v3.provenance import sha256_file, sha256_tree


LINKS = ("thumb_link5", "index_link4", "middle_link4", "ring_link4", "little_link4")


def vec(value) -> list[float]:
    return [float(v) for v in value]


def quat_xyzw(value) -> list[float]:
    imaginary = value.GetImaginary()
    return [float(imaginary[0]), float(imaginary[1]), float(imaginary[2]), float(value.GetReal())]


def bounds(cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> dict:
    value = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    minimum, maximum = value.GetMin(), value.GetMax()
    return {
        "minimum_m": vec(minimum),
        "maximum_m": vec(maximum),
        "center_m": vec((minimum + maximum) * 0.5),
        "size_m": vec(maximum - minimum),
    }


def main() -> None:
    asset = args.asset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise RuntimeError(f"cannot open {asset}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    links = {}
    all_prims = list(stage.Traverse())
    for name in LINKS:
        matches = [prim for prim in all_prims if str(prim.GetPath()).endswith("/" + name)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one {name}, found {len(matches)}")
        prim = matches[0]
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        origin = matrix.ExtractTranslation()
        link_bounds = bounds(cache, prim)
        collisions = []
        for child in Usd.PrimRange(prim):
            if child == prim or not child.HasAPI(UsdPhysics.CollisionAPI):
                continue
            collisions.append({
                "path": str(child.GetPath()),
                "type": child.GetTypeName(),
                "bounds": bounds(cache, child),
            })
        center = link_bounds["center_m"]
        links[name] = {
            "path": str(prim.GetPath()),
            "body_origin_world_m": vec(origin),
            "proxy_local_xyz_m": [0.0, 0.0, 0.0],
            "proxy_center_world_m_at_default_pose": vec(origin),
            "link_bounds": link_bounds,
            "origin_minus_bounds_center_m": [float(origin[i]) - float(center[i]) for i in range(3)],
            "collision_shapes": collisions,
        }
    joints = []
    for prim in all_prims:
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        path = str(prim.GetPath())
        if not any(token in path for token in ("thumb", "index", "middle", "ring", "little")):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        joints.append({
            "path": path,
            "axis": str(joint.GetAxisAttr().Get()),
            "lower_limit_deg": joint.GetLowerLimitAttr().Get(),
            "upper_limit_deg": joint.GetUpperLimitAttr().Get(),
            "body0": [str(v) for v in joint.GetBody0Rel().GetTargets()],
            "body1": [str(v) for v in joint.GetBody1Rel().GetTargets()],
            "local_pos0_m": vec(joint.GetLocalPos0Attr().Get()),
            "local_pos1_m": vec(joint.GetLocalPos1Attr().Get()),
            "local_rot0_xyzw": quat_xyzw(joint.GetLocalRot0Attr().Get()),
            "local_rot1_xyzw": quat_xyzw(joint.GetLocalRot1Attr().Get()),
        })
    report = {
        "version": "v3_asset_geometry_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_command_allowed": False,
        "asset": str(asset),
        "asset_sha256": sha256_file(asset),
        "asset_tree_sha256": sha256_tree(asset.parent),
        "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "stage_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "authored_proxy": {
            "shape": "sphere",
            "radius_m": 0.006,
            "local_xyz_m": [0.0, 0.0, 0.0],
            "source": "v3_large90_nominal",
        },
        "links": links,
        "revolute_joints": joints,
        "interpretation": (
            "USD/default-pose geometry audit only. It does not calibrate the real hand and does not prove "
            "that a link-body origin coincides with the physical tactile pad center."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "links": len(links), "joints": len(joints)}, indent=2), flush=True)
    simulation_app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
