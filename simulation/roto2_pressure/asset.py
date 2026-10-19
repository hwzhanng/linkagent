"""Build a *single* right L20 hand USD from LinkAgent's delivered URDF.

Run only after Isaac Lab's AppLauncher created the Kit application. No arm,
left hand or hardware SDK is imported by this task.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .core import ACTIVE_JOINTS, FINGERS, MIMICS

ROOT = Path(__file__).resolve().parents[2]
URDF = (ROOT / "botSDK/urdf/01_vendor_original/02_extracted_packages/"
        "L20_8_r_s1/L20_8_r_s1.urdf")
OUT_DIR = ROOT / "botSDK/urdf/02_project_derived/isaac_assets/single_l20_right_rotation"
MANIFEST = OUT_DIR / "asset_manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _author_native_mimics(usd_path: Path) -> list[str]:
    from pxr import PhysxSchema, Usd, UsdPhysics
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Unable to open converted hand USD: {usd_path}")
    joints = {prim.GetName(): prim for prim in Usd.PrimRange(stage.GetPseudoRoot())
              if prim.IsA(UsdPhysics.RevoluteJoint)}
    required = set(ACTIVE_JOINTS) | set(MIMICS)
    if not required.issubset(joints):
        raise RuntimeError(f"L20 importer lost joints: {sorted(required - set(joints))}")
    axis_tokens = {
        UsdPhysics.Tokens.x: UsdPhysics.Tokens.rotX,
        UsdPhysics.Tokens.y: UsdPhysics.Tokens.rotY,
        UsdPhysics.Tokens.z: UsdPhysics.Tokens.rotZ,
    }
    for follower, (driver, ratio) in MIMICS.items():
        owner = joints[follower]
        fa = axis_tokens[UsdPhysics.RevoluteJoint(owner).GetAxisAttr().Get()]
        da = axis_tokens[UsdPhysics.RevoluteJoint(joints[driver]).GetAxisAttr().Get()]
        api = PhysxSchema.PhysxMimicJointAPI.Apply(owner, fa)
        api.GetReferenceJointRel().ClearTargets(True)
        api.GetReferenceJointRel().AddTarget(joints[driver].GetPath())
        api.GetReferenceJointAxisAttr().Set(da)
        api.GetGearingAttr().Set(-ratio)
        api.GetOffsetAttr().Set(0.0)
        api.CreateNaturalFrequencyAttr().Set(150.0)
        api.CreateDampingRatioAttr().Set(1.2)

    # Suppress impossible adjacent segment/interdigit self contacts. Cross
    # finger distal collisions remain enabled for contact-aware manipulation.
    links = [p for p in Usd.PrimRange(stage.GetPseudoRoot())
             if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for i, link in enumerate(links):
        name = link.GetName()
        digit = next((d for d in FINGERS if name.startswith(d + "_")), None)
        if digit is None:
            continue
        api = UsdPhysics.FilteredPairsAPI.Apply(link)
        relation = api.GetFilteredPairsRel()
        for other in links[i + 1:]:
            if other.GetName().startswith(digit + "_"):
                relation.AddTarget(other.GetPath())
    stage.GetRootLayer().Save()
    return sorted(joints)


def ensure_single_hand_usd(force: bool = False) -> Path:
    if not URDF.is_file():
        raise FileNotFoundError(f"Source L20 URDF not checked out: {URDF}")
    if not (URDF.parent / "meshes/base_link.STL").is_file():
        raise FileNotFoundError("L20 vendor meshes are missing (check Git LFS)")
    source_hash = _sha(URDF)
    if MANIFEST.exists() and not force:
        manifest = json.loads(MANIFEST.read_text())
        asset = Path(manifest["usd_path"])
        if (manifest.get("source_sha256") == source_hash
                and asset.is_file() and _sha(asset) == manifest.get("asset_sha256")):
            return asset

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_joints = r"^(thumb|index|middle|ring|little)_joint[0-4]$"
    cfg = UrdfConverterCfg(
        asset_path=str(URDF),
        usd_dir=str(OUT_DIR),
        fix_base=True,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        # We must author native passive mimics and filter same-finger
        # self contacts: instance proxies are read-only in composed USD.
        make_instanceable=False,
        collision_from_visuals=False,
        collision_type="Convex Decomposition",
        self_collision=True,
        robot_type="Manipulator",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type={all_joints: "force"},
            target_type={all_joints: "position"},
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={all_joints: 0.0},
                damping={all_joints: 0.0},
            ),
        ),
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
    )
    usd = Path(UrdfConverter(cfg).usd_path).resolve()
    names = _author_native_mimics(usd)
    manifest = {
        "source_urdf": str(URDF.relative_to(ROOT)),
        "source_sha256": source_hash,
        "usd_path": str(usd),
        "asset_sha256": _sha(usd),
        "joint_names": names,
        "active_joint_names": list(ACTIVE_JOINTS),
        "passive_mimics": MIMICS,
        "calibration_status": "engineering-prior; hardware torque and fruit damage NOT identified",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return usd
