"""Freeze the validated G20 mass profile into the derived URDF and USD asset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import re
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = (
    REPOSITORY_ROOT
    / "botSDK/urdf/02_project_derived/ros2_packages/"
    "linkbot_bimanual_sdk_description/urdf/linkbot_bimanual_sdk_description.urdf"
)
ASSET_DIRECTORY = (
    REPOSITORY_ROOT
    / "botSDK/urdf/02_project_derived/isaac_assets/"
    "linkbot_bimanual_sdk_training_mass1kg"
)
USD_PATH = ASSET_DIRECTORY / "linkbot_bimanual_sdk_training_mass1kg.usda"
PHYSICS_PAYLOAD_PATH = ASSET_DIRECTORY / "payloads/Physics/physics.usda"
MANIFEST_PATH = ASSET_DIRECTORY / "mass_correction_manifest.json"
MASS_CSV_PATH = ASSET_DIRECTORY / "tests/body_mass_inertia.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mass_targets() -> dict[str, float]:
    assets = importlib.import_module("simulation.tomato_pick.isaac.assets")
    targets: dict[str, float] = {}
    for side, arm_prefix in (("left", "L"), ("right", "R")):
        targets[f"{arm_prefix}8_Link"] = 0.030
        targets[f"{side}_hand_base_link"] = assets.G20_PALM_MASS_FRACTION
        for index, mass in enumerate(assets.G20_THUMB_LINK_MASS_FRACTIONS):
            targets[f"{side}_hand_thumb_link{index}"] = mass
        for finger in ("index", "middle", "ring", "little"):
            for index, mass in enumerate(assets.G20_FINGER_LINK_MASS_FRACTIONS):
                targets[f"{side}_hand_{finger}_link{index}"] = mass
    return targets


def _replace_urdf_mass_properties(
    text: str,
    targets: dict[str, float],
    minimum_principal_inertia: float,
) -> str:
    updated = text
    for link_name, target_mass in targets.items():
        link_pattern = re.compile(
            rf'(<link name="{re.escape(link_name)}">)(.*?)(</link>)', re.DOTALL
        )
        match = link_pattern.search(updated)
        if match is None:
            raise RuntimeError(f"URDF link not found: {link_name}")
        body = match.group(2)
        mass_match = re.search(r'<mass value="([^"]+)"\s*/>', body)
        inertia_match = re.search(
            r'<inertia ixx="([^"]+)" ixy="([^"]+)" ixz="([^"]+)" '
            r'iyy="([^"]+)" iyz="([^"]+)" izz="([^"]+)"\s*/>',
            body,
        )
        if mass_match is None or inertia_match is None:
            raise RuntimeError(f"URDF inertial block incomplete: {link_name}")
        original_mass = float(mass_match.group(1))
        matrix_values = [float(value) for value in inertia_match.groups()]
        values = {
            "ixx": matrix_values[0],
            "ixy": matrix_values[1],
            "ixz": matrix_values[2],
            "iyy": matrix_values[3],
            "iyz": matrix_values[4],
            "izz": matrix_values[5],
        }
        if not abs(original_mass - target_mass) <= 1.0e-12:
            scale = target_mass / original_mass
            values = {name: value * scale for name, value in values.items()}
            # A Gershgorin lower bound is conservative but dependency-free:
            # lifting every diagonal by this amount guarantees that every
            # principal inertia is at least the requested floor.
            lower_bound = min(
                values["ixx"] - abs(values["ixy"]) - abs(values["ixz"]),
                values["iyy"] - abs(values["ixy"]) - abs(values["iyz"]),
                values["izz"] - abs(values["ixz"]) - abs(values["iyz"]),
            )
            diagonal_lift = max(0.0, minimum_principal_inertia - lower_bound)
            for name in ("ixx", "iyy", "izz"):
                values[name] += diagonal_lift
        body = mass_match.re.sub(f'<mass value="{target_mass:.12g}"/>', body, count=1)
        body = inertia_match.re.sub(
            '<inertia '
            + " ".join(f'{name}="{value:.12g}"' for name, value in values.items())
            + "/>",
            body,
            count=1,
        )
        updated = updated[: match.start(2)] + body + updated[match.end(2) :]
    return updated


def _author_usd_overrides(
    targets: dict[str, float], minimum_principal_inertia: float
) -> None:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdPhysics

    source_stage = Usd.Stage.Open(str(USD_PATH))
    if source_stage is not None:
        source_stage.SetEditTarget(source_stage.GetSessionLayer())
        source_stage.GetDefaultPrim().GetVariantSet("Physics").SetVariantSelection("physics")
    stage = Usd.Stage.Open(str(USD_PATH))
    if source_stage is None or stage is None:
        raise RuntimeError("could not open LinkBot USD layers")
    source_bodies = {
        prim.GetName(): prim
        for prim in source_stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    bodies = {
        prim.GetName(): prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    missing_source = sorted(set(targets).difference(source_bodies))
    missing_main = sorted(set(targets).difference(bodies))
    stage.SetEditTarget(stage.GetRootLayer())
    for name, target_mass in targets.items():
        if name not in source_bodies or name not in bodies:
            raise RuntimeError(f"USD rigid body not found: {name}")
        source_api = UsdPhysics.MassAPI(source_bodies[name])
        original_mass = float(source_api.GetMassAttr().Get())
        original_inertia = tuple(
            float(value) for value in source_api.GetDiagonalInertiaAttr().Get()
        )
        # USD stores these values as float32.  Treat an already-frozen mass as
        # exact so repeated runs do not multiply inertia by a one-ULP ratio.
        scale = (
            1.0
            if abs(original_mass - target_mass) <= 2.0e-8
            else target_mass / original_mass
        )
        target_inertia = tuple(
            max(value * scale, minimum_principal_inertia) for value in original_inertia
        )
        target_api = UsdPhysics.MassAPI(bodies[name])
        target_api.GetMassAttr().Set(target_mass)
        target_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*target_inertia))

    articulation_prim = stage.GetPrimAtPath("/linkbot_bimanual_sdk/Geometry")
    if not articulation_prim.IsValid():
        raise RuntimeError("USD articulation root not found")
    physx_api = PhysxSchema.PhysxArticulationAPI.Apply(articulation_prim)
    physx_api.CreateEnabledSelfCollisionsAttr(False).Set(False)
    mimic_joints = []
    for prim in stage.Traverse():
        if not prim.GetRelationship("newton:mimicJoint").IsValid():
            continue
        attribute = prim.GetAttribute("newton:mimicEnabled")
        if not attribute.IsValid():
            attribute = prim.CreateAttribute(
                "newton:mimicEnabled", Sdf.ValueTypeNames.Bool, custom=False
            )
        attribute.Set(False)
        mimic_joints.append(prim.GetPath().pathString)
    if len(mimic_joints) != 10:
        raise RuntimeError(f"expected 10 Newton mimic joints, found {len(mimic_joints)}")
    stage.GetRootLayer().Save()


def _verify_frozen_asset(targets: dict[str, float]) -> dict[str, float | int | bool]:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        raise RuntimeError("could not reopen frozen USD")
    bodies = {
        prim.GetName(): prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    maximum_mass_error = 0.0
    for name, target in targets.items():
        actual = float(UsdPhysics.MassAPI(bodies[name]).GetMassAttr().Get())
        maximum_mass_error = max(maximum_mass_error, abs(actual - target))
    self_collision = stage.GetPrimAtPath(
        "/linkbot_bimanual_sdk/Geometry"
    ).GetAttribute("physxArticulation:enabledSelfCollisions").Get()
    mimic_prims = [
        prim
        for prim in stage.Traverse()
        if prim.GetRelationship("newton:mimicJoint").IsValid()
    ]
    disabled_mimic_count = sum(
        prim.GetAttribute("newton:mimicEnabled").IsValid()
        and prim.GetAttribute("newton:mimicEnabled").Get() is False
        for prim in mimic_prims
    )
    return {
        "target_body_count": len(targets),
        "maximum_mass_error_kg": maximum_mass_error,
        "left_hand_mass_kg": sum(
            target for name, target in targets.items() if name.startswith("left_hand_")
        ),
        "right_hand_mass_kg": sum(
            target for name, target in targets.items() if name.startswith("right_hand_")
        ),
        "physx_self_collisions_disabled": self_collision is False,
        "newton_mimic_joint_count": len(mimic_prims),
        "newton_mimic_disabled_count": disabled_mimic_count,
    }


def _write_mass_csv() -> int:
    """Refresh the derived mass audit so it cannot retain pre-fix values."""

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        raise RuntimeError("could not open frozen USD for mass CSV")
    rows: list[tuple[object, ...]] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name = prim.GetName()
        if name == "base_link":
            group = "base"
        elif name.startswith("right_hand_"):
            group = "right_hand"
        elif name.startswith("left_hand_"):
            group = "left_hand"
        elif re.fullmatch(r"R[1-8]_Link", name):
            group = "right_arm"
        elif re.fullmatch(r"L[1-8]_Link", name):
            group = "left_arm"
        else:
            group = "other"
        mass_api = UsdPhysics.MassAPI(prim)
        mass = float(mass_api.GetMassAttr().Get())
        inertia = tuple(float(value) for value in mass_api.GetDiagonalInertiaAttr().Get())
        positive = mass > 0.0 and all(value > 0.0 for value in inertia)
        triangle = all(
            inertia[(index + 1) % 3] + inertia[(index + 2) % 3] >= inertia[index]
            for index in range(3)
        )
        rows.append((len(rows), name, group, mass, *inertia, positive, triangle))
    MASS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MASS_CSV_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "index",
                "name",
                "group",
                "mass_kg",
                "principal_i1_kg_m2",
                "principal_i2_kg_m2",
                "principal_i3_kg_m2",
                "positive_definite",
                "triangle_inequality",
            )
        )
        writer.writerows(rows)
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        minimum_inertia = 2.5e-7
        targets = _mass_targets()
        original_urdf = URDF_PATH.read_text(encoding="utf-8")
        updated_urdf = _replace_urdf_mass_properties(
            original_urdf, targets, minimum_inertia
        )
        URDF_PATH.write_text(updated_urdf, encoding="utf-8")
        _author_usd_overrides(targets, minimum_inertia)
        verification = _verify_frozen_asset(targets)
        verification["mass_csv_body_count"] = _write_mass_csv()
        manifest = {
            "schema_version": 3,
            "asset_role": "fixed_g20_training_mass_profile",
            "method": (
                "One 1.0 kg hand: 0.55 kg palm, 0.13 kg thumb, four 0.08 kg "
                "fingers; scale each source inertia once and enforce a 2.5e-7 "
                "kg m2 principal-inertia floor. Set each R8 mounting adapter to 0.03 kg."
            ),
            "target_hand_mass_kg": 1.0,
            "minimum_finger_link_mass_kg": 0.010,
            "minimum_principal_inertia_kg_m2": minimum_inertia,
            "mounting_adapter_mass_kg": 0.030,
            "self_collisions": False,
            "simulation_mimic_mode": "disabled_in_usd_software_coupled_from_urdf",
            "source_asset": str(URDF_PATH.relative_to(REPOSITORY_ROOT)),
            "generated_asset": str(USD_PATH.relative_to(REPOSITORY_ROOT)),
            "urdf": str(URDF_PATH.relative_to(REPOSITORY_ROOT)),
            "urdf_sha256": _sha256(URDF_PATH),
            "usd": str(USD_PATH.relative_to(REPOSITORY_ROOT)),
            "usd_sha256": _sha256(USD_PATH),
            "mass_csv": str(MASS_CSV_PATH.relative_to(REPOSITORY_ROOT)),
            "mass_csv_sha256": _sha256(MASS_CSV_PATH),
            "verification": verification,
        }
        MANIFEST_PATH.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "LINKBOT_MASS_PROFILE_FREEZE_REPORT="
            + json.dumps(manifest, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        return 0 if all(
            (
                verification["maximum_mass_error_kg"] <= 2.0e-8,
                verification["left_hand_mass_kg"] == 1.0,
                verification["right_hand_mass_kg"] == 1.0,
                verification["physx_self_collisions_disabled"],
                verification["newton_mimic_joint_count"] == 10,
                verification["newton_mimic_disabled_count"] == 10,
            )
        ) else 2
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
