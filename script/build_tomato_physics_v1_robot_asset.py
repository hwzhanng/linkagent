"""Build the independent Physics-v1 robot USD with baked static physics."""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_USD = (
    PROJECT_ROOT
    / "botSDK/urdf/02_project_derived/isaac_assets/"
    "linkbot_bimanual_sdk_training_mass1kg/"
    "linkbot_bimanual_sdk_training_mass1kg.usda"
)
OUTPUT_USD = SOURCE_USD.with_name(
    "linkbot_bimanual_sdk_training_mass1kg_physics_v1.usda"
)
SOURCE_BASE_USD = SOURCE_USD.parent / "payloads/base.usda"
SOURCE_INSTANCES_USD = SOURCE_USD.parent / "payloads/instances.usda"
PHYSICS_V1_BASE_USD = SOURCE_USD.parent / "payloads/base_physics_v1.usda"
PHYSICS_V1_INSTANCES_USD = (
    SOURCE_USD.parent / "payloads/instances_physics_v1.usda"
)
EXPECTED_MATRIX_SHA256 = (
    "16653f7ed1d19e6b6bf0a5e25b74ad57c584c74eb548e8af18f3b2a473cd5755"
)
SELF_COLLISION_REPRESENTATION = (
    "distal_envelope_instanced_prototypes_v5_source_convex_decomposition"
)
ACTIVE_COLLIDER_APPROXIMATION = "convexDecomposition"
PROFILE_PATH = PROJECT_ROOT / "simulation/config/tomato_pick_physics_v1.yaml"
RIGHT_HAND_MIMICS = {
    "right_hand_thumb_joint4": ("right_hand_thumb_joint3", 1.11),
    "right_hand_index_joint3": ("right_hand_index_joint2", 0.94),
    "right_hand_middle_joint3": ("right_hand_middle_joint2", 0.94),
    "right_hand_ring_joint3": ("right_hand_ring_joint2", 0.94),
    "right_hand_little_joint3": ("right_hand_little_joint2", 0.94),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalize_text_layer(path: Path) -> None:
    """Keep generated USDA stable for hashing and clean diffs."""

    payload = path.read_bytes()
    path.write_bytes(payload.rstrip(b"\n") + b"\n")


def build(source: Path, output: Path) -> dict[str, object]:
    # This builder intentionally uses only core USD schemas.  PhysxSchema is
    # registered by Kit at runtime and is not available in the lightweight
    # managed Python process used to produce reproducible derived assets.
    from pxr import Sdf, Usd, UsdPhysics, UsdShade

    if source.resolve() == output.resolve():
        raise ValueError("Physics-v1 asset must not overwrite the source USD")
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.parent.resolve() != output.parent.resolve():
        raise ValueError(
            "Physics-v1 source and output must share an asset directory so "
            "their relative payload references remain reproducible"
        )
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    hand = profile["hand"]
    contact = profile["contact"]
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_dir = output.parent / "payloads"
    source_base = payload_dir / SOURCE_BASE_USD.name
    source_instances = payload_dir / SOURCE_INSTANCES_USD.name
    physics_base = payload_dir / PHYSICS_V1_BASE_USD.name
    physics_instances = payload_dir / PHYSICS_V1_INSTANCES_USD.name
    if not source_base.is_file() or not source_instances.is_file():
        raise FileNotFoundError("source robot payload layers are incomplete")
    shutil.copy2(source_instances, physics_instances)
    physics_base.write_text(
        source_base.read_text(encoding="utf-8").replace(
            "@./instances.usda@", "@./instances_physics_v1.usda@"
        ),
        encoding="utf-8",
    )
    _canonicalize_text_layer(physics_base)
    shutil.copy2(source, output)
    output.write_text(
        output.read_text(encoding="utf-8").replace(
            "@./payloads/base.usda@", "@./payloads/base_physics_v1.usda@"
        ),
        encoding="utf-8",
    )
    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"could not open copied Physics-v1 USD: {output}")
    root = stage.GetDefaultPrim()
    if not root.IsValid():
        raise RuntimeError("source robot USD has no valid default prim")

    articulation_root = stage.GetPrimAtPath(f"{root.GetPath()}/Geometry")
    if not articulation_root.IsValid():
        raise RuntimeError("robot USD has no Geometry articulation root")
    # The source layer already carries PhysxArticulationAPI.  Author the raw
    # namespaced attribute so the output can be built without launching Kit.
    articulation_root.CreateAttribute(
        "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool
    ).Set(True)

    pair_records: list[tuple[str, str, str]] = []
    rigid_body_paths = sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    if len(rigid_body_paths) != 71:
        raise RuntimeError(
            f"expected 71 robot rigid bodies, found {len(rigid_body_paths)}"
        )

    filtered_pairs: set[frozenset[str]] = set()
    adjacent_count = 0
    hand_reporter_paths: set[str] = set()
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint) or "_hand_" not in prim.GetName():
            continue
        joint = UsdPhysics.Joint(prim)
        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        if len(body0_targets) != 1 or len(body1_targets) != 1:
            raise RuntimeError(f"invalid hand joint body relationship: {prim.GetPath()}")
        body0_path, body1_path = body0_targets[0], body1_targets[0]
        filtered_pairs.add(frozenset((str(body0_path), str(body1_path))))
        hand_reporter_paths.update((str(body0_path), str(body1_path)))
        prefix = f"{root.GetPath()}/"
        pair_records.append(
            (
                f"adjacent:{prim.GetName()}",
                str(body0_path).removeprefix(prefix),
                str(body1_path).removeprefix(prefix),
            )
        )
        adjacent_count += 1

    def hand_side(path: str) -> str | None:
        if "/right_hand_base_link" in path:
            return "right"
        if "/left_hand_base_link" in path:
            return "left"
        return None

    def digit_and_link(path: str) -> tuple[str, int] | None:
        name = path.rsplit("/", 1)[-1]
        for digit in ("thumb", "index", "middle", "ring", "little"):
            prefix = f"right_hand_{digit}_link"
            if name.startswith(prefix):
                return digit, int(name.removeprefix(prefix))
        return None

    def is_external_contact_body(path: str) -> bool:
        """Keep the simplified palm/distal-pad collision proxy.

        The visual/inertial articulation remains complete.  Contact solving is
        restricted to the palm and terminal padded segments that actually form
        the tomato cage.  Imported proximal/mounting convex hulls overlap at
        normal poses and are not suitable scalable contact geometry.
        """

        if hand_side(path) != "right":
            return False
        name = path.rsplit("/", 1)[-1]
        if name == "right_hand_base_link":
            return True
        digit_link = digit_and_link(path)
        if digit_link is None:
            return False
        digit, link_index = digit_link
        return link_index >= 3

    def is_self_collision_envelope_body(path: str) -> bool:
        """Return exposed distal bodies that bound the closing hand cage."""

        name = path.rsplit("/", 1)[-1]
        if name == "right_hand_base_link":
            return True
        digit_link = digit_and_link(path)
        if digit_link is None:
            return False
        digit, link_index = digit_link
        return link_index >= (3 if digit == "thumb" else 3)

    def keep_active_hand_collision(body0_path: str, body1_path: str) -> bool:
        if hand_side(body0_path) != "right" or hand_side(body1_path) != "right":
            return False
        if not is_self_collision_envelope_body(
            body0_path
        ) or not is_self_collision_envelope_body(body1_path):
            return False
        name0 = body0_path.rsplit("/", 1)[-1]
        name1 = body1_path.rsplit("/", 1)[-1]
        digit0 = digit_and_link(body0_path)
        digit1 = digit_and_link(body1_path)
        if name0 == "right_hand_base_link" and digit1 is not None:
            return True
        if name1 == "right_hand_base_link" and digit0 is not None:
            return True
        if digit0 is None or digit1 is None:
            return False
        return digit0[0] != digit1[0]

    enabled_count = 0
    prefix = f"{root.GetPath()}/"
    for body0_path, body1_path in combinations(rigid_body_paths, 2):
        pair = frozenset((body0_path, body1_path))
        if pair in filtered_pairs:
            continue
        if keep_active_hand_collision(body0_path, body1_path):
            enabled_count += 1
            continue
        filtered_pairs.add(pair)
        pair_records.append(
            (
                "outside_same_hand_scope",
                body0_path.removeprefix(prefix),
                body1_path.removeprefix(prefix),
            )
        )

    pair_records.sort()
    matrix_payload = json.dumps(
        pair_records, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")
    matrix_sha256 = hashlib.sha256(matrix_payload).hexdigest()
    if adjacent_count != 54 or enabled_count != 59 or len(pair_records) != 2426:
        raise RuntimeError(
            "unexpected collision matrix counts: "
            f"adjacent={adjacent_count}, enabled={enabled_count}, "
            f"filtered={len(pair_records)}"
        )
    if matrix_sha256 != EXPECTED_MATRIX_SHA256:
        raise RuntimeError(
            f"collision matrix hash drifted: {matrix_sha256}"
        )

    # PhysX supports only one collision group per articulation, so collision
    # groups cannot express the selective active-hand policy.  Instead, the
    # coarse arm, parked left hand, and overlapping hand mounting carriers are
    # removed from collision broad phase as well as contact solving.  Keeping
    # their collision meshes enabled with solveContact=false still makes PhysX
    # generate the root-wide articulation self-collision candidate set.  The
    # visual/inertial articulation is unchanged.  The
    # 12 palm/distal-pad proxy bodies collide with the fruit and one another,
    # preventing visible finger/thumb penetration without making imported
    # proximal convex hulls fight the solver during every closure.
    external_contact_paths = {
        path
        for path in rigid_body_paths
        if is_external_contact_body(path)
    }
    if len(external_contact_paths) != 12:
        raise RuntimeError(
            "expected 12 palm/distal-pad contact bodies, got "
            f"{len(external_contact_paths)}"
        )
    stored_active_hand_pairs = 0
    for pair in filtered_pairs:
        body0_path, body1_path = sorted(pair)
        if body0_path not in external_contact_paths or body1_path not in external_contact_paths:
            continue
        body0_prim = stage.GetPrimAtPath(body0_path)
        api = (
            UsdPhysics.FilteredPairsAPI(body0_prim)
            if body0_prim.HasAPI(UsdPhysics.FilteredPairsAPI)
            else UsdPhysics.FilteredPairsAPI.Apply(body0_prim)
        )
        api.CreateFilteredPairsRel().AddTarget(Sdf.Path(body1_path))
        stored_active_hand_pairs += 1
    if stored_active_hand_pairs != 7:
        raise RuntimeError(
            f"expected 7 active-hand filtered pairs, got {stored_active_hand_pairs}"
        )

    def owned_collision_shapes(body_prim):
        """Return collision shapes whose nearest rigid ancestor is body_prim."""

        owned = []
        for candidate in Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies()):
            if candidate == body_prim or not candidate.HasAPI(
                UsdPhysics.CollisionAPI
            ):
                continue
            ancestor = candidate.GetParent()
            belongs_to_child_body = False
            while ancestor and ancestor != body_prim:
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    belongs_to_child_body = True
                    break
                ancestor = ancestor.GetParent()
            if not belongs_to_child_body:
                owned.append(candidate)
        return owned

    def collision_prototype_path(shape) -> Sdf.Path:
        instance_root = shape.GetParent()
        if not instance_root.IsInstance():
            raise RuntimeError(
                f"collision shape is not backed by an instance: {shape.GetPath()}"
            )
        matching_specs = [
            spec
            for spec in instance_root.GetPrimStack()
            if Path(spec.layer.realPath or spec.layer.identifier).resolve()
            == physics_instances.resolve()
        ]
        if len(matching_specs) != 1:
            raise RuntimeError(
                f"could not resolve Physics-v1 collision prototype for {shape.GetPath()}"
            )
        return matching_specs[0].path

    active_prototype_paths: set[Sdf.Path] = set()
    active_collision_shape_count = 0
    for body_path in sorted(external_contact_paths):
        prim = stage.GetPrimAtPath(body_path)
        shapes = owned_collision_shapes(prim)
        if len(shapes) != 1:
            raise RuntimeError(
                f"expected one owned collision shape for {body_path}, got {len(shapes)}"
            )
        active_prototype_paths.add(collision_prototype_path(shapes[0]))
        active_collision_shape_count += 1
    if active_collision_shape_count != 12:
        raise RuntimeError(
            "expected 12 active collision shapes, got "
            f"{active_collision_shape_count}"
        )

    outside_solve_contact_disabled = 0
    outside_collision_shapes_disabled = 0
    outside_prototype_paths: set[Sdf.Path] = set()
    for body_path in rigid_body_paths:
        if body_path in external_contact_paths:
            continue
        prim = stage.GetPrimAtPath(body_path)
        if "PhysxRigidBodyAPI" not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema("PhysxRigidBodyAPI")
        prim.CreateAttribute(
            "physxRigidBody:solveContact", Sdf.ValueTypeNames.Bool
        ).Set(False)
        shapes = owned_collision_shapes(prim)
        if len(shapes) != 1:
            raise RuntimeError(
                f"expected one owned collision shape for {body_path}, got {len(shapes)}"
            )
        outside_prototype_paths.add(collision_prototype_path(shapes[0]))
        outside_collision_shapes_disabled += 1
        outside_solve_contact_disabled += 1
    if outside_solve_contact_disabled != 59:
        raise RuntimeError(
            "expected contact solving disabled on 59 out-of-scope bodies, got "
            f"{outside_solve_contact_disabled}"
        )
    if outside_collision_shapes_disabled != 59:
        raise RuntimeError(
            "expected collision disabled on 59 out-of-scope shapes, got "
            f"{outside_collision_shapes_disabled}"
        )
    if len(active_prototype_paths) != 8 or len(outside_prototype_paths) != 41:
        raise RuntimeError(
            "unexpected collision prototype counts: "
            f"active={len(active_prototype_paths)}, "
            f"outside={len(outside_prototype_paths)}"
        )
    if active_prototype_paths & outside_prototype_paths:
        raise RuntimeError("active and disabled collision prototypes overlap")

    # Author collision policy once on shared Physics-v1 prototypes.  Robot
    # links and all visual geometry remain instanceable in every environment.
    # This avoids materializing 71 mesh hierarchies per clone while still
    # leaving exactly twelve active palm/distal collision bodies.
    stage.GetRootLayer().Save()
    prototype_stage = Usd.Stage.Open(str(physics_instances))
    if prototype_stage is None:
        raise RuntimeError(f"could not open Physics-v1 instances: {physics_instances}")

    def prototype_collision_shape(prototype_path: Sdf.Path):
        prototype = prototype_stage.GetPrimAtPath(prototype_path)
        shapes = [
            candidate
            for candidate in Usd.PrimRange(prototype)
            if candidate.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if len(shapes) != 1:
            raise RuntimeError(
                f"expected one collider in prototype {prototype_path}, got {len(shapes)}"
            )
        return shapes[0]

    for prototype_path in sorted(active_prototype_paths, key=str):
        shape = prototype_collision_shape(prototype_path)
        mesh_collision = UsdPhysics.MeshCollisionAPI(shape)
        if not mesh_collision:
            raise RuntimeError(f"active collision proxy is not a mesh: {shape.GetPath()}")
        source_approximation = mesh_collision.GetApproximationAttr().Get()
        if source_approximation != ACTIVE_COLLIDER_APPROXIMATION:
            raise RuntimeError(
                "source active collision approximation changed: "
                f"expected={ACTIVE_COLLIDER_APPROXIMATION}, "
                f"actual={source_approximation}, path={shape.GetPath()}"
            )
        # Preserve the source collision geometry for the twelve task-active
        # palm/distal bodies.  Physics-v1 v4 forced one convex hull per mesh;
        # that changed the fingertip/palm envelope enough to make the strict
        # tail learnability gate less repeatable.  The performance win from
        # removing the other 59 collision shapes remains isolated here.
        mesh_collision.CreateApproximationAttr(ACTIVE_COLLIDER_APPROXIMATION)
    for prototype_path in sorted(outside_prototype_paths, key=str):
        shape = prototype_collision_shape(prototype_path)
        UsdPhysics.CollisionAPI(shape).CreateCollisionEnabledAttr(False)
    prototype_stage.GetRootLayer().Save()
    _canonicalize_text_layer(physics_instances)
    stage.Reload()
    root = stage.GetDefaultPrim()

    # The five right-hand passive constraints are identical for every clone.
    # Baking them into the referenced robot asset avoids authoring and storing
    # five API schemas per environment in the anonymous scene layer.
    revolute_joints = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(root)
        if prim.IsA(UsdPhysics.RevoluteJoint)
    }
    axis_instance = {
        UsdPhysics.Tokens.x: "rotX",
        UsdPhysics.Tokens.y: "rotY",
        UsdPhysics.Tokens.z: "rotZ",
    }
    mimic_records: list[dict[str, object]] = []
    for follower_name, (driver_name, multiplier) in RIGHT_HAND_MIMICS.items():
        try:
            follower = revolute_joints[follower_name]
            driver = revolute_joints[driver_name]
        except KeyError as error:
            raise RuntimeError(f"missing Physics-v1 mimic joint: {error}") from error
        follower_axis = axis_instance[
            UsdPhysics.RevoluteJoint(follower).GetAxisAttr().Get()
        ]
        driver_axis = axis_instance[
            UsdPhysics.RevoluteJoint(driver).GetAxisAttr().Get()
        ]
        schema = f"PhysxMimicJointAPI:{follower_axis}"
        if schema not in follower.GetAppliedSchemas():
            follower.AddAppliedSchema(schema)
        property_prefix = f"physxMimicJoint:{follower_axis}"
        reference = follower.CreateRelationship(f"{property_prefix}:referenceJoint")
        reference.ClearTargets(True)
        reference.AddTarget(driver.GetPath())
        follower.CreateAttribute(
            f"{property_prefix}:referenceJointAxis", Sdf.ValueTypeNames.Token
        ).Set(driver_axis)
        follower.CreateAttribute(
            f"{property_prefix}:gearing", Sdf.ValueTypeNames.Float
        ).Set(-float(multiplier))
        follower.CreateAttribute(
            f"{property_prefix}:offset", Sdf.ValueTypeNames.Float
        ).Set(-float(hand["passive_mimic_offset_rad"]))
        follower.CreateAttribute(
            f"{property_prefix}:naturalFrequency", Sdf.ValueTypeNames.Float
        ).Set(float(hand["passive_mimic_natural_frequency"]))
        follower.CreateAttribute(
            f"{property_prefix}:dampingRatio", Sdf.ValueTypeNames.Float
        ).Set(float(hand["passive_mimic_damping_ratio"]))
        mimic_records.append(
            {
                "follower": follower_name,
                "driver": driver_name,
                "follower_axis": follower_axis,
                "driver_axis": driver_axis,
                "multiplier": multiplier,
                "offset_rad": hand["passive_mimic_offset_rad"],
                "natural_frequency": hand["passive_mimic_natural_frequency"],
                "damping_ratio": hand["passive_mimic_damping_ratio"],
            }
        )
    mimic_payload = json.dumps(
        mimic_records, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    mimic_sha256 = hashlib.sha256(mimic_payload).hexdigest()

    # The hand-pad material is likewise clone-invariant.  A strong physics
    # binding at every hand rigid-body parent is inherited by the instanced
    # collision meshes without writing bindings into every scene clone.
    material_path = root.GetPath().AppendPath(
        "PhysicsMaterials/PhysicsV1FingerPad"
    )
    material = UsdShade.Material.Define(stage, material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(
        float(contact["finger_static_friction"])
    )
    material_api.CreateDynamicFrictionAttr().Set(
        float(contact["finger_dynamic_friction"])
    )
    material_api.CreateRestitutionAttr().Set(0.0)
    hand_material_body_count = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or "_hand_" not in prim.GetName():
            continue
        binding = (
            UsdShade.MaterialBindingAPI(prim)
            if prim.HasAPI(UsdShade.MaterialBindingAPI)
            else UsdShade.MaterialBindingAPI.Apply(prim)
        )
        binding.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        hand_material_body_count += 1
    # The SDK asset contains 27 rigid bodies per hand.  The separate
    # self-collision metadata counts 56 joint body paths, so it must not be
    # reused as the material-binding count.
    if hand_material_body_count != 54:
        raise RuntimeError(
            "expected Physics-v1 material on 54 hand rigid bodies, got "
            f"{hand_material_body_count}"
        )

    metadata = {
        "physicsV1SelfCollisionPolicy": "right_distal_envelope_and_palm_only",
        "physicsV1SelfCollisionMatrixSha256": matrix_sha256,
        "physicsV1FilteredPairsPerRobot": len(pair_records),
        "physicsV1EnabledPairsPerRobot": enabled_count,
        "physicsV1AdjacentPairsPerRobot": adjacent_count,
        "physicsV1SelfCollisionRepresentation": (
            SELF_COLLISION_REPRESENTATION
        ),
        "physicsV1StoredActiveHandFilteredPairCount": stored_active_hand_pairs,
        "physicsV1OutsideSolveContactDisabledBodyCount": (
            outside_solve_contact_disabled
        ),
        "physicsV1OutsideCollisionDisabledShapeCount": (
            outside_collision_shapes_disabled
        ),
        "physicsV1ActiveColliderApproximation": ACTIVE_COLLIDER_APPROXIMATION,
        "physicsV1ActiveColliderShapeCount": active_collision_shape_count,
        "physicsV1ActiveCollisionPrototypeCount": len(active_prototype_paths),
        "physicsV1DisabledCollisionPrototypeCount": len(outside_prototype_paths),
        "physicsV1BaseLayerSha256": _sha256(physics_base),
        "physicsV1InstancesLayerSha256": _sha256(physics_instances),
        # Contact reporters remain a runtime sensor concern.
        "physicsV1ExpectedHandContactBodyCount": len(hand_reporter_paths),
        "physicsV1PassiveMimicPolicy": "right_hand_native_physx_baked",
        "physicsV1PassiveMimicCount": len(mimic_records),
        "physicsV1PassiveMimicSha256": mimic_sha256,
        "physicsV1HandMaterialPolicy": "both_hands_parent_binding_baked",
        "physicsV1HandMaterialBodyCount": hand_material_body_count,
        "physicsV1HandMaterialPath": str(material_path),
        "physicsV1SourceUsdSha256": _sha256(source),
    }
    for key, value in metadata.items():
        root.SetCustomDataByKey(key, value)
    stage.GetRootLayer().Save()
    _canonicalize_text_layer(output)

    return {
        "schema_version": 3,
        "source_usd": str(source.resolve()),
        "source_usd_sha256": metadata["physicsV1SourceUsdSha256"],
        "output_usd": str(output.resolve()),
        "output_usd_sha256": _sha256(output),
        "matrix_sha256": matrix_sha256,
        "filtered_pairs_per_robot": len(pair_records),
        "enabled_pairs_per_robot": enabled_count,
        "adjacent_pairs_per_robot": adjacent_count,
        "stored_active_hand_filtered_pair_count": stored_active_hand_pairs,
        "outside_solve_contact_disabled_body_count": (
            outside_solve_contact_disabled
        ),
        "outside_collision_disabled_shape_count": outside_collision_shapes_disabled,
        "active_collider_approximation": ACTIVE_COLLIDER_APPROXIMATION,
        "active_collider_shape_count": active_collision_shape_count,
        "active_collision_prototype_count": len(active_prototype_paths),
        "disabled_collision_prototype_count": len(outside_prototype_paths),
        "physics_v1_base_usd": str(physics_base.resolve()),
        "physics_v1_base_usd_sha256": _sha256(physics_base),
        "physics_v1_instances_usd": str(physics_instances.resolve()),
        "physics_v1_instances_usd_sha256": _sha256(physics_instances),
        "expected_hand_contact_body_count": len(hand_reporter_paths),
        "passive_mimic_count": len(mimic_records),
        "passive_mimic_sha256": mimic_sha256,
        "hand_material_body_count": hand_material_body_count,
        "hand_material_path": str(material_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_USD)
    parser.add_argument("--output", type=Path, default=OUTPUT_USD)
    args = parser.parse_args()
    print("TOMATO_PHYSICS_V1_ASSET=" + json.dumps(build(args.source, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
