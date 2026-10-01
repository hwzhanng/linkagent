"""Isaac-only task-asset helpers for the single tomato-pick environment."""

from __future__ import annotations

import math
import re

import torch

import isaaclab.sim as sim_utils

from .geometry_sampling import (
    equivalent_diameter_m,
    sample_stratified_ellipsoids,
    stable_geometry_seed,
)


G20_PALM_MASS_FRACTION = 0.55
G20_THUMB_LINK_MASS_FRACTIONS = (0.028, 0.020, 0.034, 0.020, 0.018, 0.010)
G20_FINGER_LINK_MASS_FRACTIONS = (0.012, 0.024, 0.018, 0.016, 0.010)


def _linkbot_mass_targets(target_hand_mass_kg: float = 1.0) -> dict[str, float]:
    targets: dict[str, float] = {}
    for side, arm_prefix in (("left", "L"), ("right", "R")):
        targets[f"{arm_prefix}8_Link"] = 0.030
        targets[f"{side}_hand_base_link"] = target_hand_mass_kg * G20_PALM_MASS_FRACTION
        for index, fraction in enumerate(G20_THUMB_LINK_MASS_FRACTIONS):
            targets[f"{side}_hand_thumb_link{index}"] = target_hand_mass_kg * fraction
        for finger in ("index", "middle", "ring", "little"):
            for index, fraction in enumerate(G20_FINGER_LINK_MASS_FRACTIONS):
                targets[f"{side}_hand_{finger}_link{index}"] = target_hand_mass_kg * fraction
    return targets


def validate_linkbot_mass_properties(
    root_prim_path: str,
    *,
    target_hand_mass_kg: float = 1.0,
    minimum_principal_inertia_kg_m2: float = 2.5e-7,
) -> dict[str, float | int | bool]:
    """Validate the frozen asset without authoring any runtime overrides."""

    from pxr import Usd, UsdPhysics
    from isaaclab.sim.utils.stage import get_current_stage

    root = get_current_stage().GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        raise RuntimeError(f"LinkBot mass-validation root does not exist: {root_prim_path}")
    bodies = {
        prim.GetName(): prim
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    targets = _linkbot_mass_targets(target_hand_mass_kg)
    missing = sorted(set(targets).difference(bodies))
    if missing:
        raise RuntimeError(f"frozen LinkBot mass-profile bodies missing: {missing}")
    maximum_mass_error = 0.0
    minimum_inertia = float("inf")
    for name, target in targets.items():
        mass_api = UsdPhysics.MassAPI(bodies[name])
        actual = float(mass_api.GetMassAttr().Get())
        maximum_mass_error = max(maximum_mass_error, abs(actual - target))
        minimum_inertia = min(
            minimum_inertia,
            *(float(value) for value in mass_api.GetDiagonalInertiaAttr().Get()),
        )
    valid = (
        maximum_mass_error <= 2.0e-8
        and minimum_inertia >= minimum_principal_inertia_kg_m2 * (1.0 - 1.0e-6)
    )
    if not valid:
        raise RuntimeError(
            "frozen LinkBot mass profile is invalid: "
            f"maximum_mass_error={maximum_mass_error}, minimum_inertia={minimum_inertia}"
        )
    return {
        "valid": True,
        "validated_body_count": len(targets),
        "maximum_mass_error_kg": maximum_mass_error,
        "minimum_principal_inertia_kg_m2": minimum_inertia,
        "target_hand_mass_kg": target_hand_mass_kg,
    }


def validate_single_left_hand_topology(
    root_prim_path: str,
    *,
    body_names: tuple[str, ...],
    joint_names: tuple[str, ...],
) -> dict[str, object]:
    """Validate the isolated fixed-base left-hand training asset.

    The old bimanual USD can be made visually left-only while still leaving
    arm/right-hand rigid bodies in the PhysX scene.  This check is deliberately
    strict and runs before sensors are registered: exactly one articulation
    root, only the allow-listed hand bodies, and no arm/right-hand namespace.
    It is a topology boundary, not a tunable performance gate.
    """

    from pxr import Usd, UsdPhysics
    from isaaclab.sim.utils.stage import get_current_stage

    root = get_current_stage().GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        raise RuntimeError(f"single-left-hand root does not exist: {root_prim_path}")
    allow_bodies = set(body_names)
    actual_bodies = {
        prim.GetName()
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    missing = sorted(allow_bodies - actual_bodies)
    unexpected = sorted(actual_bodies - allow_bodies)
    if missing or unexpected:
        raise RuntimeError(
            "single-left-hand body allowlist mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    forbidden = re.compile(r"(?:right_hand|left_hand|(?:^|_)R[1-7]_Joint|(?:^|_)L[1-7]_Joint)")
    forbidden_bodies = sorted(name for name in actual_bodies if forbidden.search(name))
    if forbidden_bodies:
        raise RuntimeError(f"single-left-hand scene contains forbidden body names: {forbidden_bodies}")

    articulation_roots = [
        prim.GetPath().pathString
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(articulation_roots) != 1:
        raise RuntimeError(
            "single-left-hand scene must contain exactly one articulation root: "
            f"{articulation_roots}"
        )

    allowed_joints = set(joint_names)
    actual_joints = {
        prim.GetName()
        for prim in Usd.PrimRange(root)
        if prim.IsA(UsdPhysics.Joint)
    }
    unexpected_joints = sorted(actual_joints - allowed_joints)
    forbidden_joints = sorted(name for name in actual_joints if forbidden.search(name))
    # Fixed fingertip joints and the world root joint are structural and are
    # allowed in addition to the five software-mimic followers.
    if forbidden_joints or not set(joint_names).issubset(actual_joints):
        raise RuntimeError(
            "single-left-hand joint allowlist mismatch: "
            f"missing={sorted(allowed_joints - actual_joints)}, "
            f"forbidden={forbidden_joints}, unexpected={unexpected_joints}"
        )
    return {
        "valid": True,
        "body_count": len(actual_bodies),
        "joint_count": len(actual_joints),
        "articulation_root": articulation_roots[0],
        "forbidden_namespace_present": False,
    }


def condition_linkbot_mass_properties(
    root_prim_path: str,
    *,
    target_hand_mass_kg: float = 1.0,
    mounting_adapter_mass_kg: float = 0.030,
    minimum_principal_inertia_kg_m2: float = 2.5e-7,
) -> dict[str, float | int]:
    """Apply a physically plausible one-kilogram G20 mass distribution.

    The imported asset puts almost all of the published whole-hand mass in the
    palm while retaining gram- and milligram-scale vendor finger links.  That
    distribution is numerically fragile and is not representative of a normal
    motorized dexterous hand.  This runtime-only overlay assigns 55% to the
    palm, 13% to the thumb, and 8% to each other finger.  Inertia tensors keep
    their vendor principal-axis ratios and scale with mass; the tiny fixed
    fingertip bodies receive a conservative physical inertia floor.
    """

    from pxr import Gf, Usd, UsdPhysics
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        raise RuntimeError(f"LinkBot mass-conditioning root does not exist: {root_prim_path}")
    bodies = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    profile_total = (
        G20_PALM_MASS_FRACTION
        + sum(G20_THUMB_LINK_MASS_FRACTIONS)
        + 4.0 * sum(G20_FINGER_LINK_MASS_FRACTIONS)
    )
    if not math.isclose(profile_total, 1.0, abs_tol=1.0e-12):
        raise RuntimeError(f"invalid G20 mass profile total: {profile_total}")

    changed = 0
    minimum_original_mass = float("inf")
    maximum_original_to_conditioned_ratio = 1.0
    conditioned_hand_masses: list[float] = []

    def set_body_mass(prim, target_mass: float) -> None:
        nonlocal changed, minimum_original_mass, maximum_original_to_conditioned_ratio
        mass_api = UsdPhysics.MassAPI(prim)
        original_mass = float(mass_api.GetMassAttr().Get())
        if original_mass <= 0.0:
            raise RuntimeError(f"non-positive mass on {prim.GetPath()}: {original_mass}")
        minimum_original_mass = min(minimum_original_mass, original_mass)
        maximum_original_to_conditioned_ratio = max(
            maximum_original_to_conditioned_ratio, target_mass / original_mass
        )
        original_inertia = tuple(
            float(value) for value in mass_api.GetDiagonalInertiaAttr().Get()
        )
        scale = target_mass / original_mass
        conditioned_inertia = tuple(
            max(value * scale, minimum_principal_inertia_kg_m2)
            for value in original_inertia
        )
        mass_api.GetMassAttr().Set(target_mass)
        mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*conditioned_inertia))
        changed += 1

    for side in ("left", "right"):
        marker = f"/{side}_hand_"
        hand_bodies = [prim for prim in bodies if marker in prim.GetPath().pathString]
        bodies_by_name = {prim.GetName(): prim for prim in hand_bodies}
        expected_names = {f"{side}_hand_base_link"}
        expected_names.update(
            f"{side}_hand_thumb_link{index}"
            for index in range(len(G20_THUMB_LINK_MASS_FRACTIONS))
        )
        expected_names.update(
            f"{side}_hand_{finger}_link{index}"
            for finger in ("index", "middle", "ring", "little")
            for index in range(len(G20_FINGER_LINK_MASS_FRACTIONS))
        )
        missing = expected_names.difference(bodies_by_name)
        if missing:
            raise RuntimeError(f"{side} G20 mass-profile bodies missing: {sorted(missing)}")

        set_body_mass(
            bodies_by_name[f"{side}_hand_base_link"],
            target_hand_mass_kg * G20_PALM_MASS_FRACTION,
        )
        for index, fraction in enumerate(G20_THUMB_LINK_MASS_FRACTIONS):
            set_body_mass(
                bodies_by_name[f"{side}_hand_thumb_link{index}"],
                target_hand_mass_kg * fraction,
            )
        for finger in ("index", "middle", "ring", "little"):
            for index, fraction in enumerate(G20_FINGER_LINK_MASS_FRACTIONS):
                set_body_mass(
                    bodies_by_name[f"{side}_hand_{finger}_link{index}"],
                    target_hand_mass_kg * fraction,
                )
        conditioned_hand_mass = target_hand_mass_kg * profile_total
        conditioned_hand_masses.append(conditioned_hand_mass)

        adapter = next(
            (prim for prim in bodies if prim.GetName() == f"{side[0].upper()}8_Link"),
            None,
        )
        if adapter is None:
            raise RuntimeError(f"{side} mounting adapter body was not found")
        set_body_mass(adapter, mounting_adapter_mass_kg)

    return {
        "changed_body_count": changed,
        "minimum_original_mass_kg": minimum_original_mass,
        "minimum_conditioned_finger_mass_kg": (
            target_hand_mass_kg
            * min(
                min(G20_THUMB_LINK_MASS_FRACTIONS),
                min(G20_FINGER_LINK_MASS_FRACTIONS),
            )
        ),
        "minimum_conditioned_inertia_kg_m2": minimum_principal_inertia_kg_m2,
        "target_hand_mass_kg": target_hand_mass_kg,
        "conditioned_palm_mass_kg": target_hand_mass_kg * G20_PALM_MASS_FRACTION,
        "conditioned_thumb_mass_kg": target_hand_mass_kg * sum(G20_THUMB_LINK_MASS_FRACTIONS),
        "conditioned_finger_mass_kg": target_hand_mass_kg * sum(G20_FINGER_LINK_MASS_FRACTIONS),
        "conditioned_left_hand_mass_kg": conditioned_hand_masses[0],
        "conditioned_right_hand_mass_kg": conditioned_hand_masses[1],
        "conditioned_mounting_adapter_mass_kg": mounting_adapter_mass_kg,
        "maximum_mass_scale": maximum_original_to_conditioned_ratio,
    }


def fix_linkbot_base_to_world(root_prim_path: str) -> str:
    """Repair the imported root joint as a world-to-base fixed constraint."""

    from pxr import Sdf, UsdPhysics
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    base_path = f"{root_prim_path}/Geometry/base_link"
    base_prim = stage.GetPrimAtPath(base_path)
    if not base_prim.IsValid() or not base_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"LinkBot base rigid body does not exist: {base_path}")
    joint_path = f"{root_prim_path}/Physics/root_joint"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if joint_prim.IsValid() and joint_prim.IsA(UsdPhysics.FixedJoint):
        joint = UsdPhysics.FixedJoint(joint_prim)
    else:
        joint_path = f"{root_prim_path}/Geometry/root_world_joint"
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    body0_rel = joint.CreateBody0Rel()
    body0_rel.SetTargets([])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(base_path)])
    joint.CreateJointEnabledAttr().Set(True)
    if body0_rel.GetTargets():
        raise RuntimeError(f"LinkBot root joint is not fixed to world: {body0_rel.GetTargets()}")
    return joint_path


def activate_nested_contact_reporters(
    root_prim_path: str, body_names: tuple[str, ...]
) -> tuple[str, ...]:
    """Apply PhysX contact reporting only to selected nested rigid bodies.

    The LinkBot USD keeps articulated rigid bodies in a nested transform tree.
    Isaac Lab's generic helper intentionally stops at the first rigid body, which
    is correct for ordinary assets but misses every G20 fingertip in this model.
    This task-local overlay authors only runtime API metadata on the composed
    stage; it never modifies the source USD.
    """

    from pxr import Sdf, Usd, UsdPhysics
    from isaaclab.sim.utils.stage import get_current_stage

    stage = get_current_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        raise ValueError(f"contact reporter root does not exist: {root_prim_path}")
    requested = set(body_names)
    activated: list[str] = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetName() not in requested:
            continue
        schemas = prim.GetAppliedSchemas()
        if "PhysxRigidBodyAPI" not in schemas:
            prim.AddAppliedSchema("PhysxRigidBodyAPI")
        if "PhysxContactReportAPI" not in schemas:
            prim.AddAppliedSchema("PhysxContactReportAPI")
        prim.CreateAttribute(
            "physxContactReport:threshold", Sdf.ValueTypeNames.Float, custom=False
        ).Set(0.0)
        activated.append(prim.GetPath().pathString)
    if not activated:
        raise RuntimeError(f"no requested nested rigid bodies found below {root_prim_path}")
    activated_names = {stage.GetPrimAtPath(path).GetName() for path in activated}
    if activated_names != requested:
        raise RuntimeError(
            f"contact reporter mismatch: missing={sorted(requested - activated_names)}"
        )
    return tuple(activated)


def extract_collision_aabbs_by_body(
    root_prim_path: str, body_names: tuple[str, ...]
) -> tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...]:
    """Return conservative collider AABBs expressed in each rigid-body frame.

    The frozen G20 asset stores collision meshes as instance proxies below
    nested rigid bodies.  Fingertip points therefore cannot prove a reset is
    collision-free.  This helper reads every authored collision extent once
    and aggregates it in its owning body frame; the environment then performs
    the per-reset clearance calculation vectorially on the GPU.
    """

    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from isaaclab.sim.utils.stage import get_current_stage

    requested = set(body_names)
    stage = get_current_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        raise ValueError(f"collision-bound root does not exist: {root_prim_path}")
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    bounds: dict[str, Gf.Range3d] = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        extent = prim.GetAttribute("extent").Get()
        if not extent:
            continue
        body = prim.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body or body.GetName() not in requested:
            continue
        box = Gf.BBox3d(
            Gf.Range3d(Gf.Vec3d(*extent[0]), Gf.Vec3d(*extent[1]))
        )
        box.Transform(
            xforms.GetLocalToWorldTransform(prim)
            * xforms.GetLocalToWorldTransform(body).GetInverse()
        )
        aligned = box.ComputeAlignedRange()
        previous = bounds.get(body.GetName())
        bounds[body.GetName()] = (
            aligned
            if previous is None
            else Gf.Range3d.GetUnion(previous, aligned)
        )
    # Some standalone URDF-to-USD exports put CollisionAPI on an intermediate
    # Xform while the actual mesh extent is authored on a descendant Gprim.
    # In that layout the strict collision-only pass above finds no extent at
    # all, even though PhysX has a valid collider.  Fill only the missing
    # bodies from composed Gprim bounds.  This fallback is intentionally
    # conservative (it may include a visual mesh) and is used for reset
    # clearance only; it never changes collision geometry or runtime physics.
    missing = requested.difference(bounds)
    if missing:
        body_prims = {
            prim.GetName(): prim
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in missing
        }
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        for name, body in body_prims.items():
            body_inverse = xforms.GetLocalToWorldTransform(body).GetInverse()
            aggregate: Gf.Range3d | None = None
            for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                try:
                    world_box = bbox_cache.ComputeWorldBound(prim)
                    world_range = world_box.ComputeAlignedRange()
                except Exception:
                    continue
                if world_range.IsEmpty():
                    continue
                local_box = Gf.BBox3d(world_range)
                local_box.Transform(body_inverse)
                aligned = local_box.ComputeAlignedRange()
                aggregate = (
                    aligned
                    if aggregate is None
                    else Gf.Range3d.GetUnion(aggregate, aligned)
                )
            if aggregate is not None:
                bounds[name] = aggregate

    missing = requested.difference(bounds)
    if missing:
        raise RuntimeError(f"collision bounds missing for bodies: {sorted(missing)}")

    result = []
    for name in sorted(bounds):
        lower = bounds[name].GetMin()
        upper = bounds[name].GetMax()
        center = tuple(
            0.5 * (float(lower[index]) + float(upper[index])) for index in range(3)
        )
        half_extent = tuple(
            0.5 * (float(upper[index]) - float(lower[index])) for index in range(3)
        )
        if any(value <= 0.0 for value in half_extent):
            raise RuntimeError(f"non-positive collision extent for {name}: {half_extent}")
        result.append((name, center, half_extent))
    return tuple(result)


def validate_newton_mimic_constraints_disabled(root_prim_path: str) -> tuple[str, ...]:
    """Verify that the fixed training USD uses software follower coupling."""

    from pxr import Usd
    from isaaclab.sim.utils.stage import get_current_stage

    root = get_current_stage().GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        raise RuntimeError(f"missing articulation root: {root_prim_path}")
    disabled: list[str] = []
    for prim in Usd.PrimRange(root):
        if not prim.GetRelationship("newton:mimicJoint").IsValid():
            continue
        attribute = prim.GetAttribute("newton:mimicEnabled")
        if not attribute.IsValid() or attribute.Get() is not False:
            raise RuntimeError(f"Newton mimic is not disabled on {prim.GetPath()}")
        disabled.append(prim.GetPath().pathString)
    if len(disabled) != 10:
        raise RuntimeError(f"expected 10 Newton mimic constraints, found {len(disabled)}")
    return tuple(disabled)


def randomize_correlated_ellipsoid_geometry(
    env,
    env_ids,
    *,
    asset_name: str,
    axis_ranges_m: dict[str, tuple[float, float]],
    density_range_kg_m3: tuple[float, float],
    accepted_mass_kg: tuple[float, float],
    maximum_axis_ratio: float,
    diameter_bins_m: tuple[tuple[float, float], ...] | None = None,
    diameter_bin_weights: tuple[float, ...] | None = None,
    geometry_seed: int | None = None,
    geometry_cohort_id: str = "legacy",
) -> None:
    """Author one ellipsoid per environment before physics starts.

    The fruit begins as a unit-radius convex mesh sphere.  Non-uniform USD
    scale turns that one source mesh into an ellipsoid.  Axes, density, and
    mass are sampled together; impossible size/mass combinations are rejected.
    The sampled tensors are retained on the environment for mass assignment,
    actor measurement synthesis, and the privileged critic.
    """

    if env.sim.is_playing():
        raise RuntimeError("tomato geometry can only be authored before physics starts")
    if env_ids is not None:
        raise ValueError("prestartup tomato geometry must cover every environment")
    count = env.scene.num_envs
    ranges = torch.tensor(
        [axis_ranges_m[name] for name in ("x", "y", "z")],
        dtype=torch.float32,
    )
    if diameter_bins_m is not None:
        if diameter_bin_weights is None:
            diameter_bin_weights = tuple(1.0 for _ in diameter_bins_m)
        base_seed = int(getattr(env.cfg, "seed", 0) if geometry_seed is None else geometry_seed)
        resolved_seed = stable_geometry_seed(base_seed, str(geometry_cohort_id))
        sampled = sample_stratified_ellipsoids(
            count,
            diameter_bins_m=diameter_bins_m,
            diameter_bin_weights=diameter_bin_weights,
            shape_axis_ranges_m=tuple(
                tuple(float(value) for value in axis_ranges_m[name])
                for name in ("x", "y", "z")
            ),
            density_range_kg_m3=density_range_kg_m3,
            accepted_mass_kg=accepted_mass_kg,
            maximum_axis_ratio=maximum_axis_ratio,
            seed=resolved_seed,
        )
        axes = sampled["axes_m"]
        density = sampled["density_kg_m3"]
        diameter_bin_index = sampled["bin_index"]
        equivalent_diameter = sampled["equivalent_diameter_m"]
        bin_quotas = sampled["bin_quotas"]
    else:
        axes = torch.empty((count, 3), dtype=torch.float32)
        density = torch.empty(count, dtype=torch.float32)
        accepted = torch.zeros(count, dtype=torch.bool)
        # 窄尺寸域配合质量/密度约束时，4096 次只增加初始化重试，
        # 不改变历史分布或物理参数。
        for _ in range(4096):
            pending = (~accepted).nonzero(as_tuple=False).squeeze(-1)
            if pending.numel() == 0:
                break
            samples = ranges[:, 0] + torch.rand((pending.numel(), 3)) * (
                ranges[:, 1] - ranges[:, 0]
            )
            sampled_density = density_range_kg_m3[0] + torch.rand(pending.numel()) * (
                density_range_kg_m3[1] - density_range_kg_m3[0]
            )
            mass = 4.0 / 3.0 * math.pi * samples.prod(dim=1) * sampled_density
            ratio = samples.max(dim=1).values / samples.min(dim=1).values
            valid = (
                (mass >= accepted_mass_kg[0])
                & (mass <= accepted_mass_kg[1])
                & (ratio <= maximum_axis_ratio)
            )
            if valid.any():
                target_ids = pending[valid]
                axes[target_ids] = samples[valid]
                density[target_ids] = sampled_density[valid]
                accepted[target_ids] = True
        if not bool(accepted.all()):
            raise RuntimeError("could not sample correlated tomato geometry in 4096 attempts")
        diameter_bin_index = torch.full((count,), -1, dtype=torch.long)
        equivalent_diameter = equivalent_diameter_m(axes)
        bin_quotas = torch.empty(0, dtype=torch.long)

    prim_paths = sim_utils.find_matching_prim_paths(env.scene[asset_name].cfg.prim_path)
    if len(prim_paths) != count:
        raise RuntimeError(f"expected {count} tomato prims, found {len(prim_paths)}")
    from pxr import Gf, UsdGeom, UsdPhysics

    for index, prim_path in enumerate(prim_paths):
        prim = env.sim.stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        scale_op = next(
            (op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale),
            None,
        )
        if scale_op is None:
            scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3d(*(float(value) for value in axes[index])))
        mesh = env.sim.stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")
        UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Set("convexHull")

    env._prestartup_fruit_axes_m = axes
    env._prestartup_fruit_density_kg_m3 = density
    env._prestartup_fruit_equivalent_diameter_m = equivalent_diameter
    env._prestartup_fruit_diameter_bin_index = diameter_bin_index
    env._prestartup_fruit_diameter_bin_quotas = bin_quotas
    env._prestartup_geometry_cohort_id = str(geometry_cohort_id)
    scenario_seed = int(getattr(env.cfg, "seed", 0) if geometry_seed is None else geometry_seed)
    env._prestartup_geometry_scenario_ids = tuple(
        f"{geometry_cohort_id}:seed{scenario_seed}:env{index:06d}:bin{int(diameter_bin_index[index])}"
        for index in range(count)
    )


__all__ = [
    "activate_nested_contact_reporters",
    "condition_linkbot_mass_properties",
    "extract_collision_aabbs_by_body",
    "fix_linkbot_base_to_world",
    "randomize_correlated_ellipsoid_geometry",
    "validate_newton_mimic_constraints_disabled",
    "validate_linkbot_mass_properties",
]
