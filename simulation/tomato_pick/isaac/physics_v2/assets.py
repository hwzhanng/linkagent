"""Pre-startup public morphology authoring for Physics-v2."""

from __future__ import annotations

import isaaclab.sim as sim_utils

from simulation.tomato_pick.physics_v2 import (
    load_physics_v2_profile,
    quantize_engineering_contact_profiles,
    sample_physics_v2_profiles,
)


def author_public_prior_fruit_profiles(env, env_ids, *, asset_name: str) -> None:
    """Author one public-prior crop profile per cloned environment."""

    if env.sim.is_playing():
        raise RuntimeError("Physics-v2 fruit profiles must be authored prestartup")
    if env_ids is not None:
        raise ValueError("Physics-v2 prestartup profiles must cover every environment")
    v2 = load_physics_v2_profile()
    parent = v2["frozen_parent"]
    from simulation.tomato_pick.config import load_object
    from simulation.tomato_pick.physics_v2 import REPOSITORY_ROOT

    parent_profile = load_object(REPOSITORY_ROOT / parent["physics_v1_profile"])
    contact = parent_profile["contact"]
    randomization = parent_profile["randomization"]
    profiles = sample_physics_v2_profiles(env.scene.num_envs)
    contact_profiles = quantize_engineering_contact_profiles(
        profiles,
        friction_bins=randomization["contact_friction_bins"],
        compliance_bins=randomization["contact_compliance_bins"],
    )
    radii = profiles["radii_m"]
    prim_paths = sim_utils.find_matching_prim_paths(env.scene[asset_name].cfg.prim_path)
    if len(prim_paths) != env.scene.num_envs:
        raise RuntimeError(
            f"expected {env.scene.num_envs} Physics-v2 tomato prims, found {len(prim_paths)}"
        )

    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = env.sim.stage
    material_bank: dict[int, UsdShade.Material] = {}
    for index in range(env.scene.num_envs):
        material_bin = int(contact_profiles["material_bin"][index])
        if material_bin in material_bank:
            continue
        friction_bin = int(contact_profiles["friction_bin"][index])
        compliance_bin = int(contact_profiles["compliance_bin"][index])
        material_path = (
            "/World/PhysicsMaterials/"
            f"PhysicsV2Tomato_f{friction_bin}_c{compliance_bin}"
        )
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(
                contact_profiles["material_static_friction"][index]
            ),
            dynamic_friction=float(
                contact_profiles["material_dynamic_friction"][index]
            ),
            restitution=contact["restitution"],
            friction_combine_mode=contact["friction_combine_mode"],
            restitution_combine_mode=contact["restitution_combine_mode"],
            compliant_contact_stiffness=float(
                contact_profiles["material_stiffness_n_m"][index]
            ),
            compliant_contact_damping=float(
                contact_profiles["material_damping_n_s_m"][index]
            ),
        )
        material_cfg.func(material_path, material_cfg)
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        if not material:
            raise RuntimeError(
                f"Physics-v2 tomato material was not created: {material_path}"
            )
        material_bank[material_bin] = material

    inherited_collider_count = 0
    for index, prim_path in enumerate(prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        xformable = UsdGeom.Xformable(prim)
        scale_op = next(
            (
                op
                for op in xformable.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeScale
            ),
            None,
        )
        if scale_op is None:
            scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3d(*(float(value) for value in radii[index])))
        mesh = stage.GetPrimAtPath(f"{prim_path}/geometry/mesh")
        UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Set("convexHull")
        material = material_bank[int(contact_profiles["material_bin"][index])]
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
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not child.HasAPI(UsdPhysics.CollisionAPI):
                continue
            inherited = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial(
                "physics"
            )[0]
            if inherited and inherited.GetPath() == material.GetPath():
                inherited_collider_count += 1
    if inherited_collider_count < env.scene.num_envs:
        raise RuntimeError(
            "Physics-v2 tomato materials did not reach all collision shapes: "
            f"inherited={inherited_collider_count}, expected>={env.scene.num_envs}"
        )

    env._prestartup_fruit_axes_m = radii
    env._prestartup_fruit_density_kg_m3 = profiles["density_kg_m3"]
    # Compatibility key consumed by the frozen V1 base class during init.
    env._prestartup_physics_v1_profiles = profiles
    env._prestartup_physics_v2_profiles = profiles
    env._prestartup_contact_material_profiles = contact_profiles
    env._physics_v2_fruit_material_binding = {
        "material_count": len(material_bank),
        "bound_fruit_count": env.scene.num_envs,
        "inherited_collider_count": inherited_collider_count,
        "public_morphology": True,
        "engineering_pair_contact_prior": True,
        "contact_overlay_hardware_calibrated": False,
    }


__all__ = ["author_public_prior_fruit_profiles"]
