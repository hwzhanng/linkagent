"""Physics-v2 fruit profile authoring over the frozen robot composition layer."""

from __future__ import annotations

import isaaclab.sim as sim_utils

from simulation.tomato_pick.physics_v2 import load_physics_v2_profile
from simulation.tomato_pick.physics_v2_models import (
    quantize_contact_material_bank,
    sample_correlated_fruit_contact_profiles,
    sample_stem_profiles_v2,
)


def author_correlated_fruit_profiles_v2(env, env_ids, *, asset_name: str) -> None:
    """Author correlated v2 fruit/contact and explicitly selected stem priors."""

    if env.sim.is_playing():
        raise RuntimeError("Physics-v2 fruit profiles must be authored prestartup")
    if env_ids is not None:
        raise ValueError("Physics-v2 prestartup profile must cover every environment")
    profile = load_physics_v2_profile()
    contact = profile["fruit_contact_v2"]
    bank_cfg = contact["material_bank"]
    fruit = sample_correlated_fruit_contact_profiles(env.scene.num_envs)
    stem_mode = profile["stem_modes_v2"]["active_offline_mode"]
    profiles = dict(fruit)
    profiles.update(sample_stem_profiles_v2(fruit, mode=stem_mode))
    contact_profiles = quantize_contact_material_bank(
        profiles,
        friction_bins=bank_cfg["friction_bins"],
        compliance_bins=bank_cfg["compliance_bins"],
    )
    axes = profiles["axes_m"]
    prim_paths = sim_utils.find_matching_prim_paths(env.scene[asset_name].cfg.prim_path)
    if len(prim_paths) != env.scene.num_envs:
        raise RuntimeError(
            f"expected {env.scene.num_envs} Physics-v2 tomatoes, found {len(prim_paths)}"
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
            static_friction=float(contact_profiles["material_static_friction"][index]),
            dynamic_friction=float(contact_profiles["material_dynamic_friction"][index]),
            restitution=float(contact["restitution"]),
            friction_combine_mode="average",
            restitution_combine_mode="min",
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
            raise RuntimeError(f"Physics-v2 material was not created: {material_path}")
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
        scale_op.Set(Gf.Vec3d(*(float(value) for value in axes[index])))
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
            "Physics-v2 materials did not reach all tomato collision shapes: "
            f"{inherited_collider_count} < {env.scene.num_envs}"
        )
    env._prestartup_fruit_axes_m = axes
    env._prestartup_fruit_density_kg_m3 = profiles["density_kg_m3"]
    env._prestartup_physics_v1_profiles = profiles
    env._prestartup_physics_v2_profiles = profiles
    env._prestartup_contact_material_profiles = contact_profiles
    env._physics_v2_fruit_material_binding = {
        "material_count": len(material_bank),
        "bound_fruit_count": len(prim_paths),
        "inherited_collider_count": inherited_collider_count,
        "stem_mode": stem_mode,
    }


__all__ = ["author_correlated_fruit_profiles_v2"]
