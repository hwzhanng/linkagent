"""V3 fingertip proxies with explicit palm-support collision filtering.

The v2 proxy-only scene keeps the palm support insert and five fingertip
proxies physical.  A proxy represents fruit contact and must not jam against
the artificial support insert.  V3 filters only those proxy/support pairs;
tomato/proxy contact remains physical and measured.
"""

from __future__ import annotations

def author_fingertip_collision_proxies(
    env,
    env_ids,
    *,
    radius_m: float = 0.004,
    local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    physical_collision: bool = True,
    contact_offset_m: float = 0.002,
    rest_offset_m: float = 0.0,
    filter_palm_support: bool = True,
    disable_palm_cup_rims: bool = True,
    attachment_mode: str = "fixed_body",
) -> None:
    """Author v2 pads, then exclude the artificial palm insert only."""

    # V3 authors these shapes from ``_setup_scene`` before ContactSensor
    # initialization.  The inherited prestartup EventTerm calls this function
    # again; make that call a no-op so the already registered PhysX shapes are
    # never redefined after the contact views have been constructed.
    if bool(getattr(env, "_v3_fingertip_proxies_authored_early", False)):
        return

    if radius_m <= 0.0 or radius_m > 0.012:
        raise ValueError("radius_m must lie in (0, 0.012]")
    if attachment_mode not in {"fixed_body", "attached_shape"}:
        raise ValueError("attachment_mode must be fixed_body or attached_shape")
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    names = ("thumb_link5", "index_link4", "middle_link4", "ring_link4", "little_link4")
    authored: list[str] = []
    authored_relative_xyz: list[tuple[float, float, float]] = []
    for env_index in range(int(env.scene.num_envs)):
        robot_root = env.sim.stage.GetPrimAtPath(f"/World/envs/env_{env_index}/Robot")
        if robot_root.IsValid() and robot_root.IsInstance():
            robot_root.SetInstanceable(False)
        pads_root = UsdGeom.Scope.Define(
            env.sim.stage, f"/World/envs/env_{env_index}/Robot/v3_fixed_pads"
        ).GetPrim()
        del pads_root
        xform_cache = UsdGeom.XformCache()
        for name in names:
            link_path = f"/World/envs/env_{env_index}/Robot/{name}"
            link_prim = env.sim.stage.GetPrimAtPath(link_path)
            if not link_prim.IsValid():
                raise RuntimeError(f"missing terminal link for v3 pad: {link_path}")
            if attachment_mode == "attached_shape":
                collision_group = env.sim.stage.GetPrimAtPath(f"{link_path}/collisions")
                if collision_group.IsValid() and collision_group.IsInstance():
                    collision_group.SetInstanceable(False)
                path = f"{link_path}/collisions/v3_attached_fingertip_proxy"
                prim = UsdGeom.Sphere.Define(env.sim.stage, path).GetPrim()
                UsdGeom.Sphere(prim).GetRadiusAttr().Set(float(radius_m))
                if any(abs(float(value)) > 1.0e-12 for value in local_xyz):
                    UsdGeom.Xformable(prim).AddTranslateOp(opSuffix="v3_pad_offset").Set(
                        Gf.Vec3d(*(float(value) for value in local_xyz))
                    )
                UsdPhysics.CollisionAPI.Apply(prim).GetCollisionEnabledAttr().Set(bool(physical_collision))
                PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr().Set(float(contact_offset_m))
                collision.CreateRestOffsetAttr().Set(float(rest_offset_m))
                UsdGeom.Imageable(prim).MakeInvisible()
                authored.append(path)
                authored_relative_xyz.append(tuple(float(value) for value in local_xyz))
                continue
            # A dedicated rigid pad has its own PhysX contact view.  It is
            # fixed to the terminal link at an initially coincident pose.
            path = f"/World/envs/env_{env_index}/Robot/v3_fixed_pads/{name}_pad"
            prim = UsdGeom.Sphere.Define(env.sim.stage, path).GetPrim()
            UsdGeom.Sphere(prim).GetRadiusAttr().Set(float(radius_m))
            relative, _ = xform_cache.ComputeRelativeTransform(link_prim, robot_root)
            translation = relative.ExtractTranslation()
            authored_relative_xyz.append(tuple(float(value) for value in translation))
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTransformOp(opSuffix="v3_link_pose").Set(relative)
            if any(abs(float(value)) > 1.0e-12 for value in local_xyz):
                xform.AddTranslateOp(opSuffix="v3_pad_offset").Set(
                    Gf.Vec3d(*(float(value) for value in local_xyz))
                )
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(0.0001)
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(True)
            UsdPhysics.CollisionAPI.Apply(prim).GetCollisionEnabledAttr().Set(
                bool(physical_collision)
            )
            PhysxSchema.PhysxContactReportAPI.Apply(prim)
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr().Set(float(contact_offset_m))
            collision.CreateRestOffsetAttr().Set(float(rest_offset_m))
            UsdGeom.Imageable(prim).MakeInvisible()
            joint = UsdPhysics.FixedJoint.Define(
                env.sim.stage,
                f"/World/envs/env_{env_index}/Robot/joints/v3_{name}_pad_fixed",
            )
            joint.CreateBody0Rel().SetTargets([Sdf.Path(link_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            authored.append(path)
    env._v2_fingertip_proxy_paths = tuple(authored)
    env._v2_fingertip_proxy_radius_m = float(radius_m)
    env._v3_proxy_attachment_mode = attachment_mode
    env._v3_pad_authored_relative_xyz = tuple(authored_relative_xyz)
    if not filter_palm_support:
        env._v3_fingertip_proxies_authored_early = True
        return

    support_suffixes = (
        "palm_support_surface",
        "palm_support_surface_rim_pos_x",
        "palm_support_surface_rim_neg_x",
        "palm_support_surface_rim_pos_y",
        "palm_support_surface_rim_neg_y",
    )
    for proxy_path in env._v2_fingertip_proxy_paths:
        proxy = env.sim.stage.GetPrimAtPath(proxy_path)
        if not proxy.IsValid():
            raise RuntimeError(f"missing authored fingertip proxy: {proxy_path}")
        env_root = proxy_path.split("/Robot/", 1)[0]
        filtered = UsdPhysics.FilteredPairsAPI.Apply(proxy).CreateFilteredPairsRel()
        for suffix in support_suffixes:
            support_path = f"{env_root}/Robot/base_link/{suffix}"
            if env.sim.stage.GetPrimAtPath(support_path).IsValid():
                filtered.AddTarget(Sdf.Path(support_path))
    if disable_palm_cup_rims:
        for env_index in range(int(env.scene.num_envs)):
            for suffix in support_suffixes[1:]:
                rim = env.sim.stage.GetPrimAtPath(
                    f"/World/envs/env_{env_index}/Robot/base_link/{suffix}"
                )
                if rim.IsValid() and rim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(rim).GetCollisionEnabledAttr().Set(False)
    env._v3_fingertip_proxies_authored_early = True


__all__ = ["author_fingertip_collision_proxies"]
