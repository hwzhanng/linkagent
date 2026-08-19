"""Simulation-only fingertip collision proxy authoring for the v2 task.

The imported L20 USD exposes roughly 2 mm link4 collision boxes while the
visual/kinematic fingertip bodies are several centimetres from the palm.  The
proxy is an explicit, small spherical fingertip pad attached to each straight
link4 rigid body.  It is not enabled in the archived v1 environment and is
reported in every v2 config as an uncalibrated simulation prior.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils


def author_fingertip_collision_proxies(
    env,
    env_ids,
    *,
    radius_m: float = 0.004,
    local_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    physical_collision: bool = True,
    contact_offset_m: float = 0.002,
    rest_offset_m: float = 0.0,
) -> None:
    """Add one invisible collision sphere under each straight link4 body."""
    if env.sim.is_playing():
        raise RuntimeError("fingertip collision proxies must be authored before physics starts")
    if radius_m <= 0.0 or radius_m > 0.012:
        raise ValueError("radius_m must lie in (0, 0.012]")
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    # The thumb fingertip is link5 in the imported hand; the straight fingers
    # terminate at link4.  Author the same small simulation-only pad for all
    # five so the strict five-finger gate can observe a real contact path.
    names = ("thumb_link5", "index_link4", "middle_link4", "ring_link4", "little_link4")
    # The single-digit retention probe can isolate one physical proxy while
    # preserving the palm collider.  The normal v2 configuration leaves this
    # switch false, so production/training proxy authoring is unchanged.
    debug_only = bool(getattr(env.cfg, "debug_support_only_collision", False))
    debug_digit = str(getattr(env.cfg, "debug_collision_digit", ""))
    count = int(env.scene.num_envs)
    authored = []
    for env_index in range(count):
        # The imported USD is instanceable.  Disable instancing at the clone
        # root before adding a v2-only child; otherwise USD rejects authoring
        # beneath ``link4/collisions`` as an instance proxy.
        robot_root = env.sim.stage.GetPrimAtPath(f"/World/envs/env_{env_index}/Robot")
        if robot_root.IsValid() and robot_root.IsInstance():
            robot_root.SetInstanceable(False)
        for name in names:
            collision_group = env.sim.stage.GetPrimAtPath(
                f"/World/envs/env_{env_index}/Robot/{name}/collisions"
            )
            if collision_group.IsValid() and collision_group.IsInstance():
                collision_group.SetInstanceable(False)
            path = f"/World/envs/env_{env_index}/Robot/{name}/collisions/v2_fingertip_proxy"
            prim = UsdGeom.Sphere.Define(env.sim.stage, path).GetPrim()
            sphere = UsdGeom.Sphere(prim)
            sphere.GetRadiusAttr().Set(float(radius_m))
            if any(abs(float(v)) > 1.0e-12 for v in local_xyz):
                xform = UsdGeom.Xformable(prim)
                # Sphere.Define may already author a translate op in an
                # inherited template; use a suffixed op to avoid duplicate
                # xformOp names.
                xform.AddTranslateOp(opSuffix="v2").Set(
                    Gf.Vec3d(*(float(v) for v in local_xyz))
                )
            UsdPhysics.CollisionAPI.Apply(prim)
            # Contact sensors are initialized before this prestartup event,
            # so the late-authored proxy must explicitly opt into PhysX
            # contact reporting.  Without this API the sphere can generate a
            # physical impulse while the parent fingertip sensor reports zero
            # (most visible on thumb_link5), creating a false missing-thumb
            # diagnosis in the strict measured gate.
            PhysxSchema.PhysxContactReportAPI.Apply(prim)
            enabled = bool(physical_collision)
            if debug_only:
                enabled = enabled and (
                    debug_digit == "all" or
                    (bool(debug_digit) and name.startswith(f"{debug_digit}_"))
                )
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(enabled)
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr().Set(float(contact_offset_m))
            collision.CreateRestOffsetAttr().Set(float(rest_offset_m))
            UsdGeom.Imageable(prim).MakeInvisible()
            authored.append(path)
    env._v2_fingertip_proxy_paths = tuple(authored)
    env._v2_fingertip_proxy_radius_m = float(radius_m)


__all__ = ["author_fingertip_collision_proxies"]
