"""修正 CAD 壳体粗凸分解；保留原 USD 实例/mesh-merge 与全部碰撞对。"""
from __future__ import annotations

PARAMETERS = {
    'hull_vertex_limit': 64,
    'max_convex_hulls': 64,
    'voxel_resolution': 1_000_000,
    'error_percentage': 1.0,
    'min_thickness': 0.0002,
    'shrink_wrap': True,
}


def configure_hand_collision_cooking(stage, num_envs, body_names):
    """在 PhysX cooking 前精确写27个 wrapper 的分解参数；不改外形、关节或过滤。"""
    from pxr import UsdPhysics, PhysxSchema

    methods = {
        'hull_vertex_limit': 'CreateHullVertexLimitAttr',
        'max_convex_hulls': 'CreateMaxConvexHullsAttr',
        'voxel_resolution': 'CreateVoxelResolutionAttr',
        'error_percentage': 'CreateErrorPercentageAttr',
        'min_thickness': 'CreateMinThicknessAttr',
        'shrink_wrap': 'CreateShrinkWrapAttr',
    }
    for env_id in range(int(num_envs)):
        for body in body_names:
            path = f'/World/envs/env_{env_id}/Robot/{body}/collisions'
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError('碰撞 wrapper 缺失: ' + path)
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                raise RuntimeError('碰撞未开启: ' + path)
            if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != 'convexDecomposition':
                raise RuntimeError('碰撞近似不是预期凸分解: ' + path)
            api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            for key, value in PARAMETERS.items():
                attr = getattr(api, methods[key])()
                attr.Set(value)
                actual = attr.Get()
                if isinstance(value, float):
                    if abs(actual - value) > 1e-7:
                        raise RuntimeError('凸分解属性未生效: ' + path + ':' + key)
                elif actual != value:
                    raise RuntimeError('凸分解属性未生效: ' + path + ':' + key)
    return {
        'approximation': 'convexDecomposition',
        'parameters': dict(PARAMETERS),
        'bodies': list(body_names),
        'authoring': 'original_instanceable_collision_wrappers_before_PhysX_cooking',
        'geometry_and_joint_frames_changed': False,
        'added_filtered_pairs': [],
        'source': 'research.fixed_fruit_grasp_v1.collision_cooking',
    }
