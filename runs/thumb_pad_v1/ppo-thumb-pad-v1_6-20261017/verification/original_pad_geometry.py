"""按冻结三角网格和已确认面编号分类指腹几何接触。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


_TRIANGLE_CHUNK_SIZE = 2048
_FLOAT_EPS = np.finfo(np.float64).eps


@dataclass(frozen=True, eq=False)
class PadSurface:
    """刚体局部坐标系中的完整网格及其已批准指腹三角面编号。

    ``vertices_body_m`` 使用米；``faces`` 保持冻结 USD 的原始面顺序。
    ``pad_triangle_ids`` 只标记批准 JSON 中的面，不代表碰撞面已校准。
    """

    vertices_body_m: np.ndarray = field(repr=False)
    faces: np.ndarray = field(repr=False)
    pad_triangle_ids: Sequence[int] | np.ndarray = field(repr=False)
    _triangles: np.ndarray = field(init=False, repr=False)
    _face_normals: np.ndarray = field(init=False, repr=False)
    _face_normal_valid: np.ndarray = field(init=False, repr=False)
    _pad_mask: np.ndarray = field(init=False, repr=False)
    _coordinate_scale_m: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            vertices = np.array(self.vertices_body_m, dtype=np.float64, copy=True)
            faces = np.array(self.faces, copy=True)
            ids_raw = np.asarray(self.pad_triangle_ids)
        except (TypeError, ValueError) as exc:
            raise ValueError("网格顶点、面和指腹面编号必须是数值序列") from exc

        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
            raise ValueError("vertices_body_m 必须是形状为 (V, 3) 的数组")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("网格顶点必须全部为有限值")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError("faces 必须是非空、形状为 (F, 3) 的三角面数组")
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError("faces 必须使用整数顶点编号")
        faces = faces.astype(np.int64, copy=False)
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("faces 中存在超出顶点数组范围的编号")
        if ids_raw.ndim != 1:
            raise ValueError("pad_triangle_ids 必须是一维面编号序列")
        if ids_raw.size:
            if not np.issubdtype(ids_raw.dtype, np.integer):
                raise ValueError("pad_triangle_ids 必须使用整数面编号")
            ids = ids_raw.astype(np.int64, copy=True)
        else:
            ids = np.empty(0, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= len(faces)):
            raise ValueError("pad_triangle_ids 中存在超出 faces 范围的编号")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("pad_triangle_ids 中不能有重复编号")

        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(cross, axis=1)
        normal_valid = np.isfinite(lengths) & (lengths > 0.0)
        face_normals = np.full((len(faces), 3), np.nan, dtype=np.float64)
        face_normals[normal_valid] = cross[normal_valid] / lengths[normal_valid, None]
        pad_mask = np.zeros(len(faces), dtype=bool)
        pad_mask[ids] = True

        scale = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        for array in (vertices, faces, ids, triangles, face_normals, normal_valid, pad_mask):
            array.setflags(write=False)
        object.__setattr__(self, "vertices_body_m", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "pad_triangle_ids", ids)
        object.__setattr__(self, "_triangles", triangles)
        object.__setattr__(self, "_face_normals", face_normals)
        object.__setattr__(self, "_face_normal_valid", normal_valid)
        object.__setattr__(self, "_pad_mask", pad_mask)
        object.__setattr__(self, "_coordinate_scale_m", scale)


def _closest_points_for_chunk(point: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐面求精确最近点；面内投影与三条闭边段共同覆盖退化三角形。"""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    best_points = np.empty_like(a)
    best_distances_sq = np.full(len(triangles), np.inf, dtype=np.float64)

    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        edge_sq = np.einsum("ij,ij->i", edge, edge)
        numer = np.einsum("ij,ij->i", point - start, edge)
        amount = np.zeros(len(triangles), dtype=np.float64)
        np.divide(numer, edge_sq, out=amount, where=edge_sq > 0.0)
        np.clip(amount, 0.0, 1.0, out=amount)
        candidate = start + amount[:, None] * edge
        delta = point - candidate
        distance_sq = np.einsum("ij,ij->i", delta, delta)
        take = distance_sq < best_distances_sq
        best_distances_sq[take] = distance_sq[take]
        best_points[take] = candidate[take]

    ab = b - a
    ac = c - a
    ap = point - a
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    d20 = np.einsum("ij,ij->i", ap, ab)
    d21 = np.einsum("ij,ij->i", ap, ac)
    denominator = d00 * d11 - d01 * d01
    face_possible = (denominator > 0.0) & np.isfinite(denominator)
    v = np.zeros(len(triangles), dtype=np.float64)
    w = np.zeros(len(triangles), dtype=np.float64)
    np.divide(d11 * d20 - d01 * d21, denominator, out=v, where=face_possible)
    np.divide(d00 * d21 - d01 * d20, denominator, out=w, where=face_possible)
    u = 1.0 - v - w
    face_inside = face_possible & np.isfinite(u) & np.isfinite(v) & np.isfinite(w)
    face_inside &= (u >= 0.0) & (v >= 0.0) & (w >= 0.0)
    projected = a + v[:, None] * ab + w[:, None] * ac
    delta = point - projected
    projected_sq = np.einsum("ij,ij->i", delta, delta)
    take = face_inside & (projected_sq < best_distances_sq)
    best_distances_sq[take] = projected_sq[take]
    best_points[take] = projected[take]
    return best_points, best_distances_sq


def classify(
    surface: PadSurface,
    points_body_m: np.ndarray,
    normals_body_outward: np.ndarray,
    *,
    position_tolerance_m: float,
    normal_tolerance_rad: float,
) -> dict[str, np.ndarray]:
    """将每个接触点与完整网格逐三角面比较并分类。

    ``normal_alignment`` 是输入单位法向与最近面的单位外法向点积，范围为
    [-1, 1]。PAD 同时要求点到完整网格距离合格、最近面属于批准集合且法向
    夹角合格。等距面给出相互冲突的标签或法向结论时返回 UNKNOWN。
    容差由调用者显式提供；本函数不表示物理接触已经校准。
    """
    try:
        position_tolerance = float(position_tolerance_m)
        normal_tolerance = float(normal_tolerance_rad)
    except (TypeError, ValueError) as exc:
        raise ValueError("位置和法向容差必须是有限数值") from exc
    if not np.isfinite(position_tolerance) or position_tolerance < 0.0:
        raise ValueError("position_tolerance_m 必须是非负有限值")
    if not np.isfinite(normal_tolerance) or not 0.0 <= normal_tolerance <= np.pi:
        raise ValueError("normal_tolerance_rad 必须位于 [0, pi]")

    points = np.asarray(points_body_m, dtype=np.float64)
    normals = np.asarray(normals_body_outward, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_body_m 必须是形状为 (N, 3) 的数组")
    if normals.shape != points.shape:
        raise ValueError("normals_body_outward 必须与 points_body_m 形状相同")

    count = len(points)
    state = np.full(count, "UNKNOWN", dtype="<U8")
    nearest_triangle_id = np.full(count, -1, dtype=np.int64)
    distance_m = np.full(count, np.nan, dtype=np.float64)
    normal_alignment = np.full(count, np.nan, dtype=np.float64)
    closest_point_body_m = np.full((count, 3), np.nan, dtype=np.float64)
    point_valid = np.all(np.isfinite(points), axis=1)
    normal_lengths = np.linalg.norm(normals, axis=1)
    normal_valid = np.all(np.isfinite(normals), axis=1) & np.isfinite(normal_lengths) & (normal_lengths > 0.0)
    unit_normals = np.zeros_like(normals)
    unit_normals[normal_valid] = normals[normal_valid] / normal_lengths[normal_valid, None]
    alignment_limit = float(np.cos(normal_tolerance))

    for point_id in np.flatnonzero(point_valid):
        point = points[point_id]
        triangle_count = len(surface.faces)
        distances_sq = np.empty(triangle_count, dtype=np.float64)
        closest_points = np.empty((triangle_count, 3), dtype=np.float64)
        for start in range(0, triangle_count, _TRIANGLE_CHUNK_SIZE):
            stop = min(start + _TRIANGLE_CHUNK_SIZE, triangle_count)
            closest, distance_sq = _closest_points_for_chunk(point, surface._triangles[start:stop])
            closest_points[start:stop] = closest
            distances_sq[start:stop] = distance_sq

        distances = np.sqrt(np.maximum(distances_sq, 0.0))
        minimum_distance = float(np.min(distances))
        nearest_id = int(np.argmin(distances))
        nearest_triangle_id[point_id] = nearest_id
        distance_m[point_id] = distances[nearest_id]
        closest_point_body_m[point_id] = closest_points[nearest_id]

        if normal_valid[point_id] and surface._face_normal_valid[nearest_id]:
            normal_alignment[point_id] = float(
                np.clip(np.dot(unit_normals[point_id], surface._face_normals[nearest_id]), -1.0, 1.0)
            )
        if not normal_valid[point_id] or distances[nearest_id] > position_tolerance:
            continue

        # 只把浮点舍入量级的等距视为同一最近距离，不扩展物理边界带。
        tie_scale = max(1.0, surface._coordinate_scale_m, float(np.max(np.abs(point))), minimum_distance)
        tie_tolerance = 64.0 * _FLOAT_EPS * tie_scale
        tied_ids = np.flatnonzero(distances <= minimum_distance + tie_tolerance)
        tied_pad = surface._pad_mask[tied_ids]
        if np.any(tied_pad) and not np.all(tied_pad):
            continue
        if not np.any(tied_pad):
            state[point_id] = "NON_PAD"
            continue

        tied_normals_valid = surface._face_normal_valid[tied_ids]
        if not np.all(tied_normals_valid):
            continue
        tied_alignment = surface._face_normals[tied_ids] @ unit_normals[point_id]
        tied_alignment = np.clip(tied_alignment, -1.0, 1.0)
        aligned = tied_alignment >= alignment_limit
        if np.all(aligned):
            state[point_id] = "PAD"
        elif not np.any(aligned):
            state[point_id] = "NON_PAD"

    return {
        "state": state,
        "nearest_triangle_id": nearest_triangle_id,
        "distance_m": distance_m,
        "normal_alignment": normal_alignment,
        "closest_point_body_m": closest_point_body_m,
    }
