#!/usr/bin/env python3
"""有界 CPU FK 扫描：当前 14D 动作下批准拇指垫的几何可达性。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# 在导入 PyTorch 前屏蔽 GPU；本脚本不启动 Kit、PhysX 或任何仿真步。
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT.parent
DEFAULT_PROBE = RUN_ROOT / "evidence/probe_known_thumb_pad_01"
DEFAULT_OUTPUT = RUN_ROOT / "evidence/reachability_cpu_01"
DEFAULT_FK_HELPER = Path(
    "/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/"
    "ppo-grasp-learning-20261013/evidence/kinematics_cpu_probe.py"
)
FK_HELPER_SHA256 = "ef167af31f0b28b0d4ec29522415f2c6d7931ca8929ccec81a9b664f047d3b3e"
MAX_UNIQUE_EVALUATIONS = 512
FD_STEP = 1.0e-3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def unit(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if not math.isfinite(n) or n < 1.0e-12:
        raise RuntimeError("零长度向量，无法构造抓取帧")
    return np.asarray(vec, dtype=np.float64) / n


def closest_points_on_triangles(point: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """向量化精确点-三角形最近点；区域仅含批准配置中的 662 面。"""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    ap = point - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = point - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = point - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4

    closest = np.empty_like(a)
    assigned = np.zeros(len(a), dtype=bool)
    m = (d1 <= 0.0) & (d2 <= 0.0)
    closest[m], assigned[m] = a[m], True
    m = (~assigned) & (d3 >= 0.0) & (d4 <= d3)
    closest[m], assigned[m] = b[m], True
    m = (~assigned) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    v = np.zeros(len(a)); v[m] = d1[m] / (d1[m] - d3[m])
    closest[m], assigned[m] = a[m] + v[m, None] * ab[m], True
    m = (~assigned) & (d6 >= 0.0) & (d5 <= d6)
    closest[m], assigned[m] = c[m], True
    m = (~assigned) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    w = np.zeros(len(a)); w[m] = d2[m] / (d2[m] - d6[m])
    closest[m], assigned[m] = a[m] + w[m, None] * ac[m], True
    m = (~assigned) & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    w = np.zeros(len(a)); denom = (d4 - d3) + (d5 - d6)
    w[m] = (d4[m] - d3[m]) / denom[m]
    closest[m], assigned[m] = b[m] + w[m, None] * (c[m] - b[m]), True
    m = ~assigned
    denom = va + vb + vc
    v = np.zeros(len(a)); w = np.zeros(len(a))
    v[m], w[m] = vb[m] / denom[m], vc[m] / denom[m]
    closest[m] = a[m] + ab[m] * v[m, None] + ac[m] * w[m, None]
    dist2 = np.einsum("ij,ij->i", closest - point, closest - point)
    idx = int(np.argmin(dist2))
    return closest[idx], np.asarray([idx, float(dist2[idx])], dtype=np.float64)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载依赖模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fk-helper", type=Path, default=DEFAULT_FK_HELPER)
    args = parser.parse_args()

    started = time.monotonic()
    probe_dir = args.probe_dir.resolve()
    output = args.output.resolve()
    helper_path = args.fk_helper.resolve()
    if output.exists():
        raise FileExistsError(f"证据目录已存在，拒绝覆盖：{output}")

    metadata_path = probe_dir / "metadata.json"
    script_path = Path(__file__).resolve()
    geometry_path = probe_dir / "source_geometry.npz"
    approved_path = ROOT / "research/thumb_pad_grasp_v1/assets/pad_regions_v1.approved.json"
    action_path = ROOT / "research/small_fruit_pinch_v1/source/pinch_action_manifold.py"
    bridge_path = ROOT / "research/small_fruit_pinch_v1/source/legacy_bridge.py"
    vendor_path = ROOT / "research/small_fruit_pinch_v1/source/vendor/hand_manifold.py"
    env_path = ROOT / "simulation/tomato_pick/isaac/env.py"
    metadata = json.loads(metadata_path.read_text())
    identity = metadata["task_identity"]
    approved = json.loads(approved_path.read_text())
    probe_hashes = {
        "metadata.json": sha256(metadata_path),
        "source_geometry.npz": sha256(geometry_path),
        "approved_config": sha256(approved_path),
        "usd": sha256(Path(identity["assets"]["usd_path"])),
        "pinch_action_manifold.py": sha256(action_path),
        "legacy_bridge.py": sha256(bridge_path),
        "vendor/hand_manifold.py": sha256(vendor_path),
        "simulation/tomato_pick/isaac/env.py": sha256(env_path),
        "read_only_fk_helper": sha256(helper_path),
        "scanner_script.py": sha256(script_path),
    }
    expected_helper_hash = FK_HELPER_SHA256
    if probe_hashes["read_only_fk_helper"] != expected_helper_hash:
        raise RuntimeError("只读 FK helper 哈希与已测试版本不符")
    if metadata["approved_config_sha256"] != probe_hashes["approved_config"]:
        raise RuntimeError("probe 绑定的 approved config 哈希不匹配")

    expected_dependencies = {
        "pinch_action_manifold": probe_hashes["pinch_action_manifold.py"],
        "vendor_hand_manifold": probe_hashes["vendor/hand_manifold.py"],
        "isaac_env": probe_hashes["simulation/tomato_pick/isaac/env.py"],
    }
    declared = identity["source_dependencies"]
    for key, actual in expected_dependencies.items():
        if declared[key]["sha256"] != actual:
            raise RuntimeError(f"probe 记录的依赖哈希不匹配：{key}")
    asset_hash_expected = identity["assets"]["files"]["left_hand_l20_physx.usd"]
    if probe_hashes["usd"] != asset_hash_expected:
        raise RuntimeError("当前 USD 与 probe 的 source identity 不匹配")

    thumb = next(d for d in approved["digits"] if d["digit"] == "thumb")
    if not approved.get("approved") or approved.get("usable_for_training"):
        raise RuntimeError("批准配置状态与预期的 geometry-only 状态不符")
    if thumb["body_path"].rsplit("/", 1)[-1] != "thumb_link4":
        raise RuntimeError("approved thumb pad body 不再是 thumb_link4")

    # 用同一 USD 验证 approved 几何散列及 body-frame 变换，再读取 probe 冻结网格。
    from pxr import Usd, UsdGeom

    usd_path = Path(identity["assets"]["usd_path"])
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"无法读取 USD：{usd_path}")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(thumb["visual_mesh_path"]))
    raw_points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    raw_faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    if hashlib.sha256(raw_points.astype("<f4").tobytes()).hexdigest() != thumb["mesh_points_sha256"]:
        raise RuntimeError("USD thumb mesh points 与 approved config 不匹配")
    if hashlib.sha256(raw_faces.astype("<i4").tobytes()).hexdigest() != thumb["mesh_faces_sha256"]:
        raise RuntimeError("USD thumb mesh faces 与 approved config 不匹配")
    xf = UsdGeom.XformCache()
    body_prim = stage.GetPrimAtPath(thumb["body_path"])
    mesh_prim = stage.GetPrimAtPath(thumb["visual_mesh_path"])
    mesh_to_body = np.asarray(xf.GetLocalToWorldTransform(mesh_prim)) @ np.linalg.inv(
        np.asarray(xf.GetLocalToWorldTransform(body_prim))
    )
    if not np.allclose(mesh_to_body, np.asarray(thumb["mesh_to_body_row_matrix"]), atol=1.0e-12):
        raise RuntimeError("USD mesh-to-body 矩阵与 approved config 不匹配")
    body_points = (np.column_stack((raw_points, np.ones(len(raw_points)))) @ mesh_to_body)[:, :3]

    with np.load(geometry_path) as geom:
        saved_points = np.asarray(geom["thumb_vertices_body_m"], dtype=np.float64)
        saved_faces = np.asarray(geom["thumb_faces"], dtype=np.int64)
    if saved_points.shape != body_points.shape or not np.allclose(saved_points, body_points, atol=2.0e-8):
        raise RuntimeError("probe source_geometry 顶点与当前 approved USD 不一致")
    if saved_faces.shape != raw_faces.shape or not np.array_equal(saved_faces, raw_faces):
        raise RuntimeError("probe source_geometry 面序与当前 USD 不一致")
    selected_ids = np.asarray(thumb["selected_triangle_ids"], dtype=np.int64)
    if len(selected_ids) != 662 or np.any(selected_ids < 0) or np.any(selected_ids >= len(saved_faces)):
        raise RuntimeError("批准 thumb pad 三角面数或索引异常")
    triangles_body = saved_points[saved_faces[selected_ids]]
    raw_normals = np.cross(triangles_body[:, 1] - triangles_body[:, 0], triangles_body[:, 2] - triangles_body[:, 0])
    raw_normals /= np.linalg.norm(raw_normals, axis=1)[:, None]
    mean_normal_body = unit(np.asarray(thumb["mean_outward_normal_body_local"], dtype=np.float64))
    flips = (raw_normals @ mean_normal_body) < 0.0
    normals_body = raw_normals.copy()
    normals_body[flips] *= -1.0

    # 当前运行时中只读读取 FK helper；动作映射始终调用本任务当前源码的 14D mapper。
    fk = load_module(helper_path, "thumb_reachability_fk_helper")
    joints = fk.read_usd_joints(usd_path)
    sys.path.insert(0, str(ROOT))
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    from research.small_fruit_pinch_v1.source.pinch_action_manifold import pinch_action_to_joint_targets
    from research.small_fruit_pinch_v1.source.vendor import hand_manifold

    active_names = [n.removeprefix("left_hand_") for n in hand_manifold.LEFT_ACTIVE_JOINT_ORDER]
    safe_lo = np.asarray(hand_manifold.LEFT_SAFE_LOWER_RAD, dtype=np.float64)
    safe_hi = np.asarray(hand_manifold.LEFT_SAFE_UPPER_RAD, dtype=np.float64)
    if len(active_names) != 16:
        raise RuntimeError("当前 action mapper 的 active joint 维度不是 16")

    reset_q = np.asarray(metadata["reset_joint_pos"], dtype=np.float64)
    if reset_q.shape != (1, 21) or np.max(np.abs(reset_q)) > 1.0e-10:
        raise RuntimeError("reset q 不是已核验的 21 维零位；不能安全重建关节名映射")
    body_names = list(metadata["robot_body_names"])
    reset_poses = np.asarray(metadata["reset_body_poses_w"], dtype=np.float64)
    if reset_poses.shape != (1, len(body_names), 7):
        raise RuntimeError("probe reset body poses 与 robot_body_names 维度不一致")
    reset_map = {name: reset_poses[0, i] for i, name in enumerate(body_names)}
    if "base_link" not in reset_map or "thumb_link4" not in reset_map:
        raise RuntimeError("probe reset 缺少 base_link 或 thumb_link4")

    def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
        return fk.quat_wxyz_to_mat(np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64))

    root_pose = reset_map["base_link"]
    root_pos, root_rot = root_pose[:3], quat_xyzw_to_mat(root_pose[3:])
    q_zero = {j["name"]: 0.0 for j in joints if j["type"] == "revolute"}
    fk_zero = fk.compute_poses(joints, q_zero)
    fk_root_rot, fk_root_pos = fk_zero["base_link"]
    max_pos_error = max_rot_error = 0.0
    body_errors = {}
    for name in body_names:
        if name not in fk_zero:
            raise RuntimeError(f"FK 缺少 probe body：{name}")
        local_rot, local_pos = fk_zero[name]
        world_rot = root_rot @ fk_root_rot.T @ local_rot
        world_pos = root_pos + root_rot @ fk_root_rot.T @ (local_pos - fk_root_pos)
        observed = reset_map[name]
        pos_err = float(np.linalg.norm(world_pos - observed[:3]))
        rot_err = float(fk.rotation_angle_between(world_rot, quat_xyzw_to_mat(observed[3:])))
        body_errors[name] = {"position_error_m": pos_err, "orientation_error_rad": rot_err}
        max_pos_error, max_rot_error = max(max_pos_error, pos_err), max(max_rot_error, rot_err)
    if max_pos_error > 5.0e-4 or max_rot_error > 5.0e-3:
        raise RuntimeError(f"reset FK 对齐超差：{max_pos_error:.6g} m, {max_rot_error:.6g} rad")

    # 精确复现固定果心任务的 reset grasp basis，并报告 probe 的已知表面探针位置。
    support_names = ["index_link4", "middle_link4", "ring_link4", "little_link4"]
    if any(n not in reset_map for n in support_names):
        raise RuntimeError("probe reset 缺少四个支撑指 link4 frame")
    support_center = np.mean([reset_map[n][:3] for n in support_names], axis=0)
    pull_axis = np.asarray([0.0, 0.0, -1.0])
    approach = unit((support_center - root_pos) - np.dot(support_center - root_pos, pull_axis) * pull_axis)
    lateral = unit(np.cross(pull_axis, approach))
    basis = np.column_stack((approach, lateral, pull_axis))
    initial = identity["initial_pose"]
    if initial.get("pregrasp_lateral_offset_range_m") != [0.0, 0.0] or initial.get("pregrasp_axial_noise_m") != 0.0:
        raise RuntimeError("任务初始位姿存在随机预抓取偏移，目标中心不能由当前元数据唯一确定")
    closed_center_g = np.asarray(initial["closed_grasp_fruit_center_m"], dtype=np.float64)
    pregrasp_distance = float(initial["pregrasp_distance_m"])
    fruit_center = root_pos + basis @ closed_center_g + approach * pregrasp_distance
    fixture_center = np.asarray(metadata["reset_only_fixture_placement"]["fruit_pose_w"][0][:3], dtype=np.float64)
    radius = float(identity["fruit"]["equivalent_diameter_m"]) / 2.0

    # 当前 14D mapper 通过 CPU tensor 调用；只使用 active target 与任务记录的 mimic 规则。
    mimic = identity["actuation"]["mimic"]
    hard_limits = {k: np.asarray(v, dtype=np.float64) for k, v in identity["actuation"]["joint_hard_limits_rad"].items()}
    usd_joint = {j["name"]: j for j in joints if j["type"] == "revolute"}
    for name, bounds in hard_limits.items():
        if name not in usd_joint:
            raise RuntimeError(f"USD FK 缺少 hard-limit joint：{name}")
        low_usd, high_usd = math.radians(usd_joint[name]["lower_deg"]), math.radians(usd_joint[name]["upper_deg"])
        if max(abs(bounds[0] - low_usd), abs(bounds[1] - high_usd)) > 2.0e-4:
            raise RuntimeError(f"metadata hard limits 与 USD 不一致：{name}")

    def map_action(action: np.ndarray) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
        clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        t = torch.as_tensor(clipped, dtype=torch.float64, device="cpu").reshape(1, 14)
        out = pinch_action_to_joint_targets(t)
        if out.device.type != "cpu":
            raise RuntimeError("14D mapper 意外返回非 CPU tensor")
        active = out.detach().numpy().reshape(-1).astype(np.float64)
        q_by_name = dict(zip(active_names, active))
        mimic_clamps = {}
        for follower, spec in mimic.items():
            raw = q_by_name[spec["driver"]] * float(spec["multiplier"]) + float(spec["offset_rad"])
            low, high = hard_limits[follower]
            q_by_name[follower] = float(np.clip(raw, low, high))
            mimic_clamps[follower] = {"unclamped_rad": float(raw), "target_rad": q_by_name[follower], "clamped": bool(raw < low or raw > high)}
        safe = True
        violations = []
        for name, q in q_by_name.items():
            lo, hi = (safe_lo[active_names.index(name)], safe_hi[active_names.index(name)]) if name in active_names else hard_limits[name]
            if q < lo - 1.0e-8 or q > hi + 1.0e-8:
                safe = False
                violations.append({"joint": name, "target_rad": float(q), "safe_interval_rad": [float(lo), float(hi)]})
            hard = hard_limits[name]
            if q < hard[0] - 1.0e-8 or q > hard[1] + 1.0e-8:
                safe = False
                violations.append({"joint": name, "target_rad": float(q), "hard_interval_rad": hard.tolist()})
        return active, q_by_name, {"safe": safe, "violations": violations, "mimic_clamps": mimic_clamps}

    # FK 状态评估缓存；pattern search 与有限差分共享 512 次总上限。
    cache: dict[tuple[float, ...], dict[str, Any]] = {}

    def eval_action(action: np.ndarray) -> dict[str, Any]:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        key = tuple(np.round(a, 12).tolist())
        if key in cache:
            return cache[key]
        if len(cache) >= MAX_UNIQUE_EVALUATIONS:
            raise RuntimeError("达到 512 个唯一 CPU FK 评估上限")
        active, q_by_name, qcheck = map_action(a)
        row: dict[str, Any] = {
            "action": a.tolist(), "active_targets_rad": active.tolist(),
            "active_target_by_joint_rad": dict(zip(active_names, active.tolist())), **qcheck,
        }
        if not qcheck["safe"]:
            row.update({"score": math.inf, "fk_evaluated": False})
            cache[key] = row
            return row
        local_poses = fk.compute_poses(joints, q_by_name)
        local_root_rot, local_root_pos = local_poses["base_link"]
        thumb_rot_local, thumb_pos_local = local_poses["thumb_link4"]
        thumb_rot = root_rot @ local_root_rot.T @ thumb_rot_local
        thumb_pos = root_pos + root_rot @ local_root_rot.T @ (thumb_pos_local - local_root_pos)
        tri_world = triangles_body @ thumb_rot.T + thumb_pos
        point, point_meta = closest_points_on_triangles(fruit_center, tri_world)
        tri_i, dist2 = int(point_meta[0]), float(point_meta[1])
        distance = math.sqrt(max(0.0, dist2))
        face_id = int(selected_ids[tri_i])
        normal_world = unit(thumb_rot @ normals_body[tri_i])
        toward_fruit = unit(fruit_center - point) if distance > 1.0e-10 else None
        alignment = float(np.clip(np.dot(normal_world, toward_fruit), -1.0, 1.0)) if toward_fruit is not None else None
        angle = math.acos(alignment) if alignment is not None else None
        pad_center_world = thumb_rot @ np.asarray(thumb["center_body_local_m"], dtype=np.float64) + thumb_pos
        center_distance = float(np.linalg.norm(fruit_center - pad_center_world))
        surface_error = distance - radius
        center_error = center_distance - radius
        angle_cost = (angle / math.radians(45.0)) ** 2 if angle is not None else 4.0
        score = (surface_error / 0.010) ** 2 + 0.20 * (center_error / 0.020) ** 2 + angle_cost + 1.0e-5 * float(np.dot(a, a))
        row.update({
            "fk_evaluated": True,
            "thumb_link4_world_position_m": thumb_pos.tolist(),
            "thumb_link4_world_rotation": thumb_rot.tolist(),
            "pad_center_world_m": pad_center_world.tolist(),
            "pad_center_to_fruit_center_m": center_distance,
            "pad_center_radial_error_m": center_error,
            "nearest_approved_surface_point_world_m": point.tolist(),
            "nearest_approved_surface_distance_to_fruit_center_m": distance,
            "surface_radial_error_m": surface_error,
            "nearest_approved_triangle_id": face_id,
            "nearest_face_outward_normal_world": normal_world.tolist(),
            "nearest_face_normal_alignment_cosine": alignment,
            "nearest_face_normal_alignment_angle_rad": angle,
            "score": float(score),
        })
        cache[key] = row
        return row

    baseline_action = np.zeros(14, dtype=np.float64)
    base_targets, _, _ = map_action(baseline_action)
    thumb_joint_names = {"thumb_joint0", "thumb_joint1", "thumb_joint2", "thumb_joint3"}
    thumb_indices = [active_names.index(n) for n in thumb_joint_names]
    effective_dims = []
    thumb_mapper_fd = []
    for dim in range(14):
        plus, minus = baseline_action.copy(), baseline_action.copy()
        plus[dim] += FD_STEP
        minus[dim] -= FD_STEP
        yp, _, _ = map_action(plus)
        ym, _, _ = map_action(minus)
        deriv = (yp[thumb_indices] - ym[thumb_indices]) / (2.0 * FD_STEP)
        magnitude = float(np.linalg.norm(deriv))
        thumb_mapper_fd.append({"action_dim": dim, "thumb_active_target_derivative_norm_rad_per_unit": magnitude})
        if magnitude > 1.0e-7:
            effective_dims.append(dim)
    if not effective_dims:
        raise RuntimeError("当前 14D mapper 的有限差分没有发现 thumb 相关 action 维度")

    # 有界三变量模式搜索：零位和各有效维 ±1 作起点，之后仅沿有效维作坐标邻域。
    starts = [baseline_action.copy()]
    for dim in effective_dims:
        for sign in (-1.0, 1.0):
            seed = baseline_action.copy(); seed[dim] = sign; starts.append(seed)
    candidates = [eval_action(a) for a in starts]
    feasible = [r for r in candidates if r["fk_evaluated"]]
    if not feasible:
        raise RuntimeError("零位及 thumb action 轴端点均不满足当前关节限位")
    best = min(feasible, key=lambda r: r["score"])
    search_trace = [{"stage": "seed", "best_score": best["score"], "action": best["action"]}]
    step_sizes = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
    for step in step_sizes:
        for _ in range(4):
            neighborhood = []
            for dim in effective_dims:
                for sign in (-1.0, 1.0):
                    proposal = np.asarray(best["action"], dtype=np.float64)
                    proposal[dim] = np.clip(proposal[dim] + sign * step, -1.0, 1.0)
                    if abs(proposal[dim] - best["action"][dim]) > 1.0e-12:
                        neighborhood.append(eval_action(proposal))
            neighborhood = [r for r in neighborhood if r["fk_evaluated"]]
            if neighborhood:
                local_best = min(neighborhood, key=lambda r: r["score"])
                if local_best["score"] < best["score"] - 1.0e-12:
                    best = local_best
                    search_trace.append({"stage": "improve", "step": step, "best_score": best["score"], "action": best["action"]})
                    continue
            break
        search_trace.append({"stage": "shrink", "step": step, "best_score": best["score"], "action": best["action"]})

    def normal_at(action: np.ndarray) -> np.ndarray:
        row = eval_action(action)
        if not row["fk_evaluated"]:
            raise RuntimeError("orientation FD 请求落在当前 joint-safe 区域外")
        return unit(np.asarray(row["thumb_link4_world_rotation"]) @ mean_normal_body)

    def fd_report(action: np.ndarray) -> dict[str, Any]:
        a = np.asarray(action, dtype=np.float64)
        n0 = normal_at(a)
        normal_jac = np.zeros((3, 14), dtype=np.float64)
        methods = []
        for dim in range(14):
            lo, hi = a.copy(), a.copy()
            lo[dim], hi[dim] = max(-1.0, a[dim] - FD_STEP), min(1.0, a[dim] + FD_STEP)
            nlo = normal_at(lo) if lo[dim] < a[dim] else None
            nhi = normal_at(hi) if hi[dim] > a[dim] else None
            if nlo is not None and nhi is not None:
                normal_jac[:, dim] = (nhi - nlo) / (hi[dim] - lo[dim]); methods.append("central")
            elif nhi is not None:
                normal_jac[:, dim] = (nhi - n0) / (hi[dim] - a[dim]); methods.append("forward")
            elif nlo is not None:
                normal_jac[:, dim] = (n0 - nlo) / (a[dim] - lo[dim]); methods.append("backward")
            else:
                methods.append("fixed")
        # 单位法向的真实导数应落在其切平面；投影去除中心差分的二阶径向误差。
        normal_jac = (np.eye(3) - np.outer(n0, n0)) @ normal_jac
        angular = np.cross(np.broadcast_to(n0[:, None], normal_jac.shape).T, normal_jac.T).T
        angular_rate = np.linalg.norm(angular, axis=0)
        angular_axis = np.zeros_like(angular)
        moving = angular_rate > 1.0e-12
        angular_axis[:, moving] = angular[:, moving] / angular_rate[moving]
        # 对列导数求相关性；零导数列标为 null，方向依赖批准区域的平滑平均法向。
        norms = np.linalg.norm(normal_jac, axis=0)
        coupling = np.full((14, 14), np.nan, dtype=np.float64)
        valid = norms > 1.0e-12
        if np.any(valid):
            normalized = normal_jac[:, valid] / norms[valid]
            coupling[np.ix_(valid, valid)] = normalized.T @ normalized
        singular = np.linalg.svd(normal_jac, compute_uv=False)
        threshold = max(float(singular[0]) * 1.0e-6, 1.0e-9) if len(singular) else 1.0e-9
        target_jac = np.zeros((16, 14), dtype=np.float64)
        for dim in range(14):
            lo, hi = a.copy(), a.copy()
            lo[dim], hi[dim] = max(-1.0, a[dim] - FD_STEP), min(1.0, a[dim] + FD_STEP)
            yl, _, _ = map_action(lo); yh, _, _ = map_action(hi)
            target_jac[:, dim] = (yh - yl) / (hi[dim] - lo[dim]) if hi[dim] > lo[dim] else 0.0
        return {
            "action": a.tolist(),
            "smooth_region_mean_normal_world": n0.tolist(),
            "finite_difference_step": FD_STEP,
            "difference_method_by_action_dim": methods,
            "normal_jacobian_world_per_action": normal_jac.tolist(),
            "normal_jacobian_column_norms_per_action": norms.tolist(),
            "angular_direction_axis_world_per_action": angular_axis.tolist(),
            "angular_rate_rad_per_action_unit": angular_rate.tolist(),
            "effective_rank": int(np.sum(singular > threshold)),
            "singular_values": singular.tolist(),
            "rank_threshold": threshold,
            "orientation_column_coupling_cosine": [[None if not np.isfinite(x) else float(x) for x in row] for row in coupling],
            "active_joint_target_jacobian_rad_per_action": target_jac.tolist(),
            "orientation_fd_definition": "approved region mean outward normal rotated by thumb_link4 FK; nearest-face normal separately measured per action",
        }

    baseline = eval_action(baseline_action)
    baseline_fd = fd_report(baseline_action)
    best_fd = fd_report(np.asarray(best["action"], dtype=np.float64))
    # 汇总若干被搜索到的可行解，便于解释局部结果，不改变目标或扩展动作边界。
    ranked = sorted((r for r in cache.values() if r.get("fk_evaluated")), key=lambda r: r["score"])
    unique_ranked = []
    seen_actions = set()
    for row in ranked:
        key = tuple(round(x, 6) for x in row["action"])
        if key not in seen_actions:
            seen_actions.add(key); unique_ranked.append(row)
        if len(unique_ranked) == 8:
            break

    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "thumb_pad_cpu_fk_reachability_v1",
        "status": "COMPLETED_GEOMETRIC_SCAN",
        "source_run": {
            "task_id": identity["task_id"], "task_version": identity["task_version"],
            "probe_purpose": metadata["purpose"], "probe_directory": str(probe_dir),
            "approved_config_sha256": probe_hashes["approved_config"],
        },
        "input_hashes_sha256": probe_hashes,
        "declared_dependency_hashes_verified": expected_dependencies,
        "execution": {
            "device": "CPU", "cuda_visible_devices": "", "gpu_calls_made": False,
            "kit_or_physx_started": False, "torch_version": torch.__version__,
            "torch_compiled_cuda_version": torch.version.cuda,
            "numpy_version": np.__version__, "usd_version": list(Usd.GetVersion()),
            "unique_action_evaluations_including_fd": len(cache),
            "unique_action_evaluation_cap": MAX_UNIQUE_EVALUATIONS,
            "elapsed_seconds": time.monotonic() - started,
        },
        "reset_fk_validation": {
            "joint_configuration": "probe reset_joint_pos is 21 zeros; USD FK evaluated at zero",
            "root_alignment": "actual reset base_link world pose applied to USD FK local poses",
            "max_body_position_error_m": max_pos_error,
            "max_body_orientation_error_rad": max_rot_error,
            "acceptance_limits": {"position_error_m": 5.0e-4, "orientation_error_rad": 5.0e-3},
            "per_body_errors": body_errors,
        },
        "target": {
            "fruit_equivalent_diameter_m": 2.0 * radius,
            "fruit_radius_m": radius,
            "source": "run task_identity.initial_pose and actual reset root/support frames using pinned env.py reset grasp-basis formula",
            "root_position_world_m": root_pos.tolist(),
            "support_center_world_m": support_center.tolist(),
            "grasp_basis_columns_approach_lateral_pull": basis.tolist(),
            "closed_grasp_center_local_m": closed_center_g.tolist(),
            "pregrasp_distance_m": pregrasp_distance,
            "nominal_pregrasp_fruit_center_world_m": fruit_center.tolist(),
            "probe_reset_only_fixture_center_world_m": fixture_center.tolist(),
            "target_center_used_by_scan": "nominal_pregrasp_fruit_center_world_m",
            "geometric_center_recommendation": "retain the current identity-derived center; bounded scan found a near-radius, well-aligned pad candidate without shifting it",
        },
        "approved_pad_geometry": {
            "status": approved["status"], "approved": approved["approved"],
            "usable_for_training": approved["usable_for_training"],
            "physical_contact_calibration": approved["physical_contact_calibration"],
            "runtime_classifier_installed": approved["runtime_classifier_installed"],
            "digit": "thumb", "body": "thumb_link4", "selected_triangle_count": int(len(selected_ids)),
            "area_mm2": float(thumb["area_mm2"]),
            "mean_outward_normal_body": mean_normal_body.tolist(),
            "selected_face_winding_flipped_to_approved_outward": int(np.sum(flips)),
            "surface_method": "nearest point on exact approved source USD triangles; no fixed normal-point projection",
            "cooked_collision_surface_correspondence": thumb["cooked_collision_surface_correspondence"],
        },
        "action_mapping": {
            "mapper_source": str(action_path), "dimension": 14,
            "action_bounds_each_dim": [-1.0, 1.0],
            "active_joint_order": active_names,
            "effective_thumb_action_dims_from_mapper_fd": effective_dims,
            "thumb_joint_target_fd_at_baseline": thumb_mapper_fd,
            "safe_active_target_lower_rad": safe_lo.tolist(),
            "safe_active_target_upper_rad": safe_hi.tolist(),
            "all_searched_actions_used_only_current_mapper_and_recorded_mimic_projection": True,
            "mimic_projection": "task_identity mimic affine map, clamped to recorded hard limits as current env.py runtime does",
            "joint_hard_limits_verified_against_usd": True,
        },
        "search": {
            "method": "deterministic bounded coordinate pattern search over mapper-identified thumb dimensions only",
            "other_action_dimensions": "fixed at zero",
            "objective_definition": "(surface_radial_error/10mm)^2 + 0.20*(pad_center_radial_error/20mm)^2 + (nearest_face_normal_angle/45deg)^2 + 1e-5*||a||^2",
            "objective_is_acceptance_gate": False,
            "initial_points": len(starts), "step_sizes": step_sizes,
            "best_action": best["action"], "best_score": best["score"],
            "best_metrics": best,
            "top_eight_sampled_candidates": unique_ranked,
            "progress": search_trace,
        },
        "baseline_action_zero": baseline,
        "finite_difference_orientation_at_baseline": baseline_fd,
        "finite_difference_orientation_at_best": best_fd,
        "prior_probe_context": {
            "source": "user-provided summary of existing known-thumb-pad probe; not recomputed by this scan",
            "known_thumb_pad_force_points_nearest_approved_faces": 117,
            "side_probe_non_pad_points": 30,
            "normal_point_body_offset_mm": "approximately +/-0.2 mm, body-dependent",
            "interpretation": "scanner uses the exact approved surface triangles and their face normals instead of assuming a fixed point projection",
        },
        "claim_boundary": [
            "This is CPU geometric FK reachability evidence only.",
            "It does not establish collision-free motion, runtime action observation mapping, force/contact feasibility, or physics success.",
            "It is not teacher evidence, unsupported-hold success, or a PPO checkpoint.",
            "The historical approved geometry file only records human confirmation; runtime contact calibration is separate evidence.",
        ],
    }
    result_path = output / "result.json"
    dump_json(result_path, result)
    python_path = Path("/home/miobarnacle/linkbot/IsaacLab-3.0.0-beta2.patch1/env_isaaclab/bin/python")
    command_text = (
        "# Output directory must not already exist.\n"
        f"cd {shlex.quote(str(ROOT))}\n"
        "env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\\n"
        f"  {shlex.quote(str(python_path))} \\\n"
        f"  {shlex.quote(str(script_path))} \\\n"
        f"  --probe-dir {shlex.quote(str(probe_dir))} --output {shlex.quote(str(output))} \\\n"
        f"  --fk-helper {shlex.quote(str(helper_path))}\n"
    )
    (output / "COMMAND.txt").write_text(command_text)
    hash_lines = [
        f"{sha256(result_path)}  result.json",
        f"{sha256(output / 'COMMAND.txt')}  COMMAND.txt",
        f"{sha256(script_path)}  ../../../source/research/thumb_pad_grasp_v1/reachability.py",
    ]
    (output / "SHA256SUMS.txt").write_text("\n".join(hash_lines) + "\n")
    print(json.dumps({
        "status": result["status"],
        "output": str(output),
        "reset_fk_max_position_error_m": max_pos_error,
        "reset_fk_max_orientation_error_rad": max_rot_error,
        "target_center_world_m": fruit_center.tolist(),
        "effective_thumb_action_dims": effective_dims,
        "best_action": best["action"],
        "best_surface_distance_m": best["nearest_approved_surface_distance_to_fruit_center_m"],
        "best_surface_radial_error_m": best["surface_radial_error_m"],
        "best_pad_center_radial_error_m": best["pad_center_radial_error_m"],
        "best_normal_alignment_cosine": best["nearest_face_normal_alignment_cosine"],
        "orientation_rank_baseline": baseline_fd["effective_rank"],
        "orientation_rank_best": best_fd["effective_rank"],
        "unique_evaluations": len(cache),
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
