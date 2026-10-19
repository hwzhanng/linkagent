#!/usr/bin/env python3
"""Probe-backed CPU FK/PAD geometry scan for firm-grasp F1.

This reads the completed F1 probe records and approved PAD triangles only. It
does not start Isaac, PhysX, an AppLauncher, or hardware, and it never queries
or initializes CUDA.
"""

from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT.parent
DEFAULT_PROBE01 = RUN_ROOT / "evidence/F1_middle_probe01/result.json"
DEFAULT_PROBE02 = RUN_ROOT / "evidence/F1_middle_probe02/result.json"
DEFAULT_OUTPUT = RUN_ROOT / "evidence/F1_FK_SCAN.json"
DEFAULT_HELPER = Path(
    "/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/"
    "ppo-grasp-learning-20261013/evidence/kinematics_cpu_probe.py"
)
DEFAULT_ACTION_MAPPING = (
    RUN_ROOT / "evidence/F1_middle_probe02-inputs/research/thumb_pad_grasp_v1/action_mapping.py"
)
DEFAULT_GEOMETRY = ROOT / "research/thumb_pad_grasp_v1/assets/source_geometry_v1.npz"
DEFAULT_APPROVED = ROOT / "research/thumb_pad_grasp_v1/assets/pad_regions_v1.approved.json"
DEFAULT_RUNTIME_CONTACT = ROOT / "research/thumb_pad_grasp_v1/assets/runtime_contact_v2.json"
DEFAULT_REACHABILITY = ROOT / "research/thumb_pad_grasp_v1/reachability.py"
FRUIT_RADIUS_M = 0.0335
SCAN_POINTS = 21
ACTION_INDEX = {"middle_residual": 7, "middle_distal": 11}
PAD_BODIES = {"thumb": "thumb_link4", "index": "index_link3", "middle": "middle_link3"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("遇到零长度向量，无法计算法向对齐")
    return np.asarray(vector, dtype=np.float64) / norm


def xyzw_to_matrix(fk: Any, quaternion_xyzw: list[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    return fk.quat_wxyz_to_mat(np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64))


def rotation_error_rad(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))
    return abs(math.acos(cosine))


def world_poses(fk: Any, fk_poses: dict[str, tuple[np.ndarray, np.ndarray]], base_pose_xyzw: list[float]):
    base_world_position = np.asarray(base_pose_xyzw[:3], dtype=np.float64)
    base_world_rotation = xyzw_to_matrix(fk, base_pose_xyzw[3:])
    root_rotation, root_position = fk_poses["base_link"]
    root_to_world_rotation = base_world_rotation @ root_rotation.T
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (local_rotation, local_position) in fk_poses.items():
        result[name] = (
            root_to_world_rotation @ local_rotation,
            base_world_position + root_to_world_rotation @ (local_position - root_position),
        )
    return result


def joint_map(names: list[str], values: list[float] | np.ndarray) -> dict[str, float]:
    if len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError("关节名与关节值维度不一致或关节名重复")
    return {name: float(value) for name, value in zip(names, values)}


def load_approved_pad_triangles(approved_path: Path, geometry_path: Path):
    approved = json.loads(approved_path.read_text(encoding="utf-8"))
    if not approved.get("approved") or approved.get("usable_for_training"):
        raise RuntimeError("PAD 区域批准状态异常；预期为几何批准、不可直接作为训练标签")
    rows = {row["digit"]: row for row in approved["digits"]}
    pads = {}
    with np.load(geometry_path, allow_pickle=False) as geometry:
        for digit, body_name in PAD_BODIES.items():
            row = rows[digit]
            approved_body = str(row["body_path"]).rsplit("/", 1)[-1]
            if approved_body != body_name:
                raise RuntimeError(f"{digit}批准 body 不匹配：{approved_body} != {body_name}")
            vertices = np.asarray(geometry[f"{digit}_vertices_body_m"], dtype=np.float64)
            faces = np.asarray(geometry[f"{digit}_faces"], dtype=np.int64)
            selected = np.asarray(row["selected_triangle_ids"], dtype=np.int64)
            if faces.ndim != 2 or faces.shape[1] != 3 or len(selected) == 0:
                raise RuntimeError(f"{digit}批准三角面索引维度异常")
            if np.any(selected < 0) or np.any(selected >= len(faces)):
                raise RuntimeError(f"{digit}批准三角面索引越界")
            triangles = vertices[faces[selected]]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            if np.any(lengths <= 1.0e-14):
                raise RuntimeError(f"{digit}批准区域包含退化三角面")
            normals /= lengths[:, None]
            mean_outward = unit(np.asarray(row["mean_outward_normal_body_local"], dtype=np.float64))
            normals[np.einsum("ij,j->i", normals, mean_outward) < 0.0] *= -1.0
            pads[digit] = {
                "body_name": body_name,
                "triangles_body_m": triangles,
                "normals_body": normals,
                "selected_triangle_ids": selected,
                "vertex_count": int(len(vertices)),
                "source_face_count": int(len(faces)),
                "approved_triangle_count": int(len(selected)),
                "approved_mean_outward_normal_body": mean_outward,
            }
    return approved, pads


def pad_to_fruit_metric(
    closest_points_on_triangles: Any,
    pad_geometry: dict[str, Any],
    body_pose: tuple[np.ndarray, np.ndarray],
    fruit_center: np.ndarray,
) -> dict[str, Any]:
    body_rotation, body_position = body_pose
    triangles_world = pad_geometry["triangles_body_m"] @ body_rotation.T + body_position
    closest, index_and_squared_distance = closest_points_on_triangles(fruit_center, triangles_world)
    local_index = int(index_and_squared_distance[0])
    distance = float(np.linalg.norm(fruit_center - closest))
    normal_world = pad_geometry["normals_body"][local_index] @ body_rotation.T
    normal_to_fruit = None if distance <= 1.0e-12 else (fruit_center - closest) / distance
    alignment = None if normal_to_fruit is None else float(np.dot(normal_world, normal_to_fruit))
    return {
        "distance_pad_to_fruit_center_m": distance,
        "surface_gap_m_distance_minus_0p0335": distance - FRUIT_RADIUS_M,
        "normal_facing_fruit_dot": alignment,
        "nearest_point_world_m": closest.tolist(),
        "outward_normal_world": normal_world.tolist(),
        "approved_source_triangle_id": int(pad_geometry["selected_triangle_ids"][local_index]),
    }


def action_targets(action: np.ndarray, mapper: Any, torch: Any) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    tensor = torch.as_tensor(values, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        output = mapper.independent_thumb_targets(tensor)
    if output.device.type != "cpu" or output.shape != (len(values), 16):
        raise RuntimeError(f"independent_thumb_targets 输出异常：shape={tuple(output.shape)}, device={output.device}")
    return output.detach().numpy().astype(np.float64)


def build_joint_targets(active16: np.ndarray, active_names: tuple[str, ...], joint_names: list[str], mimic: dict[str, Any]):
    targets = {name: 0.0 for name in joint_names}
    for name, value in zip(active_names, active16):
        targets[name] = float(value)
    for follower, spec in mimic.items():
        driver = spec["driver"]
        if follower not in targets or driver not in targets:
            raise RuntimeError(f"mimic 关节缺失：{driver} -> {follower}")
        targets[follower] = float(spec["multiplier"]) * targets[driver] + float(spec["offset_rad"])
    return targets


def q_array_from_target(targets: dict[str, float], joint_names: list[str]) -> np.ndarray:
    return np.asarray([targets[name] for name in joint_names], dtype=np.float64)


def run_fk(fk: Any, joints: list[dict[str, Any]], joint_names: list[str], q: np.ndarray, base_pose: list[float]):
    poses = fk.compute_poses(joints, joint_map(joint_names, q))
    if "base_link" not in poses:
        raise RuntimeError("FK 结果缺少 base_link")
    return world_poses(fk, poses, base_pose)


def validate_probe_input(probe_path: Path, data: dict[str, Any], manifest_path: Path, required_keys: list[str]):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {}
    for key, path in required_keys:
        expected = manifest.get(key)
        if expected is None:
            raise RuntimeError(f"{manifest_path} 缺少输入哈希：{key}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"冻结输入哈希不匹配：{path}，期望 {expected}，实际 {actual}")
        checks[key] = actual
    if data.get("status") != "REAL_PHYSICS_FEASIBILITY_PROBE_COMPLETED":
        raise RuntimeError(f"{probe_path} 状态不是已完成的 F1 物理探测")
    return manifest, checks


def read_substeps(path: Path):
    frames = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                row = json.loads(line)
                row["_line_number"] = line_number
                frames.append(row)
    if not frames:
        raise RuntimeError(f"物理子步记录为空：{path}")
    return frames


def summarize_attempts(
    probe_name: str,
    result: dict[str, Any],
    substeps: list[dict[str, Any]],
    active_names: tuple[str, ...],
    mimic: dict[str, Any],
    mapper: Any,
    torch: Any,
    joint_names: list[str],
    fk: Any,
    joints: list[dict[str, Any]],
    pads: dict[str, Any],
    closest_points_on_triangles: Any,
):
    initial = result["initial"]
    body_names = initial["body_names"]
    base_index = body_names.index("base_link")
    declared = np.asarray(result["declared_actions"], dtype=np.float32)
    if declared.shape != (4, 14):
        raise RuntimeError(f"{probe_name} declared_actions 维度异常：{declared.shape}")
    active = action_targets(declared, mapper, torch)
    attempt_rows = []
    per_env_frames: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(declared))}
    max_pad_force = [0.0] * 5
    for frame in substeps:
        truth = frame.get("truth", {})
        for env_id, values in enumerate(truth.get("pad_force_n", [])):
            for digit_index, value in enumerate(values):
                max_pad_force[digit_index] = max(max_pad_force[digit_index], float(value))
        q_matrix = frame.get("joint_pos")
        fruits = frame.get("fruit_pose_w")
        if q_matrix is None or fruits is None or len(q_matrix) != len(declared) or len(fruits) != len(declared):
            raise RuntimeError(f"{probe_name} 子步缺少可重建 FK 的 q 或 fruit_pose_w")
        for env_id in range(len(declared)):
            world = run_fk(
                fk, joints, joint_names, np.asarray(q_matrix[env_id], dtype=np.float64),
                initial["body_pose_w"][env_id][base_index],
            )
            metric = pad_to_fruit_metric(
                closest_points_on_triangles, pads["middle"], world[PAD_BODIES["middle"]],
                np.asarray(fruits[env_id][:3], dtype=np.float64),
            )
            per_env_frames[env_id].append({"step": int(frame["sim_step"]), **metric})

    final_frame = substeps[-1]
    final_targets = final_frame.get("joint_targets", [])
    final_executed_actions = final_frame.get("executed_actions", [])
    for env_id, action in enumerate(declared):
        base_pose = initial["body_pose_w"][env_id][base_index]
        fruit_center_initial = np.asarray(initial["fruit_pose_w"][env_id][:3], dtype=np.float64)
        target_map = build_joint_targets(active[env_id], active_names, joint_names, mimic)
        target_world = run_fk(fk, joints, joint_names, q_array_from_target(target_map, joint_names), base_pose)
        target_pads = {
            digit: pad_to_fruit_metric(
                closest_points_on_triangles, pads[digit], target_world[body], fruit_center_initial
            )
            for digit, body in PAD_BODIES.items()
        }
        actual_frames = per_env_frames[env_id]
        nearest_surface = min(actual_frames, key=lambda row: abs(row["surface_gap_m_distance_minus_0p0335"]))
        smallest_signed_gap = min(actual_frames, key=lambda row: row["surface_gap_m_distance_minus_0p0335"])
        last_actual = actual_frames[-1]
        recorded_target_error = None
        if len(final_targets) > env_id:
            observed_targets = np.asarray(final_targets[env_id], dtype=np.float64)
            expected_targets = q_array_from_target(target_map, joint_names)
            if observed_targets.shape == expected_targets.shape:
                recorded_target_error = float(np.max(np.abs(observed_targets - expected_targets)))
        attempt_rows.append({
            "env_id": env_id,
            "declared_action_14d": action.astype(float).tolist(),
            "middle_residual": float(action[ACTION_INDEX["middle_residual"]]),
            "middle_distal": float(action[ACTION_INDEX["middle_distal"]]),
            "mapped_active16_rad": {name: float(value) for name, value in zip(active_names, active[env_id])},
            "mimic_follower_targets_rad": {name: float(target_map[name]) for name in mimic},
            "initial_fruit_target_geometry": target_pads,
            "last_recorded_executed_action_14d": (
                final_executed_actions[env_id] if len(final_executed_actions) > env_id else None
            ),
            "last_recorded_target_max_abs_error_rad_vs_mapper_and_mimic": recorded_target_error,
            "measured_physics_last_substep_middle_pad": last_actual,
            "measured_physics_closest_surface_approach": nearest_surface,
            "measured_physics_min_signed_gap": smallest_signed_gap,
            "max_recorded_middle_pad_force_n": max(
                float(frame.get("truth", {}).get("pad_force_n", [[0.0] * 5] * 4)[env_id][2])
                for frame in substeps
            ),
        })
    return {
        "probe_result_sha256": sha256(Path(probe_name)),
        "substeps_sha256": None,
        "sim_substep_count": len(substeps),
        "action_count": len(attempt_rows),
        "max_recorded_pad_force_n_by_digit_order_thumb_index_middle_ring_little": max_pad_force,
        "middle_pad_contact_observed": max_pad_force[2] > 0.0,
        "attempts": attempt_rows,
    }


def candidate_order(row: dict[str, Any]):
    alignment = row["normal_facing_fruit_dot"]
    return (
        abs(float(row["surface_gap_m_distance_minus_0p0335"])),
        -(float(alignment) if alignment is not None else -2.0),
        abs(float(row["middle_residual"])),
        abs(float(row["middle_distal"])),
    )


def build_scan(
    base_action: np.ndarray,
    fruit_center: np.ndarray,
    base_pose: list[float],
    active_names: tuple[str, ...],
    mimic: dict[str, Any],
    mapper: Any,
    torch: Any,
    joint_names: list[str],
    fk: Any,
    joints: list[dict[str, Any]],
    pads: dict[str, Any],
    closest_points_on_triangles: Any,
):
    values = np.linspace(-1.0, 1.0, SCAN_POINTS, dtype=np.float64)
    action_rows = []
    for residual in values:
        for distal in values:
            action = np.asarray(base_action, dtype=np.float32).copy()
            action[ACTION_INDEX["middle_residual"]] = residual
            action[ACTION_INDEX["middle_distal"]] = distal
            action_rows.append(action)
    action_batch = np.stack(action_rows)
    active_batch = action_targets(action_batch, mapper, torch)
    grid_rows = []
    detailed_rows = []
    active_reference: dict[str, float] | None = None
    frozen_thumb_index_max_delta = 0.0
    for action, active16 in zip(action_batch, active_batch):
        targets = build_joint_targets(active16, active_names, joint_names, mimic)
        if active_reference is None:
            active_reference = targets
        for name in active_names:
            if name.startswith("thumb_") or name.startswith("index_"):
                frozen_thumb_index_max_delta = max(
                    frozen_thumb_index_max_delta, abs(targets[name] - active_reference[name])
                )
        poses = run_fk(fk, joints, joint_names, q_array_from_target(targets, joint_names), base_pose)
        metric = pad_to_fruit_metric(
            closest_points_on_triangles, pads["middle"], poses[PAD_BODIES["middle"]], fruit_center
        )
        row = {
            "middle_residual": float(action[ACTION_INDEX["middle_residual"]]),
            "middle_distal": float(action[ACTION_INDEX["middle_distal"]]),
            "action_14d": action.astype(float).tolist(),
            **metric,
        }
        grid_rows.append(row)
        detailed_rows.append({"action": action, "active16": active16, "joint_targets": targets, "row": row})
    ranked = sorted(grid_rows, key=candidate_order)
    details_by_pair = {
        (round(item["row"]["middle_residual"], 8), round(item["row"]["middle_distal"], 8)): item
        for item in detailed_rows
    }
    top10 = []
    for rank, row in enumerate(ranked[:10], start=1):
        detail = details_by_pair[(round(row["middle_residual"], 8), round(row["middle_distal"], 8))]
        top10.append({
            "rank": rank,
            **row,
            "joint_targets_rad_in_probe_order": {
                name: float(detail["joint_targets"][name]) for name in joint_names
            },
        })
    near_surface = [
        row for row in grid_rows
        if abs(row["surface_gap_m_distance_minus_0p0335"]) <= 0.0005
        and row["normal_facing_fruit_dot"] is not None
        and row["normal_facing_fruit_dot"] > 0.0
    ]
    return {
        "scan_values_rad": values.tolist(),
        "grid_size": len(grid_rows),
        "action_dimensions_varied_zero_based": ACTION_INDEX,
        "all_other_action_dimensions_frozen_from_probe02_env0": base_action.astype(float).tolist(),
        "active16_and_mimic_used_for_each_candidate": True,
        "max_change_in_frozen_thumb_index_active_targets_rad": frozen_thumb_index_max_delta,
        "min_abs_surface_gap_m": float(min(abs(r["surface_gap_m_distance_minus_0p0335"]) for r in grid_rows)),
        "min_signed_surface_gap_m": float(min(r["surface_gap_m_distance_minus_0p0335"] for r in grid_rows)),
        "max_normal_facing_fruit_dot": float(max(
            r["normal_facing_fruit_dot"] for r in grid_rows if r["normal_facing_fruit_dot"] is not None
        )),
        "near_surface_aligned_candidate_count_within_0p5mm": len(near_surface),
        "top10_ranked_by_abs_gap_then_alignment": top10,
        "all_441_gap_and_normal_rows": [
            {
                "middle_residual": r["middle_residual"],
                "middle_distal": r["middle_distal"],
                "surface_gap_m": r["surface_gap_m_distance_minus_0p0335"],
                "normal_facing_fruit_dot": r["normal_facing_fruit_dot"],
                "approved_source_triangle_id": r["approved_source_triangle_id"],
            }
            for r in grid_rows
        ],
        "near_surface_candidates": near_surface,
        "_ranked": ranked,
        "_details": detailed_rows,
    }


def diagnostic_fruit_shifts(
    scan: dict[str, Any],
    fruit_center: np.ndarray,
    base_pose: list[float],
    joints: list[dict[str, Any]],
    fk: Any,
    joint_names: list[str],
    pads: dict[str, Any],
    closest_points_on_triangles: Any,
):
    if scan["near_surface_aligned_candidate_count_within_0p5mm"] > 0:
        return None
    anchor = scan["_ranked"][0]
    direction = np.asarray(anchor["nearest_point_world_m"], dtype=np.float64) - fruit_center
    if np.linalg.norm(direction) <= 1.0e-12:
        direction = -np.asarray(anchor["outward_normal_world"], dtype=np.float64)
    direction = unit(direction)
    detail = next(
        item for item in scan["_details"]
        if abs(item["row"]["middle_residual"] - anchor["middle_residual"]) < 1e-8
        and abs(item["row"]["middle_distal"] - anchor["middle_distal"]) < 1e-8
    )
    targets = detail["joint_targets"]
    poses = run_fk(fk, joints, joint_names, q_array_from_target(targets, joint_names), base_pose)
    triangles_world = pads["middle"]["triangles_body_m"] @ poses[PAD_BODIES["middle"]][0].T + poses[PAD_BODIES["middle"]][1]
    rows = []
    for millimeters in (5, 10, 15):
        shifted_center = fruit_center + direction * (millimeters / 1000.0)
        metric = pad_to_fruit_metric(
            closest_points_on_triangles, pads["middle"], poses[PAD_BODIES["middle"]], shifted_center
        )
        rows.append({
            "translation_toward_middle_pad_mm": millimeters,
            "fruit_center_world_m": shifted_center.tolist(),
            "best_scan_action_middle_residual_distal": [anchor["middle_residual"], anchor["middle_distal"]],
            **metric,
        })
    return {
        "scope": "diagnostic_only_no_ROI_or_runtime_config_change",
        "direction_world_unit": direction.tolist(),
        "alignment_with_world_positive_Y": float(np.dot(direction, np.asarray([0.0, 1.0, 0.0]))),
        "fixed_joint_target": {
            "middle_residual": anchor["middle_residual"],
            "middle_distal": anchor["middle_distal"],
        },
        "same_best_scan_pose_metrics": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe01", type=Path, default=DEFAULT_PROBE01)
    parser.add_argument("--probe02", type=Path, default=DEFAULT_PROBE02)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fk-helper", type=Path, default=DEFAULT_HELPER)
    parser.add_argument("--action-mapping", type=Path, default=DEFAULT_ACTION_MAPPING)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--approved", type=Path, default=DEFAULT_APPROVED)
    parser.add_argument("--runtime-contact", type=Path, default=DEFAULT_RUNTIME_CONTACT)
    parser.add_argument("--reachability-helper", type=Path, default=DEFAULT_REACHABILITY)
    args = parser.parse_args()
    start = time.monotonic()

    probe01_path, probe02_path, output_path = args.probe01.resolve(), args.probe02.resolve(), args.output.resolve()
    helper_path = args.fk_helper.resolve()
    mapper_path = args.action_mapping.resolve()
    geometry_path, approved_path = args.geometry.resolve(), args.approved.resolve()
    runtime_contact_path, reachability_path = args.runtime_contact.resolve(), args.reachability_helper.resolve()
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有证据：{output_path}")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"证据目录不存在：{output_path.parent}")

    probe01 = json.loads(probe01_path.read_text(encoding="utf-8"))
    probe02 = json.loads(probe02_path.read_text(encoding="utf-8"))
    result02_identity = probe02["task_identity"]
    result01_identity = probe01["task_identity"]
    usd_path = Path(result02_identity["assets"]["usd_path"]).resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"探测身份绑定的 USD 不存在：{usd_path}")

    manifest_rows = {}
    required_manifest_keys = [
        "research/thumb_pad_grasp_v1/action_mapping.py",
        "research/thumb_pad_grasp_v1/assets/pad_regions_v1.approved.json",
        "research/thumb_pad_grasp_v1/assets/source_geometry_v1.npz",
    ]
    manifest_checks = {}
    for probe_path, data in ((probe01_path, probe01), (probe02_path, probe02)):
        manifest_path = probe_path.parent.parent / f"{probe_path.parent.name}-inputs/MANIFEST.json"
        required = []
        for key in required_manifest_keys:
            if key == "research/thumb_pad_grasp_v1/action_mapping.py":
                required.append((key, mapper_path))
            elif key == "research/thumb_pad_grasp_v1/assets/pad_regions_v1.approved.json":
                required.append((key, approved_path))
            elif key == "research/thumb_pad_grasp_v1/assets/source_geometry_v1.npz":
                required.append((key, geometry_path))
        manifest, checked = validate_probe_input(probe_path, data, manifest_path, required)
        manifest_rows[probe_path.parent.name] = manifest
        manifest_checks[probe_path.parent.name] = checked
    for name, identity in (("probe01", result01_identity), ("probe02", result02_identity)):
        if identity["action_mapping"]["source_sha256"] != sha256(mapper_path):
            raise RuntimeError(f"{name} 记录的 action_mapping 哈希不匹配")
        if sha256(usd_path) != identity["assets"]["files"]["left_hand_l20_physx.usd"]:
            raise RuntimeError(f"{name} 记录的 USD 哈希不匹配")
    for dep in ("pinch_action_manifold", "vendor_hand_manifold"):
        rel = result02_identity["source_dependencies"][dep]["source_path"]
        dep_path = Path(rel)
        if sha256(dep_path) != result02_identity["source_dependencies"][dep]["sha256"]:
            raise RuntimeError(f"task_identity 依赖哈希不匹配：{dep_path}")

    helper = load_module(helper_path, "f1_fk_readonly_helper")
    # The helper contract for this task is limited to these three tested readers/FK routines.
    for required_function in ("read_usd_joints", "compute_poses", "quat_wxyz_to_mat"):
        if not callable(getattr(helper, required_function, None)):
            raise RuntimeError(f"FK helper 缺少允许复用的函数：{required_function}")
    joints = helper.read_usd_joints(usd_path)
    sys.path.insert(0, str(ROOT))
    mapper_module = load_module(mapper_path, "f1_frozen_action_mapping")
    import torch
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    legacy_hand = mapper_module.load_legacy_hand_manifold()
    active_names = tuple(name.removeprefix("left_hand_") for name in legacy_hand.LEFT_ACTIVE_JOINT_ORDER)
    if len(active_names) != 16 or len(set(active_names)) != 16:
        raise RuntimeError("active16 关节名不满足 16 个唯一关节")
    action_order = list(mapper_module.ACTION_ORDER)
    if action_order != result02_identity["action_mapping"]["action_order"]:
        raise RuntimeError("冻结映射 ACTION_ORDER 与 probe task_identity 不一致")
    if action_order[ACTION_INDEX["middle_residual"]] != "middle_residual" or action_order[ACTION_INDEX["middle_distal"]] != "middle_distal":
        raise RuntimeError("中指 residual/distal 动作索引与 probe task_identity 不一致")

    joint_names = list(probe01["initial"]["joint_names"])
    if joint_names != list(probe02["initial"]["joint_names"]):
        raise RuntimeError("两组 probe 的关节名称/顺序不一致")
    if set(active_names) | set(result02_identity["actuation"]["mimic"]) != set(joint_names):
        raise RuntimeError("active16 与 mimic followers 未覆盖 probe 的 21 个关节")
    mimic = result02_identity["actuation"]["mimic"]
    if mimic != result01_identity["actuation"]["mimic"]:
        raise RuntimeError("probe01/probe02 的 mimic task_identity 不一致")

    approved, pads_loaded = load_approved_pad_triangles(approved_path, geometry_path)
    pads = {
        key: {
            **value,
            "triangles_body_m": value["triangles_body_m"],
            "normals_body": value["normals_body"],
        }
        for key, value in pads_loaded.items()
    }
    runtime_contact = json.loads(runtime_contact_path.read_text(encoding="utf-8"))
    boundary_uncertainty_m = float(runtime_contact["boundary_uncertainty_m"])
    if abs(boundary_uncertainty_m - 0.0005) > 1.0e-12:
        raise RuntimeError("runtime_contact_v2 boundary uncertainty 与冻结预期不符")
    reachability_module = load_module(reachability_path, "f1_approved_pad_distance_helper")
    closest_points_on_triangles = reachability_module.closest_points_on_triangles

    # Initial FK alignment is checked for each of the 27 recorded bodies in all four envs.
    initial = probe01["initial"]
    if len(initial["body_names"]) != 27 or len(initial["joint_pos"]) != 4:
        raise RuntimeError("F1 initial 记录不再是 4 env x 27 bodies")
    body_names = list(initial["body_names"])
    initial_errors = []
    for env_id in range(4):
        q = np.asarray(initial["joint_pos"][env_id], dtype=np.float64)
        fk_poses = helper.compute_poses(joints, joint_map(joint_names, q))
        predicted = world_poses(
            helper, fk_poses, initial["body_pose_w"][env_id][body_names.index("base_link")]
        )
        for body_index, body_name in enumerate(body_names):
            obs = initial["body_pose_w"][env_id][body_index]
            observed_rotation = xyzw_to_matrix(helper, obs[3:])
            predicted_rotation, predicted_position = predicted[body_name]
            initial_errors.append({
                "env_id": env_id,
                "body_name": body_name,
                "position_error_m": float(np.linalg.norm(predicted_position - np.asarray(obs[:3]))),
                "orientation_error_rad": rotation_error_rad(predicted_rotation, observed_rotation),
            })
    max_position_error = max(row["position_error_m"] for row in initial_errors)
    max_orientation_error = max(row["orientation_error_rad"] for row in initial_errors)
    if probe01["initial"]["joint_pos"] != probe02["initial"]["joint_pos"]:
        raise RuntimeError("probe01 与 probe02 initial_q 不一致；本次几何扫描不合并不同初态")

    probe_results = []
    substep_sets = {}
    for result_path, result in ((probe01_path, probe01), (probe02_path, probe02)):
        substep_path = result_path.parent / "physical_substeps.jsonl"
        substeps = read_substeps(substep_path)
        substep_sets[result_path.parent.name] = substeps
        summary = summarize_attempts(
            str(result_path), result, substeps, active_names, mimic, mapper_module, torch,
            joint_names, helper, joints, pads, closest_points_on_triangles,
        )
        summary["probe_result_sha256"] = sha256(result_path)
        summary["substeps_sha256"] = sha256(substep_path)
        summary["probe_recorded_max_middle_pad_force_n"] = max(
            float(step["pad_force_n"][env_id][2])
            for step in result["steps"] for env_id in range(4)
        )
        probe_results.append({"probe": result_path.parent.name, **summary})

    # Current-position scan: probe02 env0 fruit/base frame; retain all other declared dimensions.
    probe02_initial = probe02["initial"]
    base_body_index = probe02_initial["body_names"].index("base_link")
    base_action = np.asarray(probe02["declared_actions"][0], dtype=np.float32)
    fruit_center = np.asarray(probe02_initial["fruit_pose_w"][0][:3], dtype=np.float64)
    scan = build_scan(
        base_action, fruit_center, probe02_initial["body_pose_w"][0][base_body_index],
        active_names, mimic, mapper_module, torch, joint_names, helper, joints, pads,
        closest_points_on_triangles,
    )
    baseline_active = action_targets(base_action, mapper_module, torch)[0]
    baseline_targets = build_joint_targets(baseline_active, active_names, joint_names, mimic)
    baseline_world = run_fk(
        helper, joints, joint_names, q_array_from_target(baseline_targets, joint_names),
        probe02_initial["body_pose_w"][0][base_body_index],
    )
    baseline_pad_metrics = {
        digit: pad_to_fruit_metric(
            closest_points_on_triangles, pads[digit], baseline_world[body], fruit_center
        )
        for digit, body in PAD_BODIES.items()
    }
    shift_diagnostics = diagnostic_fruit_shifts(
        scan, fruit_center, probe02_initial["body_pose_w"][0][base_body_index],
        joints, helper, joint_names, pads, closest_points_on_triangles,
    )
    scan.pop("_ranked")
    scan.pop("_details")
    if shift_diagnostics is not None:
        scan["fruit_toward_middle_pad_translation_diagnostic"] = shift_diagnostics
    else:
        scan["fruit_toward_middle_pad_translation_diagnostic"] = None

    recommended = []
    aligned_candidates = sorted(
        (row for row in scan["near_surface_candidates"] if row["normal_facing_fruit_dot"] > 0.0),
        key=candidate_order,
    )
    for row in aligned_candidates[:4]:
        recommended.append({
            "action_14d": row["action_14d"],
            "middle_residual": row["middle_residual"],
            "middle_distal": row["middle_distal"],
            "surface_gap_m": row["surface_gap_m_distance_minus_0p0335"],
            "normal_facing_fruit_dot": row["normal_facing_fruit_dot"],
        })

    output = {
        "schema": "firm_grasp.F1_FK_SCAN.v1",
        "created_unix": time.time(),
        "status": "CPU_GEOMETRY_DIAGNOSTIC_ONLY_NOT_PHYSICAL_REACHABILITY_PASS",
        "scope": {
            "cpu_only": True,
            "cuda_visible_devices": "",
            "simulation_or_app_launcher_started": False,
            "physical_probes_launched": False,
            "roi_or_friction_modified": False,
            "full_action_space_scanned": False,
        },
        "provenance": {
            "probe01_result": str(probe01_path),
            "probe02_result": str(probe02_path),
            "fk_helper": str(helper_path),
            "fk_helper_sha256": sha256(helper_path),
            "action_mapping": str(mapper_path),
            "action_mapping_sha256": sha256(mapper_path),
            "action_mapping_source": "probe02 frozen input worktree; independent_thumb_targets",
            "usd_path": str(usd_path),
            "usd_sha256": sha256(usd_path),
            "approved_pad_regions": str(approved_path),
            "approved_pad_regions_sha256": sha256(approved_path),
            "source_geometry_npz": str(geometry_path),
            "source_geometry_sha256": sha256(geometry_path),
            "closest_points_helper": str(reachability_path),
            "runtime_contact_boundary_uncertainty_m": boundary_uncertainty_m,
            "input_manifest_hash_checks": manifest_checks,
        },
        "probe_identity": {
            "task_id": result02_identity.get("task_id"),
            "task_version": result02_identity.get("task_version"),
            "action_mapping_id": result02_identity["action_mapping"].get("mapping_id"),
            "action_order": action_order,
            "probe_count": 2,
            "real_declared_action_count": 8,
            "mimic_followertarget_formula_from_task_identity": mimic,
        },
        "approved_pad_geometry": {
            "approval_status": approved.get("status"),
            "approved": approved.get("approved"),
            "usable_for_training": approved.get("usable_for_training"),
            "triangle_selection": {
                digit: {
                    "body": value["body_name"],
                    "approved_triangle_count": value["approved_triangle_count"],
                    "source_face_count": value["source_face_count"],
                }
                for digit, value in pads.items()
            },
            "normal_orientation": "raw selected triangle normals flipped to agree with approved mean outward normal",
        },
        "initial_fk_validation": {
            "validated_probe": "F1_middle_probe01",
            "environments": 4,
            "bodies_per_environment": 27,
            "joint_count": len(joint_names),
            "max_position_error_m": max_position_error,
            "max_orientation_error_rad": max_orientation_error,
            "position_error_lt_0p5mm": max_position_error < 0.0005,
            "orientation_error_lt_0p005rad": max_orientation_error < 0.005,
            "per_env_per_body_errors": initial_errors,
        },
        "real_attempt_geometry": probe_results,
        "probe02_env0_declared_target_pad_geometry_at_initial_fruit_pose": {
            "action_14d": base_action.astype(float).tolist(),
            "fruit_center_world_m": fruit_center.tolist(),
            "pads": baseline_pad_metrics,
        },
        "middle_residual_distal_grid_scan": scan,
        "recommended_next_action_candidates_at_most_four": recommended,
        "interpretation_limit": (
            "PAD triangle distance, normal alignment, and USD FK are geometric diagnostics. "
            "They do not establish contact dynamics, actuator tracking, or physical reachability PASS. "
            "The probe02 last-substep metrics use recorded physical joint_pos and fruit_pose_w."
        ),
        "elapsed_seconds": time.monotonic() - start,
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "elapsed_seconds": output["elapsed_seconds"],
        "max_initial_fk_position_error_m": max_position_error,
        "max_initial_fk_orientation_error_rad": max_orientation_error,
        "probe02_final_middle_gaps_m": [
            row["measured_physics_last_substep_middle_pad"]["surface_gap_m_distance_minus_0p0335"]
            for row in next(x for x in probe_results if x["probe"] == "F1_middle_probe02")["attempts"]
        ],
        "best_scan": recommended[0] if recommended else None,
        "fruit_shift_diagnostic": shift_diagnostics,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
