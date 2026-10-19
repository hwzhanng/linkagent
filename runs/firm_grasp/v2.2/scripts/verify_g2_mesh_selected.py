#!/usr/bin/env python3
"""G2 360步候选窗的原始full-mesh PAD重分类；仅几何核验，不声明数值G2通过。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


RUN_ROOT = Path(__file__).resolve().parent
S_ROOT = RUN_ROOT / "source-f3-v2_0"
if not S_ROOT.is_dir():
    raise FileNotFoundError(f"固定 source 根目录不存在：{S_ROOT}")
sys.path.insert(0, str(S_ROOT))

from research.firm_grasp_v1.contract import load_frozen_contract  # noqa: E402
from research.firm_grasp_v1 import verify_window as original_window  # noqa: E402


COMBINATIONS = original_window.COMBINATIONS
DIGITS = original_window.DIGITS
ORIGINAL = original_window.ORIGINAL
ASSETS = S_ROOT / "research" / "thumb_pad_grasp_v1" / "assets"
REQUIRED = 360


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-id", type=int)
    parser.add_argument("--combo-index", type=int)
    parser.add_argument("--end-sim-step", type=int)
    args = parser.parse_args()
    selectors = (args.env_id, args.combo_index, args.end_sim_step)
    selected_mode = any(value is not None for value in selectors)
    if selected_mode and not all(value is not None for value in selectors):
        parser.error("--env-id、--combo-index、--end-sim-step 必须同时提供")
    if selected_mode and any(value < 0 for value in selectors):
        parser.error("筛选参数必须为非负整数")
    contract = load_frozen_contract()
    if args.output.exists():
        raise FileExistsError(args.output)
    window_path = args.output.with_name(args.output.stem + "_window.jsonl")
    if window_path.exists():
        raise FileExistsError(window_path)

    rows = []
    raw_lines = []
    selected = None
    selected_matches = []
    peaks = None
    with args.trace.open(encoding="utf-8") as stream:
        for line in stream:
            outer = json.loads(line)
            row = outer.get("record", outer)
            rows.append(row)
            raw_lines.append(line)
            if "g2_streak" not in row:
                raise ValueError("轨迹缺少 g2_streak，无法选择 G2 360步几何窗")
            streak = np.asarray(row["g2_streak"])
            peaks = streak.copy() if peaks is None else np.maximum(peaks, streak)
            if selected_mode:
                env, combo = args.env_id, args.combo_index
                if streak.ndim != 2 or env >= streak.shape[0] or combo >= streak.shape[1]:
                    raise ValueError(
                        f"筛选 env/combo 超出 g2_streak 矩阵范围："
                        f"env_id={env}, combo_index={combo}, shape={streak.shape}"
                    )
                if row.get("sim_step") == args.end_sim_step and streak[env, combo] >= REQUIRED:
                    selected_matches.append((len(rows) - 1, env, combo))
            else:
                where = np.argwhere(streak >= REQUIRED)
                if selected is None and len(where):
                    env, combo = map(int, where[0])
                    selected = (len(rows) - 1, env, combo)

    if selected_mode:
        if len(selected_matches) > 1:
            raise ValueError(
                f"筛选条件命中多个轨迹行，无法唯一选择：{selected_matches}"
            )
        selected = selected_matches[0] if selected_matches else None

    if selected is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "NO_WINDOW" if selected_mode else "NO_G2_STREAK_WINDOW",
            "geometry_only_report": True,
            "claim_scope": "仅原始full-mesh PAD几何核验；不表示数值G2或正式G2通过",
            "required_substeps": REQUIRED,
            "max_g2_streak_by_env_combo": peaks.tolist() if peaks is not None else [],
            "trace": str(args.trace.resolve()),
        }
        if selected_mode:
            result["selection"] = {
                "env_id": args.env_id,
                "combo_index": args.combo_index,
                "end_sim_step": args.end_sim_step,
            }
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "output": str(args.output)}))
        return 2

    end, env, combo = selected
    window = rows[end - REQUIRED + 1 : end + 1]
    trio = COMBINATIONS[combo]
    assert len(window) == REQUIRED
    assert all(r["sim_step"] == window[0]["sim_step"] + k for k, r in enumerate(window))
    assert all(abs(r["dt_s"] - 1 / 120) < 1e-12 for r in window)
    if selected_mode:
        assert window[-1]["sim_step"] == args.end_sim_step
        streak_start = window[0]["g2_streak"][env][combo]
        assert all(r["g2_streak"][env][combo] == streak_start + k
                   for k, r in enumerate(window))
        assert window[-1]["g2_streak"][env][combo] >= REQUIRED
    else:
        assert all(r["g2_streak"][env][combo] == k + 1
                   for k, r in enumerate(window))
    assert all(r["released"][env] for r in window)

    expected_faults = {
        "overload", "nonfinite", "penetration", "joint_limit",
        "self_collision", "instability", "joint_effort",
    }
    assert all(expected_faults <= set(r["hard_faults"]) for r in window), "七硬故障字段缺失"
    assert all(not any(values[env] for values in r["hard_faults"].values()) for r in window)
    assert all(not any(env in channel["env_ids"] for channel in r["invalid_contact_channels"])
               for r in window), "无效接触通道"
    assert all(
        np.all(np.asarray(r["support_force_w"][env]) == 0)
        and np.all(np.asarray(r["support_torque_w"][env]) == 0)
        for r in window
    ), "support force/torque 非零"
    assert all(
        not any(r["truth"]["unknown"][env])
        and np.all(np.asarray(r["truth"]["nonpad_force_n"][env]) == 0)
        and r["palm_force_n"][env] == 0
        for r in window
    ), "unknown/NONPAD/palm 检查失败"
    force = np.asarray([r["truth"]["pad_force_n"][env] for r in window])
    share = force / force.sum(1)[:, None]
    assert (force[:, trio] >= contract.f_pad_min_n).all()
    assert (share[:, trio] >= contract.min_load_share).all()
    speed = np.asarray([r["relative_speed_m_s"][env] for r in window])
    position = np.asarray([r["fruit_in_hand_m"][env] for r in window])
    drift = np.linalg.norm(position - position[0], axis=1)
    assert (speed <= 0.05).all() and (drift <= 0.01).all()

    # 原始冻结分类器：未加速完整三角网格，独立于运行时AABB剪枝分类器。
    assert original_window.sha(ORIGINAL) == (
        "14d7412e72af348e04d56e3452c7f9b34a135a9f2777de623e8faf4d7eea58bd"
    )
    spec = importlib.util.spec_from_file_location("firm_original_full_mesh", ORIGINAL)
    geom = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = geom
    spec.loader.exec_module(geom)
    approved_path = ASSETS / "pad_regions_v1.approved.json"
    assert original_window.sha(approved_path) == (
        "abd16607249509762ac07483f9542b6b460628beee5b13a464fca31217b04258"
    )
    approved = json.loads(approved_path.read_text(encoding="utf-8"))
    geometry = np.load(ASSETS / "source_geometry_v1.npz")
    config = json.loads((ASSETS / "runtime_contact_v2.json").read_text(encoding="utf-8"))
    full_mesh = {}
    layout = []
    for digit_id in trio:
        name = DIGITS[digit_id]
        definition = next(d for d in approved["digits"] if d["digit"] == name)
        body = definition["body_path"].rsplit("/", 1)[-1]
        local, normal, world_local, world_normal, centroids = [], [], [], [], []
        for row in window:
            points, weights = [], []
            for contact in row["contacts"]:
                if contact["body"] != body:
                    continue
                for j, contact_env in enumerate(contact["env_ids"]):
                    if contact_env != env:
                        continue
                    assert contact["forces_n"][j] > 0 and contact["labels"][j] == 1
                    pose = np.asarray(contact["body_pose_w"][j])
                    rotation = Rotation.from_quat(pose[3:7])
                    local.append(contact["points_body_m"][j])
                    normal.append(contact["outward_normals_body"][j])
                    world_local.append(rotation.inv().apply(np.asarray(contact["points_w"][j]) - pose[:3]))
                    world_normal.append(rotation.inv().apply(-np.asarray(contact["raw_normals_w"][j])))
                    weights.append(contact["forces_n"][j])
                    points.append(contact["points_w"][j])
            assert weights and abs(sum(weights) - row["truth"]["pad_force_n"][env][digit_id]) < 1e-5
            centroids.append(np.average(np.asarray(points), axis=0, weights=weights))
        surface = geom.PadSurface(
            geometry[name + "_vertices_body_m"], geometry[name + "_faces"],
            definition["selected_triangle_ids"],
        )
        kwargs = {
            "position_tolerance_m": config["thumb_position_tolerance_m"] if digit_id == 0
            else config["other_position_tolerance_m"],
            "normal_tolerance_rad": math.radians(config["thumb_normal_tolerance_deg"] if digit_id == 0
            else config["other_normal_tolerance_deg"]),
        }
        local_result = geom.classify(surface, np.asarray(local), np.asarray(normal), **kwargs)
        world_result = geom.classify(
            surface, np.asarray(world_local), np.asarray(world_normal), **kwargs
        )
        assert all(label == "PAD" for label in local_result["state"])
        assert all(label == "PAD" for label in world_result["state"])
        boundary_distance = original_window.edge_distance(
            local_result["closest_point_body_m"],
            np.asarray(definition["boundary_edges_body_local_m"]),
        )
        assert (boundary_distance >= config["boundary_uncertainty_m"]).all()
        full_mesh[name] = {
            "points": len(local),
            "all_original_full_mesh_PAD": True,
            "world_to_body_recomputed_PAD": True,
            "max_source_distance_m": float(np.max(local_result["distance_m"])),
            "min_approved_boundary_distance_m": float(boundary_distance.min()),
            "max_world_body_coordinate_error_m": float(
                np.linalg.norm(np.asarray(local) - np.asarray(world_local), axis=1).max()
            ),
        }
        layout.append(np.asarray(centroids))

    fruit = np.asarray([r["fruit_pose_w"][env][:3] for r in window])
    radial = [centroid - fruit for centroid in layout]
    bottom = original_window.assert_ground_free_window(window, env)
    radial = [vector / np.linalg.norm(vector, axis=1)[:, None] for vector in radial]
    opposite = radial[1] + radial[2]
    opposite /= np.linalg.norm(opposite, axis=1)[:, None]
    opposition = (radial[0] * opposite).sum(1)
    separation = np.linalg.norm(layout[1] - layout[2], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with window_path.open("x", encoding="utf-8") as stream:
        stream.writelines(raw_lines[end - REQUIRED + 1 : end + 1])
    result = {
        "status": "PASS_G2_WINDOW_ORIGINAL_FULL_MESH_LABELS",
        "geometry_only_report": True,
        "claim_scope": "仅几何验证；不替代数值G2验证，不声明扰动/恢复或正式G2通过",
        "trace": str(args.trace.resolve()),
        "env_id": env,
        "combo_index": combo,
        "digits": [DIGITS[i] for i in trio],
        "sim_step_start": window[0]["sim_step"],
        "sim_step_end": window[-1]["sim_step"],
        "post_release_start_substep": window[0].get("post_release_substep", [[]])[env],
        "geometry_window_substeps": REQUIRED,
        "dt_sum_s": sum(r["dt_s"] for r in window),
        "window_file": str(window_path),
        "window_sha256": original_window.sha(window_path),
        "min_PAD_force_n": dict(zip(DIGITS, force.min(0).tolist())),
        "min_load_share": dict(zip(DIGITS, share.min(0).tolist())),
        "max_relative_speed_m_s": float(speed.max()),
        "max_hand_anchor_drift_m": float(drift.max()),
        "min_fruit_bottom_clearance_m": float(bottom.min()),
        "support_all_components_zero": True,
        "all_recorded_runtime_hard_fault_flags_false": True,
        "full_mesh": full_mesh,
        "geometry_diagnostic": {
            "min_nonthumb_centroid_separation_m": float(separation.min()),
            "max_thumb_opposition_dot": float(opposition.max()),
        },
        "pad_thresholds": {
            "min_force_n": contract.f_pad_min_n,
            "min_share": contract.min_load_share,
            "source_contract_sha256": original_window.sha(
                S_ROOT / "research" / "firm_grasp_v1" / "firm_grasp_contract_v1.json"
            ),
        },
        "all_required_seven_fault_fields_present": True,
        "selected_env_invalid_channels_absent": True,
        "original_classifier_sha256": original_window.sha(ORIGINAL),
        "verifier_source_sha256": original_window.sha(Path(__file__).resolve()),
    }
    for field in ("matrix_case_id", "matrix_scenario"):
        if field in window[0]:
            value = window[0][field]
            if isinstance(value, list):
                value = value[env]
            result[field] = value
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "env_id": env, "digits": result["digits"],
                      "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
