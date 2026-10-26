"""Summarize observed recontact geometry; never changes reward or success gates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from simulation.roto2_pressure.core import quat_rotate
from simulation.roto2_pressure.reward_geometry import pad_approach_quality, pad_frames, pad_contact


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--evaluation", type=Path, help="Existing evaluation JSON, including final_seed files")
    parser.add_argument("--trace", type=Path, help="Action trace NPZ")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--final-training-result", action="store_true")
    args = parser.parse_args()
    folder = args.folder
    trace_path = args.trace or folder / f"action_trace_seed{args.seed}.npz"
    evaluation_path = args.evaluation or folder / f"evaluation_seed{args.seed}.json"
    trace = np.load(trace_path)
    evaluation = json.loads(evaluation_path.read_text())
    evaluation_npz_path = evaluation_path.with_suffix(".npz")
    evaluation_trace = np.load(evaluation_npz_path)
    trace_summary = json.loads(trace_path.with_suffix(".json").read_text())
    active = trace["active"].astype(bool)
    mask = active & ~trace["done"].astype(bool)
    time = (np.arange(len(active)) + 1) * trace_summary["dt"]
    late = mask & (time[:, None] > 10)
    limits = trace["limits"]
    near = np.minimum(trace["joint"] - limits[None, :, :, 0],
                      limits[None, :, :, 1] - trace["joint"]) < .02
    local = torch.from_numpy(trace["contact_local"][mask])
    forces = torch.from_numpy(trace["forces"][mask])
    _, pad_forces, _, _ = pad_contact(local, forces)
    assert np.array_equal((pad_forces > .15).numpy(), trace["pad"][mask])

    # Raw tip poses on terminal steps can be overwritten by automatic reset;
    # all geometry here excludes those steps, including the final time limit.
    tip = torch.from_numpy(trace["tip_position_w"][mask])
    tip_quat = torch.from_numpy(trace["tip_quaternion_xyzw"][mask][..., [3, 0, 1, 2]])
    fruit = torch.from_numpy(trace["pose"][mask])
    axes = torch.from_numpy(np.broadcast_to(trace["fruit_axes_m"],
                                           (*active.shape, 3))[mask].copy())
    recomputed = pad_approach_quality(tip, tip_quat, fruit[:, :3],
                                     fruit[:, [6, 3, 4, 5]], axes).numpy()
    max_quality_error = float(np.max(np.abs(recomputed - trace["approach_quality"][mask])))
    assert max_quality_error < 1e-5, max_quality_error
    center, _, _, normal = pad_frames("cpu")
    centers = tip + quat_rotate(tip_quat, center.expand_as(tip))
    normals = quat_rotate(tip_quat, normal.expand_as(tip))
    to_fruit = fruit[:, None, :3] - centers
    alignment = (normals * torch.nn.functional.normalize(to_fruit, dim=-1)).sum(-1).numpy()
    # Current nominal fruit is spherical; do not silently approximate ellipsoids.
    assert torch.max(torch.abs(axes - axes[:, :1])) < 1e-7
    gap = (torch.linalg.vector_norm(to_fruit, dim=-1) - axes[:, :1]).numpy()
    geometry_late = late[mask]
    episodes = evaluation["per_episode"]
    assert np.max(np.abs(evaluation_trace["poses"][mask] - trace["pose"][mask])) < 1e-6
    poses = np.concatenate([evaluation_trace["initial_pose"][None], evaluation_trace["poses"]])
    x, y, z, w = (poses[..., i] for i in range(3, 7))
    heading = np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))
    delta = np.diff(heading, axis=0)
    delta = np.arctan2(np.sin(delta), np.cos(delta)) * evaluation_trace["target_direction"][None]
    valid_path = trace["term_invalid_rotation_path"] == 0
    support = evaluation_trace["support"].astype(bool)
    displacement = np.linalg.norm(trace["pose"][..., :3] -
                                  trace["center_reference_initial"][None], axis=-1)
    unsupported = late & ~support
    speed = delta / trace_summary["dt"]
    forward = np.minimum(np.maximum(speed, 0), 2 * evaluation_trace["target_speed"][None])
    expected_motion = forward * (valid_path & support) + np.minimum(speed, 0)
    expected_motion *= trace_summary.get("baseline_motion_weight", 1.)
    motion_error = float(np.max(np.abs(expected_motion[late] - trace["term_rotation"][late])))
    assert motion_error < 1e-3, motion_error
    result = {
        "diagnostic_only": True,
        "final_training_result": args.final_training_result,
        "seed": args.seed,
        "source_sha256": {str(trace_path): sha256(trace_path),
                          str(evaluation_path): sha256(evaluation_path),
                          str(evaluation_npz_path): sha256(evaluation_npz_path)},
        "episodes": len(episodes),
        "safe_full_turn_successes": sum(row["success"] for row in episodes),
        "raw_full_turns": sum(row["directed_pose_turns"] >= 1 for row in episodes),
        "physical_failures": sum(row["failed"] for row in episodes),
        "mean_total_degrees": float(np.mean([row["directed_pose_turns"] * 360 for row in episodes])),
        "mean_after10s_degrees": evaluation["mean_after_10s_directed_turns"] * 360,
        "max_total_degrees": max(row["directed_pose_turns"] * 360 for row in episodes),
        "max_after10s_degrees": max(row["after_10s_directed_turns"] * 360 for row in episodes),
        "policy_std": trace_summary["policy_std"],
        "pad_flags_recomputed_exactly": True,
        "approach_quality_max_recomputation_error": max_quality_error,
        "late_motion_reward_max_recomputation_error": motion_error,
        "late": {
            "denominator": "Active, nonterminal environment steps with post-step time strictly >10s; joint values are pre-action.",
            "environment_steps": int(late.sum()),
            "near_limit_joint_fraction_20mrad": float(near[late].mean()),
            "mean_effective_pad_count": float(trace["pad"].sum(-1)[late].mean()),
            "per_finger_effective_contact_fraction": trace["pad"][late].mean(0).tolist(),
            "mean_approach_quality": float(trace["approach_quality"][late].mean()),
            "mean_best_three_approach_quality": float(np.sort(trace["approach_quality"][late], axis=-1)[:, -3:].mean()),
            "median_center_displacement_mm": float(np.median(displacement[late]) * 1000),
            "unsupported_environment_steps": int(unsupported.sum()),
            "unsupported_steps_with_center_displacement_at_least25mm": int((unsupported & (displacement >= .025)).sum()),
            "per_finger_mean_surface_gap_mm": (gap[geometry_late].mean(0) * 1000).tolist(),
            "per_finger_mean_normal_alignment": alignment[geometry_late].mean(0).tolist(),
        },
        "limitations": [
            ("Single training seed; diagnosis supplements but does not replace independent acceptance."
             if args.final_training_result else "Single training seed and intermediate checkpoint; not a final gate."),
            "Only five distal links are measured; absent distal contact does not imply absent palm/proximal support.",
            "Geometric approach quality is not actual contact and is not a success metric.",
            "Mean gap and alignment aggregate trajectories; they do not prove a feasible recontact action.",
        ],
    }
    result["late_motion_credit_categories"] = {}
    for label, category in [
        ("eligible", valid_path & support),
        ("valid_path_without_support", valid_path & ~support),
        ("invalid_path", ~valid_path),
    ]:
        selected = late & category
        result["late_motion_credit_categories"][label] = {
            "environment_steps": int(selected.sum()),
            "net_degrees_per_initial_episode": float(np.rad2deg(delta[selected]).sum() / len(episodes)),
            "positive_degrees_per_initial_episode": float(np.rad2deg(np.maximum(delta[selected], 0)).sum() / len(episodes)),
            "rotation_reward_per_initial_episode": float(trace["term_rotation"][selected].sum() / len(episodes)),
        }
    for label, start, end in [("early1to5s", 1, 5), ("late10to24s", 10, 24)]:
        window = active & (time[:, None] > start) & (time[:, None] <= end)
        result[label + "_reward_terms_per_initial_episode"] = {
            key: float((trace[key] * window).sum() / len(episodes))
            for key in trace.files if key.startswith("term_")
        }
    out = args.output or folder / "RECONTACT_DIAGNOSIS.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
