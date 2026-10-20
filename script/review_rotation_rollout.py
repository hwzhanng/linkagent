"""Read-only phase-free behavior audit from independent evaluation pose/pad traces."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def inspect(evaluation):
    metadata = json.loads(evaluation.read_text())
    data = np.load(evaluation.with_suffix(".npz"))
    active = data["active"]
    poses = np.concatenate([data["initial_pose"][None], data["poses"]])
    # The current Isaac Lab 3 runtime records native XYZW; verified at mount/reset.
    matrices = Rotation.from_quat(poses[..., 3:7].reshape(-1, 4)).as_matrix().reshape(*poses.shape[:2], 3, 3)
    heading = np.arctan2(matrices[..., 1, 0], matrices[..., 0, 0])
    d = np.diff(heading, axis=0)
    d = np.arctan2(np.sin(d), np.cos(d)) * active * data["target_direction"]
    angle = np.cumsum(d, axis=0)
    full = data["support"].astype(bool)
    n = active.shape[1]
    seen = np.zeros(n, bool)
    armed = np.zeros(n, bool)
    on = np.zeros(n, int)
    off = np.zeros(n, int)
    recovered = np.zeros(n, int)
    progressed_recovery = np.zeros(n, int)
    last_progress_angle = np.zeros(n)
    for t in range(len(active)):
        valid = active[t]
        on = np.where(full[t] & valid, on + 1, 0)
        off = np.where(~full[t] & valid, off + 1, 0)
        seen |= on >= 4
        armed |= seen & (off >= 4)
        event = armed & (on >= 4) & valid & (t >= 20)
        recovered += event
        useful = event & (angle[t] - last_progress_angle >= np.deg2rad(10))
        progressed_recovery += useful
        last_progress_angle = np.where(useful, angle[t], last_progress_angle)
        armed &= ~event
    late = active[200:]
    result = {
        "evaluation": str(evaluation.resolve()), "episodes": n,
        "scope": "Diagnostic events only; release/recontact does not establish a full rotation or successful gait.",
        "event_definition": ">=4 supported steps before >=4 unsupported steps, then >=4 supported steps; 20Hz, after first second. Progress-linked event also requires >=10deg net advance beyond previous counted event.",
        "mean_support_recoveries": float(recovered.mean()),
        "mean_progress_linked_recoveries": float(progressed_recovery.mean()),
        "episodes_with_two_progress_linked_recoveries": int((progressed_recovery >= 2).sum()),
        "mean_after_10s_degrees": float(np.rad2deg(d[200:].sum(0)).mean()),
        "episodes_with_after_10s_net_advance_ge15deg": int((d[200:].sum(0) >= np.deg2rad(15)).sum()),
        "mean_late_strict_support_fraction": float((full[200:] * late).sum() / max(late.sum(), 1)),
        "full_task_success_fraction": metadata["success_fraction"],
    }
    if "pads" in data.files:
        result["mean_late_pad_contact_fraction"] = (
            (data["pads"][200:] * late[..., None]).sum((0, 1)) / max(late.sum(), 1)).tolist()
    result["per_episode"] = [
        dict(env=i, recoveries=int(recovered[i]), progress_linked_recoveries=int(progressed_recovery[i]),
             after_10s_degrees=float(np.rad2deg(d[200:, i].sum()))) for i in range(n)]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_episode"}, indent=2))
