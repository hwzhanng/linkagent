"""Plot first-episode physical evidence with explicit aggregation denominators."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", default="Policy diagnostic")
    a = p.parse_args()
    summary = json.loads(a.evaluation.read_text())
    e = np.load(a.evaluation.with_suffix(".npz"))
    tr = np.load(a.trace.with_suffix(".npz"))
    active = e["active"].astype(bool)
    # This project records native xyzw; require independent agreement with the
    # evaluator rather than silently accepting an incorrect quaternion layout.
    poses = np.concatenate([e["initial_pose"][None], e["poses"]])
    n = poses.shape[1]
    mats = Rotation.from_quat(poses[..., 3:7].reshape(-1, 4)).as_matrix().reshape(-1, n, 3, 3)
    h = np.arctan2(mats[..., 1, 0], mats[..., 0, 0])
    delta = np.arctan2(np.sin(np.diff(h, axis=0)), np.cos(np.diff(h, axis=0)))
    degrees = np.cumsum(delta * active * e["target_direction"][None], axis=0) * 180 / np.pi
    expected = np.array([r["directed_pose_turns"] * 360 for r in summary["per_episode"]])
    assert np.max(np.abs(degrees[-1] - expected)) < .01
    mask = active & ~tr["done"]
    assert np.max(np.abs(e["poses"][mask] - tr["pose"][mask])) < 1e-6
    t = (np.arange(len(active)) + 1) * .05
    count = active.sum(axis=1).clip(1)
    pads = (e["pads"] * active[..., None]).sum(axis=1) / count[:, None]
    q, limits = tr["joint"], tr["limits"]
    near = np.minimum(q - limits[None, :, :, 0], limits[None, :, :, 1] - q) < .02
    near_fraction = (near * active[..., None]).sum((1, 2)) / (count * q.shape[-1])
    rotation = np.cumsum((tr["term_rotation"] * active).mean(axis=1))
    other_terms = sum(tr[k] for k in tr.files if k.startswith("term_") and k != "term_rotation")
    other = np.cumsum((other_terms * active).mean(axis=1))
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax[0, 0].plot(t, degrees, color="#778da9", alpha=.16, lw=.7)
    ax[0, 0].plot(t, degrees.mean(axis=1), color="#005f73", lw=2.3, label=f"Mean of all {n} first episodes")
    ax[0, 0].axhline(360, color="#ae2012", ls="--", label="One full revolution")
    ax[0, 0].axvline(10, color="grey", ls=":")
    ax[0, 0].set(ylabel="Net directed heading (degrees)", title="A. Actual quaternion-derived rotation")
    ax[0, 0].legend(fontsize=9)
    for i, name in enumerate(["Thumb", "Index", "Middle", "Ring", "Little"]):
        ax[0, 1].plot(t, pads[:, i], label=name, lw=1)
    ax[0, 1].set(ylabel="Effective-pad contact fraction", title="B. Contact among still-active episodes", ylim=(-.02, 1.02))
    ax[0, 1].legend(fontsize=9, ncol=3)
    ax[1, 0].plot(t, near_fraction, color="#9b2226", lw=2)
    ax[1, 0].set(ylabel="Fraction of active joints within 0.02 rad", ylim=(0, 1),
                 title="C. Joint-limit proximity, active episodes")
    ax[1, 1].plot(t, rotation, label="Rotation term", color="#005f73", lw=2)
    ax[1, 1].plot(t, other, label="Other shaping terms (excludes failure cost)", color="#bb3e03", lw=2)
    ax[1, 1].set(ylabel="Cumulative mean reward term", title="D. Where reward is earned")
    ax[1, 1].legend(fontsize=9)
    for x in ax.flat:
        x.set(xlabel="Simulation time (s)", xlim=(0, 24))
        x.grid(alpha=.18)
    successes = sum(r["success"] for r in summary["per_episode"])
    fig.suptitle(f"{a.label}: {n} episodes\n"
                 f"{successes}/{n} full turns; mean {expected.mean():.2f} deg; "
                 f"mean progress after 10 s = {summary['mean_after_10s_directed_turns']*360:.3f} deg", fontsize=14)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=180)
    fig.savefig(a.output.with_suffix(".svg"))
    np.savetxt(a.output.with_suffix(".csv"),
               np.column_stack([t, active.sum(1), degrees.mean(1), pads, near_fraction, rotation, other]),
               delimiter=",", comments="",
               header="time_s,active_episodes,mean_net_degrees,thumb_pad,index_pad,middle_pad,ring_pad,little_pad,near_limit_joint_fraction,cumulative_rotation_reward,cumulative_other_shaping")
    a.output.with_suffix(".json").write_text(json.dumps({
        "evaluation": str(a.evaluation), "trace": str(a.trace),
        "pose_traces_match": True, "episodes": n,
        "rotation_and_reward_denominator": f"All{n} first episodes, freeze accumulated values after first termination.",
        "contact_and_joint_denominator": "Still-active episodes at each time; active count is in CSV.",
        "failure_cost_in_reward_panel": False,
        "final_training_result": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
