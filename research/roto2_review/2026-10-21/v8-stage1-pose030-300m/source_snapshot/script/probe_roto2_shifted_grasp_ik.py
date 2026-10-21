"""CPU-only reference-geometry tracking probe; never an accepted reset library."""
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from simulation.roto2_pressure.core import ACTIVE_JOINTS, MIMICS
from simulation.roto2_pressure.reward_geometry import pad_frames

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/simulation/rotation_v8/stage1"


def main():
    spec = importlib.util.spec_from_file_location("fk", ROOT / "simulation/.runtime/setup/search_grasp.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    anchor = json.loads((ROOT / "simulation/roto2_pressure/reset_grasps_v8_validated256.json").read_text())[0]
    q0 = np.array(anchor["joint_positions"])
    target0 = np.array(anchor["joint_targets"])
    fruit0 = np.array(anchor["fruit_position_hand"])
    loaded = np.flatnonzero(anchor["reference_pad_fingers"])
    center, _, _, normal = [x.numpy() for x in pad_frames("cpu")]
    lo, hi = g.lower[:16].copy(), g.upper[:16].copy()
    for follower, (driver, ratio) in MIMICS.items():
        i = ACTIVE_JOINTS.index(driver)
        lo[i] = max(lo[i], g.limits[follower][0] / ratio)
        hi[i] = min(hi[i], g.limits[follower][1] / ratio)
    scale = hi - lo

    def points(q):
        tf = g.fk(q)
        return np.array([tf[g.tips[i]][:3, 3] + tf[g.tips[i]][:3, :3] @ (center[i] + .032 * normal[i])
                         for i in loaded])

    reference = points(q0) - fruit0
    configs = [("original_control", np.zeros(3))]
    transform = Rotation.from_quat([0, -2 ** -.5, 0, 2 ** -.5])
    for name in ("interim_031M", "milestone_100000000", "milestone_200000000"):
        a = np.load(BASE / name / "evaluation_seed5101.npz")
        d = json.loads((BASE / name / "evaluation_seed5101.json").read_text())
        last = a["active"].sum(0) - 1
        delta = a["poses"][last, np.arange(64), :3] - a["initial_pose"][:, :3]
        survivors = np.array([not r["failed"] for r in d["per_episode"]])
        configs.append((name + "_median_vector", np.median(transform.inv().apply(delta[survivors]), axis=0)))
    rng = np.random.default_rng(6200)
    rows, proposals = [], []
    started = time.monotonic()
    for name, shift in configs:
        target = fruit0 + shift

        def residual(q):
            return np.r_[((points(q) - target - reference) / .001).ravel(),
                         .002 * (q - q0) / scale]

        fits = []
        for k in range(4):
            start = q0 if k == 0 else q0 + rng.normal(size=16) * scale * .15
            fit = least_squares(residual, np.clip(start, lo + 1e-7, hi - 1e-7),
                                bounds=(lo, hi), max_nfev=80, diff_step=1e-4)
            error = points(fit.x) - target - reference
            fits.append((float(np.linalg.norm(error)), fit, error))
        _, fit, error = min(fits, key=lambda x: x[0])
        row = dict(case=name, shift_hand_mm=(shift * 1000).tolist(),
                   reference_geometry_error_per_loaded_pad_mm=(np.linalg.norm(error, axis=1) * 1000).tolist(),
                   loaded_pad_indices=loaded.tolist(), joint_positions=fit.x.tolist(),
                   normalized_joint_change_norm=float(np.linalg.norm((fit.x - q0) / scale)),
                   solver_status=int(fit.status), restarts=4,
                   physical_validation=False, collision_checked=False)
        rows.append(row)
        # A changed pose must never inherit its parent's acceptance metadata.
        proposals.append({"id": "shifted_geometry_" + name,
                          "parent_grasp_id": anchor["id"],
                          "joint_positions": fit.x.tolist(),
                          "joint_targets": np.clip(target0 + fit.x - q0, lo, hi).tolist(),
                          "fruit_position_hand": target.tolist(),
                          "fruit_quaternion_xyzw": anchor["fruit_quaternion_xyzw"],
                          "hand_posture": anchor["hand_posture"],
                          "reference_pad_fingers": anchor["reference_pad_fingers"],
                          "proposal_only": True, "physical_validation": False,
                          "n_minus_one_passed": False,
                          "validation_scope": "Unvalidated CPU IK diagnostic only."})
        print(json.dumps(row), flush=True)
    output = dict(anchor=anchor["id"], rows=rows, elapsed_seconds=time.monotonic() - started,
                  interpretation="Tracks the same four-pad reference offsets at observed median shifts using URDF FK. "
                  "A small residual is a kinematic candidate only. A large residual from four local solves "
                  "does not prove impossibility. No collision, force, N-1 or policy validation is claimed.")
    (BASE / "SHIFTED_GRASP_IK.json").write_text(json.dumps(output, indent=2) + "\n")
    (BASE / "SHIFTED_GRASP_PROPOSALS_UNVALIDATED.json").write_text(json.dumps(proposals, indent=2) + "\n")


if __name__ == "__main__":
    main()
