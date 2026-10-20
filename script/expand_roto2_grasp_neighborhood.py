"""Generate distinct local grasp proposals around a physically validated anchor.

URDF geometry preserves approximate pad/fruit relationships; every generated row
still requires independent PhysX N-1 validation. No physics or actor changes.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import time
import numpy as np
from scipy.optimize import least_squares
from simulation.roto2_pressure.core import ACTIVE_JOINTS, MIMICS, FINGERS
from simulation.roto2_pressure.reward_geometry import pad_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anchor", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2109)
    p.add_argument("--count", type=int, default=512)
    p.add_argument("--anchor-count", type=int, default=1)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("grasp_fk", root/"simulation/.runtime/setup/search_grasp.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    source = args.anchor.read_bytes()
    available = json.loads(source)
    anchor = available[0]
    q0 = np.asarray(anchor["joint_positions"])
    target0 = np.asarray(anchor.get("joint_targets", q0))
    fruit0 = np.asarray(anchor["fruit_position_hand"])
    lower, upper = g.lower[:16].copy(), g.upper[:16].copy()
    for follower, (driver, ratio) in MIMICS.items():
        i = ACTIVE_JOINTS.index(driver)
        lower[i] = max(lower[i], g.limits[follower][0]/ratio)
        upper[i] = min(upper[i], g.limits[follower][1]/ratio)
    scale = upper-lower
    support_mask = anchor.get("reference_pad_fingers",
                              anchor.get("terminal_pad_contact", [True, False, True, True, True]))
    support = np.asarray(support_mask).nonzero()[0]
    if len(support) < 4:
        raise ValueError("Anchor must have at least four recorded effective pads")
    loaded_ids = [i for i, name in enumerate(ACTIVE_JOINTS)
                  if name.split("_")[0] in [FINGERS[j] for j in support]]
    if not 1 <= args.anchor_count <= len(available):
        raise ValueError("Invalid number of seed anchors")
    anchor_vectors = (np.asarray([r["joint_positions"] for r in available])/scale)[:, loaded_ids]
    chosen = [0]
    while len(chosen) < args.anchor_count:
        distance = np.linalg.norm(anchor_vectors[:, None]-anchor_vectors[chosen][None], axis=-1).min(1)
        distance[chosen] = -1
        chosen.append(int(distance.argmax()))
    anchors = [available[i] for i in chosen]
    for row in anchors:
        if row.get("reference_pad_fingers", row.get("terminal_pad_contact", support_mask)) != support_mask:
            raise ValueError("One expansion call must use a common recorded pad support set")
    center, _, _, normal = (a.numpy() for a in pad_frames("cpu"))

    def contact_geometry(q):
        transforms = g.fk(q)
        return np.asarray([transforms[g.tips[i]][:3, 3]
                           + transforms[g.tips[i]][:3, :3] @ (center[i]+.032*normal[i])
                           for i in support])

    reference = contact_geometry(q0) - fruit0
    lo = np.r_[lower+1e-6, fruit0-np.array([.001, .003, .003])]
    hi = np.r_[upper-1e-6, fruit0+np.array([.001, .003, .003])]
    rng = np.random.default_rng(args.seed)
    rows, vectors = [], []
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(args.count*3):
        if len(rows) >= args.count:
            break
        anchor = anchors[(attempt//4) % len(anchors)]
        q0 = np.asarray(anchor["joint_positions"])
        target0 = np.asarray(anchor.get("joint_targets", q0))
        fruit0 = np.asarray(anchor["fruit_position_hand"])
        reference = contact_geometry(q0) - fruit0
        lo = np.r_[lower+1e-6, fruit0-np.array([.001, .003, .003])]
        hi = np.r_[upper-1e-6, fruit0+np.array([.001, .003, .003])]
        tier = attempt % 4
        strength = [.006, .015, .03, .06][tier]
        desired_q = np.clip(q0+rng.normal(size=16)*scale*strength, lower+1e-5, upper-1e-5)
        desired_fruit = np.clip(fruit0+rng.normal(size=3)*[.0001, .0003, .0003], lo[16:]+1e-6, hi[16:]-1e-6)
        desired = np.r_[desired_q, desired_fruit]
        mode = "direct" if attempt % 5 == 0 else "geometry_projected"
        if mode == "geometry_projected":
            def residual(x):
                return np.r_[
                    ((contact_geometry(x[:16])-x[16:]-reference)/.001).ravel(),
                    .2*(x[:16]-desired_q)/scale,
                    .1*(x[16:]-desired_fruit)/.003]
            fit = least_squares(residual, desired, bounds=(lo, hi), max_nfev=40, diff_step=1e-4)
            x = fit.x
        else:
            x = desired
        loaded_distance = float(np.linalg.norm((x[:16]-q0)[loaded_ids]/scale[loaded_ids]))
        if loaded_distance < .005:
            continue  # unused index motion alone must not manufacture grasp count
        vector = np.r_[x[:16]/scale, x[16:]/.01]
        if vectors and np.min(np.linalg.norm(np.asarray(vectors)-vector, axis=1)) < .01:
            continue
        vectors.append(vector)
        target = np.clip(target0+x[:16]-q0, lower, upper)
        row = dict(id=f"anchor_neighborhood_s{args.seed}_{len(rows)}",
                   parent_grasp_id=anchor["id"], joint_positions=x[:16].tolist(),
                   joint_targets=target.tolist(), fruit_position_hand=x[16:].tolist(),
                   fruit_quaternion_xyzw=anchor.get("fruit_quaternion_xyzw", [0, 0, 0, 1]),
                   hand_posture="palm_up", proposal_only=True, physical_validation=False,
                   reference_pad_fingers=list(support_mask),
                   reset_schema="neighborhood_proposal_v1", perturbation_mode=mode,
                   perturbation_scale=strength, normalized_loaded_joint_distance=loaded_distance,
                   fruit_position_distance_m=float(np.linalg.norm(x[16:]-fruit0)),
                   pad_geometry_residual_m=float(np.linalg.norm(contact_geometry(x[:16])-x[16:]-reference)),
                   release_profile=anchor.get("release_profile", "flexion"),
                   release_profiles=anchor.get("release_profiles", {}),
                   release_increment_rad=anchor.get("release_increment_rad", .015),
                   release_clearance_rad=.02)
        rows.append(row)
        if len(rows) % 64 == 0:
            args.output.write_text(json.dumps(rows, indent=2)+"\n")
            print(json.dumps({"generated":len(rows),"target":args.count,"elapsed_seconds":time.monotonic()-started}),flush=True)
    args.output.write_text(json.dumps(rows, indent=2)+"\n")
    args.output.with_name(args.output.stem+"_metadata.json").write_text(json.dumps({
        "seed":args.seed, "source":str(args.anchor), "source_sha256":hashlib.sha256(source).hexdigest(),
        "generated":len(rows), "requested":args.count, "generation_completed":len(rows)==args.count,
        "seed_anchor_count":len(anchors), "seed_anchor_ids":[a["id"] for a in anchors],
        "one_anchor_family_only":True, "physical_validation":False,
        "note":"Distinct local joint/fruit proposals, not independent grasp families or accepted resets.",
        "elapsed_seconds":time.monotonic()-started},indent=2)+"\n")
    if len(rows) != args.count:
        raise RuntimeError("Insufficient distinct proposals; do not pad with duplicates")


if __name__ == "__main__":
    main()
