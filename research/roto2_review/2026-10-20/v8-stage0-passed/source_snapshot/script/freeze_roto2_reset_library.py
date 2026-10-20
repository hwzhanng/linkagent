"""Freeze only distinct grasps with matched N-1 and long-hold evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from simulation.roto2_pressure.core import ACTIVE_JOINTS, MIMICS, FINGERS

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=ROOT/"artifacts/simulation/rotation_v8/stage0")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=256)
    args = p.parse_args()
    pairs = [
        ("robust_neighborhood_intersection.json", "robust141_hold30s_seed2114.json"),
        ("expanded_robust_intersection.json", "robust196_hold30s_seed2118.json"),
    ]
    pool, evidence, validation_cache = [], {}, {}
    counts = []
    for input_name, hold_name in pairs:
        inputs = json.loads((args.base/input_name).read_text())
        hold = json.loads((args.base/hold_name).read_text())
        assert hold["validation_completed"] and hold["no_automatic_resets"]
        assert hold["duration_seconds"] >= 30 and len(inputs) == len(hold["results"])
        evidence[input_name], evidence[hold_name] = sha(args.base/input_name), sha(args.base/hold_name)
        kept = 0
        for candidate, result in zip(inputs, hold["results"]):
            if not result["stable_after_settling"] or result["physical_failure"]:
                continue
            checks = []
            for name in candidate["validation_sources"]:
                if name not in validation_cache:
                    source = args.base/name
                    doc = json.loads(source.read_text())
                    assert doc["validation_completed"]
                    evidence[name] = sha(source)
                    protocol_path = source.with_name(source.stem+"_protocol.json")
                    protocol = json.loads(protocol_path.read_text())
                    evidence[protocol_path.name] = sha(protocol_path)
                    validation_cache[name] = (doc, {r["candidate_id"]:r for r in doc["results"]}, protocol)
                doc, rows_by_id, protocol = validation_cache[name]
                row = rows_by_id[candidate["id"]]
                assert row["baseline_ok"] and row["n_minus_one_passed"]
                assert all(t["actually_released"] and t["stable_through_release_and_recovery"]
                           and not t["physical_failure"] for t in row["trials"])
                checks.append({"source":name, "seed":doc["seed"],
                               "density_kg_m3":protocol["matched_density_kg_m3"][0]})
            assert sorted(c["density_kg_m3"] for c in checks) == [850, 950, 1050]
            assert len({c["seed"] for c in checks}) == 3
            pool.append({
                **candidate, "proposal_only":False, "physical_validation":True,
                "long_hold_pending":False, "long_hold_stability_passed":True,
                "long_hold_source":hold_name, "long_hold_env":result["env"],
                "long_hold_four_pad_time_fraction":result["four_pad_time_fraction"],
                "long_hold_four_pad_occupancy_passed":result["passed"],
                "verified_n1_conditions":checks,
                "validation_scope":"Three independent density/seed N-1 tests plus continuous30s object stability. Local variation of one ancestral grasp family; nominal64mm sphere only.",
            })
            kept += 1
        counts.append({"input":input_name,"input_count":len(inputs),"stable_30s_count":kept})
    unique = {}
    for row in pool:
        key = tuple(row["joint_positions"]+row["joint_targets"]+row["fruit_position_hand"])
        unique.setdefault(key, row)
    pool = list(unique.values())
    assert len(pool) >= args.count, f"Only {len(pool)} verified distinct grasps"
    urdf = ROOT/"botSDK/urdf/01_vendor_original/02_extracted_packages/L20_8_r_s1/L20_8_r_s1.urdf"
    limits = {}
    for node in ET.parse(urdf).getroot().findall("joint"):
        lim = node.find("limit")
        if lim is not None:
            limits[node.get("name")] = [float(lim.get("lower")), float(lim.get("upper"))]
    bounds = np.asarray([limits[name] for name in ACTIVE_JOINTS])
    for follower, (driver, ratio) in MIMICS.items():
        i = ACTIVE_JOINTS.index(driver)
        bounds[i, 0] = max(bounds[i, 0], limits[follower][0]/ratio)
        bounds[i, 1] = min(bounds[i, 1], limits[follower][1]/ratio)
    scale = bounds[:, 1]-bounds[:, 0]
    support = np.asarray(pool[0]["reference_pad_fingers"])
    assert all(np.array_equal(row["reference_pad_fingers"], support) for row in pool)
    loaded = [i for i, name in enumerate(ACTIVE_JOINTS)
              if name.split("_")[0] in [FINGERS[j] for j in np.where(support)[0]]]
    # Diversity uses actual load-bearing joint states/targets and fruit center.
    # Unused-finger variation alone cannot drive the selection.
    vectors = np.asarray([
        np.r_[np.asarray(row["joint_positions"])[loaded]/scale[loaded],
              np.asarray(row["joint_targets"])[loaded]/scale[loaded],
              np.asarray(row["fruit_position_hand"])/.01] for row in pool])
    chosen = [int(np.argmax([r["long_hold_four_pad_time_fraction"] for r in pool]))]
    distance = np.full(len(pool), np.inf)
    while len(chosen) < args.count:
        distance = np.minimum(distance, np.linalg.norm(vectors-vectors[chosen[-1]],axis=1))
        distance[chosen] = -1
        chosen.append(int(distance.argmax()))
    selected = [pool[i] for i in chosen]
    v = vectors[chosen]
    pair_distance = np.linalg.norm(v[:, None]-v[None], axis=-1)
    np.fill_diagonal(pair_distance, np.inf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2)+"\n")
    report = {
        "stage0_library_gate_passed":True, "training_started":False,
        "candidate_pool":counts, "distinct_verified_pool":len(pool), "frozen_count":len(selected),
        "library":str(args.output), "library_sha256":sha(args.output),
        "selection":"Farthest-point sampling over loaded-joint positions/targets normalized by legal ranges and fruit center; no duplicated rows.",
        "nearest_neighbor_distance_quantiles":np.quantile(pair_distance.min(1),[0,.1,.5,.9,1]).tolist(),
        "nominal_sphere_diameter_m":.064, "n1_densities_kg_m3":[850,950,1050],
        "long_hold_seconds":30,
        "long_hold_acceptance":"Stable object and no physical failure; perpetual four-pad occupancy is separately reported, not required by the N-1 task.",
        "one_ancestral_grasp_family":True,
        "limitations":"Does not establish continuous rotation, different object shapes, a learned gait, or pressure-gradient benefit.",
        "evidence_sha256":evidence,
    }
    (args.base/"FROZEN_LIBRARY_GATE.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="evidence_sha256"},indent=2))


if __name__ == "__main__":
    main()
