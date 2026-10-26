"""Broad IK proposals with joint travel reserve; every result needs PhysX validation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]


def initialize(mode='normal'):
    global g, center, normal, lower, upper, scale, anchors, ACTIVE_JOINTS, contact_mode
    contact_mode=mode
    from simulation.roto2_pressure.core import ACTIVE_JOINTS, MIMICS
    from simulation.roto2_pressure.reward_geometry import pad_frames
    spec = importlib.util.spec_from_file_location('fk', ROOT/'simulation/.runtime/setup/search_grasp.py')
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    center, _, _, normal = [v.numpy() for v in pad_frames('cpu')]
    lower, upper = g.lower[:16].copy(), g.upper[:16].copy()
    for follower, (driver, ratio) in MIMICS.items():
        i = ACTIVE_JOINTS.index(driver)
        lower[i] = max(lower[i], g.limits[follower][0]/ratio)
        upper[i] = min(upper[i], g.limits[follower][1]/ratio)
    scale = upper-lower
    anchors = json.loads((ROOT/'simulation/roto2_pressure/reset_grasps_v8_validated256.json').read_text())


def solve(index):
    rng = np.random.default_rng(93000+index)
    support = list(itertools.combinations(range(5), 4))[index % 5]
    anchor = anchors[(index*17) % len(anchors)]
    q0 = np.asarray(anchor['joint_positions'])
    fruit0 = np.asarray(anchor['fruit_position_hand'])
    if contact_mode in ('reference','collision_reference'):
        support=tuple(np.flatnonzero(anchor['reference_pad_fingers']))
    tf0=g.fk(q0)
    reference=np.asarray([tf0[g.tips[i]][:3,3]+tf0[g.tips[i]][:3,:3] @ (center[i]+.032*normal[i]) for i in support])-fruit0
    def clearance(tf,fruit):
        distances=[]
        for link,(normals,offset) in g.planes.items():
            local=tf[link][:3,:3].T@(fruit-tf[link][:3,3])
            near=g.trimesh.triangles.closest_point(g.triangles[link],np.broadcast_to(local,(len(g.triangles[link]),3)))
            distance=np.linalg.norm(near-local,axis=-1).min()
            distances.append(distance if np.max(normals@local+offset)>=0 else -distance)
        return np.asarray(distances)
    collision_reference=np.minimum(clearance(tf0,fruit0),.031)-.00025 if contact_mode=='collision_reference' else None
    driving = np.array([i for i,n in enumerate(ACTIVE_JOINTS)
                        if int(n[-1]) in ((1,2,3) if n.startswith('thumb') else (1,2))])
    lo = np.r_[lower+1e-5, [.025,-.045,.10]]
    hi = np.r_[upper-1e-5, [.115,.045,.235]]
    desired = fruit0 + rng.uniform(-1,1,3)*[.012,.018,.025]
    qstart = (1-.25*(index%3))*q0 + .25*(index%3)*(lower+upper)/2
    qstart += rng.normal(size=16)*scale*.08

    def residual(x):
        tf = g.fk(x[:16])
        predicted = np.asarray([tf[g.tips[i]][:3,3]+tf[g.tips[i]][:3,:3] @ (center[i]+.032*normal[i]) for i in support])
        reserve = np.minimum(x[:16]-lower, upper-x[:16])[driving]
        geometric=predicted-x[16:]-(reference if contact_mode!='normal' else 0)
        collision=np.maximum(collision_reference-clearance(tf,x[16:]),0)/.00025 if collision_reference is not None else np.empty(0)
        return np.r_[(geometric/.001).ravel(), collision,
                     2*np.maximum(.12-reserve,0)/.12,
                     .08*(x[:16]-q0)/scale, .15*(x[16:]-desired)/.02]

    fit = least_squares(residual, np.clip(np.r_[qstart,desired],lo+1e-6,hi-1e-6),
                        bounds=(lo,hi), max_nfev=100, diff_step=1e-4)
    q, fruit = fit.x[:16], fit.x[16:]
    yaw = (index % 12)*np.pi/6
    preload=np.asarray(anchor['joint_targets'])-q0
    return dict(id=f'reserve_ik_{contact_mode}_{index}', joint_positions=q.tolist(),
                joint_targets=np.clip(q+preload,lower,upper).tolist(),
                fruit_position_hand=fruit.tolist(), fruit_quaternion_xyzw=[0,0,float(np.sin(yaw/2)),float(np.cos(yaw/2))],
                hand_posture='palm_up', proposed_support_indices=[int(i) for i in support],
                proposal_only=True, physical_validation=False, n_minus_one_passed=False,
                generator='wide_position_joint_reserve_ik_'+contact_mode,
                source_anchor_id=anchor['id'],
                active_joint_min_margin_rad=float(np.minimum(q-lower,upper-q)[driving].min()),
                geometry_residual_norm_mm=float(np.linalg.norm(residual(fit.x)[:12])),
                release_profile=anchor.get('release_profile','flexion'),
                release_profiles=anchor.get('release_profiles',{}),
                release_increment_rad=.015, release_clearance_rad=.02,
                collision_clearance_prior_used=collision_reference is not None,
                optimizer_nfev=fit.nfev)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=160);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--mode',choices=('normal','reference','collision_reference'),default='normal')
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(a.mode,)) as pool:
        for row in pool.map(solve,range(a.count)):
            rows.append(row)
            if len(rows)%20==0:
                print(json.dumps(dict(completed=len(rows),total=a.count,seconds=time.monotonic()-started)),flush=True)
    rows.sort(key=lambda x:x['geometry_residual_norm_mm'])
    a.output.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(completed=len(rows),best_geometry_error_mm=rows[0]['geometry_residual_norm_mm'],physical_validation=False)),flush=True)


if __name__=='__main__': main()
