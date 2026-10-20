"""Offline palm-near seed generation. Every result still requires PhysX/N-1.

Uses original URDF FK and effective-pad geometry; no asset modification or
controller. Deliberately constrains the object closer to the palm rather than
continuing to optimize only the previous elevated thumb/ring pinch family.
"""
from pathlib import Path
import argparse, importlib.util, json, itertools, time
import numpy as np
from scipy.optimize import least_squares
from simulation.roto2_pressure.core import ACTIVE_JOINTS, MIMICS
from simulation.roto2_pressure.reward_geometry import pad_frames

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=2087);p.add_argument('--starts',type=int,default=8)
    args=p.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('original_grasp_geometry',ROOT/'simulation/.runtime/setup/search_grasp.py')
    g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    centers,_,_,normals=[x.numpy() for x in pad_frames('cpu')]
    lo=g.lower[:16].copy();hi=g.upper[:16].copy()
    for f,(d,r) in MIMICS.items():
        i=ACTIVE_JOINTS.index(d);lo[i]=max(lo[i],g.limits[f][0]/r+.002);hi[i]=min(hi[i],g.limits[f][1]/r-.002)
    lower=np.r_[lo,[.035,-.035,.105]];upper=np.r_[hi,[.060,.035,.175]]
    base=json.loads((ROOT/'simulation/roto2_pressure/reset_pad_grasps_palm_up.json').read_text())[0]
    rng=np.random.default_rng(args.seed);results=[];started=time.monotonic()
    supports=list(itertools.combinations(range(5),4))+[tuple(range(5))]
    for support in supports:
        for trial in range(args.starts):
            q0=np.array(base['joint_positions'])
            for finger in ['index','middle','ring','little']:
                q0[ACTIVE_JOINTS.index(finger+'_joint0')]=rng.uniform(-.15,.15)
                q0[ACTIVE_JOINTS.index(finger+'_joint1')]=rng.uniform(.85,1.4)
                q0[ACTIVE_JOINTS.index(finger+'_joint2')]=rng.uniform(.35,1.0)
            fruit0=np.array([rng.uniform(.040,.055),rng.uniform(-.02,.02),rng.uniform(.115,.165)])
            def residual(x):
                tf=g.fk(x[:16]);positions=[];directions=[]
                for i in support:
                    t=tf[g.tips[i]];positions.append(t[:3,3]+t[:3,:3]@centers[i]);directions.append(t[:3,:3]@normals[i])
                positions=np.array(positions);directions=np.array(directions)
                pad_errors=(positions+.032*directions-x[16:])*100
                # Avoid resting entirely on one upward-facing contact.
                upward= directions[:,0]
                leave_one_up=[max(0,.15-np.delete(upward,i).mean()) for i in range(len(support))]
                separation=[max(0,.016-np.linalg.norm(positions[i]-positions[j]))*100
                            for i,j in itertools.combinations(range(len(support)),2)]
                # Cheap conservative hull separating-plane collision prior.
                collisions=[]
                for name,(ns,off) in g.planes.items():
                    t=tf[name];local=t[:3,:3].T@(x[16:]-t[:3,3])
                    collisions.append(min(0,np.max(ns@local+off)-.0305)*30)
                return np.r_[pad_errors.ravel(),leave_one_up,separation,collisions,
                             .005*(x[:16]-q0)/(hi-lo),2*(x[16:]-fruit0)]
            start=np.clip(np.r_[q0,fruit0],lower+1e-6,upper-1e-6)
            fit=least_squares(residual,start,bounds=(lower,upper),max_nfev=120,diff_step=1e-4)
            tf=g.fk(fit.x[:16]);errors=[]
            for i in range(5):
                t=tf[g.tips[i]];errors.append(float(np.linalg.norm(t[:3,3]+t[:3,:3]@(centers[i]+.032*normals[i])-fit.x[16:])))
            row=dict(id=f'palm_ik_{args.seed}_{"".join(map(str,support))}_{trial}',
                joint_positions=fit.x[:16].tolist(),fruit_position_hand=fit.x[16:].tolist(),
                support_fingers=list(support),pad_target_error_m=errors,cost=float(np.linalg.norm(residual(fit.x))),
                evaluations=fit.nfev,hand_posture='palm_up',proposal_only=True,physical_validation=False)
            results.append(row);results.sort(key=lambda r:r['cost'])
            temporary=args.output.with_suffix('.tmp');temporary.write_text(json.dumps(results,indent=2)+'\n');temporary.replace(args.output)
            print(json.dumps(dict(completed=len(results),total=len(supports)*args.starts,
                best_cost=results[0]['cost'],last_cost=row['cost'],last_support=support,
                last_errors_mm=[round(errors[i]*1000,2) for i in support],elapsed_s=time.monotonic()-started)),flush=True)

if __name__=='__main__':main()
