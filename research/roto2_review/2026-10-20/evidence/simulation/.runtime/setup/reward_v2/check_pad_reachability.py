"""Offline diagnostic only: do not change resets or control the trained policy."""
import sys,importlib.util,json,itertools
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('grasp',ROOT/'simulation/.runtime/setup/search_grasp.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
from simulation.roto2_pressure.reward_geometry import pad_frames
centers,_,_,normals=[x.numpy() for x in pad_frames('cpu')]
base=json.loads((ROOT/'simulation/roto2_pressure/reset_grasps.json').read_text())[0]
fruit=np.array(base['fruit_position_hand']);q0=np.array(base['joint_positions']);lo=g.lower[:16].copy();hi=g.upper[:16].copy()
for f,(d,r) in g.MIMICS.items():
 i=g.ACTIVE_JOINTS.index(d);lo[i]=max(lo[i],g.limits[f][0]/r+.002);hi[i]=min(hi[i],g.limits[f][1]/r-.002)
results=[]
for support in [(0,1,2),(0,1,4)]:
 def residual(q):
  tf=g.fk(q);c=np.array([tf[name][:3,3]+tf[name][:3,:3]@centers[i] for i,name in enumerate(g.tips)]);n=np.array([tf[name][:3,:3]@normals[i] for i,name in enumerate(g.tips)])
  errors=(c[list(support)]+n[list(support)]*.032-fruit)*100
  # The original reset's all-link penetration diagnostics are kept as a soft bound.
  original=g.residual(np.r_[q,fruit]);collision=original[5:5+len(g.names)]
  return np.r_[errors.flatten(),collision*.5,.01*(q-q0)]
 fit=least_squares(residual,np.clip(q0,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=180,diff_step=1e-4)
 tf=g.fk(fit.x);c=np.array([tf[name][:3,3]+tf[name][:3,:3]@centers[i] for i,name in enumerate(g.tips)]);n=np.array([tf[name][:3,:3]@normals[i] for i,name in enumerate(g.tips)])
 errors=np.linalg.norm(c+n*.032-fruit,axis=-1)
 r={'support_fingers':support,'pad_target_error_mm':(errors*1000).tolist(),'cost':float(np.linalg.norm(residual(fit.x))),'joint_positions':fit.x.tolist(),'fruit_position_hand':fruit.tolist(),'physical_validation':False};results.append(r);print(json.dumps(r),flush=True)
(ROOT/'simulation/.runtime/setup/reward_v2/pad_reachability.json').write_text(json.dumps(results,indent=2)+'\n')
