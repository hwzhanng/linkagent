"""用真实持握记录做有界拇指PAD中心几何修正，仅CPU候选而非物理成功。"""
from pathlib import Path
import json,importlib.util,sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from research.firm_grasp_v1.reachability_scan import world_poses

root=Path(__file__).resolve().parents[2];ev=root.parent/'evidence'
result=json.loads((ev/'F1_middle_probe05/result.json').read_text())
rows=[json.loads(l) for l in (ev/'F1_middle_probe05/physical_substeps.jsonl').open()]
r=next(x for x in rows if x['episode_policy_step'][1]==130);env_id=1
helper=Path('/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-grasp-learning-20261013/evidence/kinematics_cpu_probe.py')
spec=importlib.util.spec_from_file_location('firm_thumb_fk',helper);fk=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fk;spec.loader.exec_module(fk)
joints=fk.read_usd_joints(Path(result['task_identity']['assets']['usd_path']))
names=result['initial']['joint_names'];q=np.array(r['joint_pos'][env_id]);original=dict(zip(names,q));base=result['initial']['body_pose_w'][env_id][result['initial']['body_names'].index('base_link')]
approved=json.loads((root/'research/thumb_pad_grasp_v1/assets/pad_regions_v1.approved.json').read_text())
pad=next(x for x in approved['digits'] if x['digit']=='thumb');center=np.array(pad['center_body_local_m']);normal=np.array(pad['mean_outward_normal_body_local']);normal/=np.linalg.norm(normal)
fruit=np.array(r['fruit_pose_w'][env_id][:3]);target=center+.0335*normal
thumb_names=['thumb_joint0','thumb_joint1','thumb_joint2','thumb_joint3'];q0=np.array([original[n] for n in thumb_names])
ref=np.array([.05549413117585235,1.3393746570587945,.4531111979189145,.10039140509741924]);span=np.array([.25,.35,.3,.65])
lo=np.maximum(np.array([-1.,0.,.02,.025294118]),ref-span)-q0;hi=np.minimum(np.array([.45,1.38,.8,1.25]),ref+span)-q0
lo=np.maximum(lo,-.20);hi=np.minimum(hi,.20)
def pose(delta):
 qs=original.copy()
 for n,v in zip(thumb_names,q0+delta):qs[n]=float(v)
 qs['thumb_joint4']=float(original['thumb_joint4']+1.11*delta[3])
 wp=world_poses(fk,fk.compute_poses(joints,qs),base)
 return wp['thumb_link4']
def objective(delta):
 rot,pos=pose(delta);local=rot.T@(fruit-pos)
 return np.concatenate([local-target,.002*delta])
fit=least_squares(objective,np.zeros(4),bounds=(lo,hi),max_nfev=80,xtol=1e-10,ftol=1e-10,gtol=1e-10)
base_action=result['declared_actions'][1];candidates=[]
for scale in [.5,.75,1.,1.25]:
 action=np.array(base_action);action[2:6]+=scale*fit.x/span;action=np.clip(action,-1,1)
 candidates.append(action.tolist())
rout={'status':'CPU_CANDIDATES_NOT_PHYSICAL_SUCCESS','source_probe':'F1_middle_probe05','env_id':1,'sample_age':130,'original_q':q0.tolist(),'delta_q':fit.x.tolist(),'initial_local_fruit_error_m':objective(np.zeros(4))[:3].tolist(),'predicted_local_fruit_error_m':objective(fit.x)[:3].tolist(),'actions':candidates,'all_original_pad_geometry_unchanged':True}
(ev/'F1_THUMB_REPOSITION.json').write_text(json.dumps(rout,indent=2)+'\n');(ev/'F1_actions_group06.json').write_text(json.dumps(candidates,indent=2)+'\n')
print(json.dumps(rout,indent=2))
