"""纯CPU中央指肚逆运动学；几何候选不代表物理抓握成功。"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .reachability import load_module,closest_points_on_triangles,DEFAULT_FK_HELPER,FK_HELPER_SHA256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();source=Path(__file__).resolve().parents[2];root=source.parent
    metadata=json.loads((root/'evidence/probe_known_thumb_pad_01/metadata.json').read_text())
    old=json.loads((root/'evidence/reachability_cpu_01/result.json').read_text())
    target=np.array(old['target']['nominal_pregrasp_fruit_center_world_m']);radius=.0335
    config=json.loads((source/'research/thumb_pad_grasp_v1/assets/pad_regions_v1.approved.json').read_text())
    pad=next(d for d in config['digits'] if d['digit']=='thumb')
    geom=np.load(source/'research/thumb_pad_grasp_v1/assets/source_geometry_v1.npz')
    triangles=geom['thumb_vertices_body_m'][geom['thumb_faces']]
    ids=np.array(pad['selected_triangle_ids']);mask=np.zeros(len(triangles),bool);mask[ids]=True
    center,near=closest_points_on_triangles(np.array(pad['center_body_local_m']),triangles[ids])
    central_id=int(ids[int(near[0])]);triangle=triangles[central_id]
    normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);normal/=np.linalg.norm(normal)
    if normal@np.array(pad['mean_outward_normal_body_local'])<0:raise RuntimeError('中央面法向与批准区域不符')
    if hashlib.sha256(DEFAULT_FK_HELPER.read_bytes()).hexdigest()!=FK_HELPER_SHA256:raise RuntimeError('FK helper hash drift')
    fk=load_module(DEFAULT_FK_HELPER,'central_pad_fk_helper')
    joints=fk.read_usd_joints(Path(metadata['task_identity']['assets']['usd_path']))
    names=metadata['robot_body_names'];pose=np.array(metadata['reset_body_poses_w'][0][names.index('base_link')]);root_rot=Rotation.from_quat(pose[3:]).as_matrix();root_pos=pose[:3]
    zero=fk.compute_poses(joints,{j['name']:0. for j in joints if j['type']=='revolute'});zero_rot,zero_pos=zero['base_link']
    mimic=metadata['task_identity']['actuation']['mimic'];hard=metadata['task_identity']['actuation']['joint_hard_limits_rad']
    import torch
    torch.set_num_threads(1)
    from research.small_fruit_pinch_v1.source.pinch_action_manifold import pinch_action_to_joint_targets
    from research.small_fruit_pinch_v1.source.vendor.hand_manifold import LEFT_ACTIVE_JOINT_ORDER,LEFT_SAFE_LOWER_RAD,LEFT_SAFE_UPPER_RAD
    from .action_mapping import encode_thumb_targets
    active=[x.removeprefix('left_hand_') for x in LEFT_ACTIVE_JOINT_ORDER];thumb_names=['thumb_joint'+str(i) for i in range(4)]
    index=[active.index(n) for n in thumb_names];lo=np.array(LEFT_SAFE_LOWER_RAD)[index];hi=np.array(LEFT_SAFE_UPPER_RAD)[index]
    calls={'coupled':0,'independent':0}
    def evaluate(x,mode,full=False):
        calls[mode]+=1
        a=np.zeros(14)
        if mode=='coupled':a[[0,4,5]]=x
        values=pinch_action_to_joint_targets(torch.tensor(a,dtype=torch.float64).reshape(1,14)).numpy()[0]
        q=dict(zip(active,values))
        if mode=='independent':q.update(zip(thumb_names,x))
        for follower,spec in mimic.items():q[follower]=float(np.clip(q[spec['driver']]*spec['multiplier']+spec['offset_rad'],*hard[follower]))
        poses=fk.compute_poses(joints,q);rot,pos=poses['thumb_link4'];R=root_rot@zero_rot.T@rot;t=root_pos+root_rot@zero_rot.T@(pos-zero_pos)
        point=R@center+t;outward=R@normal;residual=(point+radius*outward-target)*1000
        if not full:return residual
        local_target=R.T@(target-t)
        padpoint,pmeta=closest_points_on_triangles(local_target,triangles[mask]);sidepoint,smeta=closest_points_on_triangles(local_target,triangles[~mask])
        surface_r=float(np.linalg.norm(padpoint-local_target));nonpad_r=float(np.linalg.norm(sidepoint-local_target))
        angles=np.array([q[n] for n in thumb_names]);action=np.zeros(14)
        if mode=='independent':action[2:6]=encode_thumb_targets(torch.tensor(angles,dtype=torch.float64)).numpy()
        else:action[[0,4,5]]=x
        return {'mode':mode,'joint0_1_2_3_rad':angles.tolist(),'action14':action.tolist(),'central_target_residual_mm':residual.tolist(),
            'central_target_error_mm':float(np.linalg.norm(residual)),'central_point_body_m':center.tolist(),'central_triangle_id':central_id,
            'central_point_world_m':point.tolist(),'central_normal_world':outward.tolist(),'fruit_world_m':target.tolist(),
            'nearest_PAD_distance_m':surface_r,'nearest_NONPAD_distance_m':nonpad_r,
            'PAD_closer_than_NONPAD':surface_r<nonpad_r,'nonpad_geometric_overlap_m':max(0.,radius-nonpad_r),
            'thumb_body_rotation_world':R.tolist(),'thumb_body_position_world_m':t.tolist()}
    rng=np.random.default_rng(108)
    result={}
    for mode in ['coupled','independent']:
        bounds=([-1.]*3,[1.]*3) if mode=='coupled' else (lo,hi)
        first=np.array([0.,.484375,-.578125]) if mode=='coupled' else np.array([-.195,.614,.331,.594])
        candidates=[]
        for start in [first,*rng.uniform(bounds[0],bounds[1],size=(11,len(first)))]:
            fit=least_squares(lambda x:evaluate(x,mode),np.clip(start,np.array(bounds[0])+1e-7,np.array(bounds[1])-1e-7),bounds=bounds,max_nfev=160,ftol=1e-10,xtol=1e-10,gtol=1e-9)
            candidates.append(evaluate(fit.x,mode,True))
        candidates.sort(key=lambda x:x['central_target_error_mm']);result[mode]=candidates
        print(mode,json.dumps(candidates[0]),flush=True)
    payload={'status':'CPU_GEOMETRY_ONLY_NOT_PHYSICAL_SUCCESS','target_original_world_m':target.tolist(),'central_triangle_id':central_id,
        'calls':calls,'elapsed_s':time.monotonic()-started,'solutions':result,'scope':'thumb_link4 full original mesh PAD vs NONPAD; other thumb bodies and actual cooked hull require real physical validation',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'result.json').write_text(json.dumps(payload,indent=2))

if __name__=='__main__':main()
