"""离线独立重算G1真实子步窗和原完整三角网格PAD标签；不启动仿真。"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from .contract import load_frozen_contract

COMBINATIONS=((0,1,2),(0,1,3),(0,1,4),(0,2,3),(0,2,4),(0,3,4))
DIGITS=('thumb','index','middle','ring','little')
ORIGINAL = Path('/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/ppo-thumb-pad-v1-20261016/delivery/ppo-thumb-pad-v1_6-20261017/verification/original_pad_geometry.py')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def edge_distance(points, edges):
    start,end=edges[:,0],edges[:,1];delta=end-start
    t=((points[:,None]-start)*delta).sum(2)/np.maximum((delta*delta).sum(1),1e-24)
    q=start+np.clip(t,0,1)[...,None]*delta
    return np.linalg.norm(points[:,None]-q,axis=2).min(1)


def assert_ground_free_window(rows, env):
    """S5 独立地面支撑排除：始终从原始pose重算，并核对缓存一致。"""

    values=[]
    for r in rows:
        if 'fruit_pose_w' not in r:
            raise AssertionError('S5轨迹缺少fruit_pose_w，不能独立重算地面净空')
        recomputed=float(r['fruit_pose_w'][env][2])-.0335
        if not np.isfinite(recomputed):
            raise AssertionError('fruit_pose_w 地面净空含非有限值')
        if 'fruit_bottom_clearance_m' in r:
            cached=float(r['fruit_bottom_clearance_m'][env])
            if not np.isfinite(cached) or abs(cached-recomputed)>1e-6:
                raise AssertionError('fruit_bottom_clearance_m 与原始pose重算不一致')
        if 'ground_free' in r:
            if bool(r['ground_free'][env]) != (recomputed>.005):
                raise AssertionError('ground_free 与原始pose重算不一致')
        values.append(recomputed)
    bottom=np.asarray(values,dtype=float)
    if not np.isfinite(bottom).all():
        raise AssertionError('地面净空含非有限值')
    if not (bottom>.005).all():
        raise AssertionError('不能排除地面支撑')
    return bottom


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    contract=load_frozen_contract()
    required=contract.required_substeps
    if args.output.exists():raise FileExistsError(args.output)
    rows=[];raw_lines=[];selected=None;peaks=None
    with args.trace.open() as stream:
        for line in stream:
            outer=json.loads(line);row=outer.get('record',outer)
            rows.append(row);raw_lines.append(line)
            if 'firm_streak' not in row:
                raise ValueError('轨迹缺少新三指守卫字段，不能用于G1验收')
            streak=np.asarray(row['firm_streak'])
            peaks=streak.copy() if peaks is None else np.maximum(peaks,streak)
            where=np.argwhere(streak>=required)
            if selected is None and len(where):
                env,combo=map(int,where[0]);selected=(len(rows)-1,env,combo)
    if selected is None:
        args.output.write_text(json.dumps({'status':'NO_G1_WINDOW','max_streak_by_env_combo':peaks.tolist() if peaks is not None else [],'trace':str(args.trace)},indent=2)+'\n')
        return 2
    end,env,combo=selected;window=rows[end-required+1:end+1];trio=COMBINATIONS[combo]
    assert len(window)==required and all(r['sim_step']==window[0]['sim_step']+k for k,r in enumerate(window))
    assert all(abs(r['dt_s']-1/120)<1e-12 for r in window)
    assert all(r['firm_streak'][env][combo]==k+1 for k,r in enumerate(window))
    assert all(r['released'][env] for r in window)
    expected_faults={'overload','nonfinite','penetration','joint_limit','self_collision','instability','joint_effort'}
    assert all(expected_faults <= set(r['hard_faults']) for r in window), '七硬故障字段缺失'
    assert all(not any(v[env] for v in r['hard_faults'].values()) for r in window)
    assert all(not any(env in c['env_ids'] for c in r['invalid_contact_channels']) for r in window), '无效接触通道'
    assert all(np.asarray(r['test_disturbance_force_w'][env]).shape==(3,) and np.all(np.asarray(r['test_disturbance_force_w'][env])==0) and np.asarray(r['test_disturbance_torque_w'][env]).shape==(3,) and np.all(np.asarray(r['test_disturbance_torque_w'][env])==0) for r in window), 'F2须为无显式扰动窗口'
    assert all(np.all(np.asarray(r['support_force_w'][env])==0) and np.all(np.asarray(r['support_torque_w'][env])==0) for r in window)
    assert all(not any(r['truth']['unknown'][env]) and np.all(np.asarray(r['truth']['nonpad_force_n'][env])==0) and r['palm_force_n'][env]==0 for r in window)
    force=np.asarray([r['truth']['pad_force_n'][env] for r in window]);share=force/force.sum(1)[:,None]
    assert (force[:,trio]>=contract.f_pad_min_n).all() and (share[:,trio]>=contract.min_load_share).all()
    speed=np.asarray([r['relative_speed_m_s'][env] for r in window]);position=np.asarray([r['fruit_in_hand_m'][env] for r in window])
    drift=np.linalg.norm(position-position[0],axis=1)
    assert (speed<=.05).all() and (drift<=.01).all()
    # 使用冻结的未加速完整网格实现，独立于运行时AABB剪枝分类器。
    assert sha(ORIGINAL)=='14d7412e72af348e04d56e3452c7f9b34a135a9f2777de623e8faf4d7eea58bd'
    spec=importlib.util.spec_from_file_location('firm_original_full_mesh',ORIGINAL)
    geom=importlib.util.module_from_spec(spec);sys.modules[spec.name]=geom;spec.loader.exec_module(geom)
    assets=Path(__file__).resolve().parents[1]/'thumb_pad_grasp_v1/assets'
    approved=json.loads((assets/'pad_regions_v1.approved.json').read_text())
    assert sha(assets/'pad_regions_v1.approved.json')=='abd16607249509762ac07483f9542b6b460628beee5b13a464fca31217b04258'
    geometry=np.load(assets/'source_geometry_v1.npz');config=json.loads((assets/'runtime_contact_v2.json').read_text())
    full_mesh={};layout=[]
    for digit_id in trio:
        name=DIGITS[digit_id];definition=next(d for d in approved['digits'] if d['digit']==name)
        body=definition['body_path'].rsplit('/',1)[-1]
        local=[];normal=[];world_local=[];world_normal=[];centroids=[];point_loads=[]
        for r in window:
            points=[];weights=[]
            for c in r['contacts']:
                if c['body']!=body:continue
                for j,i in enumerate(c['env_ids']):
                    if i!=env:continue
                    assert c['forces_n'][j]>0 and c['labels'][j]==1
                    pose=np.asarray(c['body_pose_w'][j]);rot=Rotation.from_quat(pose[3:7])
                    local.append(c['points_body_m'][j]);normal.append(c['outward_normals_body'][j])
                    world_local.append(rot.inv().apply(np.asarray(c['points_w'][j])-pose[:3]))
                    world_normal.append(rot.inv().apply(-np.asarray(c['raw_normals_w'][j])))
                    weights.append(c['forces_n'][j]);points.append(c['points_w'][j]);point_loads.append(c['forces_n'][j])
            assert weights and abs(sum(weights)-r['truth']['pad_force_n'][env][digit_id])<1e-5
            centroids.append(np.average(np.asarray(points),axis=0,weights=weights))
        surface=geom.PadSurface(geometry[name+'_vertices_body_m'],geometry[name+'_faces'],definition['selected_triangle_ids'])
        p=np.asarray(local);n=np.asarray(normal)
        kwargs={'position_tolerance_m':config['thumb_position_tolerance_m'] if digit_id==0 else config['other_position_tolerance_m'],
                'normal_tolerance_rad':math.radians(config['thumb_normal_tolerance_deg'] if digit_id==0 else config['other_normal_tolerance_deg'])}
        a=geom.classify(surface,p,n,**kwargs);b=geom.classify(surface,np.asarray(world_local),np.asarray(world_normal),**kwargs)
        assert all(x=='PAD' for x in a['state']) and all(x=='PAD' for x in b['state'])
        distance=edge_distance(a['closest_point_body_m'],np.asarray(definition['boundary_edges_body_local_m']))
        assert (distance>=config['boundary_uncertainty_m']).all()
        full_mesh[name]={'points':len(p),'all_original_full_mesh_PAD':True,'world_to_body_recomputed_PAD':True,
            'max_source_distance_m':float(np.max(a['distance_m'])),'min_approved_boundary_distance_m':float(distance.min()),
            'max_world_body_coordinate_error_m':float(np.linalg.norm(p-np.asarray(world_local),axis=1).max())}
        layout.append(np.asarray(centroids))
    fruit=np.asarray([r['fruit_pose_w'][env][:3] for r in window]);radial=[v-fruit for v in layout]
    bottom=assert_ground_free_window(window,env)
    radial=[v/np.linalg.norm(v,axis=1)[:,None] for v in radial]
    opposite=radial[1]+radial[2];opposite/=np.linalg.norm(opposite,axis=1)[:,None]
    dot=(radial[0]*opposite).sum(1);separation=np.linalg.norm(layout[1]-layout[2],axis=1)
    window_path=args.output.with_name(args.output.stem+'_window.jsonl')
    with window_path.open('x') as stream:stream.writelines(raw_lines[end-required+1:end+1])
    result={'status':'PASS_G1_ORIGINAL_FULL_MESH_THREE_PAD_120_REAL_SUBSTEPS',
        'trace':str(args.trace),'env_id':env,'combo_index':combo,'digits':[DIGITS[i] for i in trio],
        'sim_step_start':window[0]['sim_step'],'sim_step_end':window[-1]['sim_step'],'substeps':required,
        'dt_sum_s':sum(r['dt_s'] for r in window),'window_file':str(window_path),'window_sha256':sha(window_path),
        'min_PAD_force_n':dict(zip(DIGITS,force.min(0).tolist())),
        'max_PAD_force_n':dict(zip(DIGITS,force.max(0).tolist())),
        'min_load_share':dict(zip(DIGITS,share.min(0).tolist())),
        'max_relative_speed_m_s':float(speed.max()),'max_hand_anchor_drift_m':float(drift.max()),
        'min_fruit_bottom_clearance_m':float(bottom.min()),'ground_plane_z_m':0.,
        'support_all_components_zero':True,'all_recorded_runtime_hard_fault_flags_false':True,
        'full_mesh':full_mesh,'geometry_diagnostic_only_for_G1':{'min_nonthumb_centroid_separation_m':float(separation.min()),'max_thumb_opposition_dot':float(dot.max())},
        'frozen_contract':contract.__dict__,'frozen_contract_sha256':sha(Path(__file__).with_name('firm_grasp_contract_v1.json')),'all_required_seven_fault_fields_present':True,'selected_env_invalid_channels_absent':True,'test_disturbance_all_components_zero':True,'selection':'first_qualifying_env_combo_only','original_classifier_sha256':sha(ORIGINAL),'verifier_source_sha256':sha(Path(__file__)),
        'no_F3_disturbance_or_long_hold_claim':True}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'env_id':env,'digits':result['digits'],'output':str(args.output)}))
    return 0


if __name__=='__main__':raise SystemExit(main())
