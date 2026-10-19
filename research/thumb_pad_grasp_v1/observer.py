"""同一真实接触缓冲生成逐点标签、载荷及新成功条件；未知不作指肚。"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import warp as wp
from isaaclab.utils.math import quat_apply_inverse
from .packed_contacts import PhysicalStepContactCache, unpack_points
from .pad_geometry import PadSurface, classify
from .pad_contacts import summarize_contact_loads, PAD, NON_PAD, UNKNOWN


class PadObserver:
    def __init__(self, env):
        self.env = env
        assets = Path(__file__).parent/'assets'
        self.config = json.loads((assets/'runtime_contact_v2.json').read_text())
        approved_path = assets/'pad_regions_v1.approved.json'
        if self.config.get('usable_for_training') is not True:
            raise RuntimeError('运行时接触校准契约未允许使用')
        if hashlib.sha256(approved_path.read_bytes()).hexdigest() != self.config.get('geometry_approval_sha256'):
            raise RuntimeError('运行时标定引用的用户确认几何哈希不符')
        approved = json.loads(approved_path.read_text())
        geometry = np.load(assets/'source_geometry_v1.npz')
        self.bindings = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.iterdir() if p.is_file()}
        self.digits = ('thumb','index','middle','ring','little')
        self.surfaces = {}
        self.boundaries = {}
        for digit in approved['digits']:
            name = digit['digit']
            body = digit['body_path'].rsplit('/',1)[-1]
            self.surfaces[body] = PadSurface(geometry[name+'_vertices_body_m'],geometry[name+'_faces'],digit['selected_triangle_ids'])
            self.boundaries[body] = np.asarray(digit['boundary_edges_body_local_m'])
        self.cache = PhysicalStepContactCache(lambda:env._sim_step_counter)
        for sensor in [*env._raw_penetration_sensors,*env._self_penetration_sensors]:
            sensor._thumb_pad_contact_read = lambda s=sensor:self.cache.read(s)
        self.latest = None
        self.counter = None
        self.details = []
        self.invalid_details = []

    @torch.inference_mode()
    def reset(self, env_ids):
        self.cache.invalidate()
        self.counter = None
        if self.latest is not None:
            for value in self.latest.values():
                if isinstance(value,torch.Tensor):value[env_ids] = 0

    def _boundary_distance(self, points, edges):
        start,end = edges[:,0],edges[:,1]
        delta = end-start
        amount = np.sum((points[:,None]-start)*delta,axis=2)/np.maximum(np.sum(delta*delta,axis=1),1e-24)
        closest = start + np.clip(amount,0,1)[...,None]*delta
        return np.linalg.norm(points[:,None]-closest,axis=2).min(axis=1)

    def sample(self):
        env = self.env
        counter = int(env._sim_step_counter)
        if self.counter == counter and self.latest is not None:return self.latest
        labels,loads,env_ids,digit_ids = [],[],[],[]
        details = []
        invalid_details = []
        invalid_env = torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        seen = set()
        for sensor in env._raw_penetration_sensors:
            raw = self.cache.read(sensor)
            arrays = [wp.to_torch(value) for value in raw]
            paths = list(sensor.body_physx_view.prim_paths)
            body = paths[0].rsplit('/',1)[-1]
            packed = unpack_points(arrays,paths=paths,num_envs=env.num_envs,capacity=int(sensor.cfg.max_contact_data_count_per_prim),invalid_policy="mark_unknown")
            invalid_env |= packed.invalid_by_env
            if bool(packed.invalid_by_env.any()):
                invalid_details.append({"body":body,"env_ids":torch.nonzero(packed.invalid_by_env).flatten().tolist(),"reason":"INVALID_PACKED_POINT_FORCE_OR_NORMAL","raw_forces_repr":repr([arrays[0][int(start[0]):int(start[0])+int(count[0])].tolist() for count,start in zip(arrays[4],arrays[5]) if int(count[0])>0])})
            seen.add(body)
            prefix = body.split('_')[0]
            if prefix not in self.digits:continue
            digit = self.digits.index(prefix)
            # 正好为零的 manifold 槽位是预测接触，不是承载点。仍由 unpack 检查其有限性。
            active = packed.forces_n > 0
            if not bool(active.any()):continue
            ids = packed.env_ids[active]
            force = packed.forces_n[active]
            point = packed.points_w[active]
            inward = packed.normals_inward_w[active]
            pose = env._robot.data.body_pose_w.torch[ids,env._robot.body_names.index(body)]
            local = quat_apply_inverse(pose[:,3:7],point-pose[:,:3])
            outward_local = quat_apply_inverse(pose[:,3:7],-inward)
            label = torch.full_like(ids,NON_PAD)
            diagnostic = None
            if body in self.surfaces:
                c = self.config
                diagnostic = classify(self.surfaces[body],local.cpu().numpy(),outward_local.cpu().numpy(),
                    position_tolerance_m=c['thumb_position_tolerance_m'] if digit==0 else c['other_position_tolerance_m'],
                    normal_tolerance_rad=math.radians(c['thumb_normal_tolerance_deg'] if digit==0 else c['other_normal_tolerance_deg']))
                strings = diagnostic['state']
                uncertain_edge = self._boundary_distance(diagnostic['closest_point_body_m'],self.boundaries[body]) < c['boundary_uncertainty_m']
                strings[uncertain_edge] = 'UNKNOWN'
                label = torch.tensor([{'PAD':PAD,'NON_PAD':NON_PAD,'UNKNOWN':UNKNOWN}[x] for x in strings],device=env.device)
            labels.append(label);loads.append(force);env_ids.append(ids);digit_ids.append(torch.full_like(ids,digit))
            if bool(getattr(env,'pad_trace_enabled',False)):
                details.append({'body':body,'sensor_paths':paths,'counts':arrays[4].tolist(),'starts':arrays[5].tolist(),
                    'env_ids':ids.tolist(),'forces_n':force.tolist(),'points_w':point.tolist(),
                    'raw_normals_w':inward.tolist(),'separations_m':packed.separations_m[active].tolist(),
                    'body_pose_w':pose.tolist(),'points_body_m':local.tolist(),'outward_normals_body':outward_local.tolist(),
                    'labels':label.tolist(),
                    'nearest_triangle_ids':None if diagnostic is None else diagnostic['nearest_triangle_id'].tolist(),
                    'source_distance_m':None if diagnostic is None else diagnostic['distance_m'].tolist()})
        for body in self.surfaces:
            if body not in seen:raise RuntimeError('缺少已确认指肚传感器: '+body)
        def cat(items,dtype):return torch.cat(items) if items else torch.empty(0,device=env.device,dtype=dtype)
        c = self.config
        self.latest = summarize_contact_loads(labels=cat(labels,torch.long),normal_forces_n=cat(loads,torch.float32),
            env_ids=cat(env_ids,torch.long),digit_ids=cat(digit_ids,torch.long),num_envs=env.num_envs,
            channel_valid=(~invalid_env[:,None]).expand(env.num_envs,5),
            f_pad_min_n=c['f_pad_min_n'],f_nonpad_noise_max_n=c['f_nonpad_noise_max_n'],f_other_contact_min_n=c['f_other_contact_min_n'])
        self.counter = counter
        self.details = details
        self.invalid_details = invalid_details
        return self.latest
