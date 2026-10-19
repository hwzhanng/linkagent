"""P0/P1 实际 packed 接触布局与局部表面数据采集，不作为训练成功证据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--fixture-source-surface', choices=('none', 'thumb_pad', 'thumb_side'), default='none')
    parser.add_argument('--fixture-center-distance-m', type=float, default=.0334)
    parser.add_argument('--fixture-pad-target-m',nargs=3,type=float,default=None)
    parser.add_argument('--fixture-digit', choices=('thumb','index','middle','ring','little'), default='thumb')
    parser.add_argument('--task', choices=('base','thumb_pad'), default='base')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    try:
        import numpy as np
        import torch
        import warp as wp
        from pxr import Usd, UsdGeom
        from isaaclab.utils.math import quat_apply, quat_apply_inverse
        if args.task == 'thumb_pad':
            from .env import ThumbPadGraspEnv as FixedFruitGraspEnv, make_config
        else:
            from research.fixed_fruit_grasp_v1.env import FixedFruitGraspEnv, make_config

        cfg = make_config(num_envs=args.num_envs, device=args.device, seed=42,
                          stage='contact', pregrasp_distance_m=.005,
                          fruit_center_local_m=(-.07916, .01116, -.160))
        env = FixedFruitGraspEnv(cfg=cfg)
        # 导出本次场景真实配置对应的凸分解；不新增/替换任何碰撞几何。
        from pxr import UsdUtils, UsdPhysics, PhysicsSchemaTools
        from omni.physx import get_physx_cooking_interface
        runtime_stage = env.sim.stage
        stage_id = UsdUtils.StageCache.Get().GetId(runtime_stage).ToLongInt()
        cooked = {}
        for body in ('thumb_link4', 'index_link3', 'middle_link3', 'ring_link3', 'little_link3'):
            prim_path = '/World/envs/env_0/Robot/' + body + '/collisions'
            if not runtime_stage.GetPrimAtPath(prim_path).IsA(UsdGeom.Mesh):
                cooked[prim_path] = {'result': 'UNSUPPORTED_MERGED_XFORM_WRAPPER',
                    'hulls': [], 'correspondence': 'requires_actual_contact_probe_not_this_export_API'}
                continue
            def save_result(result, convexes, key=prim_path):
                cooked[key] = {'result': str(result), 'hulls': [
                    {'vertices': [[float(v.x),float(v.y),float(v.z)] for v in hull.vertices],
                     'indices': list(hull.indices),
                     'polygons': [{'index_base': int(p.index_base),'num_vertices':int(p.num_vertices)}
                                  for p in hull.polygons]} for hull in convexes]}
            get_physx_cooking_interface().request_convex_collision_representation(
                stage_id=stage_id, collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim_path),
                run_asynchronously=False, on_result=save_result)
        (args.output / 'cooked_collision.json').write_text(json.dumps(cooked))
        approved_path = Path(__file__).parent / 'assets/pad_regions_v1.approved.json'
        approved = json.loads(approved_path.read_text())
        # 冻结的源三角面导出为刚体局部米坐标，保留全部非指肚表面。
        asset = Path(__file__).parents[1] / 'small_fruit_pinch_v1/assets/left_hand_urdf/left_hand_l20_physx.usd'
        stage = Usd.Stage.Open(str(asset))
        xf = UsdGeom.XformCache()
        geometry = {}
        for digit in approved['digits']:
            prim = stage.GetPrimAtPath(digit['visual_mesh_path'])
            mesh = UsdGeom.Mesh(prim)
            vertices = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
            faces = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
            assert hashlib.sha256(vertices.astype('<f4').tobytes()).hexdigest() == digit['mesh_points_sha256']
            assert hashlib.sha256(faces.tobytes()).hexdigest() == digit['mesh_faces_sha256']
            matrix = np.array(xf.GetLocalToWorldTransform(prim)) @ np.linalg.inv(
                np.array(xf.GetLocalToWorldTransform(stage.GetPrimAtPath(digit['body_path']))))
            assert np.allclose(matrix, digit['mesh_to_body_row_matrix'], atol=1e-12)
            body_vertices = (np.c_[vertices, np.ones(len(vertices))] @ matrix)[:, :3]
            geometry[digit['digit'] + '_vertices_body_m'] = body_vertices
            geometry[digit['digit'] + '_faces'] = faces
        np.savez_compressed(args.output / 'source_geometry.npz', **geometry)

        bodies = list(env._robot.body_names)
        sensors = env._raw_penetration_sensors
        metadata = {'purpose': 'runtime_contact_calibration_not_success_evidence',
                    'task_identity': env.task_identity(), 'robot_body_names': bodies,
                    'approved_config_sha256': hashlib.sha256(approved_path.read_bytes()).hexdigest(),
                    'raw_layout': 'UNVERIFIED_6_ARRAYS', 'sensor_layout': []}
        observed = set()
        stream = (args.output / 'contacts.jsonl').open('x')
        frame_stream = (args.output / 'channel_frames.jsonl').open('x')
        original_update = env.scene.update

        def observe(dt):
            original_update(dt)
            if float(dt) <= 0:
                return
            count = int(env._sim_step_counter)
            frame = {'sim_step': count, 'dt_s': float(dt), 'channels': [],
                     'joint_pos': env._robot.data.joint_pos.torch.tolist(),
                     'joint_targets':env._joint_targets.tolist(),
                     'executed_actions':env._executed_actions.tolist()}
            if args.task == 'thumb_pad' and env.pad_observer.latest is not None:
                frame['pad_truth'] = {k:v.tolist() for k,v in env.pad_observer.latest.items()}
            for sensor in sensors:
                reader = getattr(sensor,'_thumb_pad_contact_read',None)
                raw = reader() if reader is not None else sensor.contact_view.get_contact_data(1 / 120)
                if len(raw) != 6:
                    raise RuntimeError('接触布局不是6数组')
                arrays = [wp.to_torch(value) for value in raw]
                paths = list(sensor.body_physx_view.prim_paths)
                name = paths[0].rsplit('/', 1)[-1]
                if name not in observed:
                    metadata['sensor_layout'].append({'body': name, 'paths': paths,
                        'shapes': [list(a.shape) for a in arrays],
                        'dtypes': [str(a.dtype) for a in arrays]})
                    observed.add(name)
                counts = arrays[4].reshape(-1).long()
                starts = arrays[5].reshape(-1).long()
                frame['channels'].append({'body': name,'paths':paths,
                    'counts':counts.tolist(),'starts':starts.tolist(),
                    'force_matrix_w':sensor.data.force_matrix_w.torch.tolist()})
                for row, path in enumerate(paths):
                    n = int(counts[row])
                    if n == 0:
                        continue
                    start = int(starts[row])
                    if n < 0 or n >= sensor.cfg.max_contact_data_count_per_prim or start < 0 or start+n > arrays[0].shape[0]:
                        raise RuntimeError('packed 接触缓冲越界/溢出')
                    env_id = int(path.split('/env_')[1].split('/')[0])
                    pose = env._robot.data.body_pose_w.torch[env_id, bodies.index(name)]
                    points = arrays[1][start:start+n]
                    normals = arrays[2][start:start+n]
                    local = quat_apply_inverse(pose[3:7].expand(n, -1), points - pose[:3])
                    local_normals = quat_apply_inverse(pose[3:7].expand(n, -1), normals)
                    record = {'sim_step': count, 'dt_s': float(dt), 'env_id': env_id,
                        'episode_step': int(env.episode_length_buf[env_id]), 'body': name,
                        'sensor_path': path, 'count': n, 'start': start,
                        'raw_0': arrays[0][start:start+n].tolist(), 'points_w': points.tolist(),
                        'raw_2': normals.tolist(), 'separations_m': arrays[3][start:start+n].tolist(),
                        'body_pose_w': pose.tolist(), 'points_body_m': local.tolist(),
                        'normals_body_raw': local_normals.tolist(),
                        'filtered_force_w': sensor.data.force_matrix_w.torch[env_id, 0, 0].tolist(),
                        'fruit_pose_w': env._tomato.data.root_pose_w.torch[env_id].tolist(),
                        'support_force_w': env._last_stem_force_w[env_id].tolist()}
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
            frame_stream.write(json.dumps(frame, allow_nan=False)+'\n')

        env.scene.update = observe
        env.reset()
        metadata['reset_body_poses_w'] = env._robot.data.body_pose_w.torch.tolist()
        metadata['reset_joint_pos'] = env._robot.data.joint_pos.torch.tolist()
        metadata['fixture_source_surface'] = args.fixture_source_surface
        metadata['fixture_digit'] = args.fixture_digit
        if args.fixture_source_surface != 'none':
            if args.num_envs != 1:
                raise ValueError('已知表面标定暂仅1env')
            # 独立接触标定夹具：只在 reset 边界预置球面到指定源表面。
            # 不是自然抓握/成功证据，回合中禁止改写果实或关节状态。
            digit = next(d for d in approved['digits'] if d['digit'] == args.fixture_digit)
            vertices, faces = geometry[args.fixture_digit+'_vertices_body_m'], geometry[args.fixture_digit+'_faces']
            tri = vertices[faces]
            centers = tri.mean(axis=1)
            cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
            lengths = np.linalg.norm(cross,axis=1)
            normals = cross / np.maximum(lengths[:,None],1e-15)
            if args.fixture_source_surface == 'thumb_pad':
                candidates = np.asarray(digit['selected_triangle_ids'])
                target = np.asarray(digit['center_body_local_m'] if args.fixture_pad_target_m is None else args.fixture_pad_target_m)
                candidates = candidates[lengths[candidates] > 1e-10]
                chosen = candidates[np.argmin(np.linalg.norm(centers[candidates]-target,axis=1))]
            else:
                eligible = (normals[:,0] < -.95) & (lengths > 1e-8)
                eligible[np.asarray(digit['selected_triangle_ids'])] = False
                candidates = np.flatnonzero(eligible)
                target = np.asarray(digit['center_body_local_m'])
                chosen = candidates[np.argmin(np.linalg.norm(centers[candidates]-target,axis=1))]
            body_pose = env._robot.data.body_pose_w.torch[0,bodies.index(digit['body_path'].rsplit('/',1)[-1])]
            local_point = torch.as_tensor(centers[chosen],device=args.device,dtype=torch.float32)
            local_normal = torch.as_tensor(normals[chosen],device=args.device,dtype=torch.float32)
            surface_point_w = body_pose[:3]+quat_apply(body_pose[3:7],local_point)
            normal_w = quat_apply(body_pose[3:7],local_normal)
            fruit_pose = env._tomato.data.root_pose_w.torch.clone()
            fruit_pose[0,:3] = surface_point_w + normal_w*args.fixture_center_distance_m
            env._tomato.write_root_pose_to_sim(fruit_pose)
            env._tomato.write_root_velocity_to_sim(torch.zeros((1,6),device=args.device))
            offset = torch.tensor([[0.,0.,.0335]],device=args.device)
            env._stem_rest_attachment_pos_w.copy_(fruit_pose[:,:3]+quat_apply(fruit_pose[:,3:7],offset))
            env._open_hand_baseline = True
            metadata['reset_only_fixture_placement'] = {
                'requested_pad_target_body_m':args.fixture_pad_target_m,
                'source_triangle_id':int(chosen),'source_point_body_m':centers[chosen].tolist(),
                'source_outward_normal_body':normals[chosen].tolist(),
                'fruit_pose_w':fruit_pose.tolist(),'center_distance_m':args.fixture_center_distance_m,
                'classification':'known_surface_probe_not_grasp_success'}
            env.scene.write_data_to_sim()
            if args.task == 'thumb_pad':env.pad_observer.reset(torch.arange(env.num_envs,device=env.device))
            env.scene.update(0.)
        bias = torch.tensor([.6, .5, 0, 0, .1, 0, 0, 0, 0, 0, .8, .8, .8, .8], device=args.device)
        completed = 0
        for step in range(args.steps):
            # 仅真实关节目标，闭合过程中不写果实/关节状态。
            action = bias * min(1., step / 35.)
            _, _, terminated, truncated, _ = env.step(action.unsqueeze(0).repeat(args.num_envs, 1))
            completed += 1
            if args.fixture_source_surface != 'none' and bool((terminated | truncated).any()):
                break
        stream.flush()
        stream.close()
        frame_stream.flush()
        frame_stream.close()
        metadata['completed_policy_steps'] = completed
        metadata['completed_episodes'] = env.completed_episodes
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2, default=str))
    except BaseException:
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
