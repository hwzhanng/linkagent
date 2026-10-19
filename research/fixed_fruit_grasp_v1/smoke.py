"""有限的原生 App/关节/接触检查；不执行 PPO 或硬件控制。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback


def main() -> None:
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--boot-only', action='store_true')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--stage', choices=['contact', 'hold', 'lift'], default='hold')
    parser.add_argument('--pregrasp-distance-mm', type=float, default=None)
    parser.add_argument('--center-local-mm', nargs=3, type=float, default=None)
    parser.add_argument('--action-json', type=str, default=None)
    parser.add_argument('--closure-amplitude', type=float, default=0.65)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.closure_amplitude <= 1:
        parser.error('closure-amplitude 必须在0到1之间')
    start = time.monotonic()
    app = None
    env = None
    result = {'kind': 'boot' if args.boot_only else 'bounded_physics', 'formal_grasp_acceptance': False,
              'closure_amplitude': args.closure_amplitude}
    try:
        app = AppLauncher(args).app
        result['app_created'] = True
        result['app_running'] = bool(app.is_running())
        if args.boot_only:
            return
        import torch
        from .env import FixedFruitGraspEnv, make_config
        cfg = make_config(args.num_envs, args.device, 42, stage=args.stage,
                          pregrasp_distance_m=None if args.pregrasp_distance_mm is None else args.pregrasp_distance_mm / 1000,
                          fruit_center_local_m=None if args.center_local_mm is None else [v / 1000 for v in args.center_local_mm])
        env = FixedFruitGraspEnv(cfg=cfg)
        obs, _ = env.reset(seed=42)
        result['identity'] = env.task_identity()
        result['initial_fruit_world'] = env._tomato.data.root_pos_w.torch.cpu().tolist()
        result['initial_fingertip_world'] = env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids].cpu().tolist()
        result['initial_hand_base_world'] = env._robot.data.body_pos_w.torch[:, env._ee_body_id].cpu().tolist()
        result['grasp_basis_world'] = env._grasp_basis_w.cpu().tolist()
        result['reset_pose_records'] = list(env.last_reset_poses)
        result['collider_aabb_centers_local'] = env._hand_collision_aabb_centers.cpu().tolist()
        result['collider_aabb_half_extents_local'] = env._hand_collision_aabb_half_extents.cpu().tolist()
        result['collider_body_names'] = [env._robot.body_names[j] for j in env._hand_collision_body_ids]
        result['joint_names'] = list(env._robot.joint_names)
        result['initial_collider_body_pose_w'] = env._robot.data.body_pose_w.torch[:, env._hand_collision_body_ids].cpu().tolist()
        from pxr import PhysxSchema
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        articulation_flags = []
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith('/World/envs/env_0/Robot') and prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                articulation_flags.append({'path': str(prim.GetPath()), 'enabled_self_collisions': PhysxSchema.PhysxArticulationAPI(prim).GetEnabledSelfCollisionsAttr().Get()})
        result['runtime_self_collision_flags'] = articulation_flags
        rows = []
        open_phase_steps = min(40, args.steps // 4)
        for mode in ['open', 'bounded_closure']:
            env.reset(seed=42)
            env.set_open_hand_baseline(mode == 'open')
            initial_q = env._robot.data.joint_pos.torch.clone()
            steps = args.steps if mode == 'open' else max(open_phase_steps, args.steps)
            action = env.closure_reference_action(closure=args.closure_amplitude)
            if args.action_json is not None:
                pattern = json.loads(args.action_json)
                if len(pattern) != 14 or any(not -1 <= v <= 1 for v in pattern):
                    raise ValueError("action-json 必须是14D有限归一化动作")
                action = torch.tensor(pattern, device=env.device).repeat(env.num_envs, 1)
                result["action_pattern"] = pattern
            max_force = 0.0
            max_displacement = 0.0
            max_by_joint = torch.zeros_like(initial_q)
            for step in range(steps):
                obs, reward, terminated, truncated, info = env.step(action)
                force = env._last_physical_force_n
                max_force = max(max_force, float(force.abs().max().item()))
                current_q = env._robot.data.joint_pos.torch
                max_displacement = max(max_displacement, float((current_q-initial_q).abs().max().item()))
                max_by_joint = torch.maximum(max_by_joint, (current_q-initial_q).abs())
                if step % 10 == 0 or bool((terminated | truncated).any()):
                    rows.append({'mode': mode, 'step': step, 'q': current_q.cpu().tolist(),
                                 'targets': env._joint_targets.cpu().tolist(), 'forces_n': force.cpu().tolist(),
                                 'fruit_world': env._tomato.data.root_pos_w.torch.cpu().tolist(),
                                 'fingertip_world': env._robot.data.body_pos_w.torch[:, env._fingertip_contact_body_ids].cpu().tolist(),
                                 'support_removed': env._stem_detached.cpu().tolist(),
                                 'support_force_w': env._last_stem_force_w.cpu().tolist(),
                                 'legacy_pad_force_n': env._legacy_pad_force_n.cpu().tolist(),
                                 'hold_window_drift_m': env._hold_window_drift_m.cpu().tolist(),
                                 'done': (terminated | truncated).cpu().tolist()})
                if bool((terminated | truncated).any()):
                    break
            result[mode] = {'max_force_n': max_force, 'max_joint_displacement_rad': max_displacement,
                            'max_displacement_by_joint_rad': max_by_joint.cpu().tolist(),
                            'episodes': list(env.completed_episodes)}
            env.completed_episodes.clear()
        result['trace'] = rows
        result['finite_observations'] = all(bool(torch.isfinite(v).all()) for v in obs.values())
        result['action_effect_observed'] = result['bounded_closure']['max_joint_displacement_rad'] > 0.005
        result['real_contact_observed'] = result['bounded_closure']['max_force_n'] > 0.02
        result['open_successes'] = sum(bool(r.get('success')) for r in result['open']['episodes'])
        checks = {
            'finite_observations': result['finite_observations'],
            'action_effect_observed': result['action_effect_observed'],
            'real_contact_observed': result['real_contact_observed'],
            'self_collision_enabled': bool(articulation_flags) and all(r['enabled_self_collisions'] is True for r in articulation_flags),
            'open_episode_completed': bool(result['open']['episodes']),
            'open_baseline_failed': result['open_successes'] == 0,
            'no_hard_physical_faults': not any(r.get(k, False) for mode in ['open', 'bounded_closure'] for r in result[mode]['episodes']
                                               for k in ['nonfinite', 'overload', 'joint_limit', 'joint_effort', 'self_collision', 'instability', 'penetration']),
            'closure_reached_support_release': any(r.get('support_released') for r in result['bounded_closure']['episodes']),
        }
        result['required_checks'] = checks
        result['gate_pass'] = all(checks.values())
        if not result['gate_pass']:
            raise RuntimeError('物理检查未通过：' + ', '.join(k for k, v in checks.items() if not v))
    except BaseException as error:
        result['error'] = repr(error)
        traceback.print_exc()
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        if env is not None:
            env.close()
        if app is not None:
            app.close(exit_code=1 if 'error' in result else 0)


if __name__ == '__main__':
    main()
