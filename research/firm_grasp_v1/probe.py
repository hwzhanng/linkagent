"""真实驱动下的有界三指可行性探测；不训练，不写回合内状态。"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import traceback
from isaaclab.app import AppLauncher
from .release_profile import DEFAULT_SUPPORT_RAMP_POLICY_STEPS
from .oriented_reset import DEFAULT_WRIST_TILT_AXIS_WORLD_XYZ


LEGACY_CENTER_LOCAL_MM = (-88.76065266544136, -11.325082731003075, -193.5340228574738)


def _scheduled_probe_action(base, offsets, episode_age, start_step, end_step):
    """仅产生真实动作命令；各环境 reset 后按自身年龄重新开始，不写物理状态。"""
    scale = ((episode_age.to(dtype=base.dtype) - start_step) / (end_step-start_step)).clamp(0, 1)
    return base + offsets * scale[:, None]


def _resolve_pose_arguments(parser, args):
    if args.reset_fruit_reference_hand_mm is not None and args.center_local_mm is not None:
        parser.error('reference 模式禁止同时显式给出 --center-local-mm')
    if args.legacy_guard and (args.reset_fruit_reference_hand_mm is not None
                              or args.wrist_tilt_deg != 0.0):
        parser.error('--legacy-guard 不接受 S5 定向 reset 参数')
    center = args.center_local_mm if args.center_local_mm is not None else list(LEGACY_CENTER_LOCAL_MM)
    reference = (None if args.reset_fruit_reference_hand_mm is None
                 else tuple(value / 1000.0 for value in args.reset_fruit_reference_hand_mm))
    return center, reference


def _initial_snapshot(env):
    """reset 后、首 step 前的只读初始快照；深拷贝避免后续 reset 覆盖。"""

    return {'joint_names': list(env._robot.joint_names),
            'body_names': list(env._robot.body_names),
            'joint_pos': env._robot.data.joint_pos.torch.tolist(),
            'body_pose_w': env._robot.data.body_pose_w.torch.tolist(),
            'fruit_pose_w': env._tomato.data.root_pose_w.torch.tolist(),
            'stem_rest_attachment_pos_w': env._stem_rest_attachment_pos_w.tolist(),
            'stem_plant_anchor_pos_w': env._stem_plant_anchor_pos_w.tolist(),
            'stem_anchor_quat_w': env._stem_anchor_quat_w.tolist(),
            'last_reset_poses': copy.deepcopy(env.last_reset_poses)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--actions-file', type=Path, required=True)
    parser.add_argument('--action-schedule-file', type=Path, default=None,
                        help='仅用于脚本可行性：按真实回合年龄线性引入14D动作偏移，不是PPO')
    parser.add_argument('--legacy-guard', action='store_true')
    parser.add_argument('--support-release-powers-file',type=Path,default=None)
    parser.add_argument('--support-release-power',type=float,default=1.)
    parser.add_argument('--support-ramp-policy-steps', type=int,
                        default=DEFAULT_SUPPORT_RAMP_POLICY_STEPS,
                        help='撤支撑线性渐退的策略步数；默认20步（1 s），0保留原突撤对照')
    parser.add_argument('--center-local-mm', nargs=3, type=float, default=None)
    parser.add_argument('--wrist-tilt-deg', type=float, default=0.0)
    parser.add_argument('--wrist-tilt-axis-world-xyz', nargs=3, type=float,
                        default=list(DEFAULT_WRIST_TILT_AXIS_WORLD_XYZ))
    parser.add_argument('--reset-fruit-reference-hand-mm', nargs=3, type=float, default=None)
    parser.add_argument('--base-lift-m', type=float, default=0.0,
                        help='reset 时固定基座竖直平移（米，<=0.10）')
    parser.add_argument('--reset-reference-offsets-file', type=Path, default=None,
                        help='诊断专用：{num_envs}x3 手系 reference 偏移（mm），仅 reset 写入')
    parser.add_argument('--preparation-reference-offsets-file', type=Path, default=None,
                        help='诊断专用：每env准备支撑弹簧参考位移(mm)，经真实动力学施加')
    parser.add_argument('--preparation-reference-offset-hand-mm',nargs=3,type=float,default=None)
    parser.add_argument('--reset-reference-offset-hand-mm', nargs=3, type=float, default=None,
                        help='固定任务条件：手系 reference 偏移（mm，每轴<=5）')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    center_mm, reference_m = _resolve_pose_arguments(parser, args)
    if args.reset_reference_offset_hand_mm is not None:
        if reference_m is None or args.wrist_tilt_deg == 0.0:
            parser.error('--reset-reference-offset-hand-mm 仅用于非零 tilt 的 reference reset')
        if not all(abs(value) <= 5.0 for value in args.reset_reference_offset_hand_mm):
            parser.error('--reset-reference-offset-hand-mm 每轴不得超过 5 mm')
    args.output.mkdir(exist_ok=False)
    launcher = AppLauncher(args)
    env = None
    try:
        import torch
        if args.legacy_guard:
            from research.thumb_pad_grasp_v1.env import ThumbPadGraspEnv as Env, make_config
        else:
            from .env import FirmGraspEnv as Env, make_config
        # 老守护对照不接受新任务条件参数，斜坡只在 firm_grasp_v1 分支生效。
        extra = {} if args.legacy_guard else {
            'support_ramp_policy_steps': args.support_ramp_policy_steps,
            'wrist_tilt_deg': args.wrist_tilt_deg,
            'wrist_tilt_axis_world_xyz': tuple(args.wrist_tilt_axis_world_xyz),
            'reset_fruit_reference_hand_m': reference_m,
            'base_lift_m': args.base_lift_m,
            'reset_reference_offset_hand_m': (None if args.reset_reference_offset_hand_mm is None
                else tuple(value / 1000.0 for value in args.reset_reference_offset_hand_mm))}
        cfg = make_config(num_envs=args.num_envs, device=args.device, seed=110,
                          thumb_mapping='independent', stage='contact', pregrasp_distance_m=.005,
                          fruit_center_local_m=(None if reference_m is not None
                              else tuple(x / 1000 for x in center_mm)),
                          **extra)
        env = Env(cfg=cfg)
        env.pad_trace_enabled = True
        release_powers = None
        if args.support_release_powers_file is not None:
            release_powers = torch.tensor(json.loads(args.support_release_powers_file.read_text()),device=env.device,dtype=torch.float32)
        elif args.support_release_power != 1.:
            release_powers = torch.full((args.num_envs,),args.support_release_power,device=env.device)
        if release_powers is not None:
            if release_powers.shape != (args.num_envs,) or not bool(torch.isfinite(release_powers).all()) or bool(((release_powers<1)|(release_powers>4)).any()):
                raise ValueError('退出曲线指数应为[N]且1<=power<=4')
            env._support_release_power_by_env = release_powers
        preparation_offsets = None
        if args.preparation_reference_offsets_file is not None:
            preparation_offsets = torch.tensor(json.loads(args.preparation_reference_offsets_file.read_text()),
                device=env.device,dtype=torch.float32)/1000.
        elif args.preparation_reference_offset_hand_mm is not None:
            preparation_offsets = torch.tensor(args.preparation_reference_offset_hand_mm,
                device=env.device,dtype=torch.float32).reshape(1,3).expand(args.num_envs,3).clone()/1000.
        if preparation_offsets is not None:
            if preparation_offsets.shape != (args.num_envs,3) or not bool(torch.isfinite(preparation_offsets).all()) or bool((preparation_offsets.abs()>.01).any()):
                raise ValueError('准备参考必须为[N,3]有限值且每轴<=10mm')
            env._prepare_target_offsets_m = preparation_offsets
        reference_offsets = None
        if args.reset_reference_offsets_file is not None:
            raw_offsets = json.loads(args.reset_reference_offsets_file.read_text())
            if (not isinstance(raw_offsets, list) or len(raw_offsets) != args.num_envs
                    or any(not isinstance(row, list) or len(row) != 3 for row in raw_offsets)):
                raise ValueError('reset reference offsets 必须是 num_envs x 3 数值')
            reference_offsets = torch.tensor(raw_offsets, dtype=torch.float32,
                                            device=env.device) / 1000.0
            if not bool(torch.isfinite(reference_offsets).all()):
                raise ValueError('reset reference offsets 含非有限值')
            env._reset_reference_offsets_m = reference_offsets
        obs, _ = env.reset()
        initial = _initial_snapshot(env)
        # reset 后、首 step 前的只读初始快照；避免后续 episode reset 覆盖 last_reset_poses。
        (args.output / 'initial_state.json').write_text(
            json.dumps(initial, indent=2, allow_nan=False) + '\n')
        table = json.loads(args.actions_file.read_text())
        action = torch.tensor(table, device=env.device, dtype=torch.float32)
        assert action.shape == (args.num_envs, 14) and torch.isfinite(action).all() and (action.abs() <= 1).all()
        schedule = None
        offsets = None
        if args.action_schedule_file is not None:
            schedule = json.loads(args.action_schedule_file.read_text())
            start, end = schedule['start_policy_step'], schedule['end_policy_step']
            if (type(start) is not int or type(end) is not int or start < 0 or end <= start):
                raise ValueError('动作时序必须使用非负整数起点和更晚的终点')
            offsets = torch.tensor(schedule['action_offsets'], device=env.device, dtype=action.dtype)
            if offsets.shape != action.shape or not bool(torch.isfinite(offsets).all()):
                raise ValueError('动作时序偏移必须是有限的[num_envs,14]矩阵')
            if not bool(((action+offsets).abs() <= 1).all()):
                raise ValueError('动作时序端点超出原14D动作范围')
        steps = []
        with (args.output / 'physical_substeps.jsonl').open('x') as stream:
            for step in range(args.steps):
                command = action if schedule is None else _scheduled_probe_action(
                    action, offsets, env.episode_length_buf,
                    schedule['start_policy_step'], schedule['end_policy_step'])
                obs, reward, terminated, truncated, extras = env.step(command)
                if not torch.isfinite(reward).all() or not torch.isfinite(obs['policy']).all():
                    raise RuntimeError('非有限奖励或观测')
                for row in env.drain_pad_records():
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                truth = env.pad_observer.latest
                steps.append({'step': step, 'episode_age': env.episode_length_buf.tolist(),
                              'pad_force_n': truth['pad_force_n'].tolist(),
                              'nonpad_force_n': truth['nonpad_force_n'].tolist(),
                              'unknown': truth['unknown'].tolist(),
                              'terminated': terminated.tolist(), 'truncated': truncated.tolist()})
        result = {'status': 'REAL_PHYSICS_FEASIBILITY_PROBE_COMPLETED', 'training': False,
                  'legacy_guard': args.legacy_guard, 'initial': initial,
                  'declared_actions': table, 'command_warmup': 'environment35steps_only_no_extra_ramp',
                  'declared_action_schedule': schedule,
                  'action_schedule_sha256': (None if args.action_schedule_file is None else
                      hashlib.sha256(args.action_schedule_file.read_bytes()).hexdigest()),
                  'action_schedule_semantics': 'scripted_real_20Hz_commands_not_PPO_not_state_writes',
                  'reset_reference_offsets_m': (None if reference_offsets is None
                      else reference_offsets.tolist()),
                  'preparation_reference_offsets_m':(None if preparation_offsets is None else preparation_offsets.tolist()),
                  'support_release_powers':(None if release_powers is None else release_powers.tolist()),
                  'support_ramp_policy_steps': None if args.legacy_guard else args.support_ramp_policy_steps,
                  'wrist_tilt_deg': None if args.legacy_guard else args.wrist_tilt_deg,
                  'wrist_tilt_axis_world_xyz': (None if args.legacy_guard
                      else list(args.wrist_tilt_axis_world_xyz)),
                  'reset_fruit_reference_hand_m': None if args.legacy_guard else reference_m,
                  'task_identity': env.task_identity(), 'completed_episodes': env.completed_episodes,
                  'steps': steps,
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    except BaseException:
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == '__main__':
    main()
