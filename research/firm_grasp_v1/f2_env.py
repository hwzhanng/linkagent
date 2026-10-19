"""F2三指指肚一秒任务：S7 独立食指/中指 abduction 与成功守卫分离。"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_box_minus
from research.thumb_pad_grasp_v1.env import ThumbPadGraspEnv, make_config as old_make_config
from research.thumb_pad_grasp_v1.phase_rewards import block_failure_positive_terms
from research.small_fruit_pinch_v1.source.legacy_bridge import load_legacy_hand_manifold
from .abduction_mapping import (
    ABDUCTION_ACTION_INDICES,
    ABDUCTION_CANONICAL_INDICES,
    ACTION_ORDER as ABDUCTION_ACTION_ORDER,
    LEGACY_RING_LITTLE_RESIDUAL_INDICES,
    MAPPING_ID as ABDUCTION_MAPPING_ID,
    index_middle_abduction_targets,
    mapping_identity as abduction_mapping_identity,
)
from .stable_temporal import StableCriticalDampedActionFilter, temporal_identity
from .contract import FirmGraspContract, load_frozen_contract
from .grip_metrics import compute_contact_metrics, COMBINATIONS
from .oriented_reset import (DEFAULT_WRIST_TILT_AXIS_WORLD_XYZ,
    MIN_FRUIT_BOTTOM_CLEARANCE_M, OrientedResetCondition, fruit_bottom_clearance,
    ground_free_mask, make_oriented_reset_condition, oriented_reset_targets,
    validate_reset_fruit_reference)
from .physical_guard import CombinationHoldGuard
from .release_profile import (DEFAULT_SUPPORT_RAMP_POLICY_STEPS, SCOPE_NOTE,
    scale_support_wrench, support_ramp_identity, support_release_scale,
    validate_support_ramp_policy_steps)


TASK_VERSION = 'firm_grasp_v1.8.0'
MAX_BASE_LIFT_M = 0.10
LEGACY_ACTOR_DIM = 161
LEGACY_CRITIC_DIM = 222
LEGACY_PRIVILEGED_DIM = 61
CLOCK_DIM = 1
ACTOR_DIM = LEGACY_ACTOR_DIM + CLOCK_DIM
CRITIC_DIM = LEGACY_CRITIC_DIM + CLOCK_DIM
ACTION_DIM = 14


def release_clock(episode_length_buf: torch.Tensor, support_release_step: int) -> torch.Tensor:
    """返回 ``[N, 1]`` 的真实回合年龄/释放步时钟，仅读取策略步计数。"""

    if episode_length_buf.ndim != 1:
        raise ValueError(f'episode_length_buf 必须是一维，得到 {tuple(episode_length_buf.shape)}')
    if isinstance(support_release_step, bool) or int(support_release_step) != support_release_step:
        raise ValueError(f'support_release_step 必须是整数，得到 {support_release_step!r}')
    release_step = int(support_release_step)
    if release_step <= 0:
        raise ValueError(f'support_release_step 必须为正数，得到 {release_step}')
    return torch.clamp(
        episode_length_buf.float() / float(release_step), 0.0, 3.0
    ).unsqueeze(1)


def make_config(**kwargs):
    # S7 的动作 8/9 已改为 abduction；构造入口必须显式保持 independent，
    # 旧 legacy 路径会悄悄恢复 ring/little residual，因此在这里 fail-closed。
    thumb_mapping = kwargs.pop('thumb_mapping', 'independent')
    if thumb_mapping != 'independent':
        raise ValueError(
            'firm_grasp_v1.8.0 仅接受 thumb_mapping="independent"; '
            'legacy 会把动作8/9解释回ring/little residual'
        )
    # 显式初始化高度：仅整体平移固定基座（连同 reference 果实/果梗），
    # 用于排除定向 reset 下拇指碰撞网格与地面的重叠；手果相对几何、
    # 质量、摩擦、PD、力矩上限与判据全部不变。
    base_lift = kwargs.pop('base_lift_m', 0.0)
    if isinstance(base_lift, bool) or not isinstance(base_lift, (int, float)):
        raise TypeError(f'base_lift_m 必须是数值，得到 {base_lift!r}')
    base_lift = float(base_lift)
    if not (base_lift == base_lift) or not (-1e-9 <= base_lift <= MAX_BASE_LIFT_M):
        raise ValueError(f'base_lift_m 必须位于 [0, {MAX_BASE_LIFT_M}]，得到 {base_lift!r}')
    # 显式新任务条件：默认最后20个策略步（20 Hz ⇒ 1 s）线性退出支撑，
    # 取 0 保留原“到点突撤”对照；其余物理与判据数值全不变。
    ramp = kwargs.pop('support_ramp_policy_steps', DEFAULT_SUPPORT_RAMP_POLICY_STEPS)
    tilt_deg = kwargs.pop('wrist_tilt_deg', 0.0)
    tilt_axis = kwargs.pop('wrist_tilt_axis_world_xyz', DEFAULT_WRIST_TILT_AXIS_WORLD_XYZ)
    reference = kwargs.pop('reset_fruit_reference_hand_m', None)
    condition = make_oriented_reset_condition(tilt_deg, tilt_axis, reference)
    # 固定 reference 偏移：只在 reset 时把手系果实目标整体平移，用于把三指
    # 接触点移到批准PAD内部；质量/摩擦/PD/限位/PAD阈值全部不变。
    fixed_offset = kwargs.pop('reset_reference_offset_hand_m', None)
    if fixed_offset is None:
        fixed_offset = (0.0, 0.0, 0.0)
    else:
        fixed_offset = validate_reset_fruit_reference(fixed_offset)
        if any(abs(value) > 0.005 for value in fixed_offset):
            raise ValueError(f'reset_reference_offset_hand_m 每轴不得超过 5 mm：{fixed_offset}')
    if not condition.enabled and any(fixed_offset):
        raise ValueError('reset reference 偏移只允许用于 reference reset 模式')
    if condition.enabled and kwargs.get('fruit_center_local_m') is not None:
        raise ValueError('reference reset 模式不得同时给出 fruit_center_local_m')
    cfg = old_make_config(**kwargs)
    cfg.thumb_mapping = thumb_mapping
    if condition.enabled and tuple(float(item) for item in cfg.pinch_working_offset_m) != (0., 0., 0.):
        raise ValueError('reference reset 模式要求 pinch_working_offset_m 为零，避免二次叠加')
    if condition.enabled:
        # IsaacLab 的 init_state.rot 是 XYZW；这里只固定基座初始姿态，
        # 回合内不写腕部/根姿态。
        cfg.robot.init_state.rot = condition.init_state_rot_xyzw()
    # 基座竖直平移只影响 reset 世界高度；回合内无任何状态写入。
    if base_lift != 0.0:
        x, y, z = (float(value) for value in cfg.robot.init_state.pos)
        cfg.robot.init_state.pos = (x, y, z + base_lift)
    cfg.base_lift_m = base_lift
    cfg.reset_reference_offset_hand_m = tuple(float(value) for value in fixed_offset)
    cfg.wrist_tilt_deg = condition.wrist_tilt_deg
    cfg.wrist_tilt_axis_world_xyz = condition.wrist_tilt_axis_world_xyz
    cfg.reset_fruit_reference_hand_m = condition.reset_fruit_reference_hand_m
    cfg.oriented_reset_condition = condition
    cfg.support_ramp_policy_steps = validate_support_ramp_policy_steps(
        ramp, release_step=int(cfg.support_release_step))
    cfg.task_id = 'FirmGrasp-v1'
    cfg.task_version = TASK_VERSION
    return cfg


class FirmGraspEnv(ThumbPadGraspEnv):
    def __init__(self, cfg, **kwargs):
        self._firm_ready = False
        self._firm_metrics = None
        self._firm_last_state = None
        condition = getattr(cfg, 'oriented_reset_condition', None)
        if condition is None:
            condition = make_oriented_reset_condition(
                getattr(cfg, 'wrist_tilt_deg', 0.0),
                getattr(cfg, 'wrist_tilt_axis_world_xyz', DEFAULT_WRIST_TILT_AXIS_WORLD_XYZ),
                getattr(cfg, 'reset_fruit_reference_hand_m', None))
        if not isinstance(condition, OrientedResetCondition):
            raise TypeError('oriented_reset_condition 类型错误')
        if condition.enabled and tuple(float(item) for item in cfg.pinch_working_offset_m) != (0., 0., 0.):
            raise ValueError('reference reset 模式要求 pinch_working_offset_m 为零')
        self._oriented_reset_condition = condition
        self.firm_contract = load_frozen_contract()
        self._support_ramp_policy_steps = validate_support_ramp_policy_steps(
            getattr(cfg, 'support_ramp_policy_steps', DEFAULT_SUPPORT_RAMP_POLICY_STEPS),
            release_step=int(cfg.support_release_step))
        self._support_ramp_scale = None
        super().__init__(cfg, **kwargs)
        # 仅替换新任务的命令数值积分器；原滤波器和物理执行链只读。
        self._filter = StableCriticalDampedActionFilter(
            self.num_envs, ACTION_DIM, dt_s=float(cfg.sim.dt)*int(cfg.decimation),
            device=self.device, limits=self._filter.limits)
        # 旧 Thumb 父环境在构造期把实际 Gym 空间设为 161/222；新任务的
        # 真实 tensor 为 162/223。base make_config 仍保留 canonical 114/175，
        # 因而不能把新维度写回 Fixed.validate_cfg 的输入。
        self.cfg.observation_space = ACTOR_DIM
        self.cfg.state_space = CRITIC_DIM
        self._configure_gym_env_spaces()
        self._support_ramp_scale = torch.zeros(self.num_envs, device=self.device)
        self.firm_guard = CombinationHoldGuard(self.num_envs, device=self.device,
            initial_counter=self._sim_step_counter, physics_dt_s=float(self.cfg.sim.dt),
            decimation=int(self.cfg.decimation),
            max_speed=self._physical_hold_guard.hold_max_relative_speed_m_s,
            max_drift=self._physical_hold_guard.hold_max_drift_m)
        self._firm_ready = True
        self._identity = self._build_identity()

    def _postprocess_joint_targets(self, targets):
        """在执行动作历史层覆盖父级独立拇指映射；开掌对照仍走父级。"""

        if (getattr(self.cfg, 'thumb_mapping', None) != 'independent'
                or bool(getattr(self, '_open_hand_baseline', False))):
            return super()._postprocess_joint_targets(targets)
        return index_middle_abduction_targets(self._executed_actions)

    @torch.inference_mode()
    def _check_action_effect_and_limits(self):
        """保留父级通用检查，但把动作探针报告整体替换为 S7 新映射结果。"""

        result = super()._check_action_effect_and_limits()
        if getattr(self.cfg, 'thumb_mapping', None) != 'independent':
            raise RuntimeError('firm_grasp_v1.7.0 动作检查拒绝 legacy 映射')
        probes = torch.tensor(
            (( -1.0,) * ACTION_DIM, (0.0,) * ACTION_DIM, (1.0,) * ACTION_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        targets = index_middle_abduction_targets(probes)
        if tuple(targets.shape) != (3, 16) or not bool(torch.isfinite(targets).all()):
            raise RuntimeError('S7 动作探针没有产生有限的 canonical16 目标')
        limits = self._robot.data.joint_pos_limits.torch[0, self._hand_joint_ids]
        legacy = load_legacy_hand_manifold()
        safe_low = targets.new_tensor(legacy.LEFT_SAFE_LOWER_RAD)
        safe_high = targets.new_tensor(legacy.LEFT_SAFE_UPPER_RAD)
        low = targets.amin(dim=0)
        high = targets.amax(dim=0)
        span = high - low
        outside_safe = (low < safe_low - 1e-6) | (high > safe_high + 1e-6)
        outside_usd = (low < limits[:, 0] - 1e-6) | (high > limits[:, 1] + 1e-6)
        names = [self._robot.joint_names[index] for index in self._hand_joint_ids]
        safe_offenders = [names[i] for i in torch.nonzero(outside_safe).flatten().tolist()]
        usd_offenders = [names[i] for i in torch.nonzero(outside_usd).flatten().tolist()]
        if safe_offenders or usd_offenders:
            raise RuntimeError(
                f'S7 动作目标越界：safe={safe_offenders}, USD={usd_offenders}'
            )
        if float(span.mean().item()) <= 1e-6:
            raise RuntimeError('S7 14D 动作对 canonical16 没有控制作用')
        neutral = targets[1]
        result.update(
            ok=True,
            mapping_id=ABDUCTION_MAPPING_ID,
            probe_targets_rad=targets.tolist(),
            per_joint_span_rad=span.tolist(),
            min_joint_span_rad=float(span.min()),
            mean_joint_span_rad=float(span.mean()),
            neutral_targets_rad=neutral.tolist(),
            manifold_zero_target_gap_rad=float(
                (neutral - self._nominal_joint_targets[0, self._hand_joint_ids]).abs().max()
            ),
            joint_limit_overhang_rad=0.0,
            joint_limit_overhang_names=[],
            joint_limit_note=(
                'targets checked against both the original safe endpoints and actual USD limits'
            ),
            within_original_safe_limits=True,
            within_usd_limits=True,
            action_semantics_changed=sorted(ABDUCTION_ACTION_INDICES),
            legacy_ring_little_residuals_frozen_zero=True,
        )
        return result

    def _build_identity(self):
        identity = super()._build_identity()
        identity.update(contract_id='firm_grasp_v1', task_id='FirmGrasp-v1',
                        task_version=TASK_VERSION, record_schema_version='firm_grasp_episode.v1')
        identity['firm_grasp'] = {'contract': asdict(self.firm_contract),
            'combinations': [list(c) for c in COMBINATIONS], 'scope': 'F0_F1_F2_G1_only',
            'F3_F4': 'WAIT_USER_APPROVAL', 'test_disturbances_enabled': False,
            'success_rule': 'same_thumb_plus_two_distinct_nonthumb_PAD_120_continuous_real_substeps',
            'actor_compatibility': (
                'numeric_legacy161D_prefix_and_appended_release_clock_layout_only; '
                'action_history_slots_8_9_now_index_middle_abduction; '
                'not_full_legacy_actor_action_semantic_compatibility; '
                'fresh_or_same_firm_grasp_v1.7.0_checkpoint_only; actor_warmstart_disabled; '
                'distinct_from_thumb_pad_grasp_v1.6.0'
            ),
            'new_guard_is_separate': True, 'palm_or_nonpad_support_allowed': False}
        observation_dimensions = {'actor': ACTOR_DIM, 'critic': CRITIC_DIM, 'action': ACTION_DIM}
        identity['observations'].update(
            policy_dim=ACTOR_DIM, critic_dim=CRITIC_DIM,
            legacy_policy_prefix_dim=LEGACY_ACTOR_DIM,
            legacy_critic_prefix_dim=LEGACY_CRITIC_DIM,
            legacy_critic_privileged_dim=LEGACY_PRIVILEGED_DIM,
            legacy_critic_privileged_position={
                'start': 114, 'stop': 175},
            policy_prefix_preserved=False, critic_prefix_preserved=False,
            policy_prefix_numeric_layout_preserved=True,
            policy_layout=('native_policy114(previous_executed_actions100:114) '
                           '+ pad_features47(114:161) + release_clock1(161)'),
            critic_layout=('native_policy114 + native_privileged61(114:175; '
                           'executed_actions142:156,raw_actions156:170) '
                           '+ pad_features47(175:222) + release_clock1(222)'),
            native_policy_dim=114,
            native_critic_dim=175,
            pad_feature_append_dim=47,
            pad_features_position={
                'actor': {'start': 114, 'stop': 161},
                'critic': {'start': 175, 'stop': 222},
            },
            previous_executed_actions_position={
                'actor': {'start': 100, 'stop': 114},
                'critic': {'start': 100, 'stop': 114},
            },
            critic_action_blocks={
                'executed_actions': {'start': 142, 'stop': 156},
                'raw_actions': {'start': 156, 'stop': 170},
            },
            appended_clock_dim=CLOCK_DIM,
            appended_clock_position={'policy': LEGACY_ACTOR_DIM, 'critic': LEGACY_CRITIC_DIM},
            appended_clock={
                'source': 'episode_length_buf',
                'formula': 'clamp(episode_length_buf.float()/support_release_step,0,3)',
                'support_release_step': int(self.cfg.support_release_step),
                'clamp_max': 3.0,
                'reset_value': 0.0,
                'semantics': ('known episode age relative to fixed release timing; '
                              'not a future outcome, contact age replacement or success signal'),
                'legacy_contact_age_unchanged': True,
            })
        identity['observations'].update(
            action_history_dim=ACTION_DIM,
            action_history_semantics_changed=list(ABDUCTION_ACTION_INDICES),
            action_history_slot_names=list(ABDUCTION_ACTION_ORDER),
            action_history_prefix_numeric_positions={
                'actor_previous_executed_actions': {'start': 100, 'stop': 114},
                'critic_policy_previous_executed_actions': {'start': 100, 'stop': 114},
                'critic_privileged_executed_actions': {'start': 142, 'stop': 156},
                'critic_privileged_raw_actions': {'start': 156, 'stop': 170},
            },
            action_history_semantics_changed_observation_columns={
                'actor_previous_executed_actions': [108, 109],
                'critic_policy_previous_executed_actions': [108, 109],
                'critic_privileged_executed_actions': [150, 151],
                'critic_privileged_raw_actions': [164, 165],
            },
            action_history_semantics_note=(
                'slots 8/9 were ring/little proximal residuals in the legacy mapper and are now '
                'index/middle q0 abduction; the numeric 161/222 prefix layout remains but full '
                'legacy actor/action semantics do not'
            ),
        )
        identity['observations']['new_features'] = (
            identity['observations']['new_features'] + ',release_clock1')
        identity['physics']['observation_dims'] = dict(observation_dimensions)
        identity['asset_config']['physics']['observation_dims'] = dict(observation_dimensions)
        identity['criteria'].update(nonthumb_requirement='two_distinct_PAD_load_bearing_fingers',
            required_true_substeps=120, min_pad_force_n=.15, min_pad_load_share=.05,
            hold_streak_gate='six_separate_fixed_triplet_120Hz_streaks', palm_may_carry_load=False)
        identity['stage'].update(success_criterion='three_PAD_unsupported_1s_G1',
            goal_status='G1_NOT_G2_NO_DISTURBANCE_OR_LONG_HOLD')
        identity['physics']['physical_hold_guard'].update(
            success_semantics='same_three_PADs_continuous120_substeps_with_original_seven_faults',
            legacy_guard_role='original_actor_observation_compatibility_and_shared_physical_faults',
            support_release_signal='stem_detached AND each recorded attachment force/torque component equals zero',
            support_wrench_zero_provenance='new_guard_checks_recorded_application_buffers_every_substep; not an independent sensor or all_external_force_measurement')
        ramp = support_ramp_identity(release_step=int(self.cfg.support_release_step),
            ramp_policy_steps=self._support_ramp_policy_steps,
            policy_dt_s=float(identity['physics']['policy_dt_s']))
        identity['physics']['support']['release_ramp'] = ramp
        identity['physics']['support']['release_ramp'].update(
            action_14d_semantics_unchanged=False,
            action_semantics_changed_from_legacy=list(ABDUCTION_ACTION_INDICES),
        )
        identity['physics']['physical_hold_guard']['support_ramp_note'] = (
            'ramp only scales the same declared stem wrench before the single PhysX permanent-wrench write;'
            ' release flag still flips at the fixed release step and G1 still requires exact zero support')
        identity['criteria'].update(support_ramp_policy_steps=ramp['support_ramp_policy_steps'],
            support_ramp_window_s=ramp['ramp_window_s'],
            support_zero_starts_at_policy_step=ramp['release_step'])
        identity['stage'].update(support_ramp_policy_steps=ramp['support_ramp_policy_steps'],
            support_ramp_start_step=ramp['ramp_start_step'])
        identity['preparation_condition'] = {'kind': 'support_release_ramp',
            'gain_claim': 'preparation_condition_change_not_ppo_learning_gain',
            'note': SCOPE_NOTE,
            'same_condition_rerun_required_for_any_actor_comparison': True,
            'F3_F4': 'NOT_AUTHORIZED'}
        identity['preparation_condition'].update(
            reset_condition=('oriented_reset_reference'
                if self._oriented_reset_condition.enabled else 'legacy_canonical_reset'),
            reset_gain_claim='initialization_condition_change_not_ppo_learning_gain')
        identity['initial_pose']['oriented_reset'] = self._oriented_reset_identity()
        identity['initial_pose']['base_lift_m'] = float(getattr(self.cfg, 'base_lift_m', 0.0))
        identity['initial_pose']['base_lift_semantics'] = (
            'reset_only_rigid_vertical_offset_of_fixed_base_and_reference_fruit; '
            'in_episode_state_writes=False; relative_hand_fruit_geometry_unchanged; '
            'physics_mass_friction_pd_limits_and_G1_thresholds_unchanged')
        identity['initial_pose']['reset_reference_offset_hand_m'] = list(
            getattr(self.cfg, 'reset_reference_offset_hand_m', (0.0, 0.0, 0.0)))
        identity['initial_pose']['reset_reference_offset_semantics'] = (
            'reset_only_hand_frame_translation_of_reference_fruit_and_stem_anchors; '
            'in_episode_state_writes=False; fruit_geometry_mass_friction_pd_limits_unchanged')
        identity['contact']['palm_may_carry_load'] = False
        identity['goal_status_semantics'] = 'G1_same_three_PAD_unsupported_1s_F2_only'
        identity['criteria']['contact_source'] = 'confirmed_surface_real_point_normal_loads_same_triplet'
        identity['criteria']['success_criterion'] = 'three_PAD_unsupported_1s_G1'
        identity['criteria'].update(ground_free_required_for_G1=True,
            ground_plane_z_m=0.0, min_fruit_bottom_clearance_m=MIN_FRUIT_BOTTOM_CLEARANCE_M,
            ground_semantics='ground contact is external support; every G1 substep requires fruit bottom > 0.005 m above z=0')
        identity['physics']['ground'] = {'plane_z_m': 0.0,
            'min_fruit_bottom_clearance_m': MIN_FRUIT_BOTTOM_CLEARANCE_M,
            'counts_as_support': True,
            'check': 'fruit_bottom_clearance_m > 0.005 in every physical substep',
            'changes': 'none; gravity, collision filtering and fruit mass unchanged'}
        identity['goal'] = {'deliverable': 'PPO_updated_checkpoint_verified_G1_three_PAD_1s',
                            'success_rate': 'no_high_rate_gate', 'F3_F4': 'not_authorized'}
        identity['reward'] = {'profile': 'firm_G1_v1',
            'weights': {'thumb_approach':.2, 'second_nonthumb_approach':.3,
                'thumb_pad':.3, 'nonthumb_PAD_count':.3, 'third_load':2.,
                'unsupported_triple':5., 'triplet_progress':12., 'terminal_success':100.,
                'nonpad':-1., 'unknown':-.2, 'force_excess_squared':-.03,
                'force_growth':-.03, 'action_delta':-.02, 'hard_fault':-25., 'drop':-15.},
            'failure_positive_mask': True, 'no_force_sum_reward': True}
        identity['temporal_filter'] = temporal_identity()
        identity['action_mapping'] = abduction_mapping_identity()
        identity['action_mapping'].update(
            command_warmup={
                'policy_steps': 35,
                'gain': 'clip(episode_policy_step/35,0,1)',
                'preserved_from_parent': True,
                'operation': 'before_new_stable_action_filter_original_temporal_limits',
            },
            open_hand_baseline={
                'preserved': True,
                'bypasses_new_mapper': True,
                'uses_parent_neutral_targets': True,
            },
        )
        identity['asset_config']['criteria'] = identity['criteria']
        return identity

    @torch.inference_mode()
    def _get_observations(self):
        """保留父级 161/222 前缀，并分别在各自末尾追加真实释放时钟。"""

        observations = super()._get_observations()
        policy = observations['policy']
        critic = observations['critic']
        if policy.ndim != 2 or tuple(policy.shape[1:]) != (LEGACY_ACTOR_DIM,):
            raise RuntimeError(f'父级 policy 维度漂移：{tuple(policy.shape)}')
        if critic.ndim != 2 or tuple(critic.shape[1:]) != (LEGACY_CRITIC_DIM,):
            raise RuntimeError(f'父级 critic 维度漂移：{tuple(critic.shape)}')
        if policy.shape[0] != self.num_envs or critic.shape[0] != self.num_envs:
            raise RuntimeError('父级观测批大小与 num_envs 不一致')
        clock = release_clock(self.episode_length_buf, self.cfg.support_release_step)
        if tuple(clock.shape) != (self.num_envs, CLOCK_DIM):
            raise RuntimeError(f'释放时钟形状漂移：{tuple(clock.shape)}')
        if not bool(torch.isfinite(clock).all()):
            raise RuntimeError('释放时钟包含非有限值')
        observations = dict(observations)
        observations['policy'] = torch.cat((policy, clock), dim=1)
        observations['critic'] = torch.cat((critic, clock), dim=1)
        return observations

    def _oriented_reset_identity(self) -> dict:
        """S5 初始化条件声明；实际数值落点见 ``last_reset_poses``。"""

        condition = self._oriented_reset_condition
        return {'enabled': condition.enabled,
            'mode': ('hand_frame_reference_rigid_rotation'
                if condition.enabled else 'legacy_canonical_reset'),
            'wrist_tilt_deg': condition.wrist_tilt_deg,
            'wrist_tilt_axis_world_xyz': list(condition.wrist_tilt_axis_world_xyz),
            'configured_base_init_rot_xyzw': list(condition.init_state_rot_xyzw()),
            'reset_fruit_reference_hand_m': (None if condition.reset_fruit_reference_hand_m is None
                else list(condition.reset_fruit_reference_hand_m)),
            'reference_frame': 'base_link_local_metres',
            'fruit_quat_rule': 'normalized_base_link_world_quat_XYZW',
            'stem_anchor_rule': ('fruit_pos + R*(0,0,radius); '
                'plant = rest + R*(0,0,stem_visual_height)'),
            'applies_to': ['fruit_root_position', 'fruit_root_quaternion',
                'stem_rest_attachment_anchor', 'stem_plant_anchor',
                'stem_anchor_target_quat', 'stem_visual_segment'],
            'scope': 'reset_only_no_episode_wrist_motion',
            'pregrasp_distance_defines_final_center': not condition.enabled,
            'legacy_fields_role': ('canonical intermediate placement only; final pose '
                'and actual reference are recorded in env.last_reset_poses'),
            'actual_reset_record': 'env.last_reset_poses',
            'gain_claim': 'initialization_condition_change_not_ppo_learning_gain',
            'ppo_gain_claim': False,
            'fruit_geometry': 'fixed 67 mm sphere; mass, friction, PD and gravity unchanged'}

    def _live_physical_substep_state(self, *, forces):
        state = super()._live_physical_substep_state(forces=forces)
        if self._firm_ready:
            self._firm_last_state = state
            truth = self.pad_observer.latest
            self._firm_metrics = compute_contact_metrics(truth['pad_force_n'],
                truth['nonpad_force_n'], truth['unknown'].any(1), contract=self.firm_contract,
                extra_nonpad_force_n=state.palm_force_n+truth['nonpad_force_n'].sum(1))
        return state

    def _observe_physical_hold_substep(self, scene_dt_s):
        consumed = super()._observe_physical_hold_substep(scene_dt_s)
        if consumed and self._firm_ready:
            floor_free = ground_free_mask(self._firm_last_state.fruit_pose,
                self._fruit_axes_m)
            qualified = self._firm_metrics['combination_qualified'] & floor_free
            self.firm_guard.observe(state=self._firm_last_state,
                qualified=qualified,
                hard_fault_any=self._physical_hold_guard.hard_fault_any,
                support_zero=(self._last_stem_force_w==0).all(1)&(self._last_stem_torque_w==0).all(1),
                counter=self._sim_step_counter, dt=scene_dt_s)
            if self.pad_trace_enabled:
                self._record_firm_pad_trace_row()
        return consumed

    def _record_firm_pad_trace_row(self):
        """只追加同一 guard 读值的诊断字段；不控制、不覆盖 effort。"""

        row = self._pad_trace[-1]
        bottom_clearance = fruit_bottom_clearance(
            self._firm_last_state.fruit_pose, self._fruit_axes_m)
        row.update(support_release_scale=(None if self._support_ramp_scale is None
                else self._support_ramp_scale.tolist()),
            firm_streak=self.firm_guard.streak.tolist(),
            firm_peak_streak=self.firm_guard.peak.tolist(),
            firm_combo_qualified=self._firm_metrics['combination_qualified'].tolist(),
            firm_load_share=self._firm_metrics['load_share'].tolist(),
            palm_force_n=self._firm_last_state.palm_force_n.tolist(),
            relative_speed_m_s=self._firm_last_state.relative_speed_m_s.tolist(),
            fruit_in_hand_m=self._firm_last_state.fruit_in_hand_m.tolist(),
            fruit_angular_velocity_w=self._tomato.data.root_ang_vel_w.torch.tolist(),
            hand_pose_w=self._firm_last_state.ee_pose.tolist(),
            hand_angular_velocity_w=self._robot.data.body_ang_vel_w.torch[:,self._ee_body_id].tolist(),
            joint_velocity=self._robot.data.joint_vel.torch.tolist(),
            joint_applied_effort_nm=self._firm_last_state.applied_effort.tolist(),
            effort_joint_names=[self._robot.joint_names[j] for j in self._safety_joint_ids],
            joint_effort_limits_nm=self._firm_last_state.effort_limits.tolist(),
            effort_source_name=self._effort_proxy_name,
            fruit_bottom_clearance_m=bottom_clearance.tolist(),
            ground_free=(bottom_clearance > MIN_FRUIT_BOTTOM_CLEARANCE_M).tolist(),
            test_disturbance_force_w=torch.zeros((self.num_envs,3),device=self.device).tolist(),
            test_disturbance_torque_w=torch.zeros((self.num_envs,3),device=self.device).tolist())

    # --------------------------------------------------------- 真实支撑力写入
    @torch.inference_mode()
    def _unscaled_attachment_wrench(self):
        """父级果梗柔顺扳手的逐字镜像（缩放前），供等价审计直接对比。

        公式、运算顺序、上限与 dtype 与父级
        ``FixedFruitGraspEnv._apply_attachment_wrench`` 完全一致；本方法不写
        PhysX、不写日志，也不改任何动态状态。
        """
        pose = self._tomato.data.root_pose_w.torch
        velocity = self._tomato.data.root_lin_vel_w.torch
        angular = self._tomato.data.root_ang_vel_w.torch
        local = torch.zeros((self.num_envs, 3), device=self.device)
        local[:, 2] = self._fruit_axes_m[:, 2]
        offset = quat_apply(pose[:, 3:7], local)
        attachment = pose[:, :3] + offset
        attachment_velocity = velocity + torch.linalg.cross(angular, offset, dim=1)
        omega = float(self.cfg.stem_linear_natural_frequency_rad_s)
        acceleration = omega**2 * (self._stem_rest_attachment_pos_w - attachment)
        acceleration -= 2 * float(self.cfg.stem_linear_damping_ratio) * omega * attachment_velocity
        acceleration -= torch.tensor(self.cfg.sim.gravity, device=self.device).unsqueeze(0)
        force = self._fruit_mass_kg.unsqueeze(1) * acceleration
        inertia = (0.4 * self._fruit_mass_kg * self._fruit_axes_m.square().mean(dim=1)).clamp_min(1e-8)
        angular_omega = float(self.cfg.stem_angular_natural_frequency_rad_s)
        torque = inertia.unsqueeze(1) * (angular_omega**2 * quat_box_minus(self._stem_anchor_quat_w, pose[:, 3:7]) - 2 * float(self.cfg.stem_angular_damping_ratio) * angular_omega * angular)
        force *= (float(self.cfg.maximum_stem_force_n) / force.norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp(max=1)
        torque *= (float(self.cfg.maximum_stem_torque_nm) / torque.norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp(max=1)
        return force, torque

    @torch.inference_mode()
    def _support_release_scale(self):
        """读取当前真实策略步对应的 release 缩放因子（[N] float32）。"""
        scale = support_release_scale(self.episode_length_buf,
            release_step=int(self.cfg.support_release_step),
            ramp_policy_steps=self._support_ramp_policy_steps)
        self._support_ramp_scale = scale
        return scale

    @torch.inference_mode()
    def _apply_attachment_wrench(self) -> None:
        """在真实写入 PhysX 之前缩放支撑力/力矩，日志即实际应用值。

        释放掩码语义与父级一致（``_stem_detached`` 为真则严格为零）；
        斜坡窗内支撑仍严格为正，所以渐退期间的任何物理子步都不会被
        三指守卫计成零支撑成功。本方法只写支撑外力缓存与遮蔽器，不写
        果实/关节/根状态，也不改重力、摩擦、PD、限位或物理故障。
        """
        force, torque = self._unscaled_attachment_wrench()
        attached = ~self._stem_detached
        force *= attached.unsqueeze(1)
        torque *= attached.unsqueeze(1)
        force, torque = scale_support_wrench(force, torque, self._support_release_scale())
        self._last_stem_force_w.copy_(force)
        self._last_stem_torque_w.copy_(torque)
        self._tomato.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force.unsqueeze(1), torques=torque.unsqueeze(1), is_global=True)

    def _tracker_state(self):
        state = super()._tracker_state()
        if self._firm_ready:
            guard = self.firm_guard
            state.update(required_contact=self._firm_metrics['any_qualified'] if self._firm_metrics is not None else torch.zeros(self.num_envs,device=self.device,dtype=torch.bool),
                contact_count=self._firm_metrics['qualified_finger_count'] if self._firm_metrics is not None else torch.zeros(self.num_envs,device=self.device,dtype=torch.long),
                hold_age=guard.streak.amax(1)//self.cfg.decimation,
                hold_ok=guard.last_ok.any(1), physical_hold_streak=guard.streak.amax(1),
                physical_hold_peak_streak=guard.peak.amax(1))
        return state

    def _get_dones(self):
        # 父实现保留旧观测/安全状态，之后将旧成功终止替换为新三指守卫。
        terminated, truncated = super()._get_dones()
        if not self._firm_ready:
            return terminated, truncated
        self.firm_guard.complete_policy_interval()
        old_success = self._terminal_success.clone()
        success = (self.firm_guard.streak.amax(1) >= self.firm_contract.required_substeps)
        success &= self._stem_detached & ~self._drop_now & ~self._physical_hold_guard.hard_fault_any
        terminated = (terminated & ~old_success) | success
        truncated = (self.episode_length_buf >= self.max_episode_length-1) & ~terminated
        clear_old = old_success & ~success
        self._terminal_reason[clear_old] = 0
        self._terminal_reason[truncated] = 2
        self._terminal_reason[success] = 1
        self._terminal_success.copy_(success)
        return terminated, truncated

    def _get_rewards(self):
        if not self._firm_ready or self._firm_metrics is None:
            return super()._get_rewards()
        if not self._state_ready:
            self._refresh_task_state()
        features, gap, alignment, truth = self._pad_features()
        pad, nonpad = truth['pad_force_n'], truth['nonpad_force_n']
        released = self._stem_detached.float()
        third = pad[:,1:].topk(2,dim=1).values[:,1]
        relative = features[:,:15].reshape(self.num_envs,5,3)*.08
        all_gap = torch.linalg.vector_norm(relative,dim=-1)-self._fruit_axes_m[:,0,None]
        approach = torch.exp(-all_gap.abs()/.025)*alignment.clamp_min(0).square()
        hard = self._physical_hold_guard.hard_fault_any
        terms = {'thumb_approach':.2*approach[:,0],
            'second_nonthumb_approach':.3*approach[:,1:].topk(2,dim=1).values[:,1],
            'thumb_pad':.3*truth['thumb_pad_qualified'].float(),
            'nonthumb_PAD_count':.3*(pad[:,1:]>=self.firm_contract.f_pad_min_n).float().sum(1),
            'third_load':2.*(third/self.firm_contract.f_pad_min_n).clamp(max=1),
            'unsupported_triple':5.*released*self._firm_metrics['any_qualified'].float(),
            'triplet_progress':12.*(self.firm_guard.streak.amax(1).float()/120).clamp(max=1),
            'terminal_success':100.*self._terminal_success.float(),
            'nonpad':-nonpad.sum(1).clamp(max=15),
            'unknown':-.2*truth['unknown'].any(1).float(),
            'force_excess_squared':-.03*torch.relu(self._last_physical_force_n-self._soft_high).square().sum(1),
            'force_growth':-.03*torch.relu(self._last_physical_force_n-self._previous_force_n).sum(1),
            'action_delta':-.02*(self._executed_actions-self._previous_executed_actions).square().mean(1),
            'hard_fault':-25.*hard.float(), 'drop':-15.*self._drop_now.float()}
        terms = block_failure_positive_terms(terms, hard|self._drop_now)
        self.extras['pad_reward_terms'] = {k:float(v.mean()) for k,v in terms.items()}
        return sum(terms.values())

    def _finalize_episodes(self, env_ids):
        cursor = len(getattr(self,'completed_episodes',[]))
        super()._finalize_episodes(env_ids)
        if self._firm_ready:
            for row in self.completed_episodes[cursor:]:
                i = row['env_id']
                row.update(schema_version='firm_grasp_episode.v1',task_id='FirmGrasp-v1',
                    task_version=TASK_VERSION,success_criterion='three_PAD_unsupported_1s_G1',
                    goal_status='G1_REACHED' if row['success'] else 'G1_NOT_REACHED',
                    firm_peak_streak_by_combo=self.firm_guard.peak[i].tolist(),
                    firm_final_streak_by_combo=self.firm_guard.streak[i].tolist(),
                    firm_combinations=[list(c) for c in COMBINATIONS],
                    support_ramp_policy_steps=self._support_ramp_policy_steps,
                    legacy_two_finger_streak=int(self._physical_hold_guard.streak[i]),
                    physical_hold_streak_semantics='fixed_triplet_true_PAD_every120Hz_substep')

    # ------------------------------------------------------- S5 定向 reset
    def _reset_measurement_randomization(self, env_ids):
        """canonical reset 的测量噪声钩子；随后仅对本次 ids 写实际 reference。"""

        super()._reset_measurement_randomization(env_ids)
        self._apply_oriented_reset(env_ids)

    def _move_stem_visual(self, env_ids, fruit_side_w, plant_side_w):
        """reference 模式用已更新的锚点；避免 canonical 局部 old-world-Z。"""

        if self._oriented_reset_condition.enabled:
            super()._move_stem_visual(env_ids,
                self._stem_rest_attachment_pos_w[env_ids],
                self._stem_plant_anchor_pos_w[env_ids])
            return
        super()._move_stem_visual(env_ids, fruit_side_w, plant_side_w)

    @torch.inference_mode()
    def _apply_oriented_reset(self, env_ids):
        """在 canonical 中间落点之后、首个策略步之前写入实际 reference。"""

        condition = self._oriented_reset_condition
        if not condition.enabled:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if self.episode_length_buf is not None and bool((self.episode_length_buf[ids] != 0).any()):
            raise RuntimeError('S5 reference reset 必须发生在 episode_length_buf 归零之后')
        base_pose = self._robot.data.body_pose_w.torch[ids, self._ee_body_id]
        offsets = getattr(self, '_reset_reference_offsets_m', None)
        fixed_offset = tuple(getattr(self.cfg, 'reset_reference_offset_hand_m', (0., 0., 0.)))
        if offsets is None and not any(fixed_offset):
            fruit_pos, fruit_quat, stem_rest, stem_plant = oriented_reset_targets(
                base_pose[:, :3], base_pose[:, 3:7], condition.reset_fruit_reference_hand_m,
                self._fruit_axes_m[ids, 2], float(self.cfg.stem_visual_height_m))
        else:
            # 诊断/任务条件：reference 在 reset 时按手系平移一次。
            if offsets is not None:
                offsets = offsets.to(device=self.device, dtype=base_pose.dtype)
                if tuple(offsets.shape) != (self.num_envs, 3):
                    raise ValueError(f'reset reference offsets 形状错误：{tuple(offsets.shape)}')
            reference = (base_pose.new_tensor(condition.reset_fruit_reference_hand_m).reshape(1, 3)
                         + base_pose.new_tensor(fixed_offset).reshape(1, 3)).expand(len(ids), -1)
            if offsets is not None:
                reference = reference + offsets[ids]
            base_quat = torch.nn.functional.normalize(base_pose[:, 3:7], dim=1)
            fruit_pos = base_pose[:, :3] + quat_apply(base_quat, reference)
            fruit_quat = base_quat
            up_local = torch.zeros((len(ids), 3), device=self.device, dtype=base_pose.dtype)
            up_local[:, 2] = 1.0
            up = quat_apply(fruit_quat, up_local)
            stem_rest = fruit_pos + up * self._fruit_axes_m[ids, 2].reshape(-1, 1)
            stem_plant = stem_rest + up * float(self.cfg.stem_visual_height_m)
        axes = self._fruit_axes_m[ids]
        clearance = self._hand_collision_clearance_for(fruit_pos, axes, ids)
        minimum = float(self.cfg.minimum_initial_clearance_m)
        if not bool(torch.isfinite(clearance).all()):
            raise RuntimeError('S5 reference reset 的初始 clearance 非有限，拒绝写入')
        if bool((clearance < minimum).any()):
            raise RuntimeError('S5 reference reset 破坏原初始无穿透下限')
        bottom = fruit_bottom_clearance(fruit_pos, axes)
        if not bool(torch.isfinite(bottom).all()):
            raise RuntimeError('S5 reference reset 的果实地面净空非有限，拒绝写入')
        if bool((bottom <= MIN_FRUIT_BOTTOM_CLEARANCE_M).any()):
            raise RuntimeError('S5 reference reset 使果实最低点不高于地面净空下限')
        fruit_pose = torch.cat((fruit_pos, fruit_quat), dim=1)
        self._tomato.write_root_pose_to_sim_index(root_pose=fruit_pose, env_ids=ids)
        self._tomato.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((len(ids), 6), device=self.device), env_ids=ids)
        if not hasattr(self, '_oriented_reset_written_pose_w'):
            self._oriented_reset_written_pose_w = torch.zeros(
                (self.num_envs, 7), device=self.device)
        self._oriented_reset_written_pose_w[ids] = fruit_pose
        self._stem_rest_attachment_pos_w[ids] = stem_rest
        self._stem_plant_anchor_pos_w[ids] = stem_plant
        self._stem_anchor_quat_w[ids] = fruit_quat

    def _record_oriented_reset_pose(self, ids):
        """把实际 reference 与完整刚体旋转追加到 Fixed.last_reset_poses。"""

        condition = self._oriented_reset_condition
        if not condition.enabled or not getattr(self, 'last_reset_poses', None):
            return
        wanted = {int(value) for value in torch.as_tensor(
            ids, device=self.device, dtype=torch.long).flatten().tolist()}
        base_pose = self._robot.data.body_pose_w.torch[:, self._ee_body_id]
        written = getattr(self, '_oriented_reset_written_pose_w', None)
        fruit_pose = (self._tomato.data.root_pose_w.torch if written is None else written)
        for row in self.last_reset_poses:
            index = int(row['env_id'])
            if index not in wanted:
                continue
            if not bool(torch.isfinite(base_pose[index]).all()):
                raise RuntimeError('S5 reset 记录的 base_link 位姿非有限')
            base_norm = base_pose[index, 3:7].norm()
            if not bool(torch.isfinite(base_norm)) or float(base_norm) <= 1e-8:
                raise RuntimeError('S5 reset 记录的 base_link 四元数无效')
            if not bool(torch.isfinite(fruit_pose[index]).all()):
                raise RuntimeError('S5 reset 记录的目标果实位姿非有限')
            base_q = torch.nn.functional.normalize(base_pose[index, 3:7], dim=0)
            actual = quat_apply_inverse(base_q, fruit_pose[index, :3] - base_pose[index, :3])
            row_bottom = fruit_bottom_clearance(
                fruit_pose[index].unsqueeze(0), self._fruit_axes_m[index].unsqueeze(0))
            if not bool(torch.isfinite(row_bottom).all()):
                raise RuntimeError('S5 reset 记录的果实地面净空非有限')
            row.update(oriented_reset_enabled=True,
                fruit_world_xyz_m=[float(value) for value in fruit_pose[index, :3].tolist()],
                fruit_world_quat_xyzw=[float(value) for value in fruit_pose[index, 3:7].tolist()],
                reset_fruit_reference_hand_m=list(condition.reset_fruit_reference_hand_m),
                reset_reference_frame='base_link_local_metres',
                actual_reference_hand_m=[float(value) for value in actual.tolist()],
                wrist_tilt_deg=condition.wrist_tilt_deg,
                wrist_tilt_axis_world_xyz=list(condition.wrist_tilt_axis_world_xyz),
                fruit_quat_xyzw_equals_base_link_quat_xyzw=bool(torch.allclose(
                    fruit_pose[index, 3:7], base_q)),
                fruit_bottom_clearance_m=float(row_bottom[0]),
                stem_rest_attachment_world_m=[float(value) for value in
                    self._stem_rest_attachment_pos_w[index].tolist()],
                stem_plant_anchor_world_m=[float(value) for value in
                    self._stem_plant_anchor_pos_w[index].tolist()],
                stem_anchor_quat_xyzw=[float(value) for value in
                    self._stem_anchor_quat_w[index].tolist()],
                pregrasp_distance_defines_final_center=False,
                gain_claim='initialization_condition_change_not_ppo_learning_gain')

    @torch.inference_mode()
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = self._all_env_ids if env_ids is None else env_ids
        self._record_oriented_reset_pose(ids)
        if self._firm_ready:
            self.firm_guard.reset(ids)
