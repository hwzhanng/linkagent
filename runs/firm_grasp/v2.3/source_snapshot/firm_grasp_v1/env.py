"""F3：原控制/观测前缀不变，新增三秒合围、旋转与显式抗扰守卫。"""
import torch
from isaaclab.utils.math import quat_mul, quat_conjugate
from research.thumb_pad_grasp_v1.env import ThumbPadGraspEnv
from .f2_env import FirmGraspEnv as F2Env, make_config as f2_config
from .f2_env import ground_free_mask, COMBINATIONS
from .g2_guard import G2HoldGuard
from .disturbance import pulse_wrench
from .reward_margin import third_load_margin_reward
TASK_VERSION = 'firm_grasp_v2.3.0'


def make_config(**kwargs):
    legacy = bool(kwargs.pop('legacy_thumb_baseline', False))
    release_power = float(kwargs.pop('support_release_power', 1.))
    if not 1. <= release_power <= 4.:
        raise ValueError('释放曲线指数必须位于[1,4]')
    preparation_offset = kwargs.pop('preparation_reference_offset_hand_m', None)
    if preparation_offset is None:
        preparation_offset = (0., 0., 0.)
    if (len(preparation_offset) != 3 or
            not all(__import__('math').isfinite(v) and abs(v) <= .01 for v in preparation_offset)):
        raise ValueError('准备弹簧参考位移每轴必须有限且<=10mm')
    p = {key: kwargs.pop(key, default) for key, default in {
        'goal': 'g2', 'hold_seconds': 3., 'disturbance_mode': 'quiet',
        'disturbance_axis': 0, 'disturbance_sign': 1., 'force_ratio': .1,
        'torque_ratio': .01}.items()}
    if p['goal'] not in ('hold', 'g2') or p['hold_seconds'] not in (1., 2., 3.):
        raise ValueError('无效保持课程')
    if p['goal'] == 'g2' and p['hold_seconds'] != 3.:
        raise ValueError('G2必须3秒')
    if p['disturbance_mode'] not in ('quiet', 'force', 'torque'):
        raise ValueError('未知扰动模式')
    if p['disturbance_axis'] not in (0, 1, 2) or p['disturbance_sign'] not in (-1., 1.):
        raise ValueError('扰动轴/方向错误')
    if not 0 <= p['force_ratio'] <= .1 or not 0 <= p['torque_ratio'] <= .01:
        raise ValueError('超过冻结扰动上限')
    if p['goal'] == 'hold' and p['disturbance_mode'] != 'quiet':
        raise ValueError('保持课程不混入扰动')
    cfg = f2_config(**kwargs)
    cfg.f3_protocol = p
    cfg.legacy_thumb_baseline = legacy
    cfg.preparation_reference_offset_hand_m = tuple(preparation_offset)
    cfg.support_release_power = release_power
    cfg.task_version = TASK_VERSION
    return cfg


class FirmGraspEnv(F2Env):
    def __init__(self, cfg, **kwargs):
        self._f3_ready = False
        self._prepare_target_offsets_m = None
        self._support_release_power_by_env = None
        self._f3_protocol = dict(cfg.f3_protocol)
        super().__init__(cfg, **kwargs)
        if getattr(cfg, 'legacy_thumb_baseline', False):
            from research.small_fruit_pinch_v1.source.temporal import CriticalDampedActionFilter
            self._filter = CriticalDampedActionFilter(self.num_envs, 14,
                dt_s=float(cfg.sim.dt)*int(cfg.decimation), device=self.device,
                limits=self._filter.limits)
        self.g2_guard = G2HoldGuard(self.num_envs, device=self.device, initial_counter=self._sim_step_counter)
        self._release_start_counter = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self._post_release_substep = torch.full_like(self._release_start_counter, -1)
        self._test_force = torch.zeros((self.num_envs, 3), device=self.device)
        self._test_torque = torch.zeros_like(self._test_force)
        self._pulse_counts = torch.zeros_like(self._release_start_counter)
        self._f3_geometry = None
        self._prepare_target_offsets_m = torch.tensor(cfg.preparation_reference_offset_hand_m,
            device=self.device).reshape(1,3).expand(self.num_envs,3).clone()
        self._last_preparation_offset_w = torch.zeros_like(self._prepare_target_offsets_m)
        self._support_release_power_by_env = torch.full((self.num_envs,),cfg.support_release_power,device=self.device)
        self._f3_ready = True
        self._identity = self._build_identity()

    def _build_identity(self):
        d = super()._build_identity()
        p = dict(self._f3_protocol)
        d['task_version'] = TASK_VERSION
        d['firm_grasp'].update(scope='F3_F4', F3_F4='AUTHORIZED',
            test_disturbances_enabled=p['disturbance_mode'] != 'quiet',
            success_rule='same_triplet_geometry_rotation_continuous_3s')
        d['f3_protocol'] = {**p, 'physics_hz':120, 'required_substeps':int(p['hold_seconds']*120),
            'strict_start_max_substep':60, 'pulse_start_substep':180, 'pulse_duration_substeps':12,
            'recovery_deadline_substep':252, 'recovery_max_speed_m_s':.02,
            'min_nonthumb_separation_m':.006, 'max_opposition_dot':-.3,
            'max_rotation_deg':10., 'max_relative_angular_speed_rad_s':.5,
            'frame':'hand_at_application; fruit_COM', 'support_and_test_separate':True,
            'strict_early_start':p['goal']=='g2', 'slip':'not_measured_not_hard_gated',
            'old_observation_formulas_and_action_mapping_unchanged':True}
        criterion = 'G2_3s' if p['goal']=='g2' else 'curriculum_hold_not_G2'
        d['criteria'].update(required_true_substeps=int(p['hold_seconds']*120),success_criterion=criterion,
            unsupported_hold_s=p['hold_seconds'], goal_hold_s=p['hold_seconds'],
            hold_streak_gate='separate_G2_guard_same_combo_geometry_rotation_real_120Hz')
        d['stage'].update(success_criterion=criterion, goal_status=criterion)
        d['goal_status_semantics']=criterion
        d['goal']={'deliverable':'PPO_updated_G2_candidate','F3_F4':'authorized','success_rate':'no_high_rate_gate'}
        d['preparation_condition']['F3_F4']='AUTHORIZED'
        preparation = {'kind':'explicit_physical_support_spring_reference_motion',
            'target_offset_hand_m':list(getattr(self.cfg,'preparation_reference_offset_hand_m',(0.,0.,0.))),
            'start_policy_step':35,'end_policy_step':80,'release_policy_step':120,
            'profile':'linear_target_offset_then_original_support_release_ramp',
            'initial_fruit_and_joint_poses_unchanged':True,
            'in_episode_fruit_or_joint_state_writes':False,
            'original_spring_gains_caps_PD_mass_friction_ROI_faults_unchanged':True,
            'support_after_release':'exact_zero; independent_test_pulses_separately_logged',
            'gain_claim':'preparation_condition_change_not_PPO_learning'}
        d['preparation_condition']['reference_motion'] = preparation
        d['physics']['support']['reference_motion'] = preparation
        d['physics']['support']['release_ramp'].update(
            profile_id='firm_grasp_declared_power_release_v2_1',
            ramp_semantics='original linear ratio raised to explicitly configured power',
            scope_note='preparation condition change; not PPO learning gain',
            release_power=float(getattr(self.cfg,'support_release_power',1.)),
            scale_formula='original_linear_release_scale ** release_power',
            exact_zero_begins_at_policy_step=120,
            modified_preparation_condition=True)
        d['reward'].update(profile='firm_G2_load_margin_v2_2', progress='12*clamp(G2streak/required_substeps,0,1)',
            geometry='2*released*any_geometry_and_load_qualified',
            third_load_margin={
                'formula':'2*clamp((third_load_N-0.15)/0.15,0,1)',
                'third_load_N':'pad_force_n[:,1:].topk(2,dim=1).values[:,1]',
                'range_n':[0.15,0.30], 'max_bonus':2.0,
                'qualification_gate':'true_thumb_PAD_qualified AND unknown_all_false AND nonpad_force_all_zero AND no_hard_fault AND no_drop',
                'release_required':False,
                'meaning':'鼓励第三指在释放前后建立0.15至0.30 N的真实PAD载荷余量；超过0.30 N奖励封顶。'})
        if getattr(self.cfg, 'legacy_thumb_baseline', False):
            from research.thumb_pad_grasp_v1.action_mapping import MAPPING_ID, ACTION_ORDER
            d['action_mapping'] = {'mapping_id':MAPPING_ID,'action_order':list(ACTION_ORDER),
                'implementation':'thumb_pad_grasp_v1.action_mapping.independent_thumb_targets'}
            d['observations'].update(action_history_slot_names=list(ACTION_ORDER),
                action_history_semantics_changed=[],
                action_history_semantics_note='restored_v1_6_action_history_semantics_for_readonly_comparison')
            d['temporal_filter'] = {'implementation':'original_CriticalDampedActionFilter',
                'restored_for_readonly_v1_6_comparison':True}
            d['legacy_comparison_adapter'] = {'actor_prefix_dim':161,'new_clock_connection':0.,
                'action8_9':'original_ring_little_residuals',
                'new_G2_protocol_and_F2_reset_condition':True,
                'not_original_v1_6_condition_reproduction':True}
        return d

    def _postprocess_joint_targets(self, targets):
        if getattr(self.cfg, 'legacy_thumb_baseline', False):
            return ThumbPadGraspEnv._postprocess_joint_targets(self, targets)
        return super()._postprocess_joint_targets(targets)

    def _check_action_effect_and_limits(self):
        if getattr(self.cfg, 'legacy_thumb_baseline', False):
            return ThumbPadGraspEnv._check_action_effect_and_limits(self)
        return super()._check_action_effect_and_limits()

    def _live_physical_substep_state(self, *, forces):
        # 训练也读取真实点详情；仅评估持久化全部原始日志。
        previous=getattr(self,'pad_trace_enabled',False)
        if getattr(self,'_f3_ready',False): self.pad_trace_enabled=True
        try: return super()._live_physical_substep_state(forces=forces)
        finally: self.pad_trace_enabled=previous

    @torch.inference_mode()
    def _support_release_scale(self):
        scale = super()._support_release_scale()
        if self._support_release_power_by_env is not None:
            scale = scale.pow(self._support_release_power_by_env)
            self._support_ramp_scale = scale
        return scale

    @torch.inference_mode()
    def _unscaled_attachment_wrench(self):
        # 参考值是已声明支撑控制器的命令；不改写果实、手或关节状态。
        if self._prepare_target_offsets_m is None:
            return super()._unscaled_attachment_wrench()
        from isaaclab.utils.math import quat_apply
        fraction = ((self.episode_length_buf.float()-35.)/45.).clamp(0.,1.)
        hand_q = self._robot.data.body_pose_w.torch[:,self._ee_body_id,3:7]
        offset_w = quat_apply(hand_q,self._prepare_target_offsets_m*fraction[:,None])
        self._last_preparation_offset_w = offset_w
        original = self._stem_rest_attachment_pos_w
        self._stem_rest_attachment_pos_w = original + offset_w
        try:
            return super()._unscaled_attachment_wrench()
        finally:
            self._stem_rest_attachment_pos_w = original

    def _contact_geometry(self):
        centers=torch.zeros((self.num_envs,5,3),device=self.device)
        weights=torch.zeros((self.num_envs,5),device=self.device)
        digits=('thumb','index','middle','ring','little')
        for detail in self.pad_observer.details:
            digit=digits.index(detail['body'].split('_')[0])
            ids=torch.tensor(detail['env_ids'],device=self.device,dtype=torch.long)
            load=torch.tensor(detail['forces_n'],device=self.device)
            point=torch.tensor(detail['points_w'],device=self.device)
            load=load*(torch.tensor(detail['labels'],device=self.device)==1)
            weights[:,digit].index_add_(0,ids,load)
            centers[:,digit].index_add_(0,ids,point*load[:,None])
        centers/=weights.clamp_min(1e-12)[...,None]
        delta=centers-self._firm_last_state.fruit_pose[:,None,:3]
        norm=torch.linalg.vector_norm(delta,dim=-1)
        radial=delta/norm.clamp_min(1e-12)[...,None]
        separation,opposition,valid=[],[],[]
        for a,b,c in COMBINATIONS:
            pair=radial[:,b]+radial[:,c]
            pair_norm=torch.linalg.vector_norm(pair,dim=-1)
            distance=torch.linalg.vector_norm(centers[:,b]-centers[:,c],dim=-1)
            dot=(radial[:,a]*pair).sum(-1)/pair_norm.clamp_min(1e-12)
            ok=((weights[:,[a,b,c]]>0).all(1)&(norm[:,[a,b,c]]>1e-8).all(1)
                &(pair_norm>1e-8)&(distance>=.006)&(dot<=-.3))
            separation.append(distance);opposition.append(dot);valid.append(ok)
        return {'centers_w':centers,'separation_m':torch.stack(separation,1),
            'opposition_dot':torch.stack(opposition,1),'qualified':torch.stack(valid,1)}

    def _observe_physical_hold_substep(self, scene_dt_s):
        if not self._f3_ready: return super()._observe_physical_hold_substep(scene_dt_s)
        consumed=ThumbPadGraspEnv._observe_physical_hold_substep(self,scene_dt_s)
        if consumed:
            s=self._firm_last_state
            qualified=self._firm_metrics['combination_qualified']&ground_free_mask(s.fruit_pose,self._fruit_axes_m)
            zero=(self._last_stem_force_w==0).all(1)&(self._last_stem_torque_w==0).all(1)
            common=dict(state=s,hard_fault_any=self._physical_hold_guard.hard_fault_any,
                support_zero=zero,counter=self._sim_step_counter,dt=scene_dt_s)
            self.firm_guard.observe(qualified=qualified,**common)
            self._f3_geometry=self._contact_geometry()
            q=quat_mul(quat_conjugate(s.ee_pose[:,3:7]),s.fruit_pose[:,3:7])
            angular=torch.linalg.vector_norm(self._tomato.data.root_ang_vel_w.torch-
                self._robot.data.body_ang_vel_w.torch[:,self._ee_body_id],dim=1)
            self.g2_guard.observe(qualified=qualified&self._f3_geometry['qualified'],
                relative_quat_xyzw=q,relative_angular_speed=angular,
                post_release_substep=self._post_release_substep,
                strict_early_start=self._f3_protocol['goal']=='g2',
                disturbed=self._f3_protocol['disturbance_mode']!='quiet',**common)
            if self.pad_trace_enabled:
                self._record_firm_pad_trace_row()
                self._pad_trace[-1].update(g2_streak=self.g2_guard.streak.tolist(),
                    g2_peak_streak=self.g2_guard.peak.tolist(),g2_angle_rad=self.g2_guard.angle.tolist(),
                    relative_quat_xyzw=q.tolist(),relative_angular_speed_rad_s=angular.tolist(),
                    nonthumb_separation_m=self._f3_geometry['separation_m'].tolist(),
                    opposition_dot=self._f3_geometry['opposition_dot'].tolist(),
                    contact_centers_w=self._f3_geometry['centers_w'].tolist(),
                    geometry_qualified=self._f3_geometry['qualified'].tolist(),
                    post_release_substep=self._post_release_substep.tolist(),
                    test_disturbance_force_w=self._test_force.tolist(),
                    test_disturbance_torque_w=self._test_torque.tolist(),
                    total_command_force_w=(self._last_stem_force_w+self._test_force).tolist(),
                    total_command_torque_w=(self._last_stem_torque_w+self._test_torque).tolist(),
                    pulse_applied_substeps=self._pulse_counts.tolist())
                self._pad_trace[-1].update(preparation_reference_offset_w=self._last_preparation_offset_w.tolist(),
                    preparation_spring_reference_w=(self._stem_rest_attachment_pos_w+self._last_preparation_offset_w).tolist())
        return consumed

    @torch.inference_mode()
    def _apply_attachment_wrench(self):
        if not self._f3_ready: return super()._apply_attachment_wrench()
        force,torque=self._unscaled_attachment_wrench()
        scale=self._support_release_scale()*(~self._stem_detached).float()
        force*=scale[:,None];torque*=scale[:,None]
        self._last_stem_force_w.copy_(force);self._last_stem_torque_w.copy_(torque)
        fresh=self._stem_detached&(self._release_start_counter<0)
        self._release_start_counter[fresh]=self._sim_step_counter
        self._post_release_substep=torch.where(self._stem_detached,self._sim_step_counter-self._release_start_counter,-1)
        p=self._f3_protocol
        self._test_force,self._test_torque=pulse_wrench(self._post_release_substep,
            self._robot.data.body_pose_w.torch[:,self._ee_body_id,3:7],self._fruit_mass_kg,
            self._fruit_axes_m[:,0],mode=p['disturbance_mode'],axis=p['disturbance_axis'],
            sign=p['disturbance_sign'],force_ratio=p['force_ratio'],torque_ratio=p['torque_ratio'])
        self._pulse_counts+=((self._test_force!=0).any(1)|(self._test_torque!=0).any(1)).long()
        # 唯一实际写入，防止两个setter相互覆盖。
        self._tomato.permanent_wrench_composer.set_forces_and_torques_index(
            forces=(force+self._test_force).unsqueeze(1),torques=(torque+self._test_torque).unsqueeze(1),is_global=True)

    def _get_dones(self):
        terminated,truncated=super()._get_dones()
        if not self._f3_ready: return terminated,truncated
        self.g2_guard.complete_policy_interval()
        old=self._terminal_success.clone()
        success=self.g2_guard.streak.amax(1)>=round(self._f3_protocol['hold_seconds']*120)
        success&=self._stem_detached&~self._drop_now&~self._physical_hold_guard.hard_fault_any
        if self._f3_protocol['disturbance_mode']!='quiet': success&=self._pulse_counts==12
        terminated=(terminated&~old)|success
        truncated=(self.episode_length_buf>=self.max_episode_length-1)&~terminated
        self._terminal_reason[old&~success]=0
        self._terminal_reason[truncated]=2;self._terminal_reason[success]=1
        self._terminal_success.copy_(success)
        return terminated,truncated

    def _get_rewards(self):
        reward=super()._get_rewards()
        if not self._f3_ready or self._f3_geometry is None: return reward
        allowed=(~(self._physical_hold_guard.hard_fault_any|self._drop_now)).float()
        old=12.*(self.firm_guard.streak.amax(1).float()/120).clamp(max=1)
        progress=12.*(self.g2_guard.streak.amax(1).float()/(self._f3_protocol['hold_seconds']*120)).clamp(max=1)
        geometry=2.*self._stem_detached.float()*(self._f3_geometry['qualified']&self._firm_metrics['combination_qualified']).any(1).float()
        _, _, _, truth = self._pad_features()
        margin = third_load_margin_reward(
            truth['pad_force_n'], truth['thumb_pad_qualified'], truth['unknown'],
            truth['nonpad_force_n'], self._physical_hold_guard.hard_fault_any,
            self._drop_now)
        self.extras['pad_reward_terms'].update(
            triplet_progress=float(progress.mean()), geometry=float(geometry.mean()),
            third_load_margin=float(margin.mean()))
        return reward+allowed*(progress-old+geometry)+margin

    def _finalize_episodes(self, env_ids):
        cursor=len(getattr(self,'completed_episodes',[]))
        super()._finalize_episodes(env_ids)
        if self._f3_ready:
            for row in self.completed_episodes[cursor:]:
                i=row['env_id']
                row.update(task_version=TASK_VERSION,success_criterion=self._identity['criteria']['success_criterion'],
                    goal_status=('G2_REACHED' if self._f3_protocol['goal']=='g2' else 'CURRICULUM_HOLD_REACHED') if row['success'] else 'NOT_REACHED',
                    f3_protocol=dict(self._f3_protocol),g2_peak_streak_by_combo=self.g2_guard.peak[i].tolist(),
                    g2_final_streak_by_combo=self.g2_guard.streak[i].tolist(),pulse_applied_substeps=int(self._pulse_counts[i]))

    @torch.inference_mode()
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self._f3_ready:
            ids=self._all_env_ids if env_ids is None else env_ids
            self.g2_guard.reset(ids)
            self._release_start_counter[ids]=-1;self._post_release_substep[ids]=-1
            self._test_force[ids]=0;self._test_torque[ids]=0;self._pulse_counts[ids]=0
