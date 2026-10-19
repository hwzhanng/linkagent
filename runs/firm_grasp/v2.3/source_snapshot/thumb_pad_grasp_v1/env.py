"""独立指肚抓握任务：复用真实物理守卫，改变接触语义、观测与学习信号。"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import torch
from isaaclab.utils.math import quat_apply_inverse
from research.fixed_fruit_grasp_v1.env import FixedFruitGraspEnv, make_config as base_make_config
from .observer import PadObserver
from .phase_rewards import phase_contact_terms, reward_contract, block_failure_positive_terms
from .command_warmup import apply_command_warmup, WARMUP_POLICY_STEPS
from .action_mapping import independent_thumb_targets, MAPPING_ID, ACTION_ORDER, REFERENCE_THUMB_Q_RAD, THUMB_RESIDUAL_SPAN_RAD

FEATURE_DIM = 47
ACTOR_DIM = 114 + FEATURE_DIM
CRITIC_DIM = 175 + FEATURE_DIM


def make_config(**kwargs):
    mode = kwargs.pop("thumb_mapping", "legacy")
    if mode not in {"legacy", "independent"}:raise ValueError("未知新任务拇指映射")
    # 延长接近与稳定时间；撤支撑后仍须连续120个真实子步独立持握。
    kwargs.setdefault("support_release_step", 120)
    cfg = base_make_config(**kwargs)
    cfg.thumb_mapping = mode
    cfg.task_id = 'ThumbPadGrasp-v1'
    cfg.task_version = 'thumb_pad_grasp_v1.6.0'
    return cfg


class ThumbPadGraspEnv(FixedFruitGraspEnv):
    def __init__(self,cfg,**kwargs):
        self._pad_ready = False
        self.pad_trace_enabled = False
        self._pad_trace = []
        super().__init__(cfg,**kwargs)
        if cfg.thumb_mapping == 'independent':
            # 只改变reset的无接触预抓取姿态；回合中不改写果实/关节状态。
            with torch.inference_mode():
                for axis,value in enumerate(REFERENCE_THUMB_Q_RAD):
                    joint = self._robot.joint_names.index('thumb_joint'+str(axis))
                    self._nominal_joint_targets[:,joint] = value
            self._validate_nominal_joint_margin()
            self._action_effect = self._check_action_effect_and_limits()
        self.pad_observer = PadObserver(self)
        config = json.loads((Path(__file__).parent/'assets/pad_regions_v1.approved.json').read_text())
        ordered = [next(d for d in config['digits'] if d['digit']==name) for name in self.pad_observer.digits]
        self._pad_body_ids = [self._robot.body_names.index(d['body_path'].rsplit('/',1)[-1]) for d in ordered]
        self._pad_centers = torch.tensor([d['center_body_local_m'] for d in ordered],device=self.device)
        self._pad_normals = torch.tensor([d['mean_outward_normal_body_local'] for d in ordered],device=self.device)
        self._pad_potential = torch.zeros(self.num_envs,device=self.device)
        self._pad_potential_valid = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.cfg.observation_space = ACTOR_DIM
        self.cfg.state_space = CRITIC_DIM
        self._configure_gym_env_spaces()
        self._pad_ready = True
        self._identity = self._build_identity()

    def _pre_physics_step(self, actions):
        # 和已出现真实1秒持握的脚本首回合采用同样的命令缓起步，
        # 其后仍完全由策略动作控制；原滤波、PD、限位及物理守卫照常执行。
        return super()._pre_physics_step(apply_command_warmup(actions, self.episode_length_buf))

    def _postprocess_joint_targets(self, targets):
        if getattr(self.cfg,'thumb_mapping','legacy') == 'independent' and not getattr(self,'_open_hand_baseline',False):
            return independent_thumb_targets(self._executed_actions)
        return super()._postprocess_joint_targets(targets)

    @torch.inference_mode()
    def _check_action_effect_and_limits(self):
        result = super()._check_action_effect_and_limits()
        if getattr(self.cfg,'thumb_mapping','legacy') != 'independent':return result
        probes = torch.tensor([[-1.]*14,[0.]*14,[1.]*14],device=self.device)
        targets = independent_thumb_targets(probes)
        limits = self._robot.data.joint_pos_limits.torch[0,self._hand_joint_ids]
        low,high = targets.amin(0),targets.amax(0)
        outside = (low < limits[:,0]-1e-6) | (high > limits[:,1]+1e-6)
        if bool(outside.any()):raise RuntimeError('新拇指映射目标超出USD硬限')
        span = high-low
        result.update(ok=True,joint_limit_overhang_rad=0.,joint_limit_overhang_names=[],
            mapping_id=MAPPING_ID,probe_targets_rad=targets.tolist(),per_joint_span_rad=span.tolist(),
            min_joint_span_rad=float(span.min()),mean_joint_span_rad=float(span.mean()),
            manifold_zero_target_gap_rad=float((targets[1]-self._nominal_joint_targets[0,self._hand_joint_ids]).abs().max()))
        return result

    def _build_identity(self):
        identity = super()._build_identity()
        identity.update(contract_id='thumb_pad_grasp_v1',task_id='ThumbPadGrasp-v1',
                        task_version='thumb_pad_grasp_v1.6.0',record_schema_version='thumb_pad_grasp_episode.v1')
        assets = Path(__file__).parent/'assets'
        identity['pad_contact'] = {'runtime_config':json.loads((assets/'runtime_contact_v2.json').read_text()),
            'asset_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.iterdir() if p.is_file()}}
        identity['observations'] = {'policy_dim':ACTOR_DIM,'critic_dim':CRITIC_DIM,
            'base_policy_prefix_dim':114,'new_features':'five_pad_local_fruit_offset15,direction_alignment5,pad_load5,nonpad_load5,unknown5,states5,thumb_gap1,stage1,thumb_qualified1,required1,physical_streak1,thumb_normal_error1,valid1'}
        identity['physics']['observation_dims'] = {'actor':ACTOR_DIM,'critic':CRITIC_DIM,'action':14}
        # 元数据直接来自实际策略周期与释放步，避免继承旧3秒描述值。
        identity['physics']['support'].update(
            release_step=int(self.cfg.support_release_step),
            release_time_s=float(self.cfg.support_release_step)*float(identity['physics']['policy_dt_s']))
        identity['physics']['physical_hold_guard'].update(
            success_semantics='actual thumb PAD plus nonthumb contact, no support, continuous 120 valid physical substeps',
            hold_streak_semantics='per_120Hz_verified_PAD_contact_streak')
        identity['stage'].update(success_criterion='verified_thumb_PAD_plus_nonthumb_contact_unsupported_1s',
            counts_as_goal=True,goal_status='GOAL_DEFINED_BY_VERIFIED_THUMB_PAD_HOLD',
            support_release_step=int(self.cfg.support_release_step),
            support_release_time_s=float(self.cfg.support_release_step)*float(identity['physics']['policy_dt_s']))
        identity['criteria'].update(thumb_required=True,
            approach_reward_source='pad_center_distance_and_normal_alignment_potential_only_not_contact_truth',
            goal_min_height_gain_m=0.,goal_hold_s=1.,
            contact_source='per_point_actual_normal_force_and_confirmed_surface',
            thumb_nonpad_bearing_allowed_n=0.,unknown_success_allowed=False,
            nonthumb_requirement='at_least_one_real_contact_pad_or_nonpad',
            required_true_substeps=120,unsupported_hold_s=1.,
            hold_streak_gate='per_120Hz_substep_verified_thumb_PAD_plus_nonthumb_real_contact')
        identity['goal'] = {'deliverable':'new_PPO_weight_with_verified_thumb_PAD_unsupported_hold',
                            'success_rate':'report_complete_denominator_no_strict_rate_gate'}
        identity['reward'] = reward_contract()
        identity['asset_config']['criteria'] = identity['criteria']
        identity['action_mapping'] = {'mode':getattr(self.cfg,'thumb_mapping','legacy'),
            'mapping_id':MAPPING_ID if getattr(self.cfg,'thumb_mapping','legacy')=='independent' else 'legacy_coupled_14d',
            'action_order':list(ACTION_ORDER) if getattr(self.cfg,'thumb_mapping','legacy')=='independent' else 'original_frozen_14d',
            'source_sha256':hashlib.sha256((Path(__file__).parent/'action_mapping.py').read_bytes()).hexdigest(),
            'thumb_active_joints':'0/1/2/3 independent; joint4 retains asset mimic',
            'safety':'original safe joint intervals, original physical faults, no vendor global changes'}
        identity['initial_pose']['thumb_reference_q_rad'] = list(REFERENCE_THUMB_Q_RAD) if self.cfg.thumb_mapping=='independent' else 'original_authored_zero'
        identity['action_mapping']['thumb_local_residual_span_rad'] = list(THUMB_RESIDUAL_SPAN_RAD)
        identity['action_mapping']['command_warmup'] = {
            'kind':'episode_age_gain_before_original_action_filter',
            'policy_steps':WARMUP_POLICY_STEPS,
            'duration_s':WARMUP_POLICY_STEPS*float(identity['physics']['policy_dt_s']),
            'gain':'clip(episode_policy_step/35,0,1)',
            'PPO_buffer_actions':'original policy commands before environment warmup',
            'environment_raw_actions':'warmup-scaled commands; executed actions retain original filter',
            'object_force_or_state_write':False}
        identity['initial_pose']['thumb_reference_policy'] = 'reset only; actual initial no-contact and original clearance guard required'
        identity['goal_status_semantics'] = 'thumb_PAD_unsupported_hold_1s; lift is deferred'

        return identity

    def _live_physical_substep_state(self,*,forces):
        state = super()._live_physical_substep_state(forces=forces)
        if not self._pad_ready:return state
        truth = self.pad_observer.sample()
        return replace(state,required_contact=truth['required_contact'])

    def _observe_physical_hold_substep(self,scene_dt_s):
        result = super()._observe_physical_hold_substep(scene_dt_s)
        if result and self._pad_ready and self.pad_trace_enabled:
            guard = self._physical_hold_guard
            truth = self.pad_observer.latest
            self._pad_trace.append({'sim_step':int(self._sim_step_counter),'dt_s':float(scene_dt_s),
                'episode_policy_step':self.episode_length_buf.tolist(),
                'released':self._stem_detached.tolist(),'support_force_w':self._last_stem_force_w.tolist(),
                'support_torque_w':self._last_stem_torque_w.tolist(),
                'fruit_pose_w':self._tomato.data.root_pose_w.torch.tolist(),
                'fruit_linear_velocity_w':self._tomato.data.root_lin_vel_w.torch.tolist(),
                'physical_streak':guard.streak.tolist(),
                'hard_faults':{k:v.tolist() for k,v in guard.hard_faults.items()},
                'truth':{k:v.tolist() for k,v in truth.items()},'contacts':self.pad_observer.details,
                'invalid_contact_channels':self.pad_observer.invalid_details,
                'executed_actions':self._executed_actions.tolist(),'joint_targets':self._joint_targets.tolist(),
                'joint_pos':self._robot.data.joint_pos.torch.tolist()})
        return result

    def drain_pad_records(self):
        result,self._pad_trace = self._pad_trace,[]
        return result

    def _refresh_task_state(self):
        super()._refresh_task_state()
        if self._pad_ready and self.pad_observer.latest is not None:
            required = self.pad_observer.latest['required_contact']
            self._required_contact.copy_(required)
            self._hold_ok &= required

    def _pad_features(self):
        pose = self._robot.data.body_pose_w.torch[:,self._pad_body_ids]
        fruit = self._tomato.data.root_pos_w.torch
        local = quat_apply_inverse(pose[...,3:7].reshape(-1,4),
            (fruit[:,None]-pose[...,:3]).reshape(-1,3)).reshape(self.num_envs,5,3)
        relative = local-self._pad_centers
        distance = torch.linalg.vector_norm(relative,dim=2)
        alignment = (relative*self._pad_normals).sum(2)/distance.clamp_min(1e-8)
        gap = distance[:,0]-self._fruit_axes_m[:,0]
        truth = self.pad_observer.latest
        if truth is None:
            zeros = torch.zeros((self.num_envs,5),device=self.device)
            truth = {'pad_force_n':zeros,'nonpad_force_n':zeros,'unknown':zeros.bool(),
                     'contact_state':zeros,'thumb_pad_qualified':zeros[:,0].bool(),'required_contact':zeros[:,0].bool()}
        feature = torch.cat((relative.reshape(self.num_envs,15)/.08,alignment,
            truth['pad_force_n']/4,truth['nonpad_force_n']/4,truth['unknown'].float(),
            truth['contact_state'].float()/3,(gap/.03)[:,None],self._stem_detached.float()[:,None],
            truth['thumb_pad_qualified'].float()[:,None],truth['required_contact'].float()[:,None],
            (self._physical_hold_guard.streak.float()/120)[:,None],(1-alignment[:,0])[:,None],
            torch.ones((self.num_envs,1),device=self.device)),dim=1)
        if feature.shape[1]!=FEATURE_DIM:raise RuntimeError(f'新指肚特征维数漂移: {feature.shape}')
        return feature.clamp(-5,5),gap,alignment,truth

    def _get_observations(self):
        observations = super()._get_observations()
        if self._pad_ready:
            features,*_ = self._pad_features()
            observations = {key:torch.cat((value,features),dim=1) for key,value in observations.items()}
        return observations

    def _get_rewards(self):
        if not self._pad_ready:return super()._get_rewards()
        if not self._state_ready:self._refresh_task_state()
        _,gap,alignment,truth = self._pad_features()
        potential = torch.exp(-gap.abs()/.025)*alignment[:,0].clamp_min(0).square()
        delta = torch.where(self._pad_potential_valid,.995*potential-self._pad_potential,torch.zeros_like(potential))
        self._pad_potential.copy_(potential);self._pad_potential_valid.fill_(True)
        hard = self._physical_hold_guard.hard_fault_any
        weights = reward_contract()['weights']
        force_excess = torch.relu(self._last_physical_force_n-self._soft_high)
        terms = {'thumb_surface_potential_delta':weights['thumb_surface_potential_delta']*delta,
            'normal_alignment_dense':weights['normal_alignment_dense']*potential,
            'thumb_nonpad_load':weights['thumb_nonpad_load']*truth['nonpad_force_n'][:,0].clamp(max=15),
            'unknown':weights['unknown']*truth['unknown'].any(1).float(),
            'force_excess_squared':weights['force_excess_squared']*force_excess.square().sum(1),
            'force_growth':weights['force_growth']*torch.relu(self._last_physical_force_n-self._previous_force_n).sum(1),
            'action_delta':weights['action_delta']*(self._executed_actions-self._previous_executed_actions).square().mean(1),
            'hard_fault':weights['hard_fault']*hard.float(),'drop':weights['drop']*self._drop_now.float()}
        terms.update(phase_contact_terms(truth=truth,released=self._stem_detached,
            relative_speed=self._rel_speed_m_s,physical_streak=self._physical_hold_guard.streak,
            terminal_success=self._terminal_success,hard_fault=hard,drop=self._drop_now))
        terms = block_failure_positive_terms(terms,hard | self._drop_now)
        self.extras['pad_reward_terms'] = {key:float(value.mean()) for key,value in terms.items()}
        self.extras.setdefault('log',{}).update({'PadGrasp/'+key:float(value.mean()) for key,value in terms.items()})
        return sum(terms.values())

    @torch.inference_mode()
    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        if self._pad_ready:
            self.pad_observer.reset(env_ids)
            self._pad_potential[env_ids]=0
            self._pad_potential_valid[env_ids]=False

    def _finalize_episodes(self,env_ids):
        cursor = len(getattr(self,'completed_episodes',[]))
        super()._finalize_episodes(env_ids)
        for row in getattr(self,'completed_episodes',[])[cursor:]:
            row.update(schema_version='thumb_pad_grasp_episode.v1',
                task_id='ThumbPadGrasp-v1',task_version='thumb_pad_grasp_v1.6.0',
                success_criterion='verified_thumb_PAD_plus_nonthumb_contact_unsupported_1s',
                goal_success=bool(row['success']),goal_status='THUMB_PAD_HOLD_REACHED' if row['success'] else 'THUMB_PAD_HOLD_NOT_REACHED',
                stage_is_goal=True,goal_min_height_gain_m=0.,
                action_mapping_mode=getattr(self.cfg,'thumb_mapping','legacy'),
                physical_hold_streak_semantics='true_thumb_PAD_plus_nonthumb_real_contact_every_120Hz_substep',
                pad_runtime_config_sha256=hashlib.sha256((Path(__file__).parent/'assets/runtime_contact_v2.json').read_bytes()).hexdigest())
