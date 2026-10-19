"""S8：用四个内部积分步稳定20Hz动作滤波，不增加物理步或控制权限。"""
import math

import torch

from research.small_fruit_pinch_v1.source.temporal import CriticalDampedActionFilter


class StableCriticalDampedActionFilter(CriticalDampedActionFilter):
    """保留原速度/加速度/死区，只细分原不稳定的半隐式积分。"""

    internal_steps = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        omega_dt = 2 * math.pi * self.limits.natural_frequency_hz * self.dt_s / self.internal_steps
        if not 0 < omega_dt < 2 * (math.sqrt(2) - 1):
            raise ValueError('内部滤波步长不满足临界阻尼半隐式线性稳定界')

    def step(self, raw_action, *, hold_phase, speed_scale=None):
        if raw_action.shape != self.position.shape:
            raise ValueError('raw_action形状不符')
        if hold_phase.shape != (raw_action.shape[0],):
            raise ValueError('hold_phase必须是每环境一个布尔值')
        raw = torch.nan_to_num(raw_action, nan=0., posinf=1., neginf=-1.).clamp(-1., 1.)
        hold = hold_phase.to(dtype=torch.bool, device=raw.device).unsqueeze(-1)
        speed_limit = torch.where(hold, torch.full_like(raw, self.limits.hold_speed),
                                 torch.full_like(raw, self.limits.approach_speed))
        if speed_scale is not None:
            if speed_scale.shape != raw.shape or not bool(torch.isfinite(speed_scale).all()):
                raise ValueError('speed_scale必须同形且有限')
            speed_limit = speed_limit * speed_scale.to(raw).clamp(.05, 1.)
        accel_limit = torch.where(hold, torch.full_like(raw, self.limits.hold_acceleration),
                                 torch.full_like(raw, self.limits.approach_acceleration))
        omega = 2 * math.pi * self.limits.natural_frequency_hz
        dt = self.dt_s / self.internal_steps
        previous_position = self.position.clone()
        for _ in range(self.internal_steps):
            acceleration = (omega**2 * (raw - self.position) - 2 * omega * self.velocity).clamp(
                -accel_limit, accel_limit)
            velocity = (self.velocity + acceleration * dt).clamp(-speed_limit, speed_limit)
            reversal = ((velocity * self.velocity < 0)
                        & (velocity.abs() < self.limits.reversal_deadband))
            velocity = torch.where(reversal, torch.zeros_like(velocity), velocity)
            self.position.copy_((self.position + velocity * dt).clamp(-1., 1.))
            self.velocity.copy_(velocity)
        jerk = (acceleration - self.previous_acceleration) / self.dt_s
        self.previous_acceleration.copy_(acceleration)
        self.acceleration.copy_(acceleration)
        self.last_stats = {
            'action_delta': (self.position - previous_position).abs(),
            'tracking_error': (self.position - raw).abs(),
            'velocity': self.velocity.abs(), 'acceleration': acceleration.abs(),
            'jerk': jerk.abs(), 'reversal': reversal,
        }
        return self.position


def temporal_identity():
    return {
        'implementation': 'firm_grasp_v1.stable_temporal.StableCriticalDampedActionFilter',
        'integration': 'four_internal_semi_implicit_steps_per_20Hz_filter_call',
        'internal_steps': 4, 'policy_dt_s': .05, 'internal_dt_s': .0125,
        'physical_substeps_added': 0, 'original_temporal_limits_preserved': True,
        'original_parent_filter_source_modified': False,
        'old_checkpoint_action_chain_compatible': False,
        'scope': 'numerical_command_filter_change_not_new_physical_actuator_or_force_cap',
    }
