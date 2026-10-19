"""撤支撑阶段的真实接触奖励；奖励从不替代120Hz成功守卫。"""
from copy import deepcopy
import torch


PHASE_REWARD_CONTRACT = {
    'version': 'pad_phase_focused_low_impact_v3',
    'unsupported_signal': 'actual stem_detached; no support schedule changes',
    'dense_contact_is_not_success': True,
    'terminal_failure_blocks_all_positive_terms': True,
    'stability_speed_scale_m_s': .05,
    'speed_excess_squared_cap': 25.,
    'weights': {
        'thumb_surface_potential_delta': 2., 'normal_alignment_dense': .25,
        'thumb_pad_supported': .3, 'thumb_pad_unsupported': 4.,
        'other_contact': .3, 'unsupported_pair': 2.,
        'unsupported_stability': 4., 'unsupported_speed_squared': -.25,
        'thumb_nonpad_load': -.5, 'unknown': -.2, 'hold_progress': 12.,
        'force_excess_squared': -.03, 'force_growth': -.03,
        'action_delta': -.02, 'hard_fault': -25., 'drop': -15., 'success': 100.,
    },
}


def reward_contract():
    return deepcopy(PHASE_REWARD_CONTRACT)


def phase_contact_terms(*, truth, released, relative_speed, physical_streak,
                        terminal_success, hard_fault, drop):
    """只使用真实部位观测、实际释放掩码和原守卫输出构造稠密学习信号。"""
    weights = PHASE_REWARD_CONTRACT['weights']
    scale = PHASE_REWARD_CONTRACT['stability_speed_scale_m_s']
    clean = ~truth['unknown'].any(dim=1) & ~hard_fault & ~drop
    pair = released & truth['required_contact'] & clean
    thumb = truth['thumb_pad_qualified'] & clean
    other = ((truth['pad_force_n'][:, 1:] + truth['nonpad_force_n'][:, 1:]) >= .08).float().sum(1)
    thumb_weight = torch.where(released, relative_speed.new_tensor(weights['thumb_pad_unsupported']),
                               relative_speed.new_tensor(weights['thumb_pad_supported']))
    speed_excess = (torch.relu(relative_speed - scale) / scale).square().clamp(
        max=PHASE_REWARD_CONTRACT['speed_excess_squared_cap'])
    return {
        'thumb_pad': thumb_weight * thumb.float(),
        'other_contact': weights['other_contact'] * other * clean.float(),
        'unsupported_pair': weights['unsupported_pair'] * pair.float(),
        'unsupported_stability': weights['unsupported_stability'] * pair.float() * torch.exp(-(relative_speed / scale).square()),
        'unsupported_speed_squared': weights['unsupported_speed_squared'] * released.float() * speed_excess,
        'hold_progress': weights['hold_progress'] * (physical_streak.float() / 120).clamp(max=1) * clean.float(),
        'success': weights['success'] * (terminal_success & clean).float(),
    }


def block_failure_positive_terms(terms, failed):
    """失败状态保留全部负项，并屏蔽所有正向塑形和成功奖金。"""
    return {key: torch.where(failed & (value > 0), torch.zeros_like(value), value)
            for key, value in terms.items()}
