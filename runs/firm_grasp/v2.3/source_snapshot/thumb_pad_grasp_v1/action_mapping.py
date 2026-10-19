"""新任务候选14D映射：四根拇指主动轴独立，四指保留闭合及远端控制。

此模块不改变历史vendor映射；是否启用由新任务显式配置和身份声明决定。
"""
from __future__ import annotations
import torch
from research.small_fruit_pinch_v1.source.pinch_action_manifold import pinch_action_to_joint_targets
from research.small_fruit_pinch_v1.source.legacy_bridge import load_legacy_hand_manifold

MAPPING_ID = 'thumb_pad_upward_reference_residual_14d_v3'
REFERENCE_THUMB_Q_RAD = (0.05549413117585235, 1.3393746570587945, 0.4531111979189145, 0.10039140509741924)
THUMB_RESIDUAL_SPAN_RAD = (.25, .35, .3, .65)
THUMB_INDICES = (5, 10, 0, 11)  # canonical16中的thumb joint0/1/2/3
THUMB_ACTION_INDICES = (2, 3, 4, 5)
ACTION_ORDER = ('finger_flexion','finger_curl_shape','thumb_joint0','thumb_joint1',
                'thumb_joint2','thumb_joint3','index_residual','middle_residual',
                'ring_residual','little_residual','index_distal','middle_distal',
                'ring_distal','little_distal')


def independent_thumb_targets(action: torch.Tensor) -> torch.Tensor:
    if action.shape[-1] != 14 or not bool(torch.isfinite(action).all()):
        raise ValueError('新14D映射需要有限的14维动作')
    bounded = action.clamp(-1, 1)
    finger_action = bounded.clone()
    finger_action[..., 2:6] = 0
    result = pinch_action_to_joint_targets(finger_action)
    legacy = load_legacy_hand_manifold()
    low = action.new_tensor(legacy.LEFT_SAFE_LOWER_RAD)
    high = action.new_tensor(legacy.LEFT_SAFE_UPPER_RAD)
    reference = action.new_tensor(REFERENCE_THUMB_Q_RAD)
    residual_span = action.new_tensor(THUMB_RESIDUAL_SPAN_RAD)
    for axis,(component,joint) in enumerate(zip(THUMB_ACTION_INDICES, THUMB_INDICES)):
        result[...,joint] = reference[axis] + bounded[...,component]*residual_span[axis]
    return torch.maximum(low, torch.minimum(high, result))


def encode_thumb_targets(joint_values: torch.Tensor) -> torch.Tensor:
    """将joint0/1/2/3参考角转换成新动作的2:6四维，范围外明确拒绝。"""
    if joint_values.shape[-1] != 4 or not bool(torch.isfinite(joint_values).all()):
        raise ValueError('需要有限的四根拇指主动轴目标')
    legacy = load_legacy_hand_manifold()
    low = joint_values.new_tensor(legacy.LEFT_SAFE_LOWER_RAD)[list(THUMB_INDICES)]
    high = joint_values.new_tensor(legacy.LEFT_SAFE_UPPER_RAD)[list(THUMB_INDICES)]
    if bool(((joint_values < low) | (joint_values > high)).any()):
        raise ValueError('拇指目标超出保留的安全范围')
    reference = joint_values.new_tensor(REFERENCE_THUMB_Q_RAD)
    span = joint_values.new_tensor(THUMB_RESIDUAL_SPAN_RAD)
    encoded = (joint_values-reference)/span
    if bool((encoded.abs()>1+1e-6).any()):
        raise ValueError('目标超出当前参考姿态的局部残差范围')
    return encoded.clamp(-1,1)
