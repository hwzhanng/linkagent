"""小果捏取的 14D 手部动作流形。

前十维逐字复用并冻结历史 10D 语义；新增的四维只作用于食指、中指、无名指、
小指的远端关节。这样不以“强制动关节”换取表面自由度，而是在接触后给策略
独立卷曲指尖、维持低力抓握和扰动恢复所需的真实控制余量。

本模块只在仿真中使用，不改变任何实机控制接口或历史 89% 权重。
"""

from __future__ import annotations

import torch

from .legacy_bridge import load_legacy_hand_manifold


PINCH_ACTION_DIM = 14
_STRAIGHT_DISTAL_JOINTS = (12, 13, 14, 15)
# 旧值 0.25 只能微调远端，实际 link3 碰撞壳较长时会由根部先碰球。
# 提升到 0.50，使远端指垫可先卷入球面；仍由安全上下限裁剪，且近端
# 禁触/过载判据保持不变。
_DISTAL_RESIDUAL_SPAN_FRACTION = 0.50


def pinch_action_to_joint_targets(action: torch.Tensor) -> torch.Tensor:
    """将 14D 小果动作映射到 16 个左手关节安全区间。"""

    if action.ndim < 1 or action.shape[-1] != PINCH_ACTION_DIM:
        raise ValueError(f"action 最后一维必须为 {PINCH_ACTION_DIM}")
    if not bool(torch.isfinite(action).all()):
        raise ValueError("action 包含 NaN 或 Inf")
    legacy = load_legacy_hand_manifold()
    target = legacy.hybrid_action_to_joint_targets(action[..., :10])
    lower = action.new_tensor(legacy.LEFT_SAFE_LOWER_RAD)
    upper = action.new_tensor(legacy.LEFT_SAFE_UPPER_RAD)
    span = upper - lower
    for component, joint_index in enumerate(_STRAIGHT_DISTAL_JOINTS):
        target[..., joint_index] += (
            action[..., 10 + component].clamp(-1.0, 1.0)
            * _DISTAL_RESIDUAL_SPAN_FRACTION
            * span[joint_index]
        )
    return torch.maximum(lower, torch.minimum(upper, target))


__all__ = ["PINCH_ACTION_DIM", "pinch_action_to_joint_targets"]
