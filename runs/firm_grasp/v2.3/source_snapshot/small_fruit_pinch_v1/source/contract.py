"""冻结的小果三指捏取时序与张量合同。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


TASK_VERSION: Final = "small_fruit_pinch_v1.2.0"
VERIFICATION_CONTRACT_VERSION: Final = "support_retention_v1"
# 10D 历史动作 + 4D 直指远端残差（食/中/无名/小指）。
ACTION_DIM = 14
ACTOR_OBSERVATION_DIM = 114
CRITIC_STATE_DIM = 175
POLICY_RATE_HZ = 20
POLICY_DT_S = 1.0 / POLICY_RATE_HZ
EPISODE_STEPS = 320
ACQUISITION_MAX_STEPS = 140
SETTLE_MIN_STEPS = 20
STRICT_HOLD_STEPS = 80
DISTURBANCE_STEP_IN_HOLD = 40
POST_DISTURBANCE_HOLD_STEPS = 40
LONG_HOLD_STEPS = 120
REQUIRED_DIGIT_INDICES = (0, 1, 2)  # thumb, index, middle


@dataclass(frozen=True)
class FingerForceBand:
    """一根手指的仿真压力带；不是实机损伤阈值。"""

    soft_low_n: float
    soft_high_n: float
    hard_high_n: float


FORCE_BANDS = (
    FingerForceBand(0.20, 1.80, 4.00),  # thumb
    FingerForceBand(0.15, 1.50, 3.50),  # index
    FingerForceBand(0.12, 1.30, 3.50),  # middle
    FingerForceBand(0.08, 1.20, 3.00),  # ring, diagnostic only
    FingerForceBand(0.05, 1.00, 3.00),  # little, diagnostic only
)


def validate_contract(*, action_dim: int, actor_dim: int, critic_dim: int) -> None:
    """拒绝悄悄改变当前 v1.2 小果策略的训练张量接口。"""

    expected = (ACTION_DIM, ACTOR_OBSERVATION_DIM, CRITIC_STATE_DIM)
    actual = (int(action_dim), int(actor_dim), int(critic_dim))
    if actual != expected:
        raise ValueError(
            "小果捏取 v1 合同不匹配："
            f"expected action/actor/critic={expected}, got {actual}"
        )
