"""小果多指捏取 PPO 的独立实现。

本包不修改历史 89% 权重、历史训练脚本或大果掌托抓握实现。历史
PhysX 左手环境只在运行时以只读方式作为资产和低层物理接口来源。
"""

from .contract import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_STATE_DIM,
    POLICY_RATE_HZ,
    TASK_VERSION,
)

__all__ = [
    "ACTION_DIM",
    "ACTOR_OBSERVATION_DIM",
    "CRITIC_STATE_DIM",
    "POLICY_RATE_HZ",
    "TASK_VERSION",
]
