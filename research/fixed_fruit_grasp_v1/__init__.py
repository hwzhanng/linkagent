"""固定单果抓握 v1（holding-only 交付）。

包内划分：

* :mod:`research.fixed_fruit_grasp_v1.contract` —— 纯 Python 合同（阶段判据、
  终止原因、逐回合记录 schema），不导入 torch/Isaac/USD。
* :mod:`research.fixed_fruit_grasp_v1.env` —— ``FixedFruitGraspCfg`` /
  ``FixedFruitGraspEnv`` / ``make_config``，导入 isaaclab 但不启动 Kit。

本包只声明与实现“固定手 + 固定时刻撤支撑”的保持阶段；30 mm 提举需要一根
明确声明的受驱动升降关节，尚未实现（``stage='lift'`` fail-closed）。
"""

from __future__ import annotations

from typing import Any

from .contract import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    CRITIC_STATE_DIM,
    DEFAULT_STAGE,
    EPISODE_RECORD_FIELDS,
    EPISODE_STEPS,
    GOAL_STATUS_LIFT_NOT_IMPLEMENTED,
    LIFT_PROGRAM,
    LIFT_STAGE_STATUS,
    POLICY_DT_S,
    REASON_CODES,
    RECORD_SCHEMA_VERSION,
    STAGES,
    STAGE_SPECS,
    TASK_ID,
    TASK_VERSION,
    LiftStageUnavailable,
    OpenHandActionUnavailable,
    json_safe,
    lift_stage_available,
    reason_name,
    stage_spec,
    summarize_records,
)

_LAZY_ENV_EXPORTS = {
    "FixedFruitGraspCfg",
    "FixedFruitGraspEnv",
    "FixedFruitGraspPDActuator",
    "make_config",
    "register_tasks",
    "validate_cfg",
}


def __getattr__(name: str) -> Any:
    """延迟导入 env 模块，使 ``import research.fixed_fruit_grasp_v1`` 保持轻量。"""

    if name in _LAZY_ENV_EXPORTS:
        from . import env as _env

        return getattr(_env, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ACTION_DIM",
    "ACTOR_OBSERVATION_DIM",
    "CRITIC_STATE_DIM",
    "DEFAULT_STAGE",
    "EPISODE_RECORD_FIELDS",
    "EPISODE_STEPS",
    "GOAL_STATUS_LIFT_NOT_IMPLEMENTED",
    "LIFT_PROGRAM",
    "LIFT_STAGE_STATUS",
    "POLICY_DT_S",
    "REASON_CODES",
    "RECORD_SCHEMA_VERSION",
    "STAGES",
    "STAGE_SPECS",
    "TASK_ID",
    "TASK_VERSION",
    "LiftStageUnavailable",
    "OpenHandActionUnavailable",
    "json_safe",
    "lift_stage_available",
    "reason_name",
    "stage_spec",
    "summarize_records",
]
