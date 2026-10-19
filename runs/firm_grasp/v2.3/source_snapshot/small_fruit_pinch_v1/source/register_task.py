"""训练框架的外部任务注册入口。"""

from __future__ import annotations

import sys

import gymnasium as gym


TASK_ID = "LinkAgent-SmallFruitPinch-v1"


def register_task() -> list[str]:
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point="research.small_fruit_pinch_v1.source.native_env:NativeSmallFruitPinchEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": "research.small_fruit_pinch_v1.source.native_env:NativeSmallFruitPinchCfg",
                "rsl_rl_cfg_entry_point": "research.small_fruit_pinch_v1.source.agents:SmallFruitPinchPPORunnerCfg",
            },
        )
    return list(sys.argv[1:])


register_task()

__all__ = ["TASK_ID", "register_task"]
