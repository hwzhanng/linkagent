"""Registration for the offline Physics-v2 improvement lineage."""

from __future__ import annotations

import sys

import gymnasium as gym


TASK_ID = "LinkAgent-TomatoPick-Physics-v2"
IMPLEMENTATION_STATUS = "stage9_bounded_learning_artifact_and_evaluation_pending"


def register_task() -> list[str]:
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point=f"{__name__}.env:TomatoPickPhysicsV2Env",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:TomatoPickPhysicsV2EnvCfg",
                "rsl_rl_cfg_entry_point": (
                    f"{__name__}.agents.rsl_rl_ppo_cfg:"
                    "TomatoPickPhysicsV2PPORunnerCfg"
                ),
            },
        )
    return list(sys.argv[1:])


register_task()

__all__ = ["IMPLEMENTATION_STATUS", "TASK_ID", "register_task"]
