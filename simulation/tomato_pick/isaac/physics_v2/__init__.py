"""Registration for the public-prior Physics-v2 task."""

from __future__ import annotations

import sys

import gymnasium as gym


TASK_ID = "LinkAgent-TomatoPick-Physics-v2-PublicPrior-v1"
IMPLEMENTATION_STATUS = (
    "physics_v2_qualified_ready_for_external_audit_training_not_started"
)


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
