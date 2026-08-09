"""Explicit registration for the calibration-gated Physics-v1 task."""

from __future__ import annotations

import sys

import gymnasium as gym


TASK_ID = "LinkAgent-TomatoPick-Physics-v1"
IMPLEMENTATION_STATUS = (
    "physics_v1_provisional_physics_qualified_formal_training_ready_not_started"
)


def register_task() -> list[str]:
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point=f"{__name__}.env:TomatoPickPhysicsV1Env",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:TomatoPickPhysicsV1EnvCfg",
                "rsl_rl_cfg_entry_point": (
                    f"{__name__}.agents.rsl_rl_ppo_cfg:"
                    "TomatoPickPhysicsV1PPORunnerCfg"
                ),
            },
        )
    return list(sys.argv[1:])


register_task()

__all__ = ["IMPLEMENTATION_STATUS", "TASK_ID", "register_task"]
