"""Explicit Isaac task registration; imported only by the training callback."""

from __future__ import annotations

import sys

import gymnasium as gym

TASK_ID = "LinkAgent-TomatoPick-Direct-v0"
IMPLEMENTATION_STATUS = "eigengrasp_11d_direct_rl_env_ready_for_formal_training"


def register_task() -> list[str]:
    """Register the external Gym task and return unconsumed CLI arguments."""

    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point=f"{__name__}.env:TomatoPickEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.env_cfg:TomatoPickEnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:TomatoPickPPORunnerCfg",
                "rsl_rl_finetune_cfg_entry_point": (
                    f"{__name__}.agents.rsl_rl_finetune_ppo_cfg:"
                    "TomatoPickFineTunePPORunnerCfg"
                ),
                "rsl_rl_recurrent_cfg_entry_point": (
                    f"{__name__}.agents.rsl_rl_recurrent_ppo_cfg:"
                    "TomatoPickRecurrentPPORunnerCfg"
                ),
                "rsl_rl_residual_cfg_entry_point": (
                    f"{__name__}.agents.rsl_rl_residual_lstm_ppo_cfg:"
                    "TomatoPickResidualLSTMRunnerCfg"
                ),
            },
        )
    return list(sys.argv[1:])

register_task()

__all__ = ["IMPLEMENTATION_STATUS", "TASK_ID", "register_task"]
