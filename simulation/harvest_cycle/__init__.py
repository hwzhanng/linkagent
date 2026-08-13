"""Isaac-only harvest-cycle acceptance environment.

This package deliberately lives below :mod:`simulation` so importing the
Agent runtime never imports Isaac Lab, Torch, or a renderer.
"""

import gymnasium as gym

from .isaac_env import HarvestCycleIsaacEnv
from .isaac_env_cfg import HarvestCycleIsaacEnvCfg
from .physics_env import HarvestCyclePhysicsIsaacEnv

TASK_ID = "LinkAgent-HarvestCycle-Acceptance-v1"
PHYSICS_TASK_ID = "LinkAgent-HarvestCycle-Physics-Acceptance-v1"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="simulation.harvest_cycle.isaac_env:HarvestCycleIsaacEnv",
        disable_env_checker=True,
    )

if PHYSICS_TASK_ID not in gym.registry:
    gym.register(
        id=PHYSICS_TASK_ID,
        entry_point=(
            "simulation.harvest_cycle.physics_env:HarvestCyclePhysicsIsaacEnv"
        ),
        disable_env_checker=True,
    )

__all__ = [
    "HarvestCycleIsaacEnv",
    "HarvestCycleIsaacEnvCfg",
    "HarvestCyclePhysicsIsaacEnv",
    "PHYSICS_TASK_ID",
    "TASK_ID",
]
