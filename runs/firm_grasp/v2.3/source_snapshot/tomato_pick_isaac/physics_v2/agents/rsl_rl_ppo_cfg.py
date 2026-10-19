"""Frozen PPO contract for Physics-v2; formal training remains gated."""

from isaaclab.utils.configclass import configclass

from simulation.tomato_pick.isaac.physics_v1.agents.rsl_rl_ppo_cfg import (
    TomatoPickPhysicsV1PPORunnerCfg,
)


@configclass
class TomatoPickPhysicsV2PPORunnerCfg(TomatoPickPhysicsV1PPORunnerCfg):
    experiment_name = "tomato_pick_physics_v2"
    run_name = "tomato_pick_physics_v2_from_scratch"


__all__ = ["TomatoPickPhysicsV2PPORunnerCfg"]
