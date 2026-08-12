"""Frozen Clean-v5 PPO hyperparameters with a new V2 experiment identity."""

from isaaclab.utils.configclass import configclass

from simulation.tomato_pick.isaac.physics_v1.agents.rsl_rl_ppo_cfg import (
    TomatoPickPhysicsV1PPORunnerCfg,
)


@configclass
class TomatoPickPhysicsV2PPORunnerCfg(TomatoPickPhysicsV1PPORunnerCfg):
    experiment_name = "tomato_pick_physics_v2_public_prior_v1"
    run_name = "tomato_pick_physics_v2_public_prior_v1_from_scratch"
    # One invocation is one auditable curve-controller segment.  This removes
    # the inherited Physics-v1 1800-update default without imposing a total
    # V2 training cap; the controller performs full-state serial continuation.
    max_iterations = 400
    load_run = None
    load_checkpoint = None
    resume = False


__all__ = ["TomatoPickPhysicsV2PPORunnerCfg"]
