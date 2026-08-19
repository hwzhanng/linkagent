"""Frozen Clean-v5 PPO with V2-only recurrent actor memory."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlRNNModelCfg

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
    # Repair44 changes only the V2 actor model: the fracture-to-retention
    # transition is partially observable to the deployable 103D policy, so a
    # compact recurrent state supplies phase memory without adding privileged
    # detachment state or more hand-built observation history.
    # Critic, PPO optimizer/losses, reward, physics, and observation contracts
    # remain inherited and unchanged.
    actor = RslRlRNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.60,
            std_type="log",
        ),
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )


__all__ = ["TomatoPickPhysicsV2PPORunnerCfg"]
