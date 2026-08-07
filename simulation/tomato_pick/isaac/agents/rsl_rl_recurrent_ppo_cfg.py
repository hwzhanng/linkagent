"""Recurrent PPO for deployable tactile and visual temporal inference.

The environment, observation vector, action space, reward, and privileged critic
state are identical to the feed-forward task.  Only the actor gains memory so
it can infer slip and load changes from the existing sensor stream.
"""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)

from simulation.tomato_pick.reward import SHAPING_GAMMA


@configclass
class TomatoPickRecurrentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "tomato_pick_minimal_v3_recurrent"
    run_name = "tomato_pick_minimal_v3_recurrent"
    logger = "wandb"
    wandb_project = "PPO"
    clip_actions = 1.0
    check_for_nan = True
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}

    # A single 256-state LSTM is enough to span many 50 ms control periods.
    # The actor still receives only the unchanged 66 deployable dimensions.
    actor = RslRlRNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.60,
            std_type="log",
        ),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    # The asymmetric critic already observes exact fruit velocity, mass, stem
    # load, and detachment, so recurrence is deliberately limited to the actor.
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=SHAPING_GAMMA,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


__all__ = ["TomatoPickRecurrentPPORunnerCfg"]
