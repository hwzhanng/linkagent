"""Incremental PPO configuration for the proven minimal-v3 MLP policy."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from simulation.tomato_pick.reward import SHAPING_GAMMA


@configclass
class ResidualLSTMModelCfg(RslRlMLPModelCfg):
    class_name: str = (
        "simulation.tomato_pick.isaac.agents.residual_lstm_model:ResidualLSTMModel"
    )
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    freeze_base: bool = True
    freeze_obs_normalization: bool = True


@configclass
class TomatoPickResidualLSTMRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 500
    # Residual adaptation can change deterministic behavior before aggregate
    # stochastic training metrics react, so retain fine-grained rollback points.
    save_interval = 5
    experiment_name = "tomato_pick_minimal_v3_residual_lstm"
    run_name = "tomato_pick_minimal_v3_residual_lstm"
    logger = "wandb"
    wandb_project = "PPO"
    clip_actions = 1.0
    check_for_nan = True
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = ResidualLSTMModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.60,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0005,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=SHAPING_GAMMA,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


__all__ = ["ResidualLSTMModelCfg", "TomatoPickResidualLSTMRunnerCfg"]
