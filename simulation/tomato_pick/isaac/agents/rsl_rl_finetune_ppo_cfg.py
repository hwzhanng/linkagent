"""Conservative PPO configuration for adapting a proven tomato policy."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from simulation.tomato_pick.reward import SHAPING_GAMMA


@configclass
class TomatoPickFineTunePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Same policy interface as minimal-v3 with conservative update sizes."""

    num_steps_per_env = 32
    max_iterations = 100
    save_interval = 25
    experiment_name = "tomato_pick_minimal_v3_finetune"
    run_name = "tomato_pick_minimal_v3_finetune"
    logger = "wandb"
    wandb_project = "PPO"
    clip_actions = 1.0
    check_for_nan = True
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
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
        entropy_coef=5.0e-4,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=SHAPING_GAMMA,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


__all__ = ["TomatoPickFineTunePPORunnerCfg"]
