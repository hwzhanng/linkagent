"""小果捏取 v1 的 PPO 配置；与旧权重维度兼容但不复用其优化器状态。"""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SmallFruitPinchPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 2400
    save_interval = 50
    experiment_name = "small_fruit_pinch_v1"
    run_name = "from_scratch"
    logger = "wandb"
    wandb_project = "linkagent-small-fruit-pinch"
    clip_actions = 1.0
    check_for_nan = True
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            # P1 已证明初始工作区存在严格三指持稳解。14D 中 0.45 的独立
            # 高斯扰动会几乎每帧破坏至少一个接触，因此采用保守初值并仍让
            # PPO 自主学习/调节方差，而非固定执行参考动作。
            init_std=0.20,
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
        entropy_coef=0.003,
        num_learning_epochs=4,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.75,
    )


__all__ = ["SmallFruitPinchPPORunnerCfg"]
