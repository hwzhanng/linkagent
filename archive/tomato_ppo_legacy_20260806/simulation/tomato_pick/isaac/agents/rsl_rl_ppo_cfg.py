"""PPO configuration only; importing this module does not start training."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class SquashedGaussianDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
    class_name: str = (
        "simulation.tomato_pick.isaac.agents.squashed_gaussian:"
        "SquashedGaussianDistribution"
    )
    init_std: float = 0.25
    min_std: float = 0.05
    max_std: float = 0.80
    # Training-only rank-one noise shared by the ten G20 flexion drivers.
    # Every one of the 22 deterministic action means remains independent.
    hand_closure_correlation_std: float = 0.0
    max_latent_mean: float = 4.0
    epsilon: float = 1.0e-6


@configclass
class TomatoPickPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # A 64-step rollout covers the 20-step post-detachment retention invariant
    # without imposing a scripted phase sequence.
    num_steps_per_env = 64
    max_iterations = 1500
    # Keep a recent finite rollback point during contact-rich training.  The
    # checkpoint is only ~2.8 MB, so ten-update snapshots are cheap and avoid
    # losing millions of valid samples if a later numerical guard trips.
    save_interval = 10
    experiment_name = "tomato_pick_v1"
    run_name = "tomato_pick_v1"
    logger = "wandb"
    wandb_project = "PPO"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    # Actions are already bounded by the squashed distribution.  A second
    # wrapper clamp would make executed actions diverge from stored log-probs.
    clip_actions = None
    check_for_nan = True
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=SquashedGaussianDistributionCfg(),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        # Scale optimizer batches with the 8192 x 64 rollout.  Four batches
        # would create 131072-sample Adam steps, over five times the audited
        # 4096 x 24 setup and only weakly update the policy per collected
        # transition.  Sixteen keeps a 32768-sample minibatch while retaining
        # the full 64-step credit horizon.
        num_mini_batches=16,
        learning_rate=1.0e-4,
        # RSL-RL's adaptive rule may raise this as high as 1e-2.  On the
        # calibrated grasp MDP it raised the learning rate and corrupted the
        # policy with NaNs. Keep the audited conservative base rate fixed.
        schedule="fixed",
        gamma=0.99,
        # The lambda=0.98 single-variable experiment did not improve fixed
        # full-chain success, so retain the audited baseline.
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Normalize once over the complete fresh on-policy rollout.  Per-
        # minibatch normalization would make the same sample depend on the
        # arbitrary minibatch partition.
        normalize_advantage_per_mini_batch=False,
    )


__all__ = ["SquashedGaussianDistributionCfg", "TomatoPickPPORunnerCfg"]
