"""Clean-v5 PPO hyperparameters retained for Physics-v1 future training."""

from isaaclab.utils.configclass import configclass

from simulation.tomato_pick.isaac.agents.rsl_rl_ppo_cfg import TomatoPickPPORunnerCfg


@configclass
class TomatoPickPhysicsV1PPORunnerCfg(TomatoPickPPORunnerCfg):
    experiment_name = "tomato_pick_physics_v1"
    run_name = "tomato_pick_physics_v1_from_scratch"


__all__ = ["TomatoPickPhysicsV1PPORunnerCfg"]
