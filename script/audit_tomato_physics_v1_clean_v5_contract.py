"""Compare the frozen Physics-v1 training contract with released Clean-v5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.tomato_pick.isaac.agents.rsl_rl_ppo_cfg import (
    TomatoPickPPORunnerCfg,
)
from simulation.tomato_pick.isaac.env_cfg import TomatoPickEnvCfg
from simulation.tomato_pick.isaac.physics_v1.agents.rsl_rl_ppo_cfg import (
    TomatoPickPhysicsV1PPORunnerCfg,
)
from simulation.tomato_pick.isaac.physics_v1.env_cfg import (
    TomatoPickPhysicsV1EnvCfg,
)
from simulation.tomato_pick.reward import CURRICULUM_POINTS


OUTPUT = (
    PROJECT_ROOT
    / "logs/qualification/tomato_physics_v1/clean_v5_training_reward_comparison.json"
)
RELEASE = (
    PROJECT_ROOT
    / "logs/evaluations/tomato_pick_clean_v5_2026-08-08_fixed/release_manifest.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ppo_snapshot(config) -> dict:
    return {
        "num_steps_per_env": config.num_steps_per_env,
        "max_iterations": config.max_iterations,
        "save_interval": config.save_interval,
        "clip_actions": config.clip_actions,
        "actor_hidden_dims": list(config.actor.hidden_dims),
        "critic_hidden_dims": list(config.critic.hidden_dims),
        "activation": config.actor.activation,
        "actor_obs_normalization": config.actor.obs_normalization,
        "critic_obs_normalization": config.critic.obs_normalization,
        "initial_std": config.actor.distribution_cfg.init_std,
        "std_type": config.actor.distribution_cfg.std_type,
        "value_loss_coef": config.algorithm.value_loss_coef,
        "clip_param": config.algorithm.clip_param,
        "entropy_coef": config.algorithm.entropy_coef,
        "num_learning_epochs": config.algorithm.num_learning_epochs,
        "num_mini_batches": config.algorithm.num_mini_batches,
        "learning_rate": config.algorithm.learning_rate,
        "schedule": config.algorithm.schedule,
        "gamma": config.algorithm.gamma,
        "lambda": config.algorithm.lam,
        "desired_kl": config.algorithm.desired_kl,
        "max_grad_norm": config.algorithm.max_grad_norm,
    }


def _env_snapshot(config) -> dict:
    result = {
        "action_dim": config.action_space,
        "actor_observation_dim": config.observation_space,
        "critic_state_dim": config.state_space,
        "physics_hz": round(1.0 / float(config.sim.dt)),
        "policy_hz": round(1.0 / (float(config.sim.dt) * config.decimation)),
        "episode_seconds": config.episode_length_s,
        "success_hold_steps": config.success_hold_steps,
        "stable_detach_credit_steps": config.stable_detach_credit_steps,
        "reach_distance_scale_m": config.reach_distance_scale_m,
        "grasp_full_contact_force_n": config.grasp_full_contact_force_n,
        "maximum_retained_distance_m": config.maximum_retained_distance_m,
        "maximum_retained_relative_speed_m_s": (
            config.maximum_retained_relative_speed_m_s
        ),
        "maximum_fingertip_force_n": config.maximum_fingertip_force_n,
        "reward_rollout_steps": config.reward_rollout_steps,
    }
    for name in (
        "minimum_support_force_n",
        "minimum_supported_fingertips",
        "thumb_opposition_required",
        "stem_damage_onset_ratio",
        "stem_damage_recovery_time_s",
    ):
        if hasattr(config, name):
            result[name] = getattr(config, name)
    return result


def main() -> int:
    clean_agent = TomatoPickPPORunnerCfg()
    physics_agent = TomatoPickPhysicsV1PPORunnerCfg()
    clean_env = TomatoPickEnvCfg()
    physics_env = TomatoPickPhysicsV1EnvCfg()
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    best = release["best_summary"]
    reward_path = PROJECT_ROOT / "simulation/tomato_pick/reward.py"
    physics_env_source = (
        PROJECT_ROOT / "simulation/tomato_pick/isaac/physics_v1/env.py"
    ).read_text(encoding="utf-8")

    clean_ppo = _ppo_snapshot(clean_agent)
    physics_ppo = _ppo_snapshot(physics_agent)
    clean_env_values = _env_snapshot(clean_env)
    physics_env_values = _env_snapshot(physics_env)
    common_interface_keys = (
        "action_dim",
        "actor_observation_dim",
        "critic_state_dim",
        "physics_hz",
        "policy_hz",
        "episode_seconds",
        "success_hold_steps",
        "stable_detach_credit_steps",
        "reach_distance_scale_m",
        "grasp_full_contact_force_n",
        "reward_rollout_steps",
    )
    checks = {
        "clean_v5_release_success_above_90_percent": best["success_rate"] >= 0.90,
        "ppo_hyperparameters_exactly_reused": clean_ppo == physics_ppo,
        "learning_interface_exactly_reused": all(
            clean_env_values[name] == physics_env_values[name]
            for name in common_interface_keys
        ),
        "discovery_reward_is_exact_through_update_250": (
            tuple(CURRICULUM_POINTS[:2]) == ((0.0, 0.0), (250.0, 0.0))
        ),
        "aligned_reward_reaches_full_strength": CURRICULUM_POINTS[-1] == (1300.0, 1.0),
        "physics_success_requires_secure_support": (
            "self._retained &= self._shadow_secure_support" in physics_env_source
        ),
        "physics_downstream_credit_requires_secure_support": (
            "secure_support_credit_gate" in physics_env_source
            and "self._break_substep_secure_support" in physics_env_source
        ),
        "physics_retention_is_stricter_than_clean_v5": (
            physics_env_values["maximum_retained_distance_m"]
            < clean_env_values["maximum_retained_distance_m"]
            and physics_env_values["maximum_retained_relative_speed_m_s"]
            < clean_env_values["maximum_retained_relative_speed_m_s"]
        ),
    }
    report = {
        "schema_version": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "clean_v5_release": {
            "best_checkpoint": release["best_checkpoint"],
            "checkpoint_sha256": release["best_checkpoint_sha256"],
            "completed_episodes": best["completed_episodes"],
            "success_rate": best["success_rate"],
            "success_wilson_95": [
                best["success_wilson_95_low"],
                best["success_wilson_95_high"],
            ],
            "drop_rate": best["drop_rate"],
            "minimum_seed_success_rate": best["minimum_seed_success_rate"],
        },
        "reward": {
            "shared_source": str(reward_path.relative_to(PROJECT_ROOT)),
            "shared_source_sha256": _sha256(reward_path),
            "curriculum_points": CURRICULUM_POINTS,
            "interpretation": {
                "updates_0_to_250": (
                    "exact Clean-v5 discovery endpoint; bounded reach, grasp, opposition, "
                    "stem-load and qualified-break exploration remains available only "
                    "under measured Physics-v1 secure support"
                ),
                "updates_250_to_1300": (
                    "smoothly removes immediate break credit and adds stable-detach, "
                    "outcome penalties, stronger success and time pressure"
                ),
                "updates_1300_plus": "fully aligned endpoint",
            },
        },
        "ppo": {"clean_v5": clean_ppo, "physics_v1": physics_ppo},
        "environment": {
            "clean_v5": clean_env_values,
            "physics_v1": physics_env_values,
            "intentional_physics_v1_changes": [
                "user-calibrated collision-safe five-dimensional hand manifold",
                "passive follower mimics and cross-digit self collision",
                "compliant low-restitution literature-prior contact",
                "correlated fruit and stem randomization with duration-based fracture",
                "broad reachable arm-relative reset workspace",
                "secure thumb-opposed retention and success predicate",
            ],
        },
        "decision": (
            "The Physics-v1-only secure-support credit alignment passed its fresh "
            "from-scratch 8192x50 first-success gate; keep reward weights and PPO frozen."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
