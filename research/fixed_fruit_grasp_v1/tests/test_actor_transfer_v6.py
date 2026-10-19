"""保守 PPO（v1.5.0 -> v1.6.0）显式 actor 迁移与 log_std 重置的纯 CPU 回归。

覆盖四件事：
* 新迁移只接受“源 3.0e-4/0.20 -> 目标 1.0e-4/0.08”两个 agent 字段变化；
  其余 agent 字段、未知/缺失字段、架构尺寸、奖励子树与真实物理判据一律拒绝。
* 源 metadata 内的 agent config/architecture 必须与自身 SHA 自洽，目标实际
  映射必须与传入 SHA 绑定，任务身份除 task_version 外（含 reward）逐字节一致。
* 载入 actor 后只对唯一 ``distribution.log_std_param`` 显式重置一次，均值权重
  与 obs 归一化缓冲逐字节保留；重置是显式探索重初始化，不是学习成功。
* 正常 resume/evaluate 静态调用路径完全不放宽、不重置。

不启动 Kit/PhysX/GPU，不做训练或评估。
"""

from __future__ import annotations

import ast
import copy
import math
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from research.fixed_fruit_grasp_v1 import contract as C
from research.fixed_fruit_grasp_v1.run_utils import (
    ACTOR_EXPLORATION_STD_RESET_MARKER,
    ACTOR_LOG_STD_STATE_KEY,
    AGENT_CONFIG_INITIAL_STD_PATH,
    AGENT_CONFIG_LEARNING_RATE_PATH,
    CONSERVATIVE_PPO_SOURCE_INITIAL_STD,
    CONSERVATIVE_PPO_SOURCE_LEARNING_RATE,
    CONSERVATIVE_PPO_TRANSITION,
    CONSERVATIVE_PPO_AGENT_CONFIG_PATHS,
    CONSERVATIVE_PPO_ARCHITECTURE_PATHS,
    FIXED_FRUIT_ACTOR_TRANSFER_TRANSITIONS,
    FIXED_FRUIT_INITIAL_STD,
    FIXED_FRUIT_PPO_LEARNING_RATE,
    agent_config_sha256,
    contract_sha256,
    reset_actor_exploration_std,
    sha256_json,
    validate_checkpoint_compatibility,
    validate_explicit_actor_transfer,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SOURCE_VERSION = "fixed_fruit_grasp_v1.5.0"
DESTINATION_VERSION = "fixed_fruit_grasp_v1.6.0"
PHYSICAL_SOURCES = (
    "physical_hold.py",
    "physical_contacts.py",
    "collision_cooking.py",
)
SOURCE_ROOT = "/tmp/run/ppo-grasp-learning/source-v5"
DESTINATION_ROOT = "/tmp/run/ppo-grasp-learning/source-v6"


# ------------------------------------------------------------ agent 配置样本
def _source_agent_config() -> dict:
    """与真实 v1.5.0 checkpoint 内 agent_config 同结构的样本（含全部 PPO 超参）。"""

    return {
        "class_name": "RslRlOnPolicyRunnerCfg",
        "device": "cuda:0",
        "logger": "tensorboard",
        "num_steps_per_env": 64,
        "max_iterations": 2400,
        "save_interval": 50,
        "experiment_name": "small_fruit_pinch_v1",
        "run_name": "fixed_fruit_grasp_v1",
        "clip_actions": 1.0,
        "check_for_nan": True,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {
                "init_std": CONSERVATIVE_PPO_SOURCE_INITIAL_STD,
                "std_type": "log",
            },
        },
        "critic": {
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": None,
        },
        "algorithm": {
            "num_learning_epochs": 4,
            "num_mini_batches": 8,
            "learning_rate": CONSERVATIVE_PPO_SOURCE_LEARNING_RATE,
            "schedule": "fixed",
            "gamma": 0.995,
            "lam": 0.95,
            "entropy_coef": 0.003,
            "desired_kl": 0.01,
            "max_grad_norm": 0.75,
            "optimizer": "adam",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "multi_gpu": None,
        "seed": 42,
    }


def _architecture_of(config: dict) -> dict:
    """与 train.py::_architecture_config 同构的架构身份样本。"""

    return {
        "actor": copy.deepcopy(config["actor"]),
        "critic": copy.deepcopy(config["critic"]),
        "obs_groups": copy.deepcopy(config["obs_groups"]),
        "clip_actions": config["clip_actions"],
        "check_for_nan": config["check_for_nan"],
        "num_actions": 14,
        "actor_obs_dim": 114,
    }


def _declared_destination(config: dict) -> dict:
    """按已声明迁移改写目标配置：只动 learning_rate 与 init_std。"""

    destination = copy.deepcopy(config)
    destination["algorithm"]["learning_rate"] = FIXED_FRUIT_PPO_LEARNING_RATE
    destination["actor"]["distribution_cfg"]["init_std"] = FIXED_FRUIT_INITIAL_STD
    return destination


# --------------------------------------------------------------- 任务身份样本
def _identity(version: str, root: str) -> dict:
    assets = {
        "root": root + "/research/small_fruit_pinch_v1/assets/left_hand_urdf",
        "usd_path": root + "/research/small_fruit_pinch_v1/assets/left_hand_urdf/left_hand.usd",
        "selected": "canonical isolated left hand USD",
    }
    actuation = {
        "source": "cfg.robot.actuators（canonical 孤立左手 USD 驱动）",
        "effort_limit_sim": 0.2,
        "velocity_limit_sim": 1.2,
    }
    fruit = {"shape": "sphere", "equivalent_diameter_m": 0.067, "density_kg_m3": 900.0}
    physics = {
        "sim_dt_s": 1.0 / 120.0,
        "policy_dt_s": 0.05,
        "decimation": 6,
        "hand_collision_cooking": {
            "approximation": "convexDecomposition",
            "source": "research.fixed_fruit_grasp_v1.collision_cooking",
        },
        "support": {
            "kind": "declared_compliant_support_physx_wrench_no_state_write",
            "release_step": 60,
        },
        "observation_dims": {"actor": 114, "critic": 175, "action": 14},
    }
    criteria = {
        "thumb_required": False,
        "hold_max_drift_m": 0.01,
        "hold_max_relative_speed_m_s": 0.05,
        "drop_height_tolerance_m": 0.08,
        "drop_max_drift_m": 0.12,
        "instability_max_fruit_speed_m_s": 5.0,
        "instability_max_joint_velocity_rad_s": 4.0,
        "joint_limit_margin_rad": 0.01,
        "penetration_limit_m": 0.005,
    }
    contact = {
        "force_threshold_n": 0.08,
        "hard_high_n": [15.0] * 5,
        "min_contact_fingers": 2,
        "real_self_collisions_enabled": True,
        "palm_fruit_collision_filter_enabled": False,
        "palm_only_counts_as_success": False,
        "declared_self_collision_pairs": [["thumb_link4", "index_link3"]],
        "effective_self_collision_pairs": [["thumb_link4", "index_link3"]],
        "self_collision_penetration_limit_m": 0.01,
        "skipped_self_collision_pairs": [],
    }
    reward = {
        "profile": "reward_only_v1.5.0",
        "hard_fault_penalty_flags": [
            "overload",
            "nonfinite",
            "penetration",
            "joint_limit",
            "self_collision",
            "instability",
            "joint_effort",
        ],
        "weights": {
            "approach": 1.0,
            "contact": 0.25,
            "opposition": 0.25,
            "band": 0.35,
            "closure": 0.0,
            "unsupported_hold": 6.0,
            "stability": 0.5,
            "height": 0.0,
            "palm_only_penalty": -0.1,
            "overload_penalty": -0.3,
            "forbidden_penalty": -0.0,
            "smoothness_penalty": -0.12,
            "hard_fault_penalty": -8.0,
            "drop_penalty": -6.0,
            "success_bonus": 30.0,
        },
    }
    stage = {
        "stage": "contact",
        "success_criterion": "contact_grasp_hold_1s_without_support",
        "support_release_step": 60,
        "support_release_time_s": 3.0,
        "required_unsupported_hold_s": 1.0,
        "required_unsupported_hold_steps": 20,
        "required_unsupported_hold_substeps": 120,
        "min_height_gain_m": 0.0,
        "requires_lift_program": False,
        "counts_as_goal": False,
        "goal_status": "GOAL_NOT_REACHED_LIFT_PROGRAM_NOT_IMPLEMENTED",
    }
    return {
        "contract_id": "fixed_fruit_grasp_v1",
        "task_id": "LinkAgent-FixedFruitGrasp-v1",
        "task_version": version,
        "record_schema_version": "fixed_fruit_grasp_episode_record.v1",
        "run": {"num_envs": 4, "device": "cuda:0", "seed": 42},
        "stage": stage,
        "physics": physics,
        "fruit": fruit,
        "actuation": actuation,
        "contact": contact,
        "reward": reward,
        "criteria": criteria,
        "goal": {"goal_status": "GOAL_NOT_REACHED_LIFT_PROGRAM_NOT_IMPLEMENTED"},
        "randomization": {"fruit_position": "fixed", "action_delay_steps": [0, 0]},
        "safety_guard": {"guarded_joint_count": 21, "policy_observation_joints": 16},
        "assets": assets,
        "initial_pose": {
            "deterministic": True,
            "pregrasp_distance_m": 0.005,
            "closed_grasp_fruit_center_m": [-0.07916, 0.01116, -0.16],
        },
        "declared_limitations": ["holding-only; lift program not implemented"],
        "asset_config": {
            "usd_path": assets["usd_path"],
            "asset_config_path": assets["root"] + "/configuration/left_hand_physics.usd",
            "asset_files": {"left_hand.usd": "b5ab656b3b58b2e1"},
            "fruit": copy.deepcopy(fruit),
            "actuation": copy.deepcopy(actuation),
            "physics": copy.deepcopy(physics),
            "contact": copy.deepcopy(contact),
            "criteria": copy.deepcopy(criteria),
        },
    }


def _destination_identity(source_identity: dict) -> dict:
    """按已声明迁移构造目标身份：只改 task_version 与 source-v6 路径前缀。"""

    destination = copy.deepcopy(source_identity)
    destination["task_version"] = DESTINATION_VERSION
    return _rewrite_root(destination, "/source-v5/", "/source-v6/")


def _rewrite_root(value, old: str, new: str):
    if isinstance(value, dict):
        return {key: _rewrite_root(child, old, new) for key, child in value.items()}
    if isinstance(value, list):
        return [_rewrite_root(child, old, new) for child in value]
    if isinstance(value, str):
        return value.replace(old, new)
    return value


def transfer_inputs():
    """构造 (源 metadata, 目标身份, 目标 hashes, 目标 config, 目标 architecture)。"""

    source_identity = _identity(SOURCE_VERSION, SOURCE_ROOT)
    source_config = _source_agent_config()
    source_architecture = _architecture_of(source_config)
    physical_files = {
        "research/fixed_fruit_grasp_v1/" + name: "a" * 64 for name in PHYSICAL_SOURCES
    }
    metadata = {
        "contract_id": "fixed_fruit_grasp_v1",
        "task_identity": source_identity,
        "task_identity_sha256": sha256_json(source_identity),
        "contract_sha256": contract_sha256(source_identity),
        "stage": "contact",
        "agent_config": source_config,
        "agent_config_sha256": sha256_json(source_config),
        "agent_config_comparison_sha256": agent_config_sha256(source_config),
        "agent_architecture": source_architecture,
        "agent_architecture_sha256": sha256_json(source_architecture),
        "source_and_asset_hashes": {
            "training_semantics_source_files": dict(physical_files),
            "referenced_asset_files": {SOURCE_ROOT + "/assets/left_hand.usd": "bytes"},
            "missing_referenced_paths": [],
        },
        "progress": {"total_updates_completed": 85},
    }
    hashes = {
        "training_semantics_source_files": dict(physical_files),
        "referenced_asset_files": {DESTINATION_ROOT + "/assets/left_hand.usd": "bytes"},
        "missing_referenced_paths": [],
    }
    destination_config = _declared_destination(source_config)
    destination_architecture = _architecture_of(destination_config)
    return (
        metadata,
        _destination_identity(source_identity),
        hashes,
        destination_config,
        destination_architecture,
    )


def _refresh_source_agent_digests(metadata: dict) -> None:
    metadata["agent_config_sha256"] = sha256_json(metadata["agent_config"])
    metadata["agent_config_comparison_sha256"] = agent_config_sha256(
        metadata["agent_config"]
    )
    metadata["agent_architecture_sha256"] = sha256_json(metadata["agent_architecture"])


def _refresh_source_identity_digests(metadata: dict) -> None:
    metadata["task_identity_sha256"] = sha256_json(metadata["task_identity"])
    metadata["contract_sha256"] = contract_sha256(metadata["task_identity"])


def validate(metadata, identity, hashes, config, architecture, actual="a" * 64,
             expected="a" * 64, architecture_sha=None, config_sha=None):
    return validate_explicit_actor_transfer(
        metadata,
        identity,
        "contact",
        hashes,
        sha256_json(architecture) if architecture_sha is None else architecture_sha,
        agent_config_sha256(config) if config_sha is None else config_sha,
        actual,
        expected,
        actual_agent_config=config,
        actual_architecture=architecture,
    )


# ------------------------------------------------- 新允许迁移（唯一合法组合）
def test_conservative_transfer_accepts_only_the_two_declared_fields() -> None:
    metadata, identity, hashes, config, architecture = transfer_inputs()
    result = validate(metadata, identity, hashes, config, architecture)

    assert result["mode"] == "actor_only_task_semantics_transfer"
    assert result["source_task_version"] == SOURCE_VERSION
    assert result["destination_task_version"] == DESTINATION_VERSION
    assert result["source_total_updates_completed"] == 85
    assert result["new_semantics_updates_before"] == 0
    assert result["critic_optimizer_and_iteration_reset"] is True
    assert result["actor_normalizer_transferred"] is True
    assert result["is_new_semantics_success_evidence"] is False
    assert result["physical_criterion_unchanged"] is True
    assert result["reward_profile_transfer_only"] is False
    assert result["exploration_std_reset_required"] is True
    assert result["agent_config_validation"]["changed_agent_config_paths"] == [
        "actor.distribution_cfg.init_std",
        "algorithm.learning_rate",
    ]
    assert result["agent_config_validation"]["changed_agent_architecture_paths"] == [
        "actor.distribution_cfg.init_std"
    ]
    assert result["agent_config_validation"]["source_learning_rate"] == (
        CONSERVATIVE_PPO_SOURCE_LEARNING_RATE
    )
    assert result["agent_config_validation"]["destination_learning_rate"] == (
        FIXED_FRUIT_PPO_LEARNING_RATE
    )
    assert result["agent_config_validation"]["source_initial_std"] == (
        CONSERVATIVE_PPO_SOURCE_INITIAL_STD
    )
    assert result["agent_config_validation"]["destination_initial_std"] == (
        FIXED_FRUIT_INITIAL_STD
    )


def test_declared_transition_table_is_exactly_the_three_reviewed_pairs() -> None:
    assert FIXED_FRUIT_ACTOR_TRANSFER_TRANSITIONS == frozenset(
        {
            ("fixed_fruit_grasp_v1.3.0", "fixed_fruit_grasp_v1.4.0"),
            ("fixed_fruit_grasp_v1.4.0", "fixed_fruit_grasp_v1.5.0"),
            CONSERVATIVE_PPO_TRANSITION,
        }
    )
    assert CONSERVATIVE_PPO_TRANSITION == (SOURCE_VERSION, DESTINATION_VERSION)
    assert CONSERVATIVE_PPO_AGENT_CONFIG_PATHS == frozenset(
        {AGENT_CONFIG_LEARNING_RATE_PATH, AGENT_CONFIG_INITIAL_STD_PATH}
    )
    assert CONSERVATIVE_PPO_ARCHITECTURE_PATHS == frozenset({AGENT_CONFIG_INITIAL_STD_PATH})


def test_older_transitions_never_request_an_exploration_std_reset() -> None:
    from research.fixed_fruit_grasp_v1.tests.test_actor_transfer_v4 import (
        identities,
        validate as validate_v4,
    )
    from research.fixed_fruit_grasp_v1.tests.test_actor_transfer_v5 import (
        reward_identities,
        validate as validate_v5,
    )

    earlier = validate_v4(*identities())
    assert earlier["exploration_std_reset_required"] is False
    assert earlier["agent_config_validation"] is None
    assert earlier["physical_criterion_unchanged"] is False

    reward = validate_v5(*reward_identities())
    assert reward["exploration_std_reset_required"] is False
    assert reward["agent_config_validation"] is None
    assert reward["reward_profile_transfer_only"] is True
    assert reward["physical_criterion_unchanged"] is True


def test_declared_device_move_is_the_only_other_permitted_runtime_field() -> None:
    metadata, identity, hashes, config, architecture = transfer_inputs()
    config["device"] = "cpu"
    result = validate(metadata, identity, hashes, config, architecture)
    assert result["exploration_std_reset_required"] is True


@pytest.mark.parametrize(
    "change",
    [
        "learning_rate_value",
        "source_learning_rate",
        "initial_std_value",
        "source_initial_std",
        "std_type",
        "entropy",
        "clip_param",
        "num_learning_epochs",
        "num_mini_batches",
        "gamma",
        "lam",
        "desired_kl",
        "max_grad_norm",
        "value_loss_coef",
        "use_clipped_value_loss",
        "schedule",
        "num_steps_per_env",
        "save_interval",
        "seed",
        "run_name",
        "obs_groups",
        "actor_hidden_dims",
        "actor_activation",
        "actor_obs_normalization",
        "critic_hidden_dims",
        "critic_obs_normalization",
        "critic_distribution_cfg",
        "unknown_new_field",
        "dropped_field",
        "architecture_num_actions",
        "architecture_actor_obs_dim",
        "architecture_actor_hidden_dims",
    ],
)
def test_conservative_transfer_rejects_anything_outside_the_two_fields(change) -> None:
    metadata, identity, hashes, config, architecture = transfer_inputs()
    actor = config["actor"]
    algorithm = config["algorithm"]
    if change == "learning_rate_value":
        algorithm["learning_rate"] = 1.0e-3
    elif change == "source_learning_rate":
        metadata["agent_config"]["algorithm"]["learning_rate"] = 1.0e-3
        _refresh_source_agent_digests(metadata)
    elif change == "initial_std_value":
        actor["distribution_cfg"]["init_std"] = 0.1
    elif change == "source_initial_std":
        metadata["agent_config"]["actor"]["distribution_cfg"]["init_std"] = 0.1
        _refresh_source_agent_digests(metadata)
    elif change == "std_type":
        actor["distribution_cfg"]["std_type"] = "scalar"
    elif change == "entropy":
        algorithm["entropy_coef"] = 0.006
    elif change == "clip_param":
        algorithm["clip_param"] = 0.3
    elif change == "num_learning_epochs":
        algorithm["num_learning_epochs"] = 8
    elif change == "num_mini_batches":
        algorithm["num_mini_batches"] = 16
    elif change == "gamma":
        algorithm["gamma"] = 0.99
    elif change == "lam":
        algorithm["lam"] = 0.9
    elif change == "desired_kl":
        algorithm["desired_kl"] = 0.02
    elif change == "max_grad_norm":
        algorithm["max_grad_norm"] = 1.0
    elif change == "value_loss_coef":
        algorithm["value_loss_coef"] = 0.5
    elif change == "use_clipped_value_loss":
        algorithm["use_clipped_value_loss"] = False
    elif change == "schedule":
        algorithm["schedule"] = "adaptive"
    elif change == "num_steps_per_env":
        config["num_steps_per_env"] = 32
    elif change == "save_interval":
        config["save_interval"] = 25
    elif change == "seed":
        config["seed"] = 7
    elif change == "run_name":
        config["run_name"] = "fixed_fruit_grasp_v1_conservative"
    elif change == "obs_groups":
        config["obs_groups"] = {"actor": ["policy"], "critic": ["critic", "policy"]}
    elif change == "actor_hidden_dims":
        actor["hidden_dims"] = [512, 256]
    elif change == "actor_activation":
        actor["activation"] = "relu"
    elif change == "actor_obs_normalization":
        actor["obs_normalization"] = False
    elif change == "critic_hidden_dims":
        config["critic"]["hidden_dims"] = [512, 256]
    elif change == "critic_obs_normalization":
        config["critic"]["obs_normalization"] = False
    elif change == "critic_distribution_cfg":
        config["critic"]["distribution_cfg"] = {"init_std": 0.2, "std_type": "log"}
    elif change == "unknown_new_field":
        config["unreviewed_switch"] = True
    elif change == "dropped_field":
        config.pop("check_for_nan")
    elif change == "architecture_num_actions":
        architecture["num_actions"] = 15
    elif change == "architecture_actor_obs_dim":
        architecture["actor_obs_dim"] = 175
    elif change == "architecture_actor_hidden_dims":
        architecture["actor"]["hidden_dims"] = [512, 256]
    else:  # pragma: no cover - 参数表与分支必须保持一致
        raise AssertionError("unhandled change: " + change)
    with pytest.raises(ValueError):
        validate(metadata, identity, hashes, config, architecture)


@pytest.mark.parametrize(
    "change",
    [
        "reward_weight",
        "reward_profile_text",
        "reward_hard_fault_flags",
        "dropped_reward",
        "criteria_speed",
        "contact_force",
        "stage_release_step",
        "physics_sim_dt",
        "support_kind",
        "randomization",
        "safety_guard",
        "observation_dims",
        "initial_pose",
        "assets_usd",
        "unknown_new_field",
    ],
)
def test_conservative_transfer_rejects_reward_physics_or_new_identity_change(change) -> None:
    metadata, identity, hashes, config, architecture = transfer_inputs()
    if change == "reward_weight":
        identity["reward"]["weights"]["contact"] = 1.0
    elif change == "reward_profile_text":
        identity["reward"]["profile"] = "reward_only_v1.6.0"
    elif change == "reward_hard_fault_flags":
        identity["reward"]["hard_fault_penalty_flags"] = ["overload"]
    elif change == "dropped_reward":
        identity.pop("reward")
    elif change == "criteria_speed":
        identity["criteria"]["hold_max_relative_speed_m_s"] = 0.1
    elif change == "contact_force":
        identity["contact"]["hard_high_n"] = [30.0] * 5
    elif change == "stage_release_step":
        identity["stage"]["support_release_step"] = 120
    elif change == "physics_sim_dt":
        identity["physics"]["sim_dt_s"] = 1.0 / 60.0
    elif change == "support_kind":
        identity["physics"]["support"]["kind"] = "scripted_position"
    elif change == "randomization":
        identity["randomization"]["fruit_position"] = "uniform"
    elif change == "safety_guard":
        identity["safety_guard"]["policy_observation_joints"] = 20
    elif change == "observation_dims":
        identity["physics"]["observation_dims"]["critic"] = 174
    elif change == "initial_pose":
        identity["initial_pose"]["pregrasp_distance_m"] = 0.006
    elif change == "assets_usd":
        identity["assets"]["usd_path"] = identity["assets"]["usd_path"] + ".bak"
    elif change == "unknown_new_field":
        identity["unreviewed_identity_field"] = {"added": True}
    else:  # pragma: no cover
        raise AssertionError("unhandled change: " + change)
    with pytest.raises(ValueError):
        validate(metadata, identity, hashes, config, architecture)


@pytest.mark.parametrize("name", PHYSICAL_SOURCES)
@pytest.mark.parametrize("missing", [False, True])
def test_conservative_transfer_rejects_modified_or_missing_physical_source(name, missing) -> None:
    metadata, identity, hashes, config, architecture = transfer_inputs()
    key = "research/fixed_fruit_grasp_v1/" + name
    if missing:
        hashes["training_semantics_source_files"].pop(key)
    else:
        hashes["training_semantics_source_files"][key] = "b" * 64
    with pytest.raises(ValueError):
        validate(metadata, identity, hashes, config, architecture)


@pytest.mark.parametrize(
    "change",
    [
        "source_config_digest",
        "source_config_comparison_digest",
        "source_architecture_digest",
        "source_identity_digest",
        "source_contract_digest",
        "destination_config_digest",
        "destination_architecture_digest",
        "checkpoint_pin",
        "destination_asset_bytes",
        "source_asset_bytes",
        "missing_referenced_path",
        "unreviewed_transition_pair",
        "source_config_edited_with_recomputed_digest",
        "source_identity_edited_with_recomputed_digest",
        "source_update_count_bool",
    ],
)
def test_conservative_transfer_rejects_digest_hash_and_pin_tampering(change) -> None:
    metadata, identity, hashes, config, architecture = transfer_inputs()
    kwargs = {}
    if change == "source_config_digest":
        metadata["agent_config_sha256"] = "wrong"
    elif change == "source_config_comparison_digest":
        metadata["agent_config_comparison_sha256"] = "wrong"
    elif change == "source_architecture_digest":
        metadata["agent_architecture_sha256"] = "wrong"
    elif change == "source_identity_digest":
        metadata["task_identity_sha256"] = "wrong"
    elif change == "source_contract_digest":
        metadata["contract_sha256"] = "wrong"
    elif change == "destination_config_digest":
        kwargs["config_sha"] = "wrong"
    elif change == "destination_architecture_digest":
        kwargs["architecture_sha"] = "wrong"
    elif change == "checkpoint_pin":
        kwargs["actual"] = "b" * 64
    elif change == "destination_asset_bytes":
        hashes["referenced_asset_files"][DESTINATION_ROOT + "/assets/left_hand.usd"] = "changed"
    elif change == "source_asset_bytes":
        metadata["source_and_asset_hashes"]["referenced_asset_files"][
            SOURCE_ROOT + "/assets/left_hand.usd"
        ] = "changed"
    elif change == "missing_referenced_path":
        hashes["missing_referenced_paths"] = [DESTINATION_ROOT + "/assets/missing.usd"]
    elif change == "unreviewed_transition_pair":
        metadata["task_identity"]["task_version"] = "fixed_fruit_grasp_v1.4.0"
        _refresh_source_identity_digests(metadata)
    elif change == "source_config_edited_with_recomputed_digest":
        metadata["agent_config"]["algorithm"]["entropy_coef"] = 0.006
        _refresh_source_agent_digests(metadata)
    elif change == "source_identity_edited_with_recomputed_digest":
        metadata["task_identity"]["contact"]["hard_high_n"] = [30.0] * 5
        _refresh_source_identity_digests(metadata)
    elif change == "source_update_count_bool":
        metadata["progress"]["total_updates_completed"] = True
    else:  # pragma: no cover
        raise AssertionError("unhandled change: " + change)
    with pytest.raises(ValueError):
        validate(
            metadata,
            identity,
            hashes,
            config,
            architecture,
            actual=kwargs.get("actual", "a" * 64),
            expected="a" * 64,
            architecture_sha=kwargs.get("architecture_sha"),
            config_sha=kwargs.get("config_sha"),
        )


def test_conservative_transfer_requires_the_actual_config_mapping() -> None:
    metadata, identity, hashes, config, architecture = transfer_inputs()
    with pytest.raises(ValueError):
        validate_explicit_actor_transfer(
            metadata,
            identity,
            "contact",
            hashes,
            sha256_json(architecture),
            agent_config_sha256(config),
            "a" * 64,
            "a" * 64,
        )
    with pytest.raises(ValueError):
        validate_explicit_actor_transfer(
            metadata,
            identity,
            "contact",
            hashes,
            sha256_json(architecture),
            agent_config_sha256(config),
            "a" * 64,
            "a" * 64,
            actual_agent_config=config,
        )


def test_normal_resume_and_evaluate_paths_never_accept_the_old_identity() -> None:
    metadata, identity, hashes, _, _ = transfer_inputs()
    with pytest.raises(ValueError):
        validate_checkpoint_compatibility(metadata, identity, "contact")
    evaluate_source = (PACKAGE_ROOT / "evaluate.py").read_text(encoding="utf-8")
    assert "reset_actor_exploration_std" not in evaluate_source
    assert "exploration_std_reset" not in evaluate_source


def test_evaluate_applies_the_same_conservative_agent_constants() -> None:
    evaluate_source = (PACKAGE_ROOT / "evaluate.py").read_text(encoding="utf-8")
    assert "configure_fixed_fruit_agent_cfg" in evaluate_source
    assert "fixed_fruit_agent_settings" in evaluate_source
    assert FIXED_FRUIT_PPO_LEARNING_RATE == 1.0e-4
    assert FIXED_FRUIT_INITIAL_STD == 0.08


# ------------------------------------------------------------ log_std 重置行为
class _Distribution(nn.Module):
    def __init__(self, *, std_type: str = "log", init_std: float = 0.20, size: int = 14):
        super().__init__()
        self.std_type = std_type
        if std_type == "log":
            self.log_std_param = nn.Parameter(torch.full((size,), math.log(init_std)))
        else:
            self.std_param = nn.Parameter(torch.full((size,), init_std))


class _KeyedModule(nn.Module):
    """复刻 RSL-RL MLPModel 的 13 项 state_dict 名称形状（纯 CPU 样本）。"""

    def __init__(self, *, std_type: str = "log", init_std: float = 0.20):
        super().__init__()
        normalizer = nn.Module()
        normalizer.register_buffer("_mean", torch.zeros(1, 114))
        normalizer.register_buffer("_var", torch.ones(1, 114))
        normalizer.register_buffer("_std", torch.ones(1, 114))
        normalizer.register_buffer("count", torch.zeros((), dtype=torch.int64))
        self.obs_normalizer = normalizer
        self.distribution = _Distribution(std_type=std_type, init_std=init_std)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("0", nn.Linear(114, 16)),
                    ("2", nn.Linear(16, 8)),
                    ("4", nn.Linear(8, 4)),
                    ("6", nn.Linear(4, 14)),
                ]
            )
        )


def _raw_state_bytes(module: nn.Module) -> dict:
    return {
        key: value.detach().cpu().contiguous().reshape(-1).numpy().tobytes()
        for key, value in module.state_dict().items()
    }


def test_std_reset_sets_log_std_once_and_preserves_every_other_byte() -> None:
    policy = _KeyedModule()
    transferred = {key: value.clone() for key, value in policy.state_dict().items()}
    before = _raw_state_bytes(policy)
    result = reset_actor_exploration_std(torch, policy, transferred_state=transferred)

    assert result["applied"] is True and result["applied_once"] is True
    assert result["parameter_name"] == ACTOR_LOG_STD_STATE_KEY
    assert result["std_type"] == "log"
    assert result["target_std"] == FIXED_FRUIT_INITIAL_STD
    assert result["target_log_std"] == math.log(FIXED_FRUIT_INITIAL_STD)
    assert result["mean_weights_and_obs_normalizer_preserved"] is True
    assert result["preserved_parameters_byte_identical"] is True
    assert result["transferred_state_keys_checked"] == len(before)
    assert result["parameter_sha256_before"] != result["parameter_sha256_after"]
    assert getattr(policy, ACTOR_EXPLORATION_STD_RESET_MARKER) is True

    after = _raw_state_bytes(policy)
    assert set(after) == set(before)
    assert after[ACTOR_LOG_STD_STATE_KEY] != before[ACTOR_LOG_STD_STATE_KEY]
    for key in sorted(set(after) - {ACTOR_LOG_STD_STATE_KEY}):
        assert after[key] == before[key], key
    log_std = policy.distribution.log_std_param.detach().cpu()
    assert torch.all(log_std == math.log(FIXED_FRUIT_INITIAL_STD))
    assert set(result["preserved_parameter_keys"]) == set(after) - {ACTOR_LOG_STD_STATE_KEY}
    assert len(result["preserved_parameter_keys"]) == 12
    assert len(result["log_std_before"]) == len(result["log_std_after"]) == 14


def test_std_reset_applies_exactly_once() -> None:
    policy = _KeyedModule()
    reset_actor_exploration_std(torch, policy)
    digest = _raw_state_bytes(policy)
    with pytest.raises(ValueError):
        reset_actor_exploration_std(torch, policy)
    assert _raw_state_bytes(policy) == digest


def test_std_reset_refuses_ambiguous_or_non_log_structure() -> None:
    ambiguous = _KeyedModule()
    ambiguous.auxiliary = _KeyedModule()
    with pytest.raises(ValueError):
        reset_actor_exploration_std(torch, ambiguous)

    scalar = _KeyedModule(std_type="scalar")
    assert not hasattr(scalar.distribution, "log_std_param")
    with pytest.raises(ValueError):
        reset_actor_exploration_std(torch, scalar)


@pytest.mark.parametrize("target", [0.0, -0.08, float("nan"), float("inf")])
def test_std_reset_refuses_invalid_target(target) -> None:
    policy = _KeyedModule()
    with pytest.raises(ValueError):
        reset_actor_exploration_std(torch, policy, target_std=target)


def test_std_reset_refuses_when_transferred_bytes_already_differ() -> None:
    policy = _KeyedModule()
    transferred = {key: value.clone() for key, value in policy.state_dict().items()}
    with torch.no_grad():
        policy.mlp[0].weight.add_(1.0)
    with pytest.raises(ValueError):
        reset_actor_exploration_std(torch, policy, transferred_state=transferred)


def test_std_reset_refuses_a_silent_no_op_reset() -> None:
    policy = _KeyedModule(init_std=FIXED_FRUIT_INITIAL_STD)
    transferred = {key: value.clone() for key, value in policy.state_dict().items()}
    with pytest.raises(ValueError):
        reset_actor_exploration_std(torch, policy, transferred_state=transferred)


def test_std_reset_refuses_a_transferred_state_with_unknown_keys() -> None:
    policy = _KeyedModule()
    transferred = {key: value.clone() for key, value in policy.state_dict().items()}
    transferred["unknown.weight"] = torch.zeros(1)
    with pytest.raises(ValueError):
        reset_actor_exploration_std(torch, policy, transferred_state=transferred)


# ----------------------------------------------------- train.py 静态调用路径
def _train_tree() -> ast.Module:
    return ast.parse((PACKAGE_ROOT / "train.py").read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("train.py has no function " + name)


def _guarded_blocks(function: ast.AST, needle: str) -> list[str]:
    """返回测试源码包含 needle 的 If 语句体源码。"""

    blocks: list[str] = []
    for node in ast.walk(function):
        if isinstance(node, ast.If) and needle in ast.unparse(node.test):
            blocks.append("\n".join(ast.unparse(statement) for statement in node.body))
    return blocks


def test_train_reset_call_happens_once_only_in_the_actor_transfer_branch() -> None:
    tree = _train_tree()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "reset_actor_exploration_std"
    ]
    assert len(calls) == 1
    source = (PACKAGE_ROOT / "train.py").read_text(encoding="utf-8")
    assert source.count("reset_actor_exploration_std(") == 1
    target = calls[0]
    parent_stack: list[ast.AST] = []

    def walk(node: ast.AST, stack: tuple[ast.AST, ...]) -> None:
        nonlocal parent_stack
        if node is target:
            parent_stack = list(stack)
            return
        for child in ast.iter_child_nodes(node):
            walk(child, stack + (node,))

    walk(tree, ())
    guards = [ast.unparse(node.test) for node in parent_stack if isinstance(node, ast.If)]
    assert any("exploration_std_reset_required" in test for test in guards), guards
    assert any("actor_transfer" in test for test in guards), guards
    assert not any("args.resume" in test for test in guards), guards


def test_train_binds_the_actual_mapping_and_keeps_full_resume_untouched() -> None:
    source = (PACKAGE_ROOT / "train.py").read_text(encoding="utf-8")
    assert "actual_agent_config=agent_config" in source
    assert "actual_architecture=architecture" in source
    run_function = _function(_train_tree(), "_run")
    resume_blocks = _guarded_blocks(run_function, "args.resume is not None")
    assert len(resume_blocks) == 1
    assert "runner.load(" in resume_blocks[0]
    assert "reset_actor_exploration_std" not in resume_blocks[0]
    assert "exploration_std_reset_required" not in resume_blocks[0]
    transfer_blocks = _guarded_blocks(run_function, "args.actor_transfer is not None")
    assert len(transfer_blocks) == 1
    assert "reset_actor_exploration_std" in transfer_blocks[0]
    assert "validate_explicit_actor_transfer" in transfer_blocks[0]


def test_contract_version_and_conservative_constants_are_declared() -> None:
    assert C.TASK_VERSION == DESTINATION_VERSION
    assert CONSERVATIVE_PPO_SOURCE_LEARNING_RATE == 3.0e-4
    assert CONSERVATIVE_PPO_SOURCE_INITIAL_STD == 0.20
    assert FIXED_FRUIT_PPO_LEARNING_RATE == 1.0e-4
    assert FIXED_FRUIT_INITIAL_STD == 0.08
