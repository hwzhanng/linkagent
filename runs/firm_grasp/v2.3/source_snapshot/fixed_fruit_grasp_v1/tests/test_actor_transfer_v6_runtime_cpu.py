"""真实 RSL-RL 5.0.1 + 真实 v1.5.0 initial.pt 的纯 CPU 实物核验。

本模块不做任何仿真：只用 CPU 张量、真实 isaaclab_rl/rsl_rl 类构造目标 agent
（SmallFruitPinchPPORunnerCfg + handle_deprecated_rsl_rl_cfg +
configure_fixed_fruit_agent_cfg + PPO.construct_algorithm），并读入外部盘上
已核验 SHA 的真实 v1.5.0 ``initial.pt``。它证明：

* RSL 5.0.1 实际探索方差参数名与形状（``distribution.log_std_param``, 14）；
* 目标 agent 配置相对源 checkpoint 的逐叶子差异恰好是已声明的两个字段；
* 真实源身份只差 task_version 与 source-v6 路径前缀，显式迁移可接受而
  正常 resume/evaluate 仍被拒绝；
* 迁移后除唯一 log_std 外 actor 均值权重与 obs 归一化缓冲逐字节保留，
  critic/optimizer/iteration 不继承。

真实 checkpoint 缺失时显式 skip（绝不自动 PASS）；不启动 Kit/PhysX/GPU。
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path

import pytest
import torch

from research.fixed_fruit_grasp_v1 import contract as C
from research.fixed_fruit_grasp_v1 import train as train_module
from research.fixed_fruit_grasp_v1.run_utils import (
    ACTOR_LOG_STD_STATE_KEY,
    CONSERVATIVE_PPO_SOURCE_INITIAL_STD,
    CONSERVATIVE_PPO_SOURCE_LEARNING_RATE,
    FIXED_FRUIT_INITIAL_STD,
    FIXED_FRUIT_PPO_LEARNING_RATE,
    agent_config_sha256,
    comparable_agent_config,
    configure_fixed_fruit_agent_cfg,
    contract_sha256,
    json_safe,
    reset_actor_exploration_std,
    sha256_file,
    sha256_json,
    validate_checkpoint_compatibility,
    validate_explicit_actor_transfer,
    _leaf_paths,
)

TensorDict = pytest.importorskip("tensordict").TensorDict
PPO = pytest.importorskip("rsl_rl.algorithms.ppo").PPO
handle_deprecated_rsl_rl_cfg = pytest.importorskip("isaaclab_rl.rsl_rl").handle_deprecated_rsl_rl_cfg
SmallFruitPinchPPORunnerCfg = pytest.importorskip(
    "research.small_fruit_pinch_v1.source.agents"
).SmallFruitPinchPPORunnerCfg

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
SOURCE_VERSION = "fixed_fruit_grasp_v1.5.0"
DESTINATION_VERSION = "fixed_fruit_grasp_v1.6.0"
V5_CHECKPOINT_ENV = "FIXED_FRUIT_GRASP_V5_CHECKPOINT"
DEFAULT_V5_CHECKPOINT = Path(
    "/tmp/linkbot-ppo-grasp-learning-20261013/evidence/init_reward_transfer_01/initial.pt"
)
V5_CHECKPOINT_SHA256 = (
    "5d16495a1f16b4c63a157da35f74c424a249d83f2a4f7ff9990308d9e2064534"
)
V5_LINEAGE_UPDATES = 85
PHYSICAL_SOURCES = (
    "physical_hold.py",
    "physical_contacts.py",
    "collision_cooking.py",
)
CHANGED_AGENT_CONFIG_PATHS = [
    "actor.distribution_cfg.init_std",
    "algorithm.learning_rate",
]
CHANGED_AGENT_ARCHITECTURE_PATHS = ["actor.distribution_cfg.init_std"]


class _StaticEnv:
    """PPO.construct_algorithm 需要的最小环境契约（无仿真、无 Kit）。"""

    num_actions = 14
    num_envs = 2


def _checkpoint_path() -> Path:
    return Path(os.environ.get(V5_CHECKPOINT_ENV, str(DEFAULT_V5_CHECKPOINT)))


@pytest.fixture(scope="module")
def source_checkpoint() -> dict:
    path = _checkpoint_path()
    if not path.is_file():
        pytest.skip(
            "real v1.5.0 initial.pt is unavailable for CPU runtime verification: "
            + str(path)
            + " (set "
            + V5_CHECKPOINT_ENV
            + ")"
        )
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def source_metadata(source_checkpoint: dict) -> dict:
    return dict(source_checkpoint["infos"]["fixed_fruit_grasp_v1"])


@pytest.fixture(scope="module")
def destination() -> dict:
    """真实目标 agent：与 train.py 相同的配置路径与真实 RSL 构造。"""

    agent_cfg = SmallFruitPinchPPORunnerCfg()
    agent_cfg.logger = "tensorboard"
    agent_cfg.run_name = "fixed_fruit_grasp_v1"
    agent_cfg.device = "cuda:0"
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, importlib.metadata.version("rsl-rl-lib")
    )
    agent_cfg = configure_fixed_fruit_agent_cfg(agent_cfg)
    agent_config = json_safe(agent_cfg.to_dict())
    agent_config["multi_gpu"] = None
    observations = TensorDict(
        {
            "policy": torch.zeros(2, C.ACTOR_OBSERVATION_DIM),
            "critic": torch.zeros(2, C.CRITIC_STATE_DIM),
        },
        batch_size=[2],
    )
    algorithm = PPO.construct_algorithm(observations, _StaticEnv(), agent_config, "cpu")
    policy = algorithm.get_policy()
    architecture = train_module._architecture_config(agent_config, C.ACTION_DIM, policy)
    return {
        "agent_config": agent_config,
        "architecture": architecture,
        "algorithm": algorithm,
        "policy": policy,
    }


def _destination_identity(source_identity: dict) -> dict:
    destination = copy.deepcopy(source_identity)
    destination["task_version"] = DESTINATION_VERSION
    return json.loads(
        json.dumps(destination, ensure_ascii=False).replace("/source-v5/", "/source-v6/")
    )


def _destination_hashes(source_metadata: dict) -> dict:
    recorded = source_metadata["source_and_asset_hashes"]["training_semantics_source_files"]
    current = {
        relative: sha256_file(REPOSITORY_ROOT / relative) for relative in sorted(recorded)
    }
    referenced = {
        path.replace("/source-v5/", "/source-v6/"): digest
        for path, digest in source_metadata["source_and_asset_hashes"][
            "referenced_asset_files"
        ].items()
    }
    return {
        "training_semantics_source_files": current,
        "referenced_asset_files": referenced,
        "missing_referenced_paths": [],
    }


def test_real_v5_checkpoint_is_the_declared_85_update_lineage(
    source_checkpoint, source_metadata
) -> None:
    assert sha256_file(_checkpoint_path()) == V5_CHECKPOINT_SHA256
    assert source_metadata["task_identity"]["task_version"] == SOURCE_VERSION
    assert source_metadata["progress"]["total_updates_completed"] == V5_LINEAGE_UPDATES
    assert source_metadata["progress"]["stage_updates_completed"] == 0
    assert source_checkpoint["iter"] == 0
    assert set(source_checkpoint["actor_state_dict"]) == {
        ACTOR_LOG_STD_STATE_KEY,
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "obs_normalizer.count",
        "mlp.0.weight",
        "mlp.0.bias",
        "mlp.2.weight",
        "mlp.2.bias",
        "mlp.4.weight",
        "mlp.4.bias",
        "mlp.6.weight",
        "mlp.6.bias",
    }


def test_real_source_config_and_architecture_digests_are_self_consistent(
    source_metadata,
) -> None:
    assert sha256_json(source_metadata["task_identity"]) == source_metadata["task_identity_sha256"]
    assert contract_sha256(source_metadata["task_identity"]) == source_metadata["contract_sha256"]
    assert sha256_json(source_metadata["agent_config"]) == source_metadata["agent_config_sha256"]
    assert agent_config_sha256(source_metadata["agent_config"]) == source_metadata[
        "agent_config_comparison_sha256"
    ]
    assert sha256_json(source_metadata["agent_architecture"]) == source_metadata[
        "agent_architecture_sha256"
    ]
    assert source_metadata["agent_config"]["algorithm"]["entropy_coef"] == 0.003
    assert source_metadata["agent_config"]["algorithm"]["learning_rate"] == (
        CONSERVATIVE_PPO_SOURCE_LEARNING_RATE
    )
    assert source_metadata["agent_architecture"]["actor"]["distribution_cfg"]["init_std"] == (
        CONSERVATIVE_PPO_SOURCE_INITIAL_STD
    )


def test_real_rsl_actor_exposes_the_declared_log_std_parameter(destination) -> None:
    policy = destination["policy"]
    assert policy.distribution.std_type == "log"
    names = [name for name, _ in policy.named_parameters()]
    assert names.count(ACTOR_LOG_STD_STATE_KEY) == 1
    parameter = policy.distribution.log_std_param
    assert tuple(parameter.shape) == (C.ACTION_DIM,)
    assert parameter.dtype == torch.float32
    assert torch.all(parameter == math.log(FIXED_FRUIT_INITIAL_STD))
    # output_std 需要一次 forward 才会填充 _distribution，因此这里直接核验持有
    # log_std 的实参：exp(log_std_param) 必须逐元素等于声明的 0.08。
    assert torch.allclose(
        torch.exp(policy.distribution.log_std_param),
        torch.full((C.ACTION_DIM,), FIXED_FRUIT_INITIAL_STD),
    )


def test_real_destination_config_is_more_conservative_and_nothing_else(
    source_metadata, destination
) -> None:
    agent_config = destination["agent_config"]
    architecture = destination["architecture"]
    assert agent_config["algorithm"]["learning_rate"] == FIXED_FRUIT_PPO_LEARNING_RATE
    assert agent_config["algorithm"]["entropy_coef"] == 0.003
    assert agent_config["algorithm"]["schedule"] == "fixed"
    assert agent_config["algorithm"]["gamma"] == 0.995
    assert agent_config["algorithm"]["lam"] == 0.95
    assert agent_config["algorithm"]["desired_kl"] == 0.01
    assert agent_config["algorithm"]["max_grad_norm"] == 0.75
    assert agent_config["algorithm"]["clip_param"] == 0.2
    assert agent_config["algorithm"]["num_learning_epochs"] == 4
    assert agent_config["algorithm"]["num_mini_batches"] == 8
    assert agent_config["algorithm"]["value_loss_coef"] == 1.0
    assert agent_config["algorithm"]["use_clipped_value_loss"] is True
    assert agent_config["actor"]["distribution_cfg"]["init_std"] == FIXED_FRUIT_INITIAL_STD
    assert agent_config["actor"]["hidden_dims"] == [512, 256, 128]
    assert agent_config["critic"]["hidden_dims"] == [512, 256, 128]
    assert agent_config["actor"]["obs_normalization"] is True
    assert agent_config["critic"]["obs_normalization"] is True

    changed = sorted(
        ".".join(path)
        for path in _leaf_paths(
            comparable_agent_config(source_metadata["agent_config"]),
            comparable_agent_config(agent_config),
        )
    )
    assert changed == CHANGED_AGENT_CONFIG_PATHS
    assert sorted(
        ".".join(path)
        for path in _leaf_paths(source_metadata["agent_architecture"], architecture)
    ) == CHANGED_AGENT_ARCHITECTURE_PATHS


def test_real_destination_identity_only_changes_version_and_root(source_metadata) -> None:
    source_identity = source_metadata["task_identity"]
    identity = _destination_identity(source_identity)
    restored = json.loads(
        json.dumps(identity, ensure_ascii=False).replace("/source-v6/", "/source-v5/")
    )
    restored["task_version"] = SOURCE_VERSION
    assert restored == source_identity
    assert identity["task_version"] == DESTINATION_VERSION
    assert contract_sha256(identity) != source_metadata["contract_sha256"]
    assert sha256_json(identity) != source_metadata["task_identity_sha256"]


def test_real_destination_physical_sources_and_assets_are_byte_identical(
    source_metadata,
) -> None:
    recorded = source_metadata["source_and_asset_hashes"]["training_semantics_source_files"]
    for name in PHYSICAL_SOURCES:
        relative = "research/fixed_fruit_grasp_v1/" + name
        assert sha256_file(REPOSITORY_ROOT / relative) == recorded[relative], relative
    for recorded_path, digest in source_metadata["source_and_asset_hashes"][
        "referenced_asset_files"
    ].items():
        destination_path = Path(recorded_path.replace("/source-v5/", "/source-v6/"))
        assert destination_path.is_file(), destination_path
        assert sha256_file(destination_path) == digest, destination_path


def test_real_validator_accepts_the_actual_migration_and_resume_stays_rejected(
    source_metadata, destination
) -> None:
    identity = _destination_identity(source_metadata["task_identity"])
    hashes = _destination_hashes(source_metadata)
    result = validate_explicit_actor_transfer(
        source_metadata,
        identity,
        "contact",
        hashes,
        sha256_json(destination["architecture"]),
        agent_config_sha256(destination["agent_config"]),
        V5_CHECKPOINT_SHA256,
        V5_CHECKPOINT_SHA256,
        actual_agent_config=destination["agent_config"],
        actual_architecture=destination["architecture"],
    )
    assert result["source_total_updates_completed"] == V5_LINEAGE_UPDATES
    assert result["new_semantics_updates_before"] == 0
    assert result["exploration_std_reset_required"] is True
    assert result["physical_criterion_unchanged"] is True
    assert result["reward_profile_transfer_only"] is False
    assert result["is_new_semantics_success_evidence"] is False
    assert result["agent_config_validation"]["changed_agent_config_paths"] == (
        CHANGED_AGENT_CONFIG_PATHS
    )
    assert result["agent_config_validation"]["changed_agent_architecture_paths"] == (
        CHANGED_AGENT_ARCHITECTURE_PATHS
    )
    with pytest.raises(ValueError):
        validate_checkpoint_compatibility(source_metadata, identity, "contact")


def test_real_actor_transfer_preserves_bytes_and_resets_log_std_once(
    source_checkpoint, source_metadata, destination
) -> None:
    policy = destination["policy"]
    actor_state = source_checkpoint["actor_state_dict"]
    missing, unexpected = policy.load_state_dict(actor_state, strict=True)
    assert not missing and not unexpected
    before = {
        key: value.detach().cpu().contiguous().reshape(-1).numpy().tobytes()
        for key, value in policy.state_dict().items()
    }
    initial_std_bytes = (
        torch.full((C.ACTION_DIM,), math.log(FIXED_FRUIT_INITIAL_STD))
        .numpy()
        .tobytes()
    )
    assert before[ACTOR_LOG_STD_STATE_KEY] != initial_std_bytes

    result = reset_actor_exploration_std(torch, policy, transferred_state=actor_state)
    after = {
        key: value.detach().cpu().contiguous().reshape(-1).numpy().tobytes()
        for key, value in policy.state_dict().items()
    }
    assert result["transferred_state_keys_checked"] == 13
    assert len(result["preserved_parameter_keys"]) == 12
    assert result["mean_weights_and_obs_normalizer_preserved"] is True
    for key in sorted(set(after) - {ACTOR_LOG_STD_STATE_KEY}):
        assert after[key] == before[key], key
        assert after[key] == (
            actor_state[key].detach().cpu().contiguous().reshape(-1).numpy().tobytes()
        ), key
    assert after[ACTOR_LOG_STD_STATE_KEY] != before[ACTOR_LOG_STD_STATE_KEY]
    assert torch.all(policy.distribution.log_std_param == math.log(FIXED_FRUIT_INITIAL_STD))
    assert torch.allclose(
        torch.exp(policy.distribution.log_std_param),
        torch.full((C.ACTION_DIM,), FIXED_FRUIT_INITIAL_STD),
    )
    assert result["log_std_before"] == pytest.approx(
        [float(value) for value in actor_state[ACTOR_LOG_STD_STATE_KEY].tolist()]
    )
    assert float(torch.exp(actor_state[ACTOR_LOG_STD_STATE_KEY]).mean()) == pytest.approx(
        CONSERVATIVE_PPO_SOURCE_INITIAL_STD, abs=0.01
    )
    with pytest.raises(ValueError):
        reset_actor_exploration_std(torch, policy, transferred_state=actor_state)


def test_real_critic_optimizer_and_iteration_are_not_inherited(
    source_checkpoint, destination
) -> None:
    algorithm = destination["algorithm"]
    saved = algorithm.save()
    assert saved["optimizer_state_dict"]["state"] == {}
    assert set(saved["critic_state_dict"]) == set(source_checkpoint["critic_state_dict"])
    shared = [
        key
        for key in saved["critic_state_dict"]
        if torch.equal(
            saved["critic_state_dict"][key], source_checkpoint["critic_state_dict"][key]
        )
    ]
    # 源 v1.5.0 checkpoint 是 0 新更新的迁移初始化，其 critic 归一化缓冲仍是初始
    # 值；新 critic 也一样。因此只有这 4 个空缓冲可能相同，任何 critic 权重都
    # 不能被继承。
    normalizer_keys = {
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
        "obs_normalizer.count",
    }
    assert set(shared) <= normalizer_keys
    assert sorted(key for key in saved["critic_state_dict"] if key.startswith("mlp.")) == [
        "mlp.0.bias",
        "mlp.0.weight",
        "mlp.2.bias",
        "mlp.2.weight",
        "mlp.4.bias",
        "mlp.4.weight",
        "mlp.6.bias",
        "mlp.6.weight",
    ]
    for key in saved["critic_state_dict"]:
        if key.startswith("mlp."):
            assert not torch.equal(
                saved["critic_state_dict"][key], source_checkpoint["critic_state_dict"][key]
            ), key
    assert source_checkpoint["iter"] == 0


def test_real_destination_actor_key_layout_matches_the_source(
    source_checkpoint, destination
) -> None:
    source_actor = source_checkpoint["actor_state_dict"]
    destination_actor = destination["policy"].state_dict()
    assert set(destination_actor) == set(source_actor)
    for key in sorted(source_actor):
        assert tuple(destination_actor[key].shape) == tuple(source_actor[key].shape), key
