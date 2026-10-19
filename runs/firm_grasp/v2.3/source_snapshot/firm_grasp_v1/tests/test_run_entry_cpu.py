"""F3/F4 运行入口与 actor-only 初始化的 CPU 合同测试。"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from research.firm_grasp_v1 import actor_seed, run


class _CpuAppArgs:
    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--device", default="cpu")


def _model(obs_dim: int, output_dim: int, seed: int) -> MLPModel:
    torch.manual_seed(seed)
    observations = TensorDict(
        {"policy": torch.zeros((2, obs_dim), dtype=torch.float32)}, batch_size=[2]
    )
    return MLPModel(
        observations,
        {"actor": ["policy"]},
        "actor",
        output_dim=output_dim,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
            "init_std": 0.02,
            "std_type": "log",
        },
    )


def _identity(version: str, *, goal: str = "g2", disturbance: str = "quiet") -> dict:
    return {
        "contract_id": "firm_grasp_v1",
        "task_id": "FirmGrasp-v1",
        "task_version": version,
        "goal": {"goal": goal},
        "disturbance": {"mode": disturbance},
        "observations": {
            "policy_dim": 162,
            "critic_dim": 223,
            "policy_layout": "policy-layout-v1",
            "critic_layout": "critic-layout-v1",
            "action_history_dim": 14,
            "action_history_slot_names": [f"a{i}" for i in range(14)],
            "action_history_semantics_changed": [8, 9],
            "action_history_prefix_numeric_positions": {"actor": [100, 114]},
            "action_history_semantics_changed_observation_columns": {"actor": [108, 109]},
            "appended_clock_dim": 1,
            "appended_clock_position": {"actor": 161, "critic": 222},
            "appended_clock": {
                "source": "episode_length_buf",
                "formula": "clamp(step/release,0,3)",
                "support_release_step": 120,
                "clamp_max": 3.0,
                "semantics": "episode age",
            },
        },
        "action_mapping": {
            "mode": "independent",
            "mapping_id": "firm_grasp_index_middle_abduction_14d_v1",
            "action_dim": 14,
            "action_order": [f"a{i}" for i in range(14)],
            "action_semantics_changed_from_legacy": [8, 9],
            "exploration_normalized_sigma": {"8_9": 0.2},
        },
        "action_effect": {
            "mapping_id": "firm_grasp_index_middle_abduction_14d_v1",
            "action_semantics_changed": [8, 9],
            "legacy_ring_little_residuals_frozen_zero": True,
            "neutral_targets_rad": [0.0] * 16,
            "per_joint_span_rad": [0.1] * 16,
            "within_original_safe_limits": True,
            "within_usd_limits": True,
        },
        "actuation": {"effort_limit": 0.2, "stiffness": 20.0},
        "temporal_filter": {"implementation": "stable-filter-v1", "policy_dt_s": 0.05},
        "physics": {
            "policy_dt_s": 0.05,
            "sim_dt_s": 1 / 120,
            "decimation": 6,
            "support": {"release_step": 120},
        },
    }


def _save_seed(path: Path, source_actor: MLPModel, source_critic: MLPModel) -> None:
    identity = _identity(actor_seed.F2_TASK_VERSION)
    identity_hash = hashlib.sha256(
        json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    torch.save(
        {
            "actor_state_dict": source_actor.state_dict(),
            "critic_state_dict": source_critic.state_dict(),
            "infos": {
                "firm_grasp_v1": {
                    "task_identity": identity,
                    "identity_sha256": identity_hash,
                    "observation_dimensions": {
                        "actor_dim": 162,
                        "critic_dim": 223,
                        "action_dim": 14,
                    },
                }
            },
        },
        path,
    )


def _runner(actor: MLPModel, critic: MLPModel) -> SimpleNamespace:
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=2e-5)
    return SimpleNamespace(
        alg=SimpleNamespace(
            get_policy=lambda: actor,
            critic=critic,
            optimizer=optimizer,
        )
    )


def _mean(actor: MLPModel, observation: torch.Tensor) -> torch.Tensor:
    batch = TensorDict({"policy": observation}, batch_size=[observation.shape[0]])
    with torch.no_grad():
        return actor(batch).detach()


def _parse(tmp_path: Path, *extra: str):
    parser = run._make_parser(_CpuAppArgs)
    args = parser.parse_args(
        ["--mode", "train", "--output", str(tmp_path / "run"), "--device", "cpu", *extra]
    )
    run._validate_args(args)
    return args


def test_f3_defaults_and_cli_conditions_reach_make_config(tmp_path, monkeypatch):
    f2_seed = tmp_path / "f2.pt"
    f2_seed.touch()
    monkeypatch.setattr(run, "DEFAULT_F2_ACTOR_SEED", f2_seed)
    args = _parse(tmp_path)

    assert run.TASK_VERSION == "firm_grasp_v2.3.0"
    assert run.ROLLOUT_STEPS == 256
    assert args.actor_seed_checkpoint == f2_seed
    assert (args.goal, args.hold_seconds, args.disturbance_mode) == ("g2", 3.0, "quiet")
    assert args.exploration_std == pytest.approx(0.002)
    assert run._environment_config_kwargs(args)["goal"] == "g2"

    legacy_seed = tmp_path / "legacy-v1.6.pt"
    legacy_seed.touch()
    args = _parse(
        tmp_path,
        "--goal", "hold",
        "--hold-seconds", "2",
        "--disturbance-mode", "quiet",
        "--disturbance-axis", "2",
        "--disturbance-sign", "-1",
        "--force-ratio", "0.08",
        "--torque-ratio", "0.008",
        "--legacy-thumb-baseline",
        "--preparation-reference-offset-hand-mm", "1.2", "-2.4", "-6",
        "--actor-seed-checkpoint", str(legacy_seed),
    )
    cfg = run._environment_config_kwargs(args)
    assert cfg["goal"] == "hold"
    assert cfg["hold_seconds"] == 2.0
    assert cfg["disturbance_axis"] == 2
    assert cfg["disturbance_sign"] == -1.0
    assert cfg["force_ratio"] == 0.08
    assert cfg["torque_ratio"] == 0.008
    assert cfg["legacy_thumb_baseline"] is True
    assert cfg["preparation_reference_offset_hand_m"] == (.0012, -.0024, -.006)


def test_actor_only_seed_preserves_mean_normalizer_and_fresh_critic(tmp_path):
    source_actor = _model(162, 14, 11).eval()
    source_critic = _model(223, 1, 12).eval()
    path = tmp_path / "actor-seed.pt"
    _save_seed(path, source_actor, source_critic)
    source_state = {key: value.clone() for key, value in source_actor.state_dict().items()}

    target_actor = _model(162, 14, 21).eval()
    target_critic = _model(223, 1, 22).eval()
    runner = _runner(target_actor, target_critic)
    critic_before = {key: value.clone() for key, value in target_critic.state_dict().items()}
    optimizer_before = deepcopy(runner.alg.optimizer.state_dict())
    target_identity = _identity(actor_seed.F3_TASK_VERSION, goal="g2", disturbance="force")

    metadata = actor_seed.initialize_actor_seed(
        runner,
        path,
        target_identity=target_identity,
        target_dimensions={"actor_dim": 162, "critic_dim": 223, "action_dim": 14},
    )
    target_state = target_actor.state_dict()
    assert all(torch.equal(target_state[key], source_state[key]) for key in source_state)
    assert all(torch.equal(target_state[key], source_state[key]) for key in source_state if key.startswith("mlp."))
    assert all(torch.equal(target_state[key], source_state[key]) for key in source_state if key.startswith("obs_normalizer."))
    assert all(torch.equal(target_critic.state_dict()[key], value) for key, value in critic_before.items())
    assert runner.alg.optimizer.state_dict() == optimizer_before
    assert metadata["actor_state_copied_exactly"] is True
    assert metadata["critic_fresh_unchanged"] is True
    assert metadata["optimizer_fresh_unchanged"] is True
    assert metadata["strict_full_checkpoint_resume"] is False

    observation = torch.linspace(-0.5, 0.5, 3 * 162).reshape(3, 162)
    assert torch.equal(_mean(source_actor, observation), _mean(target_actor, observation))
    exploration = actor_seed.set_fresh_exploration_std(target_actor, 0.002)
    assert exploration["mean_unchanged"] is True
    assert exploration["normalizer_unchanged"] is True
    assert exploration["actual_std_by_action"] == pytest.approx([0.002] * 14, rel=1e-5)
    assert torch.equal(_mean(source_actor, observation), _mean(target_actor, observation))


def test_actor_only_seed_rejects_control_semantic_and_dimension_mismatch(tmp_path):
    source_actor = _model(162, 14, 31)
    source_critic = _model(223, 1, 32)
    path = tmp_path / "actor-seed.pt"
    _save_seed(path, source_actor, source_critic)

    wrong_control = _identity(actor_seed.F3_TASK_VERSION)
    wrong_control["action_mapping"]["action_order"][8] = "legacy_ring_residual"
    with pytest.raises(ValueError, match="控制语义"):
        actor_seed.initialize_actor_seed(
            _runner(_model(162, 14, 41), _model(223, 1, 42)),
            path,
            target_identity=wrong_control,
            target_dimensions={"actor_dim": 162, "critic_dim": 223, "action_dim": 14},
        )

    with pytest.raises(ValueError, match="目标策略维度"):
        actor_seed.initialize_actor_seed(
            _runner(_model(162, 14, 51), _model(223, 1, 52)),
            path,
            target_identity=_identity(actor_seed.F3_TASK_VERSION),
            target_dimensions={"actor_dim": 161, "critic_dim": 223, "action_dim": 14},
        )

