"""严格校验并 actor-only 导入相同控制语义的 firm-grasp checkpoint。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_F2_ACTOR_SEED = Path(
    "/home/miobarnacle/linkbot/linkagent/.omo/evidence/ppo-firm-grasp-f2-takeover-20261018/"
    "received/evidence/F2_train_handover01/update_000001.pt"
)
DEFAULT_F2_ACTOR_SEED_SHA256 = "a5997ed42977b4cd66bcba021510a95445818153b6de5b0fcb219fa8c98154b7"
DEFAULT_LEGACY_THUMB_SEED = Path(
    "/media/miobarnacle/Seagate Expansion Drive/linkbot-runs/baselines/"
    "ppo-thumb-pad-v1_6-frozen-20261017/thumb_pad_ppo_v1_6_update1.pt"
)
F2_TASK_VERSION = "firm_grasp_v1.8.0"
F3_TASK_VERSION = "firm_grasp_v2.2.0"
ACTOR_DIM = 162
CRITIC_DIM = 223
ACTION_DIM = 14
STD_STATE_KEY = "distribution.log_std_param"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_bytes(value: Any) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.reshape(-1).view(__import__("torch").uint8).numpy().tobytes()


def _state_sha256(state: Mapping[str, Any], *, prefix: str | None = None) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if prefix is not None and not name.startswith(prefix):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _tree_equal(left: Any, right: Any) -> bool:
    if hasattr(left, "detach") or hasattr(right, "detach"):
        return (
            hasattr(left, "detach")
            and hasattr(right, "detach")
            and left.shape == right.shape
            and left.dtype == right.dtype
            and bool((left.detach().cpu() == right.detach().cpu()).all())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} 必须是 mapping")
    return value


def _selected(value: Mapping[str, Any], keys: tuple[str, ...], label: str) -> dict[str, Any]:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{label} 缺少控制语义字段：{missing}")
    return {key: value[key] for key in keys}


def control_semantics(identity: Mapping[str, Any]) -> dict[str, Any]:
    """提取 actor 所依赖的观测布局、动作映射和低层控制语义。

    goal、hold 时长和 force/torque pulse 配置故意不在此投影内；它们可在
    actor-only 新身份间变化。控制器、动作维度和时钟含义必须逐字段相同。
    """

    observations = _require_mapping(identity.get("observations"), "observations")
    mapping = _require_mapping(identity.get("action_mapping"), "action_mapping")
    mapping = dict(mapping)
    mapping.pop("exploration_normalized_sigma", None)
    effect = _require_mapping(identity.get("action_effect"), "action_effect")
    effect_keys = (
        "mapping_id",
        "action_semantics_changed",
        "legacy_ring_little_residuals_frozen_zero",
        "neutral_targets_rad",
        "per_joint_span_rad",
        "within_original_safe_limits",
        "within_usd_limits",
    )
    temporal = _require_mapping(identity.get("temporal_filter"), "temporal_filter")
    physics = _require_mapping(identity.get("physics"), "physics")
    support = _require_mapping(physics.get("support"), "physics.support")
    clock = _require_mapping(observations.get("appended_clock"), "observations.appended_clock")

    observation_keys = (
        "policy_dim",
        "critic_dim",
        "policy_layout",
        "critic_layout",
        "action_history_dim",
        "action_history_slot_names",
        "action_history_semantics_changed",
        "action_history_prefix_numeric_positions",
        "action_history_semantics_changed_observation_columns",
        "appended_clock_dim",
        "appended_clock_position",
    )
    clock_keys = ("source", "formula", "support_release_step", "clamp_max", "semantics")
    physics_keys = ("policy_dt_s", "sim_dt_s", "decimation")
    return {
        "observations": _selected(observations, observation_keys, "observations"),
        "appended_clock": _selected(clock, clock_keys, "observations.appended_clock"),
        "action_mapping": mapping,
        "action_effect": _selected(effect, effect_keys, "action_effect"),
        "actuation": _require_mapping(identity.get("actuation"), "actuation"),
        "temporal_filter": temporal,
        "physics": _selected(physics, physics_keys, "physics"),
        "support_release_step": support.get("release_step"),
    }


def _state_dimensions(actor_state: Mapping[str, Any], critic_state: Mapping[str, Any]) -> dict[str, int]:
    try:
        actor_norm = actor_state["obs_normalizer._mean"]
        critic_norm = critic_state["obs_normalizer._mean"]
        first_weight = actor_state["mlp.0.weight"]
        last_bias = actor_state["mlp.6.bias"]
    except KeyError as exc:
        raise ValueError(f"checkpoint state 缺少维度键：{exc.args[0]}") from exc
    if len(actor_norm.shape) != 2 or len(critic_norm.shape) != 2:
        raise ValueError("checkpoint observation normalizer 形状错误")
    if first_weight.shape[1] != actor_norm.shape[1]:
        raise ValueError("actor 第一层输入维度与 observation normalizer 不一致")
    return {
        "actor_dim": int(actor_norm.shape[1]),
        "critic_dim": int(critic_norm.shape[1]),
        "action_dim": int(last_bias.shape[0]),
    }


def initialize_actor_seed(
    runner: Any,
    checkpoint_path: Path,
    *,
    target_identity: Mapping[str, Any],
    target_dimensions: Mapping[str, int],
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """只导入 actor 与 observation normalizer，保留新 critic/optimizer。"""

    import torch

    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    digest = _sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("默认 F2 actor seed SHA256 不匹配")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload = _require_mapping(payload, "checkpoint")
    infos = _require_mapping(payload.get("infos"), "checkpoint.infos")
    info = _require_mapping(infos.get("firm_grasp_v1"), "checkpoint.infos.firm_grasp_v1")
    source_identity = _require_mapping(info.get("task_identity"), "checkpoint.task_identity")
    source_version = source_identity.get("task_version")
    if source_identity.get("contract_id") != "firm_grasp_v1" or source_identity.get("task_id") != "FirmGrasp-v1":
        raise ValueError("actor seed 必须来自 firm_grasp_v1")
    if source_version not in {F2_TASK_VERSION, F3_TASK_VERSION,
                              "firm_grasp_v2.3.0",
                              "firm_grasp_v2.0.0", "firm_grasp_v2.1.0"}:
        raise ValueError(f"不支持的 actor seed 任务版本：{source_version!r}")
    identity_sha = _json_sha256(source_identity)
    if info.get("identity_sha256") != identity_sha:
        raise ValueError("actor seed task_identity SHA256 自校验失败")

    actor_state = _require_mapping(payload.get("actor_state_dict"), "actor_state_dict")
    critic_state = _require_mapping(payload.get("critic_state_dict"), "critic_state_dict")
    source_dimensions = _state_dimensions(actor_state, critic_state)
    if source_dimensions != {"actor_dim": ACTOR_DIM, "critic_dim": CRITIC_DIM, "action_dim": ACTION_DIM}:
        raise ValueError(f"actor seed state dimensions 不匹配：{source_dimensions}")
    saved_dimensions = _require_mapping(info.get("observation_dimensions"), "observation_dimensions")
    if {key: int(saved_dimensions.get(key, -1)) for key in source_dimensions} != source_dimensions:
        raise ValueError("actor seed 声明维度与实际模型 state 不一致")
    source_obs = _require_mapping(source_identity.get("observations"), "source observations")
    if int(source_obs.get("policy_dim", -1)) != ACTOR_DIM or int(source_obs.get("critic_dim", -1)) != CRITIC_DIM:
        raise ValueError("actor seed task_identity 维度不是 162/223")
    expected_target = {key: int(target_dimensions[key]) for key in source_dimensions}
    if source_dimensions != expected_target:
        raise ValueError(f"actor seed 与目标策略维度不同：source={source_dimensions}, target={expected_target}")

    source_control = control_semantics(source_identity)
    target_control = control_semantics(target_identity)
    if source_control != target_control:
        raise ValueError("actor seed 与目标任务的维度/控制语义不匹配")

    actor = runner.alg.get_policy()
    critic = runner.alg.critic
    optimizer = runner.alg.optimizer
    destination = actor.state_dict()
    if set(destination) != set(actor_state):
        missing = sorted(set(destination) - set(actor_state))
        unexpected = sorted(set(actor_state) - set(destination))
        raise ValueError(f"actor seed state_dict key 不匹配：missing={missing}, unexpected={unexpected}")
    for name, target_tensor in destination.items():
        source_tensor = actor_state[name]
        if not isinstance(source_tensor, torch.Tensor) or tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(f"actor seed state shape/type 不匹配：{name}")
    normalizer_keys = {name for name in destination if name.startswith("obs_normalizer.")}
    if not normalizer_keys or not {"obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std", "obs_normalizer.count"} <= normalizer_keys:
        raise ValueError("actor seed 未包含完整 matching observation normalizer")

    critic_before = copy.deepcopy(critic.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    actor.load_state_dict(actor_state, strict=True)
    copied = actor.state_dict()
    mismatches = [name for name, value in actor_state.items() if not torch.equal(copied[name].cpu(), value.cpu())]
    if mismatches:
        raise RuntimeError(f"actor/normalizer 未逐位复制：{mismatches}")
    if not _tree_equal(critic_before, critic.state_dict()):
        raise RuntimeError("actor-only seed 修改了 fresh critic")
    if not _tree_equal(optimizer_before, optimizer.state_dict()):
        raise RuntimeError("actor-only seed 修改了 fresh optimizer")

    return {
        "kind": "actor_only_seed_fresh_critic_optimizer_new_identity",
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": digest,
        "source_task_version": source_version,
        "source_task_identity_sha256": identity_sha,
        "source_observation_dimensions": source_dimensions,
        "target_observation_dimensions": expected_target,
        "control_semantics_sha256": _json_sha256(source_control),
        "actor_state_copied_exactly": True,
        "actor_mean_mlp_sha256": _state_sha256(copied, prefix="mlp."),
        "observation_normalizer_sha256": _state_sha256(copied, prefix="obs_normalizer."),
        "critic_fresh_unchanged": True,
        "optimizer_fresh_unchanged": True,
        "strict_full_checkpoint_resume": False,
    }


def set_fresh_exploration_std(actor: Any, std: float) -> dict[str, Any]:
    """设置 fresh policy 所有动作维度的实际 Gaussian std，不改 mean。"""

    import torch

    value = float(std)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("--exploration-std 必须是正的有限值")
    distribution = getattr(actor, "distribution", None)
    log_std = getattr(distribution, "log_std_param", None)
    if not isinstance(log_std, torch.nn.Parameter) or tuple(log_std.shape) != (ACTION_DIM,):
        raise TypeError("fresh actor 必须提供 [14] distribution.log_std_param Parameter")
    before = {name: tensor.detach().clone() for name, tensor in actor.state_dict().items()}
    current = log_std.detach().clone()
    with torch.no_grad():
        log_std.copy_(torch.full_like(log_std, math.log(value)))
    after = actor.state_dict()
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    if any(name != STD_STATE_KEY for name in changed):
        raise RuntimeError(f"探索 std 修改了 mean/normalizer 或其他 actor state：{changed}")
    if not torch.equal(current, log_std.detach()) and changed != [STD_STATE_KEY]:
        raise RuntimeError("探索 std 修改的 actor state 键不符合预期")
    mean_unchanged = all(
        torch.equal(before[name], after[name])
        for name in before
        if name.startswith("mlp.")
    )
    normalizer_unchanged = all(
        torch.equal(before[name], after[name])
        for name in before
        if name.startswith("obs_normalizer.")
    )
    if not mean_unchanged or not normalizer_unchanged:
        raise RuntimeError("设置探索 std 改变了 actor mean 或 observation normalizer")
    return {
        "applied": True,
        "scope": "fresh_policy_all_action_dimensions",
        "requested_std": value,
        "actual_std_by_action": log_std.detach().exp().cpu().tolist(),
        "mean_mlp_sha256": _state_sha256(after, prefix="mlp."),
        "observation_normalizer_sha256": _state_sha256(after, prefix="obs_normalizer."),
        "mean_unchanged": mean_unchanged,
        "normalizer_unchanged": normalizer_unchanged,
        "only_state_key_changed": STD_STATE_KEY if changed else None,
    }


__all__ = [
    "DEFAULT_F2_ACTOR_SEED",
    "DEFAULT_F2_ACTOR_SEED_SHA256",
    "DEFAULT_LEGACY_THUMB_SEED",
    "control_semantics",
    "initialize_actor_seed",
    "set_fresh_exploration_std",
]
