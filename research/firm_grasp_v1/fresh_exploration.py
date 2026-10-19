"""S7 fresh actor 的食指/中指 abduction 探索尺度初始化。

只在 fresh 初始化路径改写实际 ``distribution.log_std_param[8:10]``。
配置中的通用 ``init_std=0.02`` 保持不变；本 helper 明确记录配置值与
构造后实际高斯 sigma 的区别。
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any

import torch


ABDUCTION_STD_INDICES = (8, 9)
ABDUCTION_RUNTIME_STD = 0.20
DEFAULT_STD = 0.02
STD_STATE_KEY = "distribution.log_std_param"


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _tree(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_tree(child) for child in value)
    return copy.deepcopy(value)


def _tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
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


def configure_fresh_exploration_std(
    runner: Any,
    *,
    policy: str,
    checkpoint: Path | None,
    declared_init_std: float = DEFAULT_STD,
) -> dict[str, Any]:
    """在 fresh actor 初始化完成后设置实际 8/9 维探索 sigma。

    checkpoint 续训/评估返回 ``applied=False`` 且不触碰 actor。通用
    ``init_std`` 仍为 0.02；只有运行时 distribution 的 8/9 维被覆盖。
    """

    if policy != "fresh" or checkpoint is not None:
        return {
            "applied": False,
            "reason": "checkpoint_or_nonfresh_path_does_not_reset_distribution",
            "policy": policy,
            "checkpoint": None if checkpoint is None else str(Path(checkpoint)),
            "declared_cfg_init_std": float(declared_init_std),
        }
    if not math.isfinite(float(declared_init_std)) or float(declared_init_std) <= 0.0:
        raise ValueError("declared_init_std 必须是正的有限值")

    actor = runner.alg.get_policy()
    distribution = getattr(actor, "distribution", None)
    log_std_param = getattr(distribution, "log_std_param", None)
    if not isinstance(log_std_param, torch.nn.Parameter):
        raise TypeError("fresh actor.distribution 缺少 log_std_param Parameter")
    if tuple(log_std_param.shape) != (14,):
        raise ValueError(f"log_std_param 必须为 [14]，得到 {tuple(log_std_param.shape)}")

    actor_state_before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    critic = getattr(runner.alg, "critic", None)
    if critic is None or not callable(getattr(critic, "state_dict", None)):
        raise TypeError("runner.alg.critic 必须是可读取 state_dict 的模块")
    optimizer = getattr(runner.alg, "optimizer", None)
    if optimizer is None or not hasattr(optimizer, "state"):
        raise TypeError("runner.alg.optimizer 必须提供 state")
    critic_before = {name: value.detach().clone() for name, value in critic.state_dict().items()}
    optimizer_before = _tree(optimizer.state)
    sigma_before = log_std_param.detach().exp().cpu().tolist()
    actor_sha256_before = _state_sha256(actor_state_before)

    updated = log_std_param.detach().clone()
    updated[list(ABDUCTION_STD_INDICES)] = math.log(ABDUCTION_RUNTIME_STD)
    with torch.no_grad():
        log_std_param.copy_(updated)

    actor_state_after = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    changed_state_keys = [
        name for name in actor_state_before
        if not torch.equal(actor_state_before[name], actor_state_after[name])
    ]
    if changed_state_keys != [STD_STATE_KEY]:
        raise RuntimeError(f"探索 std 初始化修改了非唯一状态键：{changed_state_keys}")
    sigma_after = log_std_param.detach().exp().cpu().tolist()
    changed_sigma_indices = [
        index for index, (before, after) in enumerate(zip(sigma_before, sigma_after, strict=True))
        if before != after
    ]
    if changed_sigma_indices != list(ABDUCTION_STD_INDICES):
        raise RuntimeError(f"探索 std 初始化修改了非预期 sigma 索引：{changed_sigma_indices}")

    critic_after = {name: value.detach().clone() for name, value in critic.state_dict().items()}
    optimizer_after = _tree(optimizer.state)
    mean_mlp_unchanged = all(
        torch.equal(actor_state_before[name], actor_state_after[name])
        for name in actor_state_before
        if name.startswith("mlp.")
    )
    normalizer_unchanged = all(
        torch.equal(actor_state_before[name], actor_state_after[name])
        for name in actor_state_before
        if name.startswith("obs_normalizer.")
    )
    critic_unchanged = all(
        torch.equal(critic_before[name], critic_after[name]) for name in critic_before
    )
    optimizer_unchanged = _tree_equal(optimizer_before, optimizer_after)
    if not (mean_mlp_unchanged and normalizer_unchanged and critic_unchanged and optimizer_unchanged):
        raise RuntimeError("探索 std 初始化改变了 mean MLP/normalizer/critic/optimizer")

    return {
        "applied": True,
        "classification": "declared_exploration_scale_not_learned_success",
        "declared_cfg_init_std_remains": float(declared_init_std),
        "cfg_init_std_effective_for_indices_8_9": False,
        "normalized_sigma_before": sigma_before,
        "normalized_sigma_after": sigma_after,
        "normalized_sigma_overrides": {"8": 0.20, "9": 0.20},
        "other_sigma_indices_unchanged": [
            index for index in range(14) if index not in ABDUCTION_STD_INDICES
        ],
        "only_sigma_indices_changed": list(ABDUCTION_STD_INDICES),
        "changed_state_keys": changed_state_keys,
        "only_state_key_changed": changed_state_keys == [STD_STATE_KEY],
        "mean_mlp_unchanged": mean_mlp_unchanged,
        "observation_normalizer_unchanged": normalizer_unchanged,
        "critic_unchanged": critic_unchanged,
        "optimizer_unchanged": optimizer_unchanged,
        "actor_sha256_before_std": actor_sha256_before,
        "actor_sha256_after_std": _state_sha256(actor_state_after),
        "saved_distribution_semantics": "strict checkpoint state remains authoritative",
        "checkpoint_or_resume_std_reset": False,
        "action_clamp": [-1.0, 1.0],
        "physical_target_bounds": "original mapper LEFT_SAFE_LOWER_RAD/UPPER_RAD",
    }


__all__ = [
    "ABDUCTION_RUNTIME_STD",
    "ABDUCTION_STD_INDICES",
    "DEFAULT_STD",
    "STD_STATE_KEY",
    "configure_fresh_exploration_std",
]
