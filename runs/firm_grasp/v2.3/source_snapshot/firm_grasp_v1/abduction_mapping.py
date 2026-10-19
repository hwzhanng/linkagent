"""S7 14D 动作映射：动作 8/9 独立控制食指/中指近端 q0。

本模块只在 S7 的 ``firm_grasp_v1.7.0`` 中启用。旧的
``research/thumb_pad_grasp_v1/action_mapping.py`` 保持只读并继续承担
拇指、其余近端残差和四指远端残差映射。动作 8/9 的旧 ring/little
残差在调用旧映射前被冻结为零，然后仅把 canonical16 的 index/middle
q0 改写为围绕旧零动作参考角的单段线性 abduction。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from research.small_fruit_pinch_v1.source.legacy_bridge import load_legacy_hand_manifold
from research.thumb_pad_grasp_v1.action_mapping import independent_thumb_targets


MAPPING_ID = "firm_grasp_index_middle_abduction_14d_v1"
ACTION_DIM = 14
ACTION_ORDER = (
    "finger_flexion",
    "finger_curl_shape",
    "thumb_joint0",
    "thumb_joint1",
    "thumb_joint2",
    "thumb_joint3",
    "index_residual",
    "middle_residual",
    "index_abduction",
    "middle_abduction",
    "index_distal",
    "middle_distal",
    "ring_distal",
    "little_distal",
)
ABDUCTION_ACTION_INDICES = (8, 9)
ABDUCTION_CANONICAL_INDICES = (6, 7)
LEGACY_RING_LITTLE_RESIDUAL_INDICES = (8, 9)
HERE = Path(__file__).resolve().parent
NEW_HELPER_PATH = HERE / "abduction_mapping.py"
LEGACY_HELPER_PATH = HERE.parent / "thumb_pad_grasp_v1" / "action_mapping.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_action(action: torch.Tensor) -> None:
    if not isinstance(action, torch.Tensor):
        raise TypeError("动作映射输入必须是 torch.Tensor")
    if action.ndim < 1 or action.shape[-1] != ACTION_DIM:
        raise ValueError(f"动作最后一维必须为 {ACTION_DIM}，得到 {tuple(action.shape)}")
    if not bool(torch.isfinite(action).all()):
        raise ValueError("动作包含 NaN 或 Inf")


def index_middle_abduction_targets(action: torch.Tensor) -> torch.Tensor:
    """返回 canonical16 目标，仅让动作 8/9 改写 q0 canonical index 6/7。"""

    _validate_action(action)
    bounded = action.clamp(-1.0, 1.0)
    legacy = bounded.clone()
    legacy[..., list(LEGACY_RING_LITTLE_RESIDUAL_INDICES)] = 0.0
    targets = independent_thumb_targets(legacy)

    legacy_manifold = load_legacy_hand_manifold()
    low = action.new_tensor(legacy_manifold.LEFT_SAFE_LOWER_RAD)
    high = action.new_tensor(legacy_manifold.LEFT_SAFE_UPPER_RAD)
    for action_index, joint_index in zip(
        ABDUCTION_ACTION_INDICES, ABDUCTION_CANONICAL_INDICES, strict=True
    ):
        # 旧 mapper 对已冻结 ring/little 残差的动作输出就是当前 q0 reference。
        reference = targets[..., joint_index].clone()
        value = bounded[..., action_index]
        mapped = torch.where(
            value >= 0.0,
            reference + value * (high[joint_index] - reference),
            reference + value * (reference - low[joint_index]),
        )
        # 显式命中端点，避免浮点加减使 ±1 与安全边界相差一个 ulp。
        mapped = torch.where(value == 1.0, high[joint_index], mapped)
        mapped = torch.where(value == -1.0, low[joint_index], mapped)
        targets[..., joint_index] = mapped

    return torch.maximum(low, torch.minimum(high, targets))


def mapping_identity() -> dict[str, object]:
    """返回可序列化的映射来源、参考角和端点声明。"""

    legacy_manifold = load_legacy_hand_manifold()
    low = legacy_manifold.LEFT_SAFE_LOWER_RAD
    high = legacy_manifold.LEFT_SAFE_UPPER_RAD
    zero = torch.zeros((1, ACTION_DIM), dtype=torch.float32)
    zero_legacy = zero.clone()
    zero_legacy[:, list(LEGACY_RING_LITTLE_RESIDUAL_INDICES)] = 0.0
    zero_targets = independent_thumb_targets(zero_legacy)
    references = {
        "index_q0": float(zero_targets[0, ABDUCTION_CANONICAL_INDICES[0]]),
        "middle_q0": float(zero_targets[0, ABDUCTION_CANONICAL_INDICES[1]]),
    }
    return {
        "mode": "independent",
        "mapping_id": MAPPING_ID,
        "action_dim": ACTION_DIM,
        "action_order": list(ACTION_ORDER),
        "action_semantics_changed_from_legacy": [8, 9],
        "source_sha256": _sha256_file(NEW_HELPER_PATH),
        "legacy_helper_path": (
            "research/thumb_pad_grasp_v1/action_mapping.py"
        ),
        "legacy_helper_sha256": _sha256_file(LEGACY_HELPER_PATH),
        "legacy_helper_role": "read_only_thumb_and_remaining_residual_mapping",
        "abduction": {
            "action_indices": list(ABDUCTION_ACTION_INDICES),
            "canonical16_q0_indices": list(ABDUCTION_CANONICAL_INDICES),
            "q0_reference_source": "old_mapper_zero_action_with_ring_little_zeroed",
            "q0_reference_rad": references,
            "mapping": "piecewise_linear_around_reference_with_exact_safe_endpoints",
            "zero_matches_old_q0_reference": True,
            "positive_one_equals_safe_upper": True,
            "negative_one_equals_safe_lower": True,
        },
        "safe_endpoint_source": (
            "frozen_small_fruit_left_hand_manifold.LEFT_SAFE_LOWER_RAD/UPPER_RAD"
        ),
        "safe_lower_rad": list(low),
        "safe_upper_rad": list(high),
        "all_targets_clipped_to_original_safe_endpoints": True,
        "wider_usd_machine_limits_not_used_as_targets": True,
        "legacy_ring_little_residuals_frozen_to_zero_before_old_mapper": [8, 9],
        "ring_little_proximal_residual_rad": 0.0,
        "index_middle_distal_actions_preserved": [10, 11],
        "ring_little_distal_actions_preserved": [12, 13],
        "global_flexion_curl_and_remaining_mapping_unchanged": True,
        "command_warmup_preserved": True,
        "open_hand_baseline_preserved": True,
        "exploration_normalized_sigma": {
            "indices_8_9": 0.20,
            "other_indices": 0.02,
            "applies_to_fresh_actor_only": True,
            "checkpoint_or_resume_reset": False,
            "physical_targets_still_clipped_to_original_safe_endpoints": True,
            "all_actions_clamped": [-1.0, 1.0],
            "cfg_init_std_remains": 0.02,
            "runtime_distribution_differs_from_cfg_for_indices_8_9": True,
        },
        "vendor_parent_modified": False,
    }


__all__ = [
    "ABDUCTION_ACTION_INDICES",
    "ABDUCTION_CANONICAL_INDICES",
    "ACTION_ORDER",
    "LEGACY_RING_LITTLE_RESIDUAL_INDICES",
    "MAPPING_ID",
    "index_middle_abduction_targets",
    "mapping_identity",
]
