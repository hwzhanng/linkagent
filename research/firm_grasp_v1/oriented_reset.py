"""S5 定向 reset：只在 reset 内整体旋转手、果实四元数与果梗锚点。

这里只计算初始化条件，不改变 S3 奖励/PPO、回合内根姿态、旧 v1.6 源码、
质量摩擦、PD、限位、七故障或 120 子步 G1 判据。reference 模式收益属于
初始化条件变化，不构成 PPO 学习收益。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from isaaclab.utils.math import quat_apply


MAX_WRIST_TILT_DEG = 45.0
MIN_FRUIT_BOTTOM_CLEARANCE_M = 0.005
GROUND_PLANE_Z_M = 0.0
DEFAULT_WRIST_TILT_AXIS_WORLD_XYZ = (
    0.3713906763541037,
    0.9284766908852594,
    0.0,
)


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是有限数值，得到 {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限数值，得到 {result!r}")
    return result


def _finite_vec3(name: str, value: Sequence[float]) -> tuple[float, float, float]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} 必须是有限 3 元组") from exc
    if len(items) != 3:
        raise ValueError(f"{name} 必须是有限 3 元组，得到 {items!r}")
    return tuple(_finite_float(name, item) for item in items)


def validate_wrist_tilt_deg(value: object) -> float:
    deg = _finite_float("wrist_tilt_deg", value)
    if abs(deg) > MAX_WRIST_TILT_DEG:
        raise ValueError(f"wrist_tilt_deg 必须位于 [-45, 45] 度，得到 {deg!r}")
    return deg


def validate_wrist_tilt_axis(
    value: Sequence[float],
) -> tuple[float, float, float]:
    axis = _finite_vec3("wrist_tilt_axis_world_xyz", value)
    norm = math.hypot(*axis)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("wrist_tilt_axis_world_xyz 不得为零向量")
    return tuple(item / norm for item in axis)


def validate_reset_fruit_reference(
    value: Sequence[float] | None,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    return _finite_vec3("reset_fruit_reference_hand_m", value)


@dataclass(frozen=True)
class OrientedResetCondition:
    """校验后的 S5 reset 条件；``enabled`` 即 reference 模式。"""

    wrist_tilt_deg: float
    wrist_tilt_axis_world_xyz: tuple[float, float, float]
    reset_fruit_reference_hand_m: tuple[float, float, float] | None
    tilt_quat_xyzw: tuple[float, float, float, float]

    @property
    def enabled(self) -> bool:
        return self.reset_fruit_reference_hand_m is not None

    def init_state_rot_xyzw(self) -> tuple[float, float, float, float]:
        """IsaacLab ``init_state.rot`` 的 XYZW 固定基座姿态。"""

        return self.tilt_quat_xyzw


def make_oriented_reset_condition(
    wrist_tilt_deg: object = 0.0,
    wrist_tilt_axis_world_xyz: Sequence[float] = DEFAULT_WRIST_TILT_AXIS_WORLD_XYZ,
    reset_fruit_reference_hand_m: Sequence[float] | None = None,
) -> OrientedResetCondition:
    """校验 S5 输入并返回固定的轴角旋转。fail-closed。"""

    deg = validate_wrist_tilt_deg(wrist_tilt_deg)
    axis = validate_wrist_tilt_axis(wrist_tilt_axis_world_xyz)
    reference = validate_reset_fruit_reference(reset_fruit_reference_hand_m)
    if deg != 0.0 and reference is None:
        raise ValueError("非零 wrist_tilt_deg 必须显式给出 reset_fruit_reference_hand_m")
    half = math.radians(deg) * 0.5
    sine = math.sin(half)
    quat = (axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(half))
    return OrientedResetCondition(deg, axis, reference, quat)


def oriented_reset_targets(
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
    reference_hand_m: Sequence[float],
    radius_m: torch.Tensor,
    stem_visual_height_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """由实际 base_link 世界位姿给出果实与果梗的完整刚体旋转目标。

    返回 ``(fruit_pos, fruit_quat, stem_rest_attachment, stem_plant_anchor)``。
    果实四元数等于归一化 base_link 四元数；果梗锚点沿同一旋转后的局部 +Z。
    """

    if not bool(torch.isfinite(base_pos).all()) or not bool(torch.isfinite(base_quat).all()):
        raise ValueError("base_link 世界位姿或四元数含非有限值，拒绝 reset")
    quat_norm = torch.linalg.vector_norm(base_quat, dim=1)
    if not bool(torch.isfinite(quat_norm).all()) or bool((quat_norm <= 1e-8).any()):
        raise ValueError("base_link 四元数范数为零或非有限，拒绝 reset")
    if not bool(torch.isfinite(radius_m).all()) or bool((radius_m <= 0.0).any()):
        raise ValueError("果实半径必须有限且为正")
    if not math.isfinite(float(stem_visual_height_m)):
        raise ValueError("stem_visual_height_m 必须有限")
    reference_checked = validate_reset_fruit_reference(reference_hand_m)
    if reference_checked is None:
        raise ValueError("oriented reset 需要有限 3 维 reference")
    base_quat = torch.nn.functional.normalize(base_quat, dim=1)
    count = base_pos.shape[0]
    reference = base_pos.new_tensor(reference_checked).reshape(1, 3).expand(count, -1)
    fruit_pos = base_pos + quat_apply(base_quat, reference)
    fruit_quat = base_quat
    local_up = torch.zeros((count, 3), device=base_pos.device, dtype=base_pos.dtype)
    local_up[:, 2] = 1.0
    up = quat_apply(fruit_quat, local_up)
    stem_rest = fruit_pos + up * radius_m.reshape(-1, 1)
    stem_plant = stem_rest + up * float(stem_visual_height_m)
    return fruit_pos, fruit_quat, stem_rest, stem_plant


def fruit_bottom_clearance(fruit_pose_w: torch.Tensor, fruit_axes_m: torch.Tensor) -> torch.Tensor:
    """果实最低点相对 z=0 地面的净空，单位米；只读。"""

    return fruit_pose_w[:, 2] - fruit_axes_m.amax(dim=1) - GROUND_PLANE_Z_M


def ground_free_mask(fruit_pose_w: torch.Tensor, fruit_axes_m: torch.Tensor) -> torch.Tensor:
    """``[N,1]`` 布尔：地面不参与承载；每个物理子步都必须为真。"""

    return (fruit_bottom_clearance(fruit_pose_w, fruit_axes_m) > MIN_FRUIT_BOTTOM_CLEARANCE_M).unsqueeze(1)
