"""纯 Torch 的有界释放后外力/力矩脉冲计算。"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor


def _batch_parameter(
    name: str,
    value: float | int | Tensor,
    *,
    batch: int,
    reference: Tensor,
) -> Tensor:
    """将标量或逐环境参数变为与四元数同设备/类型的 [N] 张量。"""
    if isinstance(value, Tensor):
        if not value.is_floating_point():
            raise TypeError(f"{name} tensor 必须使用浮点 dtype")
        if value.ndim == 0 or value.shape == (1,):
            result = value.to(device=reference.device, dtype=reference.dtype).reshape(1).expand(batch)
        elif value.shape == (batch,):
            result = value.to(device=reference.device, dtype=reference.dtype)
        else:
            raise ValueError(f"{name} 必须是标量或形状为 [N] 的张量")
    else:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} 必须是实数或浮点 torch.Tensor")
        scalar = float(value)
        result = torch.full(
            (batch,), scalar, dtype=reference.dtype, device=reference.device
        )
    if not bool(torch.isfinite(result).all().item()) or bool((result <= 0).any().item()):
        raise ValueError(f"{name} 必须为正有限数")
    return result


def quat_apply_xyzw(quaternion_xyzw: Tensor, vector: Tensor) -> Tensor:
    """用 XYZW 四元数将向量从手坐标系旋转到 world 坐标系。"""
    if not isinstance(quaternion_xyzw, Tensor) or not isinstance(vector, Tensor):
        raise TypeError("quaternion_xyzw 和 vector 必须是 torch.Tensor")
    if quaternion_xyzw.ndim != 2 or quaternion_xyzw.shape[-1] != 4:
        raise ValueError("quaternion_xyzw 必须具有 [N, 4] 形状")
    if vector.shape != (quaternion_xyzw.shape[0], 3):
        raise ValueError("vector 必须具有 [N, 3] 形状")
    if not quaternion_xyzw.is_floating_point() or not vector.is_floating_point():
        raise TypeError("四元数和向量必须使用浮点 dtype")
    if quaternion_xyzw.device != vector.device:
        raise ValueError("四元数和向量必须位于同一设备")
    if not bool(torch.isfinite(quaternion_xyzw).all().item()) or not bool(torch.isfinite(vector).all().item()):
        raise ValueError("四元数和向量必须为有限值")

    q = quaternion_xyzw.to(dtype=vector.dtype)
    q_norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    if bool((q_norm <= 1e-12).any().item()):
        raise ValueError("四元数范数必须大于零")
    q = q / q_norm
    q_xyz = q[:, :3]
    q_w = q[:, 3:4]
    # 等价于 q * [v, 0] * conjugate(q)，避免构造临时四元数。
    t = 2.0 * torch.cross(q_xyz, vector, dim=-1)
    return vector + q_w * t + torch.cross(q_xyz, t, dim=-1)


def pulse_wrench(
    post_release_substep: Tensor,
    hand_quat_xyzw: Tensor,
    mass_kg: float | int | Tensor,
    radius_m: float | int | Tensor,
    mode: str = "quiet",
    axis: int = 0,
    sign: float = 1.0,
    force_ratio: float = 0.1,
    torque_ratio: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """生成 180–191 子步（恰好 12 步）的 world 系力和力矩。

    力幅值为 ``force_ratio * mass * 9.81``，力矩幅值为
    ``torque_ratio * mass * 9.81 * radius``。X/Y/Z 轴先在手坐标系
    构造，再由当前手部 XYZW 四元数旋转到 world 系。
    """
    if mode not in ("quiet", "force", "torque"):
        raise ValueError("mode 必须为 'quiet'、'force' 或 'torque'")
    if isinstance(axis, bool) or not isinstance(axis, int) or axis not in (0, 1, 2):
        raise ValueError("axis 必须是 0、1 或 2")
    if not math.isfinite(float(sign)) or float(sign) not in (-1.0, 1.0):
        raise ValueError("sign 必须为 -1.0 或 1.0")
    if not math.isfinite(float(force_ratio)) or force_ratio < 0:
        raise ValueError("force_ratio 必须为非负有限数")
    if not math.isfinite(float(torque_ratio)) or torque_ratio < 0:
        raise ValueError("torque_ratio 必须为非负有限数")
    if not isinstance(hand_quat_xyzw, Tensor) or not hand_quat_xyzw.is_floating_point():
        raise TypeError("hand_quat_xyzw 必须是浮点 torch.Tensor")
    if hand_quat_xyzw.ndim == 1 and hand_quat_xyzw.shape == (4,):
        hand_quat_xyzw = hand_quat_xyzw.unsqueeze(0)
    if hand_quat_xyzw.ndim != 2 or hand_quat_xyzw.shape[-1] != 4:
        raise ValueError("hand_quat_xyzw 必须具有 [N, 4] 形状")
    if not isinstance(post_release_substep, Tensor):
        post_release_substep = torch.as_tensor(
            post_release_substep, device=hand_quat_xyzw.device
        )
    elif post_release_substep.device != hand_quat_xyzw.device:
        raise ValueError("post_release_substep 和手部四元数必须位于同一设备")
    if post_release_substep.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    ):
        raise TypeError("post_release_substep 必须为整数")

    batch = hand_quat_xyzw.shape[0]
    if post_release_substep.ndim == 0:
        post_release_substep = post_release_substep.expand(batch)
    if post_release_substep.ndim != 1:
        raise ValueError("post_release_substep 必须为标量或 [N]")
    if post_release_substep.shape[0] == 1 and batch > 1:
        post_release_substep = post_release_substep.expand(batch)
    if post_release_substep.shape[0] != batch:
        raise ValueError("post_release_substep 与 hand_quat_xyzw 批次不匹配")

    dtype = hand_quat_xyzw.dtype
    device = hand_quat_xyzw.device
    mass = _batch_parameter("mass_kg", mass_kg, batch=batch, reference=hand_quat_xyzw)
    radius = _batch_parameter("radius_m", radius_m, batch=batch, reference=hand_quat_xyzw)
    zeros = torch.zeros((batch, 3), dtype=dtype, device=device)
    if mode == "quiet":
        return zeros, zeros.clone()

    axis_vector = torch.zeros((batch, 3), dtype=dtype, device=device)
    axis_vector[:, axis] = float(sign)
    active = (post_release_substep >= 180) & (post_release_substep < 192)
    if mode == "force":
        magnitude = float(force_ratio) * mass * 9.81
        hand_force = axis_vector * magnitude[:, None]
        world_force = quat_apply_xyzw(hand_quat_xyzw, hand_force)
        return world_force * active[:, None], zeros

    magnitude = float(torque_ratio) * mass * 9.81 * radius
    hand_torque = axis_vector * magnitude[:, None]
    world_torque = quat_apply_xyzw(hand_quat_xyzw, hand_torque)
    return zeros, world_torque * active[:, None]
