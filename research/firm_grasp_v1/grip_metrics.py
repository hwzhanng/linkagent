"""firm_grasp_v1 的逐步 PAD 抓握指标。"""
from __future__ import annotations

import torch
from torch import Tensor

from .contract import COMBINATIONS, DIGITS, FirmGraspContract


def _validate_force_tensor(name: str, value: Tensor, batch_size: int | None = None) -> None:
    """验证力张量的形状、设备、类型和数值范围。"""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    if value.ndim != 2 or value.shape[1] != len(DIGITS):
        raise ValueError(f"{name} 必须具有 [N, 5] 形状")
    if batch_size is not None and value.shape[0] != batch_size:
        raise ValueError(f"{name} 的批次长度必须为 {batch_size}")
    if not value.is_floating_point():
        raise TypeError(f"{name} 必须使用浮点 dtype")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} 含 NaN 或无穷值")
    if bool((value < 0).any().item()):
        raise ValueError(f"{name} 含负力")


def compute_contact_metrics(
    pad_force_n: Tensor,
    nonpad_force_n: Tensor,
    unknown: Tensor,
    *,
    contract: FirmGraspContract,
    extra_nonpad_force_n: Tensor | None = None,
) -> dict[str, Tensor]:
    """按样本判断六种“拇指 + 两个不同非拇指”的 PAD 承载组合。

    两种力输入的五列顺序均为 DIGITS。每个组合要求其三个独立手指各自
    达到 PAD 力和五指 PAD 总力份额阈值，且该组合内各指的非 PAD 力不
    超噪声上限。未知接触会拒绝整行；extra_nonpad_force_n 是未归属到
    五指列的额外非 PAD 力，超过噪声上限也会拒绝整行。

    返回值只描述每一行观测；连续 contract.required_substeps 步必须由
    上层状态 guard 对同一组合持续确认。本函数不累计时间，也不把同指
    多个接触点计作多个手指；输入应已按五个逻辑手指汇总 PAD 力。
    """
    if not isinstance(contract, FirmGraspContract):
        raise TypeError("contract 必须是 FirmGraspContract")

    _validate_force_tensor("pad_force_n", pad_force_n)
    _validate_force_tensor("nonpad_force_n", nonpad_force_n, pad_force_n.shape[0])
    if nonpad_force_n.device != pad_force_n.device:
        raise ValueError("pad_force_n 与 nonpad_force_n 必须位于同一设备")

    batch_size = pad_force_n.shape[0]
    if not isinstance(unknown, Tensor):
        raise TypeError("unknown 必须是 torch.Tensor")
    if unknown.dtype is not torch.bool:
        raise TypeError("unknown 必须使用 torch.bool dtype")
    if unknown.ndim != 1 or unknown.shape[0] != batch_size:
        raise ValueError("unknown 必须具有 [N] 形状")
    if unknown.device != pad_force_n.device:
        raise ValueError("unknown 必须与力张量位于同一设备")

    if extra_nonpad_force_n is not None:
        if not isinstance(extra_nonpad_force_n, Tensor):
            raise TypeError("extra_nonpad_force_n 必须是 torch.Tensor")
        if not extra_nonpad_force_n.is_floating_point():
            raise TypeError("extra_nonpad_force_n 必须使用浮点 dtype")
        if extra_nonpad_force_n.device != pad_force_n.device:
            raise ValueError("extra_nonpad_force_n 必须与力张量位于同一设备")
        if extra_nonpad_force_n.ndim == 1:
            if extra_nonpad_force_n.shape[0] != batch_size:
                raise ValueError("extra_nonpad_force_n 的形状必须是 [N] 或 [N, 5]")
        elif extra_nonpad_force_n.ndim == 2:
            if extra_nonpad_force_n.shape != (batch_size, len(DIGITS)):
                raise ValueError("extra_nonpad_force_n 的形状必须是 [N] 或 [N, 5]")
        else:
            raise ValueError("extra_nonpad_force_n 的形状必须是 [N] 或 [N, 5]")
        if not bool(torch.isfinite(extra_nonpad_force_n).all().item()):
            raise ValueError("extra_nonpad_force_n 含 NaN 或无穷值")
        if bool((extra_nonpad_force_n < 0).any().item()):
            raise ValueError("extra_nonpad_force_n 含负力")

    row_allowed = ~unknown
    if extra_nonpad_force_n is not None:
        extra = extra_nonpad_force_n.to(dtype=torch.float64)
        if extra.ndim == 2:
            extra = extra.sum(dim=1)
        row_allowed = row_allowed & (extra <= contract.f_nonpad_noise_max_n)

    # 用 float64 计算总力和份额，避免低精度输入求和溢出。
    pad = pad_force_n.to(dtype=torch.float64)
    nonpad = nonpad_force_n.to(dtype=torch.float64)
    total_pad = pad.sum(dim=1)
    safe_total_pad = torch.where(total_pad > 0, total_pad, torch.ones_like(total_pad))
    load_share = torch.where(
        total_pad[:, None] > 0,
        pad / safe_total_pad[:, None],
        torch.zeros_like(pad),
    )

    finger_qualified = (
        (pad >= contract.f_pad_min_n)
        & (load_share >= contract.min_load_share)
        & (nonpad <= contract.f_nonpad_noise_max_n)
        & row_allowed[:, None]
    )

    combination_indices = torch.tensor(
        COMBINATIONS, dtype=torch.long, device=pad_force_n.device
    )
    combo_pad = pad[:, combination_indices]
    combo_share = load_share[:, combination_indices]
    combo_nonpad = nonpad[:, combination_indices]
    min_combo_load_n = combo_pad.amin(dim=2)
    min_combo_share = combo_share.amin(dim=2)
    combination_qualified = (
        finger_qualified[:, combination_indices].all(dim=2)
        & (combo_nonpad <= contract.f_nonpad_noise_max_n).all(dim=2)
    )

    return {
        "combination_qualified": combination_qualified,
        "any_qualified": combination_qualified.any(dim=1),
        "qualified_finger_count": finger_qualified.sum(dim=1),
        "load_share": load_share,
        "min_combo_load_n": min_combo_load_n,
        "min_combo_share": min_combo_share,
    }

