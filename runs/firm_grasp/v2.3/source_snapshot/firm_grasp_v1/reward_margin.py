"""F3第三指PAD载荷余量奖励的纯Torch helper，可在CPU直接测试。"""

from __future__ import annotations

import torch


def third_load_margin_reward(
    pad_force_n: torch.Tensor,
    thumb_pad_qualified: torch.Tensor,
    unknown: torch.Tensor,
    nonpad_force_n: torch.Tensor,
    hard_fault: torch.Tensor,
    dropped: torch.Tensor,
) -> torch.Tensor:
    """按非拇指PAD载荷第二大值计算门控奖励，即拇指加两指组合中的第三指。

    奖励在0.15 N时为0，从0.15 N线性增加到0.30 N时的2分，并在更高载荷处封顶。
    函数不接收释放状态，因此释放前建立载荷余量也可获得奖励。
    """

    if not isinstance(pad_force_n, torch.Tensor) or not pad_force_n.is_floating_point():
        raise TypeError("pad_force_n must be a floating-point tensor")
    if pad_force_n.ndim != 2 or pad_force_n.shape[1] < 3:
        raise ValueError("pad_force_n must have shape [N, at least 3]")

    batch = pad_force_n.shape[0]
    device = pad_force_n.device

    def check_mask(name: str, value: torch.Tensor, *, rowwise: bool = True) -> None:
        if not isinstance(value, torch.Tensor) or value.dtype is not torch.bool:
            raise TypeError(f"{name} must be a bool tensor")
        if value.ndim < 1 or value.shape[0] != batch:
            raise ValueError(f"{name} must have batch dimension {batch}")
        if rowwise and value.numel() != batch:
            raise ValueError(f"{name} must have one value per batch row")
        if value.device != device:
            raise ValueError(f"{name} must be on {device}")

    check_mask("thumb_pad_qualified", thumb_pad_qualified)
    check_mask("unknown", unknown, rowwise=False)
    check_mask("hard_fault", hard_fault)
    check_mask("dropped", dropped)
    if not isinstance(nonpad_force_n, torch.Tensor) or not nonpad_force_n.is_floating_point():
        raise TypeError("nonpad_force_n must be a floating-point tensor")
    if nonpad_force_n.ndim < 1 or nonpad_force_n.shape[0] != batch:
        raise ValueError(f"nonpad_force_n must have batch dimension {batch}")
    if nonpad_force_n.numel() < batch or nonpad_force_n.device != device:
        raise ValueError("nonpad_force_n must contain same-device values for each row")

    third = pad_force_n[:, 1:].topk(2, dim=1).values[:, 1]
    margin = 2.0 * ((third - 0.15) / 0.15).clamp(0.0, 1.0)
    clear_unknown = ~unknown.reshape(batch, -1).any(dim=1)
    zero_nonpad = nonpad_force_n.reshape(batch, -1).eq(0).all(dim=1)
    valid = (
        thumb_pad_qualified
        & clear_unknown
        & zero_nonpad
        & ~hard_fault
        & ~dropped
        & torch.isfinite(pad_force_n).all(dim=1)
    )
    return torch.where(valid, margin, torch.zeros_like(margin))
