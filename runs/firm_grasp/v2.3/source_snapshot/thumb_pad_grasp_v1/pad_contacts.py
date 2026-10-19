"""将一个物理子步的接触点归约为逐环境、逐指的载荷与保守状态。"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch


PAD = 1
NON_PAD = 2
UNKNOWN = 3

NO_CONTACT = 0

_DIGIT_COUNT = 5
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _require_vector(name: str, value: torch.Tensor, *, dtype_kind: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} 必须是一维张量")
    if dtype_kind == "integer" and value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} 必须使用整数 dtype")
    if dtype_kind == "float" and not value.is_floating_point():
        raise TypeError(f"{name} 必须使用浮点 dtype")


def _finite_threshold(name: str, value: float, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} 必须显式传入有限实数")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 必须是有限值")
    if result < 0.0 or (result == 0.0 and not allow_zero):
        relation = "非负" if allow_zero else "正数"
        raise ValueError(f"{name} 必须是{relation}")
    return result


def summarize_contact_loads(
    *,
    labels: torch.Tensor,
    normal_forces_n: torch.Tensor,
    env_ids: torch.Tensor,
    digit_ids: torch.Tensor,
    num_envs: int,
    channel_valid: torch.Tensor,
    f_pad_min_n: float,
    f_nonpad_noise_max_n: float,
    f_other_contact_min_n: float,
) -> dict[str, torch.Tensor]:
    """归约单个实际物理子步的接触点，不包含跨子步或时序逻辑。

    ``digit_ids`` 顺序为拇指、食指、中指、无名指、小指（0 至 4）。点标签由
    上游提供；拇指确认指腹以外的所有身体应已标为 ``NON_PAD``。
    阈值必须由调用方显式提供，函数不推断或声称其已经校准。
    ``channel_valid`` 中的缺失通道和任意 ``UNKNOWN`` 点都会阻止该环境
    的 ``required_contact``。无点且通道完整时，每个通道状态为 ``NO_CONTACT``。
    """
    _require_vector("labels", labels, dtype_kind="integer")
    _require_vector("normal_forces_n", normal_forces_n, dtype_kind="float")
    _require_vector("env_ids", env_ids, dtype_kind="integer")
    _require_vector("digit_ids", digit_ids, dtype_kind="integer")

    if isinstance(num_envs, bool) or not isinstance(num_envs, Integral) or num_envs <= 0:
        raise ValueError("num_envs 必须是正整数")
    num_envs = int(num_envs)
    if not isinstance(channel_valid, torch.Tensor):
        raise TypeError("channel_valid 必须是 torch.Tensor")
    if channel_valid.dtype != torch.bool:
        raise TypeError("channel_valid 必须使用 torch.bool")
    if channel_valid.shape != (num_envs, _DIGIT_COUNT):
        raise ValueError("channel_valid 必须具有形状 (num_envs, 5)")

    tensors = (normal_forces_n, env_ids, digit_ids, channel_valid)
    if any(value.device != labels.device for value in tensors):
        raise ValueError("所有输入张量必须位于同一设备")
    count = labels.numel()
    if any(value.numel() != count for value in (normal_forces_n, env_ids, digit_ids)):
        raise ValueError("labels、normal_forces_n、env_ids 和 digit_ids 长度必须相同")

    f_pad_min = _finite_threshold("f_pad_min_n", f_pad_min_n, allow_zero=False)
    f_nonpad_noise_max = _finite_threshold(
        "f_nonpad_noise_max_n", f_nonpad_noise_max_n, allow_zero=True
    )
    f_other_contact_min = _finite_threshold(
        "f_other_contact_min_n", f_other_contact_min_n, allow_zero=False
    )

    if not torch.isfinite(normal_forces_n).all().item():
        raise ValueError("normal_forces_n 必须全部为有限值")
    if (normal_forces_n < 0).any().item():
        raise ValueError("normal_forces_n 不得为负值")
    known_label = (labels == PAD) | (labels == NON_PAD) | (labels == UNKNOWN)
    if not known_label.all().item():
        raise ValueError("labels 只能包含 PAD=1、NON_PAD=2 或 UNKNOWN=3")
    if ((env_ids < 0) | (env_ids >= num_envs)).any().item():
        raise ValueError("env_ids 中存在越界编号")
    if ((digit_ids < 0) | (digit_ids >= _DIGIT_COUNT)).any().item():
        raise ValueError("digit_ids 中存在越界编号")

    # 半精度输入提升到 float32，避免载荷求和和阈值比较过早舍入。
    load_dtype = torch.promote_types(normal_forces_n.dtype, torch.float32)
    flat_size = num_envs * _DIGIT_COUNT
    flat_channel_ids = env_ids.to(torch.int64) * _DIGIT_COUNT + digit_ids.to(torch.int64)
    loads = normal_forces_n.to(dtype=load_dtype)
    pad_force_flat = torch.zeros(flat_size, dtype=load_dtype, device=labels.device)
    nonpad_force_flat = torch.zeros_like(pad_force_flat)
    pad_mask = labels == PAD
    nonpad_mask = labels == NON_PAD
    unknown_mask = labels == UNKNOWN
    pad_force_flat.scatter_add_(0, flat_channel_ids[pad_mask], loads[pad_mask])
    nonpad_force_flat.scatter_add_(0, flat_channel_ids[nonpad_mask], loads[nonpad_mask])

    if not torch.isfinite(pad_force_flat).all().item() or not torch.isfinite(nonpad_force_flat).all().item():
        raise ValueError("聚合后的接触载荷超出可表示范围")

    pad_force_n = pad_force_flat.reshape(num_envs, _DIGIT_COUNT)
    nonpad_force_n = nonpad_force_flat.reshape(num_envs, _DIGIT_COUNT)
    pad_seen_flat = torch.zeros(flat_size, dtype=torch.bool, device=labels.device)
    nonpad_seen_flat = torch.zeros_like(pad_seen_flat)
    unknown_seen_flat = torch.zeros_like(pad_seen_flat)
    pad_seen_flat[flat_channel_ids[pad_mask]] = True
    nonpad_seen_flat[flat_channel_ids[nonpad_mask]] = True
    unknown_seen_flat[flat_channel_ids[unknown_mask]] = True

    pad_seen = pad_seen_flat.reshape(num_envs, _DIGIT_COUNT)
    nonpad_seen = nonpad_seen_flat.reshape(num_envs, _DIGIT_COUNT)
    unknown = unknown_seen_flat.reshape(num_envs, _DIGIT_COUNT) | ~channel_valid

    # 先设置已知标签；混合 PAD/NON_PAD 保留 NON_PAD，未知或缺失最后覆盖为 UNKNOWN。
    contact_state = torch.zeros(
        (num_envs, _DIGIT_COUNT), dtype=torch.int64, device=labels.device
    )
    contact_state[pad_seen] = PAD
    contact_state[nonpad_seen] = NON_PAD
    contact_state[unknown] = UNKNOWN

    thumb_pad_qualified = (
        channel_valid[:, 0]
        & ~unknown[:, 0]
        & (pad_force_n[:, 0] >= f_pad_min)
        & (nonpad_force_n[:, 0] <= f_nonpad_noise_max)
    )
    # 新硬规则对非拇指只要求真实接触，不能把同一指分散到两类的载荷漏掉。
    other_real_contact = (
        pad_force_n[:, 1:] + nonpad_force_n[:, 1:] >= f_other_contact_min
    ).any(dim=1)
    all_channels_known = channel_valid.all(dim=1) & ~unknown.any(dim=1)
    required_contact = thumb_pad_qualified & other_real_contact & all_channels_known

    return {
        "pad_force_n": pad_force_n,
        "nonpad_force_n": nonpad_force_n,
        "contact_state": contact_state,
        "unknown": unknown,
        "thumb_pad_qualified": thumb_pad_qualified,
        "required_contact": required_contact,
    }
