"""番茄等效直径分桶采样的纯张量工具。

本模块不导入 Isaac、USD 或 PhysX，便于在启动 SimulationApp 前验证分桶、
配额、质量和可复现性。正式场景的 prestartup event 只负责把这里生成的
半轴写入 USD。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import torch


DiameterBin = tuple[float, float]


def parse_diameter_bins_mm(raw: str) -> tuple[DiameterBin, ...]:
    """解析 ``52:54,54:56``，并拒绝重叠、乱序或非有限区间。"""

    if not raw.strip():
        raise ValueError("diameter-bins-mm 不能为空")
    bins: list[DiameterBin] = []
    for token in raw.split(","):
        fields = token.strip().split(":")
        if len(fields) != 2:
            raise ValueError("diameter-bins-mm 必须使用 low:high 逗号列表")
        try:
            low, high = (float(value.strip()) for value in fields)
        except ValueError as exc:
            raise ValueError("diameter-bins-mm 包含非数值端点") from exc
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("diameter-bins-mm 端点必须有限")
        if not 0.0 < low < high:
            raise ValueError("每个直径桶必须满足 0 < low < high")
        if bins and low < bins[-1][1]:
            raise ValueError("直径桶必须按升序给出且不能重叠")
        bins.append((low, high))
    return tuple(bins)


def parse_bin_weights(raw: str | None, count: int) -> tuple[float, ...]:
    """解析并归一化分桶权重；省略时使用等权。"""

    if count < 1:
        raise ValueError("直径桶数量必须为正数")
    if raw is None:
        return tuple(1.0 / count for _ in range(count))
    try:
        values = tuple(float(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError("diameter-bin-weights 必须是浮点数列表") from exc
    if len(values) != count:
        raise ValueError("diameter-bin-weights 数量必须与直径桶数量一致")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("diameter-bin-weights 必须是有限非负数")
    total = sum(values)
    if total <= 0.0:
        raise ValueError("diameter-bin-weights 至少有一个正值")
    return tuple(value / total for value in values)


def allocate_bin_counts(total: int, weights: Sequence[float]) -> tuple[int, ...]:
    """用最大余数法生成确定性整数配额，总数严格等于 ``total``。"""

    if total < 0:
        raise ValueError("total 不能为负数")
    normalized = parse_bin_weights(",".join(str(value) for value in weights), len(weights))
    exact = [total * value for value in normalized]
    counts = [math.floor(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(
        range(len(counts)),
        key=lambda index: (-(exact[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


def stable_geometry_seed(base_seed: int, cohort_id: str) -> int:
    """从训练 seed 与 cohort 名生成跨进程稳定的 CPU torch seed。"""

    payload = f"small-fruit-geometry-v1\0{int(base_seed)}\0{cohort_id}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**63 - 1)


def equivalent_diameter_m(axes_m: torch.Tensor) -> torch.Tensor:
    """计算 ``2*cbrt(a*b*c)``，输入最后一维必须为 3。"""

    if axes_m.ndim < 1 or axes_m.shape[-1] != 3:
        raise ValueError("axes_m 最后一维必须为 3")
    if not bool(torch.isfinite(axes_m).all()) or bool((axes_m <= 0.0).any()):
        raise ValueError("axes_m 必须是有限正数")
    return 2.0 * axes_m.prod(dim=-1).pow(1.0 / 3.0)


def sample_stratified_ellipsoids(
    count: int,
    *,
    diameter_bins_m: Sequence[DiameterBin],
    diameter_bin_weights: Sequence[float],
    shape_axis_ranges_m: Sequence[DiameterBin],
    density_range_kg_m3: DiameterBin,
    accepted_mass_kg: DiameterBin,
    maximum_axis_ratio: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """按精确整数配额采样等效直径、B0 形状比例、密度和质量。

    形状先从 ``shape_axis_ranges_m`` 独立采样，再除以自身几何平均值，
    最后乘目标等效半径。因此尺寸和轴比解耦，生成的 ``D_eq`` 会严格落入
    指定桶，而不是因三个半轴独立均匀采样而向区间中部聚集。
    """

    if count < 1:
        raise ValueError("count 必须为正数")
    bins = tuple((float(low), float(high)) for low, high in diameter_bins_m)
    if not bins:
        raise ValueError("diameter_bins_m 不能为空")
    for low, high in bins:
        if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
            raise ValueError("diameter_bins_m 必须是有限正区间")
    for previous, current in zip(bins, bins[1:]):
        if current[0] < previous[1]:
            raise ValueError("diameter_bins_m 必须按升序给出且不能重叠")
    if len(shape_axis_ranges_m) != 3:
        raise ValueError("shape_axis_ranges_m 必须包含 x/y/z 三个范围")
    shape_ranges = torch.tensor(shape_axis_ranges_m, dtype=torch.float64)
    if not bool(torch.isfinite(shape_ranges).all()):
        raise ValueError("shape_axis_ranges_m 必须有限")
    if bool((shape_ranges[:, 0] <= 0.0).any()) or bool(
        (shape_ranges[:, 0] > shape_ranges[:, 1]).any()
    ):
        raise ValueError("shape_axis_ranges_m 必须满足 0 < min <= max")
    density_low, density_high = (float(value) for value in density_range_kg_m3)
    mass_low, mass_high = (float(value) for value in accepted_mass_kg)
    if not 0.0 < density_low <= density_high:
        raise ValueError("density_range_kg_m3 无效")
    if not 0.0 < mass_low <= mass_high:
        raise ValueError("accepted_mass_kg 无效")
    if maximum_axis_ratio < 1.0 or not math.isfinite(maximum_axis_ratio):
        raise ValueError("maximum_axis_ratio 必须是有限值且不小于 1")

    weights = parse_bin_weights(",".join(str(value) for value in diameter_bin_weights), len(bins))
    quotas = allocate_bin_counts(count, weights)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    ordered_bins = torch.cat(
        [torch.full((quota,), index, dtype=torch.long) for index, quota in enumerate(quotas)]
    )
    permutation = torch.randperm(count, generator=generator)
    bin_index = ordered_bins[permutation]
    low = torch.tensor([bins[index][0] for index in bin_index.tolist()], dtype=torch.float64)
    high = torch.tensor([bins[index][1] for index in bin_index.tolist()], dtype=torch.float64)
    diameter = low + torch.rand(count, generator=generator, dtype=torch.float64) * (high - low)

    axes = torch.empty((count, 3), dtype=torch.float64)
    density = torch.empty(count, dtype=torch.float64)
    mass = torch.empty(count, dtype=torch.float64)
    accepted = torch.zeros(count, dtype=torch.bool)
    for _ in range(4096):
        pending = torch.nonzero(~accepted, as_tuple=False).flatten()
        if pending.numel() == 0:
            break
        raw_shape = shape_ranges[:, 0] + torch.rand(
            (pending.numel(), 3), generator=generator, dtype=torch.float64
        ) * (shape_ranges[:, 1] - shape_ranges[:, 0])
        geometric_mean = raw_shape.prod(dim=1).pow(1.0 / 3.0)
        normalized_shape = raw_shape / geometric_mean.unsqueeze(1)
        candidate_axes = normalized_shape * (diameter[pending] * 0.5).unsqueeze(1)
        candidate_density = density_low + torch.rand(
            pending.numel(), generator=generator, dtype=torch.float64
        ) * (density_high - density_low)
        candidate_mass = (
            4.0
            / 3.0
            * math.pi
            * candidate_axes.prod(dim=1)
            * candidate_density
        )
        ratio = candidate_axes.amax(dim=1) / candidate_axes.amin(dim=1)
        valid = (
            (candidate_mass >= mass_low)
            & (candidate_mass <= mass_high)
            & (ratio <= maximum_axis_ratio)
        )
        if bool(valid.any()):
            selected = pending[valid]
            axes[selected] = candidate_axes[valid]
            density[selected] = candidate_density[valid]
            mass[selected] = candidate_mass[valid]
            accepted[selected] = True
    if not bool(accepted.all()):
        missing = int((~accepted).sum().item())
        raise RuntimeError(f"4096 次后仍有 {missing} 个几何无法满足质量/轴比约束")

    sampled_diameter = equivalent_diameter_m(axes)
    tolerance = 2.0e-12
    sampled_low = torch.tensor(
        [bins[index][0] for index in bin_index.tolist()], dtype=torch.float64
    )
    sampled_high = torch.tensor(
        [bins[index][1] for index in bin_index.tolist()], dtype=torch.float64
    )
    if bool((sampled_diameter < sampled_low - tolerance).any()) or bool(
        (sampled_diameter >= sampled_high + tolerance).any()
    ):
        raise RuntimeError("等效直径越出指定桶")

    return {
        "axes_m": axes.to(torch.float32),
        "density_kg_m3": density.to(torch.float32),
        "mass_kg": mass.to(torch.float32),
        "equivalent_diameter_m": sampled_diameter.to(torch.float32),
        "bin_index": bin_index,
        "bin_quotas": torch.tensor(quotas, dtype=torch.long),
    }


__all__ = [
    "allocate_bin_counts",
    "equivalent_diameter_m",
    "parse_bin_weights",
    "parse_diameter_bins_mm",
    "sample_stratified_ellipsoids",
    "stable_geometry_seed",
]
