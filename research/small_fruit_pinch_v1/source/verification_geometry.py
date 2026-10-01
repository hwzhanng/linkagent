"""使用果实过滤后的平均接触点检查三指对向几何，不替代动力学承载验证。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch


MAX_OPPOSITION_DEG: Final = 35.0
MAX_CHORD_RADIUS_RATIO: Final = 0.45
MIN_SUPPORT_SPAN_RADIUS_RATIO: Final = 0.40


@dataclass(frozen=True, slots=True)
class ContactTopology:
    valid: torch.Tensor
    opposition_angle_deg: torch.Tensor
    chord_radius_ratio: torch.Tensor
    support_span_radius_ratio: torch.Tensor


def contact_topology(
    points_w: torch.Tensor, center_w: torch.Tensor, radius_m: torch.Tensor
) -> ContactTopology:
    """拒绝同侧、重合或非有限接触；输入顺序为拇指、食指、中指。"""

    relative = points_w - center_w[:, None, :]
    thumb = relative[:, 0]
    support = relative[:, 1:3].mean(dim=1)
    thumb_norm = torch.linalg.vector_norm(thumb, dim=1)
    support_norm = torch.linalg.vector_norm(support, dim=1)
    cosine = -(thumb * support).sum(dim=1) / (thumb_norm * support_norm).clamp_min(1.0e-12)
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
    line = support - thumb
    fraction = (-(thumb * line).sum(dim=1) / line.square().sum(dim=1).clamp_min(1.0e-12)).clamp(0.0, 1.0)
    chord = torch.linalg.vector_norm(thumb + fraction[:, None] * line, dim=1)
    span = torch.linalg.vector_norm(relative[:, 1] - relative[:, 2], dim=1)
    radius = radius_m.clamp_min(1.0e-12)
    chord_ratio = chord / radius
    span_ratio = span / radius
    finite = torch.isfinite(points_w).all(dim=(1, 2)) & torch.isfinite(center_w).all(dim=1)
    valid = (
        finite & torch.isfinite(radius_m) & (radius_m > 0.0)
        & (thumb_norm > 1.0e-6) & (support_norm > 1.0e-6)
        & (angle <= MAX_OPPOSITION_DEG)
        & (chord_ratio <= MAX_CHORD_RADIUS_RATIO)
        & (span_ratio >= MIN_SUPPORT_SPAN_RADIUS_RATIO)
    )
    return ContactTopology(valid, angle, chord_ratio, span_ratio)
