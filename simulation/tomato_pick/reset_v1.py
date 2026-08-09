"""Pure reset-distribution primitives for the independent Physics-v1 task."""

from __future__ import annotations

import math

import torch


def sample_joint_ellipsoid_offsets(
    count: int,
    radii_rad: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample uniformly inside a continuous joint-space ellipsoid."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be an integer >= 1")
    if radii_rad.ndim != 1 or radii_rad.numel() < 1:
        raise ValueError("radii_rad must be a non-empty one-dimensional tensor")
    if not bool(torch.isfinite(radii_rad).all()) or bool((radii_rad <= 0.0).any()):
        raise ValueError("joint ellipsoid radii must be finite and positive")
    direction = torch.randn(
        (count, radii_rad.numel()),
        generator=generator,
        device=radii_rad.device,
        dtype=radii_rad.dtype,
    )
    direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(
        1.0e-8
    )
    # r**dimension is uniform in ellipsoid volume.
    radius = torch.rand(
        (count, 1),
        generator=generator,
        device=radii_rad.device,
        dtype=radii_rad.dtype,
    ).pow(1.0 / float(radii_rad.numel()))
    return direction * radius * radii_rad.unsqueeze(0)


def sample_bounded_joint_ellipsoid_offsets(
    count: int,
    radii_rad: torch.Tensor,
    *,
    lower_offset_rad: torch.Tensor,
    upper_offset_rad: torch.Tensor,
    generator: torch.Generator | None = None,
    maximum_attempts: int = 64,
) -> torch.Tensor:
    """Sample an ellipsoid without clipping probability mass onto joint limits.

    The lower and upper tensors may contain one bound per joint or one bound
    per environment and joint.  Invalid samples are independently redrawn;
    unlike ``clamp``, this keeps the reset distribution continuous and avoids
    artificial piles of poses exactly at a joint-limit margin.
    """

    if isinstance(maximum_attempts, bool) or not isinstance(maximum_attempts, int):
        raise ValueError("maximum_attempts must be an integer")
    if maximum_attempts < 1:
        raise ValueError("maximum_attempts must be >= 1")
    if lower_offset_rad.device != radii_rad.device or upper_offset_rad.device != radii_rad.device:
        raise ValueError("joint radii and bounds must use the same device")
    if lower_offset_rad.dtype != radii_rad.dtype or upper_offset_rad.dtype != radii_rad.dtype:
        raise ValueError("joint radii and bounds must use the same dtype")
    target_shape = (count, radii_rad.numel())
    try:
        lower = torch.broadcast_to(lower_offset_rad, target_shape)
        upper = torch.broadcast_to(upper_offset_rad, target_shape)
    except RuntimeError as error:
        raise ValueError("joint offset bounds must broadcast to (count, joints)") from error
    if not bool(torch.isfinite(lower).all()) or not bool(torch.isfinite(upper).all()):
        raise ValueError("joint offset bounds must be finite")
    if bool((lower >= upper).any()):
        raise ValueError("every lower joint offset bound must be below its upper bound")

    offsets = torch.empty(target_shape, device=radii_rad.device, dtype=radii_rad.dtype)
    remaining = torch.arange(count, device=radii_rad.device)
    for _ in range(maximum_attempts):
        if remaining.numel() == 0:
            break
        candidate = sample_joint_ellipsoid_offsets(
            int(remaining.numel()), radii_rad, generator=generator
        )
        valid = (
            (candidate >= lower[remaining]) & (candidate <= upper[remaining])
        ).all(dim=1)
        if bool(valid.any()):
            offsets[remaining[valid]] = candidate[valid]
        remaining = remaining[~valid]
    if remaining.numel() != 0:
        raise RuntimeError(
            "could not sample the requested joint ellipsoid inside the joint limits"
        )
    return offsets


def sample_bounded_joint_workspace_offsets(
    count: int,
    radii_rad: torch.Tensor,
    *,
    correlated_modes_rad: torch.Tensor,
    mode_coefficient_ranges: torch.Tensor | None = None,
    lower_offset_rad: torch.Tensor,
    upper_offset_rad: torch.Tensor,
    generator: torch.Generator | None = None,
    maximum_attempts: int = 64,
) -> dict[str, torch.Tensor]:
    """Add continuous correlated workspace modes to a joint ellipsoid.

    Each mode receives one coefficient sampled uniformly from ``[-1, 1]``.
    The full joint offset is accepted only when it stays inside the requested
    joint-limit margin.  This supports a task-space-directed reach mode without
    widening unrelated Cartesian axes or clipping any sample to a limit.
    """

    if correlated_modes_rad.ndim != 2:
        raise ValueError("correlated_modes_rad must have shape (modes, joints)")
    if correlated_modes_rad.shape[1] != radii_rad.numel():
        raise ValueError("every correlated mode must cover every joint")
    if correlated_modes_rad.device != radii_rad.device:
        raise ValueError("joint radii and correlated modes must use the same device")
    if correlated_modes_rad.dtype != radii_rad.dtype:
        raise ValueError("joint radii and correlated modes must use the same dtype")
    if not bool(torch.isfinite(correlated_modes_rad).all()):
        raise ValueError("correlated joint modes must be finite")
    if correlated_modes_rad.shape[0] < 1:
        raise ValueError("at least one correlated joint mode is required")
    if maximum_attempts < 1:
        raise ValueError("maximum_attempts must be >= 1")

    target_shape = (count, radii_rad.numel())
    try:
        lower = torch.broadcast_to(lower_offset_rad, target_shape)
        upper = torch.broadcast_to(upper_offset_rad, target_shape)
    except RuntimeError as error:
        raise ValueError("joint offset bounds must broadcast to (count, joints)") from error
    if not bool(torch.isfinite(lower).all()) or not bool(torch.isfinite(upper).all()):
        raise ValueError("joint offset bounds must be finite")
    if bool((lower >= upper).any()):
        raise ValueError("every lower joint offset bound must be below its upper bound")

    mode_count = correlated_modes_rad.shape[0]
    if mode_coefficient_ranges is None:
        mode_coefficient_ranges = torch.tensor(
            [[-1.0, 1.0]], device=radii_rad.device, dtype=radii_rad.dtype
        ).expand(mode_count, -1)
    if mode_coefficient_ranges.shape != (mode_count, 2):
        raise ValueError("mode coefficient ranges must have shape (modes, 2)")
    if mode_coefficient_ranges.device != radii_rad.device:
        raise ValueError("mode coefficient ranges must use the joint-radii device")
    if mode_coefficient_ranges.dtype != radii_rad.dtype:
        raise ValueError("mode coefficient ranges must use the joint-radii dtype")
    if not bool(torch.isfinite(mode_coefficient_ranges).all()):
        raise ValueError("mode coefficient ranges must be finite")
    if bool(
        (mode_coefficient_ranges[:, 0] >= mode_coefficient_ranges[:, 1]).any()
    ):
        raise ValueError("every mode coefficient range must be increasing")
    offsets = torch.empty(target_shape, device=radii_rad.device, dtype=radii_rad.dtype)
    base_offsets = torch.empty_like(offsets)
    coefficients = torch.empty(
        (count, mode_count), device=radii_rad.device, dtype=radii_rad.dtype
    )
    remaining = torch.arange(count, device=radii_rad.device)
    for _ in range(maximum_attempts):
        if remaining.numel() == 0:
            break
        candidate_base = sample_joint_ellipsoid_offsets(
            int(remaining.numel()), radii_rad, generator=generator
        )
        coefficient_unit = torch.rand(
            (int(remaining.numel()), mode_count),
            generator=generator,
            device=radii_rad.device,
            dtype=radii_rad.dtype,
        )
        candidate_coefficient = mode_coefficient_ranges[:, 0] + coefficient_unit * (
            mode_coefficient_ranges[:, 1] - mode_coefficient_ranges[:, 0]
        )
        candidate = candidate_base + candidate_coefficient @ correlated_modes_rad
        valid = (
            (candidate >= lower[remaining]) & (candidate <= upper[remaining])
        ).all(dim=1)
        if bool(valid.any()):
            selected = remaining[valid]
            offsets[selected] = candidate[valid]
            base_offsets[selected] = candidate_base[valid]
            coefficients[selected] = candidate_coefficient[valid]
        remaining = remaining[~valid]
    if remaining.numel() != 0:
        raise RuntimeError(
            "could not sample the correlated workspace inside the joint limits"
        )
    return {
        "offset_rad": offsets,
        "base_ellipsoid_offset_rad": base_offsets,
        "mode_coefficients": coefficients,
    }


def sample_pregrasp_fruit_offsets(
    count: int,
    *,
    closed_grasp_center_m: torch.Tensor,
    distance_range_m: tuple[float, float],
    yaw_range_rad: tuple[float, float],
    pitch_range_rad: tuple[float, float],
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Sample fruit centres in the palm's grasp frame.

    Local axes are approach, lateral, and pull/stem respectively.  The sampled
    direction is a bounded cone around approach, so every reset remains an
    attainable local pre-grasp rather than an arbitrary point around the arm.
    """

    if count < 1 or closed_grasp_center_m.shape != (3,):
        raise ValueError("count must be positive and closed_grasp_center_m must be 3D")
    distance_lo, distance_hi = distance_range_m
    yaw_lo, yaw_hi = yaw_range_rad
    pitch_lo, pitch_hi = pitch_range_rad
    if not 0.0 < distance_lo < distance_hi:
        raise ValueError("distance_range_m must be positive and increasing")
    if not -0.5 * math.pi < yaw_lo < yaw_hi < 0.5 * math.pi:
        raise ValueError("yaw cone must lie within (-pi/2, pi/2)")
    if not -0.5 * math.pi < pitch_lo < pitch_hi < 0.5 * math.pi:
        raise ValueError("pitch cone must lie within (-pi/2, pi/2)")
    kwargs = {
        "generator": generator,
        "device": closed_grasp_center_m.device,
        "dtype": closed_grasp_center_m.dtype,
    }
    unit = torch.rand((count, 3), **kwargs)
    distance = distance_lo + unit[:, 0] * (distance_hi - distance_lo)
    yaw = yaw_lo + unit[:, 1] * (yaw_hi - yaw_lo)
    pitch = pitch_lo + unit[:, 2] * (pitch_hi - pitch_lo)
    direction = torch.stack(
        (torch.ones_like(yaw), torch.tan(yaw), torch.tan(pitch)), dim=1
    )
    direction = torch.nn.functional.normalize(direction, dim=1)
    local_position = closed_grasp_center_m.unsqueeze(0) + direction * distance.unsqueeze(1)
    return {
        "local_position_m": local_position,
        "direction_local": direction,
        "distance_m": distance,
        "yaw_rad": yaw,
        "pitch_rad": pitch,
    }


def _quat_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(dim=1)
    rw, rx, ry, rz = right.unbind(dim=1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=1,
    )


def sample_hanging_fruit_quaternions(
    count: int,
    *,
    maximum_tilt_rad: float,
    yaw_range_rad: tuple[float, float],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Sample full yaw and a bounded stem tilt, returning wxyz quaternions."""

    if count < 1 or not 0.0 <= maximum_tilt_rad < 0.5 * math.pi:
        raise ValueError("count must be positive and tilt must lie in [0, pi/2)")
    yaw_lo, yaw_hi = yaw_range_rad
    if not -math.pi <= yaw_lo < yaw_hi <= math.pi:
        raise ValueError("fruit yaw range must lie in [-pi, pi]")
    unit = torch.rand(
        (count, 3), generator=generator, device=device, dtype=dtype
    )
    # sqrt produces a uniform disk of small tilt vectors instead of crowding
    # samples at the upright centre.
    tilt = maximum_tilt_rad * torch.sqrt(unit[:, 0])
    tilt_azimuth = 2.0 * math.pi * unit[:, 1]
    yaw = yaw_lo + unit[:, 2] * (yaw_hi - yaw_lo)
    half_tilt = 0.5 * tilt
    tilt_quaternion = torch.stack(
        (
            torch.cos(half_tilt),
            torch.sin(half_tilt) * torch.cos(tilt_azimuth),
            torch.sin(half_tilt) * torch.sin(tilt_azimuth),
            torch.zeros_like(tilt),
        ),
        dim=1,
    )
    half_yaw = 0.5 * yaw
    yaw_quaternion = torch.stack(
        (
            torch.cos(half_yaw),
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
            torch.sin(half_yaw),
        ),
        dim=1,
    )
    quaternion = _quat_multiply_wxyz(tilt_quaternion, yaw_quaternion)
    quaternion = torch.nn.functional.normalize(quaternion, dim=1)
    return {"quaternion_wxyz": quaternion, "tilt_rad": tilt, "yaw_rad": yaw}


__all__ = [
    "sample_bounded_joint_ellipsoid_offsets",
    "sample_bounded_joint_workspace_offsets",
    "sample_hanging_fruit_quaternions",
    "sample_joint_ellipsoid_offsets",
    "sample_pregrasp_fruit_offsets",
]
