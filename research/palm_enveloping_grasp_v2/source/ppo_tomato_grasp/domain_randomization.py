"""Correlated CPU-side priors for the hand-only tomato task.

The fruit remains a rigid or linearly compliant ellipsoid in this phase.  The
numbers are broad numerical priors, not cultivar measurements.  A curriculum
fraction contracts every randomized physical quantity to its nominal value at
zero and expands it continuously to the declared envelope at one.
"""

from __future__ import annotations

import math

import torch


def _rand(
    shape: tuple[int, ...] | list[int],
    *,
    generator: torch.Generator | None,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample reproducibly on CPU when a CPU generator is supplied, then move."""

    source_device = "cpu" if generator is not None else device
    return torch.rand(shape, generator=generator, dtype=dtype, device=source_device).to(device)


def _randint(
    low: int,
    high: int,
    shape: tuple[int, ...] | list[int],
    *,
    generator: torch.Generator | None,
    device: torch.device | str,
) -> torch.Tensor:
    source_device = "cpu" if generator is not None else device
    return torch.randint(low, high, shape, generator=generator, device=source_device).to(device)


RANDOMIZATION_VERSION = "correlated_ellipsoid_v2_persistent_tactile_faults"

ROBUSTNESS_BUCKET_ORDER = (
    "small",
    "large",
    "ripe_soft",
    "firm",
    "wet_low_friction",
    "dry_high_friction",
    "sensor_degraded",
    "thumb_sensor_failed",
    "multiple_sensors_failed",
    "actuator_sluggish",
)


def _validate_count_fraction(count: int, curriculum_fraction: float) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be an integer >= 1")
    if not 0.0 <= curriculum_fraction <= 1.0:
        raise ValueError("curriculum_fraction must lie in [0, 1]")


def _expanded_uniform(
    latent: torch.Tensor,
    low: float,
    nominal: float,
    high: float,
    fraction: float,
) -> torch.Tensor:
    if not low <= nominal <= high:
        raise ValueError("nominal must lie inside [low, high]")
    effective_low = nominal + fraction * (low - nominal)
    effective_high = nominal + fraction * (high - nominal)
    return effective_low + latent * (effective_high - effective_low)


def sample_hand_grasp_profiles(
    count: int,
    *,
    curriculum_fraction: float,
    tail_focus_fraction: float = 0.0,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Sample correlated fruit, contact, actuator, and tactile parameters."""

    _validate_count_fraction(count, curriculum_fraction)
    if not 0.0 <= tail_focus_fraction <= 1.0:
        raise ValueError("tail_focus_fraction must lie in [0, 1]")
    fraction = float(curriculum_fraction)
    latent = _rand((count, 12), generator=generator, dtype=dtype, device=device)
    size = latent[:, 0]
    ripeness = latent[:, 1]
    wetness = latent[:, 2]
    texture = 2.0 * latent[:, 3] - 1.0
    # Optional curriculum reweighting inside the same declared envelope.  It
    # deliberately increases joint small/ripe/wet/low-texture cases without
    # narrowing the ordinary full-range component or leaking privileged labels
    # into the Actor.  At curriculum zero it is disabled automatically.
    tail_focus_mask = _rand(
        (count,), generator=generator, dtype=dtype, device=device
    ) < (float(tail_focus_fraction) * fraction)
    size = torch.where(tail_focus_mask, 0.25 * size, size)
    ripeness = torch.where(tail_focus_mask, 0.75 + 0.25 * ripeness, ripeness)
    wetness = torch.where(tail_focus_mask, 0.65 + 0.35 * wetness, wetness)
    texture = torch.where(tail_focus_mask, -1.0 + latent[:, 3], texture)
    equatorial_shape = 2.0 * latent[:, 4] - 1.0
    polar_shape = 2.0 * latent[:, 5] - 1.0
    biology = 2.0 * latent[:, 6] - 1.0

    radius = _expanded_uniform(size, 0.025, 0.0325, 0.040, fraction)
    shape_scale = 0.055 * fraction
    axis_x = radius * (1.0 + shape_scale * equatorial_shape)
    axis_y = radius * (1.0 - shape_scale * equatorial_shape)
    aspect_fraction = (
        0.50 + 0.22 * polar_shape + 0.16 * (size - 0.5) + 0.12 * (0.5 - ripeness)
    ).clamp(0.0, 1.0)
    aspect = _expanded_uniform(aspect_fraction, 0.86, 0.97, 1.08, fraction)
    axis_z = radius * aspect
    axes = torch.stack((axis_x, axis_y, axis_z), dim=1)

    density_fraction = (
        0.5 + 0.22 * (0.5 - ripeness) + 0.14 * biology
    ).clamp(0.0, 1.0)
    density = _expanded_uniform(density_fraction, 880.0, 970.0, 1060.0, fraction)
    mass = 4.0 / 3.0 * math.pi * axes.prod(dim=1) * density

    firmness = (
        0.68 * (1.0 - ripeness)
        + 0.20 * (0.5 * (biology + 1.0))
        + 0.12 * size
    ).clamp(0.0, 1.0)
    stiffness = _expanded_uniform(firmness, 2500.0, 5600.0, 9000.0, fraction)
    damping = _expanded_uniform(
        (0.72 * firmness + 0.28 * size).clamp(0.0, 1.0),
        12.0,
        30.0,
        65.0,
        fraction,
    )

    friction_fraction = (
        0.56 + 0.22 * texture - 0.46 * wetness + 0.06 * (1.0 - ripeness)
    ).clamp(0.0, 1.0)
    static_friction = _expanded_uniform(
        friction_fraction, 0.25, 0.55, 0.90, fraction
    )
    dynamic_ratio = _expanded_uniform(latent[:, 7], 0.72, 0.82, 0.92, fraction)
    dynamic_friction = static_friction * dynamic_ratio

    centered_pose = 2.0 * latent[:, 8:11] - 1.0
    position_scale = torch.tensor((0.008, 0.008, 0.004), dtype=dtype, device=device)
    object_position_offset_m = centered_pose * position_scale * fraction
    object_yaw_rad = (2.0 * latent[:, 11] - 1.0) * math.radians(25.0) * fraction

    # Episode-fixed uncertainty.  Per-channel draws are intentionally separate
    # from biological latents while their magnitude follows the same curriculum.
    tactile_random = _rand((count, 5, 5), generator=generator, dtype=dtype, device=device)
    tactile_gain = _expanded_uniform(tactile_random[:, :, 0], 0.82, 1.0, 1.18, fraction)
    tactile_bias = _expanded_uniform(tactile_random[:, :, 1], -0.03, 0.0, 0.03, fraction)
    tactile_drift_per_s = _expanded_uniform(
        tactile_random[:, :, 2], -0.01, 0.0, 0.01, fraction
    )
    tactile_dropout_probability = tactile_random[:, :, 3] * (0.04 * fraction)
    # Episode-fixed outages teach explicit degraded operation.  They remain
    # distinguishable from true no-contact through the stale feature flag.
    tactile_failure_mask = _rand(
        (count, 5), generator=generator, dtype=dtype, device=device
    ) < (0.06 * fraction)
    tactile_release_tau_s = _expanded_uniform(
        tactile_random[:, :, 4], 0.02, 0.05, 0.14, fraction
    )
    maximum_delay = int(round(3.0 * fraction))
    tactile_delay_steps = _randint(
        0,
        maximum_delay + 1,
        (count,),
        generator=generator,
        device=device,
    )

    actuator_random = _rand((count, 16, 4), generator=generator, dtype=dtype, device=device)
    actuator_response_scale = _expanded_uniform(
        actuator_random[:, :, 0], 0.72, 1.0, 1.08, fraction
    )
    actuator_bias_rad = _expanded_uniform(
        actuator_random[:, :, 1], -0.012, 0.0, 0.012, fraction
    )
    actuator_deadband_rad = actuator_random[:, :, 2] * (0.010 * fraction)
    actuator_backlash_rad = actuator_random[:, :, 3] * (0.015 * fraction)

    return {
        "tail_focus_mask": tail_focus_mask,
        "size_latent": size,
        "ripeness": ripeness,
        "wetness": wetness,
        "texture_latent": texture,
        "biology_latent": biology,
        "axes_m": axes,
        "density_kg_m3": density,
        "mass_kg": mass,
        "contact_stiffness_n_m": stiffness,
        "contact_damping_n_s_m": damping,
        "static_friction": static_friction,
        "dynamic_friction": dynamic_friction,
        "object_position_offset_m": object_position_offset_m,
        "object_yaw_rad": object_yaw_rad,
        "tactile_gain": tactile_gain,
        "tactile_bias_normalized": tactile_bias,
        "tactile_drift_normalized_s": tactile_drift_per_s,
        "tactile_dropout_probability": tactile_dropout_probability,
        "tactile_failure_mask": tactile_failure_mask,
        "tactile_release_tau_s": tactile_release_tau_s,
        "tactile_delay_steps": tactile_delay_steps,
        "actuator_response_scale": actuator_response_scale,
        "actuator_bias_rad": actuator_bias_rad,
        "actuator_deadband_rad": actuator_deadband_rad,
        "actuator_backlash_rad": actuator_backlash_rad,
    }


def robustness_bucket_masks(profiles: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return overlapping, interpretable evaluation buckets."""

    required = {
        "size_latent",
        "ripeness",
        "wetness",
        "static_friction",
        "tactile_dropout_probability",
        "tactile_failure_mask",
        "actuator_response_scale",
    }
    if not required.issubset(profiles):
        raise ValueError(f"profiles missing {sorted(required - set(profiles))}")
    size = profiles["size_latent"]
    ripeness = profiles["ripeness"]
    wetness = profiles["wetness"]
    friction = profiles["static_friction"]
    dropout = profiles["tactile_dropout_probability"].mean(dim=1)
    failed = profiles["tactile_failure_mask"].any(dim=1)
    response = profiles["actuator_response_scale"].mean(dim=1)
    return {
        "small": size <= 0.25,
        "large": size >= 0.75,
        "ripe_soft": ripeness >= 0.75,
        "firm": ripeness <= 0.25,
        "wet_low_friction": (wetness >= 0.65) & (friction <= 0.50),
        "dry_high_friction": (wetness <= 0.35) & (friction >= 0.55),
        "sensor_degraded": (dropout >= 0.018) | failed,
        "thumb_sensor_failed": profiles["tactile_failure_mask"][:, 0],
        "multiple_sensors_failed": profiles["tactile_failure_mask"].sum(dim=1) >= 2,
        "actuator_sluggish": response <= 0.91,
    }


def pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Small dependency-free correlation helper for audit reports."""

    if left.ndim != 1 or right.shape != left.shape or left.numel() < 2:
        raise ValueError("correlation inputs must be aligned one-dimensional tensors")
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    denominator = torch.sqrt(centered_left.square().sum() * centered_right.square().sum())
    if float(denominator) == 0.0:
        return float("nan")
    return float((centered_left * centered_right).sum() / denominator)


__all__ = [
    "RANDOMIZATION_VERSION",
    "ROBUSTNESS_BUCKET_ORDER",
    "pearson_correlation",
    "robustness_bucket_masks",
    "sample_hand_grasp_profiles",
]
