"""Vectorized 12x6 G20 tactile features for CPU, simulation, and deployment.

No force calibration is claimed here.  Inputs may be calibrated force taxels
or baseline-corrected raw units as long as the declared full scales use the
same units.  Invalid/stale fingers are zeroed and explicitly flagged instead
of being silently interpreted as no contact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from simulation.ppo_tomato_grasp.hand_ppo_contract import (
    TACTILE_FEATURE_DIM,
    TACTILE_FEATURES_PER_FINGER,
)


FINGER_COUNT = 5
TAXEL_ROWS = 12
TAXEL_COLUMNS = 6


@dataclass(frozen=True)
class TactileFeatureBatch:
    features: torch.Tensor
    total_load: torch.Tensor
    peak_taxel: torch.Tensor
    contact_mask: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class TactileBaseline:
    baseline: torch.Tensor
    noise_mad: torch.Tensor
    bad_taxel_mask: torch.Tensor


def _matrix_shape(value: torch.Tensor, name: str) -> tuple[int, ...]:
    expected = (FINGER_COUNT, TAXEL_ROWS, TAXEL_COLUMNS)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim < 3 or tuple(value.shape[-3:]) != expected:
        raise ValueError(f"{name} must end with {expected}, got {tuple(value.shape)}")
    return tuple(value.shape[:-3])


def _broadcast_matrix(
    value: torch.Tensor | None,
    reference: torch.Tensor,
    *,
    name: str,
    default: float | bool,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if value is None:
        return torch.full_like(reference, default, dtype=dtype or reference.dtype)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    candidate = value.to(device=reference.device, dtype=dtype or reference.dtype)
    try:
        return torch.broadcast_to(candidate, reference.shape)
    except RuntimeError as error:
        raise ValueError(f"{name} cannot broadcast to {tuple(reference.shape)}") from error


def fit_tactile_baseline(
    no_contact_samples: torch.Tensor,
    *,
    maximum_nonfinite_fraction: float = 0.01,
    maximum_noise_mad: float | None = None,
) -> TactileBaseline:
    """Fit a robust per-taxel baseline from a leading sample dimension."""

    if not isinstance(no_contact_samples, torch.Tensor) or no_contact_samples.ndim != 4:
        raise ValueError("no_contact_samples must have shape (samples, 5, 12, 6)")
    if tuple(no_contact_samples.shape[1:]) != (FINGER_COUNT, TAXEL_ROWS, TAXEL_COLUMNS):
        raise ValueError("no_contact_samples must have shape (samples, 5, 12, 6)")
    if no_contact_samples.shape[0] < 5:
        raise ValueError("at least five no-contact frames are required")
    if not 0.0 <= maximum_nonfinite_fraction < 1.0:
        raise ValueError("maximum_nonfinite_fraction must lie in [0, 1)")
    if maximum_noise_mad is not None and maximum_noise_mad <= 0.0:
        raise ValueError("maximum_noise_mad must be positive")

    finite = torch.isfinite(no_contact_samples)
    nonfinite_fraction = 1.0 - finite.float().mean(dim=0)
    clean = torch.where(finite, no_contact_samples, torch.nan)
    baseline = torch.nanmedian(clean, dim=0).values
    deviation = (clean - baseline).abs()
    noise_mad = torch.nanmedian(deviation, dim=0).values
    bad = nonfinite_fraction > maximum_nonfinite_fraction
    bad |= ~torch.isfinite(baseline) | ~torch.isfinite(noise_mad)
    if maximum_noise_mad is not None:
        bad |= noise_mad > maximum_noise_mad
    baseline = torch.nan_to_num(baseline)
    noise_mad = torch.nan_to_num(noise_mad)
    return TactileBaseline(baseline=baseline, noise_mad=noise_mad, bad_taxel_mask=bad)


def extract_tactile_features(
    matrices: torch.Tensor,
    *,
    previous_total_load: torch.Tensor,
    dt_s: float,
    baseline: torch.Tensor | None = None,
    bad_taxel_mask: torch.Tensor | None = None,
    finger_valid: torch.Tensor | None = None,
    sample_age_s: torch.Tensor | None = None,
    maximum_age_s: float = 0.15,
    contact_taxel_threshold: float = 1.0,
    total_load_full_scale: float = 1000.0,
    peak_taxel_full_scale: float = 100.0,
    load_rate_full_scale_per_s: float = 5000.0,
) -> TactileFeatureBatch:
    """Convert five tactile matrices into 5x10 deployable features."""

    batch_shape = _matrix_shape(matrices, "matrices")
    if not matrices.is_floating_point():
        matrices = matrices.float()
    expected_finger_shape = (*batch_shape, FINGER_COUNT)
    if previous_total_load.shape != expected_finger_shape:
        raise ValueError(
            f"previous_total_load must have shape {expected_finger_shape}, "
            f"got {tuple(previous_total_load.shape)}"
        )
    if dt_s <= 0.0 or maximum_age_s <= 0.0:
        raise ValueError("dt_s and maximum_age_s must be positive")
    positive_scales = (
        contact_taxel_threshold,
        total_load_full_scale,
        peak_taxel_full_scale,
        load_rate_full_scale_per_s,
    )
    if any(value <= 0.0 for value in positive_scales):
        raise ValueError("tactile thresholds and scales must be positive")

    baseline_full = _broadcast_matrix(
        baseline, matrices, name="baseline", default=0.0
    )
    bad_full = _broadcast_matrix(
        bad_taxel_mask,
        matrices,
        name="bad_taxel_mask",
        default=False,
        dtype=torch.bool,
    )
    finite_or_bad = torch.isfinite(matrices) | bad_full
    finite_finger = finite_or_bad.flatten(-2).all(dim=-1)
    corrected = (torch.nan_to_num(matrices) - baseline_full).clamp_min(0.0)
    corrected = torch.where(bad_full, torch.zeros_like(corrected), corrected)

    if finger_valid is None:
        supplied_valid = torch.ones(expected_finger_shape, dtype=torch.bool, device=matrices.device)
    else:
        if finger_valid.shape != expected_finger_shape:
            raise ValueError(f"finger_valid must have shape {expected_finger_shape}")
        supplied_valid = finger_valid.to(device=matrices.device, dtype=torch.bool)
    if sample_age_s is None:
        age = torch.zeros(expected_finger_shape, device=matrices.device, dtype=matrices.dtype)
    else:
        if sample_age_s.shape != expected_finger_shape:
            raise ValueError(f"sample_age_s must have shape {expected_finger_shape}")
        age = sample_age_s.to(device=matrices.device, dtype=matrices.dtype)
    age_valid = torch.isfinite(age) & (age >= 0.0) & (age <= maximum_age_s)
    valid = supplied_valid & finite_finger & age_valid

    total = corrected.sum(dim=(-2, -1))
    peak = corrected.amax(dim=(-2, -1))
    active = (corrected >= contact_taxel_threshold) & ~bad_full
    good_count = (~bad_full).sum(dim=(-2, -1)).clamp_min(1)
    area = active.sum(dim=(-2, -1)).to(matrices.dtype) / good_count.to(matrices.dtype)

    row = torch.linspace(-1.0, 1.0, TAXEL_ROWS, device=matrices.device, dtype=matrices.dtype)
    column = torch.linspace(-1.0, 1.0, TAXEL_COLUMNS, device=matrices.device, dtype=matrices.dtype)
    row_grid = row.view(*((1,) * (matrices.ndim - 2)), TAXEL_ROWS, 1)
    col_grid = column.view(*((1,) * (matrices.ndim - 2)), 1, TAXEL_COLUMNS)
    denominator = total.clamp_min(torch.finfo(matrices.dtype).eps)
    centroid_row = (corrected * row_grid).sum(dim=(-2, -1)) / denominator
    centroid_col = (corrected * col_grid).sum(dim=(-2, -1)) / denominator
    has_load = total > 0.0
    centroid_row = torch.where(has_load, centroid_row, torch.zeros_like(centroid_row))
    centroid_col = torch.where(has_load, centroid_col, torch.zeros_like(centroid_col))
    row_delta = row_grid - centroid_row.unsqueeze(-1).unsqueeze(-1)
    col_delta = col_grid - centroid_col.unsqueeze(-1).unsqueeze(-1)
    spread_row = torch.sqrt(
        ((corrected * row_delta.square()).sum(dim=(-2, -1)) / denominator).clamp_min(0.0)
    ).clamp(0.0, 1.0)
    spread_col = torch.sqrt(
        ((corrected * col_delta.square()).sum(dim=(-2, -1)) / denominator).clamp_min(0.0)
    ).clamp(0.0, 1.0)
    spread_row = torch.where(has_load, spread_row, torch.zeros_like(spread_row))
    spread_col = torch.where(has_load, spread_col, torch.zeros_like(spread_col))
    load_rate = (
        (total - previous_total_load.to(total)) / dt_s / load_rate_full_scale_per_s
    ).clamp(-1.0, 1.0)

    physical = torch.stack(
        (
            (total / total_load_full_scale).clamp(0.0, 1.0),
            (peak / peak_taxel_full_scale).clamp(0.0, 1.0),
            area.clamp(0.0, 1.0),
            centroid_row.clamp(-1.0, 1.0),
            centroid_col.clamp(-1.0, 1.0),
            spread_row,
            spread_col,
            load_rate,
        ),
        dim=-1,
    )
    physical = torch.where(valid.unsqueeze(-1), physical, torch.zeros_like(physical))
    per_finger = torch.cat(
        (
            physical,
            valid.to(matrices.dtype).unsqueeze(-1),
            (~valid).to(matrices.dtype).unsqueeze(-1),
        ),
        dim=-1,
    )
    if per_finger.shape[-1] != TACTILE_FEATURES_PER_FINGER:
        raise AssertionError("tactile feature arithmetic drifted")
    features = per_finger.reshape(*batch_shape, TACTILE_FEATURE_DIM)
    return TactileFeatureBatch(
        features=features,
        total_load=torch.where(valid, total, torch.zeros_like(total)),
        peak_taxel=torch.where(valid, peak, torch.zeros_like(peak)),
        contact_mask=valid & (area > 0.0),
        valid_mask=valid,
    )


__all__ = [
    "FINGER_COUNT",
    "TAXEL_COLUMNS",
    "TAXEL_ROWS",
    "TactileBaseline",
    "TactileFeatureBatch",
    "extract_tactile_features",
    "fit_tactile_baseline",
]
