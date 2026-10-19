from __future__ import annotations

import math

import pytest
import torch

from research.small_fruit_pinch_v1.source.size_evidence import (
    summarize_first_episode_rows,
    wilson_interval,
)
from simulation.tomato_pick.isaac.geometry_sampling import (
    allocate_bin_counts,
    equivalent_diameter_m,
    parse_bin_weights,
    parse_diameter_bins_mm,
    sample_stratified_ellipsoids,
    stable_geometry_seed,
)


FULL_BINS_MM = tuple((float(low), float(low + 2)) for low in range(52, 80, 2))
FULL_BINS_M = tuple((low / 1000.0, high / 1000.0) for low, high in FULL_BINS_MM)


def _sample(seed: int = 1234) -> dict[str, torch.Tensor]:
    return sample_stratified_ellipsoids(
        140,
        diameter_bins_m=FULL_BINS_M,
        diameter_bin_weights=(1.0,) * len(FULL_BINS_M),
        shape_axis_ranges_m=((0.032, 0.035),) * 3,
        density_range_kg_m3=(650.0, 1050.0),
        accepted_mass_kg=(0.03, 0.50),
        maximum_axis_ratio=1.6,
        seed=seed,
    )


def test_parse_bins_weights_and_exact_largest_remainder_quota() -> None:
    assert parse_diameter_bins_mm("52:54,54:56") == ((52.0, 54.0), (54.0, 56.0))
    assert parse_bin_weights("4,2,2,2", 4) == pytest.approx((0.4, 0.2, 0.2, 0.2))
    assert allocate_bin_counts(11, (0.4, 0.2, 0.2, 0.2)) == (5, 2, 2, 2)
    with pytest.raises(ValueError, match="不能重叠"):
        parse_diameter_bins_mm("52:55,54:56")
    with pytest.raises(ValueError, match="数量"):
        parse_bin_weights("1,1", 3)


def test_sampler_is_deterministic_balanced_and_inside_every_bin() -> None:
    first = _sample()
    second = _sample()
    for key in first:
        assert torch.equal(first[key], second[key]), key
    assert first["bin_quotas"].tolist() == [10] * 14
    assert torch.bincount(first["bin_index"], minlength=14).tolist() == [10] * 14

    recomputed = equivalent_diameter_m(first["axes_m"].double())
    assert torch.allclose(
        recomputed, first["equivalent_diameter_m"].double(), atol=2.0e-8, rtol=0.0
    )
    for row, bin_index in zip(recomputed.tolist(), first["bin_index"].tolist()):
        low, high = FULL_BINS_M[bin_index]
        assert low <= row < high

    axes = first["axes_m"].double()
    density = first["density_kg_m3"].double()
    recomputed_mass = 4.0 / 3.0 * math.pi * axes.prod(dim=1) * density
    assert torch.allclose(recomputed_mass, first["mass_kg"].double(), atol=2.0e-7)
    assert bool(((recomputed_mass >= 0.03) & (recomputed_mass <= 0.50)).all())
    assert float((axes.amax(dim=1) / axes.amin(dim=1)).max()) <= 1.6


def test_cohort_seed_is_stable_and_separates_cohorts() -> None:
    assert stable_geometry_seed(7, "cohort-a") == stable_geometry_seed(7, "cohort-a")
    assert stable_geometry_seed(7, "cohort-a") != stable_geometry_seed(7, "cohort-b")
    assert not torch.equal(_sample(1)["axes_m"], _sample(2)["axes_m"])


def test_programmatic_bins_reject_overlap_and_empty_input() -> None:
    common = dict(
        count=2,
        diameter_bin_weights=(1.0,),
        shape_axis_ranges_m=((0.032, 0.035),) * 3,
        density_range_kg_m3=(650.0, 1050.0),
        accepted_mass_kg=(0.03, 0.50),
        maximum_axis_ratio=1.6,
        seed=1,
    )
    with pytest.raises(ValueError, match="不能为空"):
        sample_stratified_ellipsoids(
            **{**common, "diameter_bins_m": (), "diameter_bin_weights": ()}
        )
    with pytest.raises(ValueError, match="不能重叠"):
        sample_stratified_ellipsoids(
            **{
                **common,
                "diameter_bins_m": ((0.052, 0.055), (0.054, 0.056)),
                "diameter_bin_weights": (1.0, 1.0),
            }
        )


def test_first_episode_summary_uses_mutually_exclusive_reasons_and_wilson() -> None:
    rows = [
        {"bin_index": 0, "termination_reason": "success"},
        {"bin_index": 0, "termination_reason": "drop"},
        {"bin_index": 1, "termination_reason": "success"},
    ]
    result = summarize_first_episode_rows(rows, ((52.0, 54.0), (54.0, 56.0)))
    assert result[0]["first_episode_count"] == 2
    assert result[0]["termination_counts"]["success"] == 1
    assert result[0]["termination_counts"]["drop"] == 1
    assert result[0]["strict_success_rate"] == 0.5
    lower, upper = result[0]["strict_success_wilson95"]
    assert 0.0 < lower < 0.5 < upper < 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)
