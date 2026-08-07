from __future__ import annotations

import torch

from simulation.tomato_pick.stem_fracture import (
    directional_load_ratio,
    directional_overload,
    update_overload_streak,
)


def _overload(force: tuple[float, float, float], torque: tuple[float, float, float]) -> bool:
    return bool(
        directional_overload(
            torch.tensor([force]),
            torch.tensor([torque]),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([5.0]),
            torch.tensor([0.008]),
        ).item()
    )


def test_stem_ignores_compression_and_lateral_bending() -> None:
    assert not _overload((0.0, 0.0, -8.0), (0.02, 0.0, 0.0))
    assert not _overload((8.0, 0.0, 0.0), (0.0, 0.02, 0.0))


def test_stem_breaks_under_axial_pull_or_twist() -> None:
    assert _overload((0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
    assert _overload((0.0, 0.0, 0.0), (0.0, 0.0, -0.008))


def test_directional_load_ratio_matches_fracture_semantics() -> None:
    ratio = directional_load_ratio(
        torch.tensor(
            [
                [0.0, 0.0, 2.5],
                [20.0, 0.0, 0.0],
                [0.0, 0.0, -20.0],
            ]
        ),
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.020, 0.0],
                [0.0, 0.0, 0.004],
            ]
        ),
        torch.tensor([[0.0, 0.0, 1.0]]).repeat(3, 1),
        torch.full((3,), 5.0),
        torch.full((3,), 0.008),
    )
    assert torch.allclose(ratio, torch.tensor([0.5, 0.0, 0.5]))


def test_overload_requires_three_consecutive_substeps() -> None:
    streak = torch.zeros(2, dtype=torch.long)
    active = torch.tensor([True, True])
    for _ in range(2):
        streak = update_overload_streak(
            streak,
            torch.tensor([True, False]),
            active,
        )
    assert streak.tolist() == [2, 0]
    assert not bool((streak >= 3).any())

    streak = update_overload_streak(
        streak,
        torch.tensor([True, True]),
        active,
    )
    assert streak.tolist() == [3, 1]
    assert (streak >= 3).tolist() == [True, False]

    streak = update_overload_streak(
        streak,
        torch.tensor([False, True]),
        active,
    )
    assert streak.tolist() == [0, 2]
