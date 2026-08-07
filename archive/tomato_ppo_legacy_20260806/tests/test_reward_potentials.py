from __future__ import annotations

import torch

from simulation.tomato_pick.reward_potentials import positive_support_potential


def test_positive_support_potential_has_no_first_contact_barrier() -> None:
    support_ratio = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])
    potential = positive_support_potential(support_ratio)

    assert torch.equal(potential[:3], torch.zeros(3))
    assert torch.allclose(potential[3:], torch.tanh(support_ratio[3:]))
    assert bool((potential >= 0.0).all())
