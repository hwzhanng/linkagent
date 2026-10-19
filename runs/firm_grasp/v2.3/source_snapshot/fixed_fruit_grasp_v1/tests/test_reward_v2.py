"""CPU-only regression checks for the v1.2 reward shaping patch."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from research.fixed_fruit_grasp_v1.env import (
    FixedFruitGraspEnv,
    _APPROACH_SHAPING_GAMMA,
    _HARD_FAULT_REWARD_NAMES,
    _approach_potential_from_digit_gaps,
    _clear_selected_approach_potential,
    _gated_stability_reward,
    _hard_fault_mask,
    _hard_fault_penalty,
    _potential_difference_reward,
)
from research.small_fruit_pinch_v1.source.agents import SmallFruitPinchPPORunnerCfg


def test_approach_uses_agent_gamma_and_static_potential_is_not_positive() -> None:
    gamma = float(SmallFruitPinchPPORunnerCfg().algorithm.gamma)
    assert gamma == pytest.approx(0.995)
    assert _APPROACH_SHAPING_GAMMA == pytest.approx(gamma)

    gaps = torch.tensor([[0.020, 0.020, 0.080, 0.100, 0.120]], device="cpu")
    potential = _approach_potential_from_digit_gaps(gaps)
    previous = torch.zeros_like(potential)
    ready = torch.zeros_like(potential, dtype=torch.bool)

    first = _potential_difference_reward(
        potential, previous, ready, gamma=_APPROACH_SHAPING_GAMMA
    )
    assert first.item() == 0.0
    assert torch.equal(previous, potential)
    assert ready.item()

    stationary = _potential_difference_reward(
        potential, previous, ready, gamma=_APPROACH_SHAPING_GAMMA
    )
    assert stationary.item() <= 0.0
    assert stationary.item() == pytest.approx(
        (gamma - 1.0) * potential.item(), abs=5.0e-9
    )

    closer = _approach_potential_from_digit_gaps(gaps - 0.001)
    improvement = _potential_difference_reward(
        closer, previous, ready, gamma=_APPROACH_SHAPING_GAMMA
    )
    assert improvement.item() > 0.0


def test_partial_reset_clears_only_selected_environment_potential() -> None:
    previous = torch.tensor([0.25, 0.50, 0.75], device="cpu")
    ready = torch.tensor([True, True, True], device="cpu")

    _clear_selected_approach_potential(previous, ready, torch.tensor([1]))

    assert previous.tolist() == pytest.approx([0.25, 0.0, 0.75])
    assert ready.tolist() == [True, False, True]
    current = torch.tensor([0.25, 0.90, 0.75], device="cpu")
    delta = _potential_difference_reward(current, previous, ready)
    assert delta[1].item() == 0.0
    assert delta[0].item() < 0.0
    assert delta[2].item() < 0.0


def test_stability_reward_requires_release_and_real_hold_ok() -> None:
    drift = torch.tensor([0.0, 0.0, 0.010], device="cpu")
    hold_ok = torch.tensor([False, True, True], device="cpu")
    released = torch.tensor([True, False, True], device="cpu")

    stability = _gated_stability_reward(drift, hold_ok, released)

    # Zero drift during free fall cannot pass without the required-contact/speed/drift gate.
    assert stability[0].item() == 0.0
    assert stability[1].item() == 0.0
    assert stability[2].item() == pytest.approx(torch.exp(torch.tensor(-0.5)).item())


def _fault_names_in_termination_mask() -> set[str]:
    source = textwrap.dedent(inspect.getsource(FixedFruitGraspEnv._get_dones))
    tree = ast.parse(source)
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "hard_fault" for target in node.targets)
    ]
    assert len(assignments) == 1
    names: set[str] = set()
    for node in ast.walk(assignments[0].value):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "_faults"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            names.add(node.slice.value)
    return names


def test_every_termination_hard_fault_receives_reward_penalty() -> None:
    assert set(_HARD_FAULT_REWARD_NAMES) == _fault_names_in_termination_mask()

    count = len(_HARD_FAULT_REWARD_NAMES)
    faults = {
        name: torch.zeros(count, dtype=torch.bool, device="cpu")
        for name in _HARD_FAULT_REWARD_NAMES
    }
    for index, name in enumerate(_HARD_FAULT_REWARD_NAMES):
        faults[name][index] = True

    mask = _hard_fault_mask(faults)
    penalty = _hard_fault_penalty(mask, 2.0)
    assert mask.tolist() == [True] * count
    assert penalty.tolist() == [-2.0] * count


def test_reward_hooks_use_the_regression_helpers() -> None:
    source = textwrap.dedent(inspect.getsource(FixedFruitGraspEnv._get_rewards))
    tree = ast.parse(source)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "_approach_potential_from_digit_gaps",
        "_potential_difference_reward",
        "_gated_stability_reward",
        "_hard_fault_mask",
        "_hard_fault_penalty",
    } <= called
