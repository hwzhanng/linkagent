from __future__ import annotations

import math

import torch

from module.skills.tomato_pick import (
    ACTION_DIM,
    CRITIC_STATE_DIM,
    OBSERVATION_DIM,
    PolicyManifest,
    TomatoPickAction,
    TomatoPickObservation,
)
from module.skills.tomato_pick.hand_synergy import (
    HAND_LOWER_RAD,
    HAND_NEUTRAL_RAD,
    HAND_UPPER_RAD,
    PREGRASP_SYNERGY,
    THUMB_ROLL_FIXED_RAD,
    THUMB_ROOT_MAX_FRACTION,
    g20_synergy_to_joint_targets,
)
from simulation.tomato_pick.reward import (
    GRASP_PROGRESS_WEIGHT,
    REACH_PROGRESS_WEIGHT,
    RETENTION_PROGRESS_WEIGHT,
    SHAPING_GAMMA,
    STABLE_DETACH_WEIGHT,
    STEM_LOAD_PROGRESS_WEIGHT,
    SUCCESS_WEIGHT,
    TERMINAL_FAILURE_WEIGHT,
    TIME_COST_WEIGHT,
    best_ever_progress,
    break_grasp_quality,
    grasp_contact_score,
    opposed_contact_score,
    tomato_pick_reward,
)
from simulation.tomato_pick.stem_fracture import attachment_load_ratio, update_overload_streak


def test_frozen_dimensions_and_named_observation_blocks() -> None:
    assert (OBSERVATION_DIM, ACTION_DIM, CRITIC_STATE_DIM) == (66, 11, 83)
    observation = TomatoPickObservation.pack(
        fruit_center_in_palm=(0.0,) * 3,
        fruit_semi_axes_norm=(0.0,) * 3,
        pull_axis_in_palm=(0.0, 0.0, 1.0),
        ee_twist_in_palm_norm=(0.0,) * 6,
        hand_joint_position_norm=(0.0,) * 16,
        hand_joint_velocity_norm=(0.0,) * 16,
        finger_pressure_norm=(0.0,) * 5,
        gravity_in_palm=(0.0, 0.0, -1.0),
        previous_action=(0.0,) * 11,
    )
    assert len(observation.values) == 66
    assert observation.block("finger_pressure") == (0.0,) * 5
    assert TomatoPickAction.zero().hand_synergy == (0.0,) * 5
    assert PolicyManifest.mock_default().policy_rate_hz == 20


def test_g20_synergy_is_absolute_bounded_and_low_rank() -> None:
    for synergy in (
        (-1.0,) * 5,
        (0.0,) * 5,
        (1.0,) * 5,
        (0.2, -0.4, 0.7, -0.3, 0.5),
    ):
        first = g20_synergy_to_joint_targets(synergy)
        second = g20_synergy_to_joint_targets(synergy)
        assert first == second
        assert len(first) == 16
        assert all(lo <= value <= hi for value, lo, hi in zip(first, HAND_LOWER_RAD, HAND_UPPER_RAD))
    assert g20_synergy_to_joint_targets((-1, -1, 0, 0, 0)) != g20_synergy_to_joint_targets((1, 1, 0, 0, 0))
    assert g20_synergy_to_joint_targets((0, 0, 0, 0, 0)) == g20_synergy_to_joint_targets(PREGRASP_SYNERGY)


def test_thumb_roll_is_fixed_and_opposition_uses_safe_coupled_range() -> None:
    open_root = g20_synergy_to_joint_targets((0, 0, 0, 0, -1))
    closed_root = g20_synergy_to_joint_targets((0, 0, 0, 0, 1))
    assert open_root[5] == closed_root[5] == HAND_NEUTRAL_RAD[5]
    assert open_root[10] == closed_root[10] == THUMB_ROLL_FIXED_RAD
    assert open_root[0] == HAND_LOWER_RAD[0]
    expected_half = HAND_LOWER_RAD[0] + THUMB_ROOT_MAX_FRACTION * (
        HAND_UPPER_RAD[0] - HAND_LOWER_RAD[0]
    )
    assert math.isclose(closed_root[0], expected_half, abs_tol=1.0e-12)
    assert open_root[11] == HAND_LOWER_RAD[11]
    assert closed_root[11] == HAND_UPPER_RAD[11]


def test_radial_ulnar_balance_can_correct_which_fingers_contact_first() -> None:
    radial = g20_synergy_to_joint_targets((0.35, 0.0, 1.0, 0.0, 0.0))
    ulnar = g20_synergy_to_joint_targets((0.35, 0.0, -1.0, 0.0, 0.0))
    assert radial[1] > ulnar[1]
    assert radial[2] > ulnar[2]
    assert radial[3] < ulnar[3]
    assert radial[4] < ulnar[4]
    assert radial[12] > ulnar[12]
    assert radial[15] < ulnar[15]


def test_curl_shape_changes_distal_relative_to_proximal_flexion() -> None:
    straight = g20_synergy_to_joint_targets((0.45, -1.0, 0.0, 0.0, 0.0))
    curled = g20_synergy_to_joint_targets((0.45, 1.0, 0.0, 0.0, 0.0))
    assert curled[1] < straight[1]
    assert curled[12] > straight[12]


def test_each_hand_manifold_coordinate_remains_effective() -> None:
    baseline = g20_synergy_to_joint_targets((0.0,) * 5)
    for coordinate in range(5):
        action = [0.0] * 5
        action[coordinate] = 1.0
        assert g20_synergy_to_joint_targets(action) != baseline


def test_stem_coordinate_counts_pull_but_not_compression_lateral_or_twist() -> None:
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    force_limit = torch.tensor([5.0])
    pull = attachment_load_ratio(
        torch.tensor([[0.0, 0.0, -5.0]]), axis, force_limit
    )
    compression = attachment_load_ratio(
        torch.tensor([[0.0, 0.0, 5.0]]), axis, force_limit
    )
    lateral = attachment_load_ratio(
        torch.tensor([[5.0, 0.0, 0.0]]), axis, force_limit
    )
    assert torch.allclose(pull, torch.ones(1))
    assert torch.allclose(compression, torch.zeros(1))
    assert torch.allclose(lateral, torch.zeros(1))


def test_stem_break_requires_more_than_one_policy_interval_of_overload() -> None:
    streak = torch.zeros(2, dtype=torch.long)
    attached = torch.tensor([True, True])
    for _ in range(6):
        streak = update_overload_streak(
            streak,
            torch.tensor([True, False]),
            attached,
        )
    assert streak.tolist() == [6, 0]
    assert not bool((streak >= 7).any())

    streak = update_overload_streak(
        streak,
        torch.tensor([True, True]),
        attached,
    )
    assert streak.tolist() == [7, 1]

    streak = update_overload_streak(
        streak,
        torch.tensor([False, True]),
        attached,
    )
    assert streak.tolist() == [0, 2]


def test_opposed_contact_requires_thumb_and_a_support_finger() -> None:
    forces = torch.tensor(
        [
            [0.6, 0.6, 0.0, 0.0, 0.0],
            [0.6, 0.6, 0.6, 0.0, 0.0],
            [0.3, 0.6, 0.3, 0.0, 0.0],
        ]
    )
    score = opposed_contact_score(forces, full_contact_force_n=0.6)
    assert torch.allclose(score, torch.tensor([1.0, 1.0, 0.5]))
    shaped = grasp_contact_score(forces, full_contact_force_n=0.6)
    assert torch.allclose(shaped, torch.tensor([0.88, 0.92, 0.48]))


def test_break_grasp_quality_requires_two_fingers_and_prefers_thumb() -> None:
    forces = torch.tensor(
        [
            [0.0, 0.6, 0.0, 0.0, 0.0],
            [0.0, 0.6, 0.6, 0.0, 0.0],
            [0.6, 0.6, 0.0, 0.0, 0.0],
        ]
    )
    quality = break_grasp_quality(forces, contact_force_n=0.6)
    assert torch.allclose(quality, torch.tensor([0.0, 0.4, 1.0]))


def test_reward_is_non_repeatable_hierarchical_and_outcome_aligned() -> None:
    best = torch.zeros(1)
    total = torch.zeros(1)
    for score in (0.2, 0.8, 0.3, 0.8, 0.1):
        progress, best = best_ever_progress(torch.tensor([score]), best)
        total += progress
    assert torch.allclose(total, torch.tensor([0.8]))

    zeros = torch.zeros(1)
    false = torch.tensor([False])
    wait, *_ = tomato_pick_reward(
        reach_score=zeros,
        previous_best_reach=zeros,
        grasp_score=zeros,
        previous_best_grasp=zeros,
        stem_load_score=zeros,
        previous_best_stem_load=zeros,
        stable_detach_score=zeros,
        previous_best_stable_detach=zeros,
        retention_score=zeros,
        previous_best_retention=zeros,
        previous_credit_balance=zeros,
        success=false,
        terminal_failure=false,
    )
    assert math.isclose(wait.item(), TIME_COST_WEIGHT, abs_tol=1.0e-6)
    assert math.isclose(TIME_COST_WEIGHT, -1.0, abs_tol=1.0e-12)

    one = torch.ones(1)
    shaping, _, _, _, _, _, balance, shaping_terms = tomato_pick_reward(
        reach_score=one,
        previous_best_reach=zeros,
        grasp_score=one,
        previous_best_grasp=zeros,
        stem_load_score=one,
        previous_best_stem_load=zeros,
        stable_detach_score=one,
        previous_best_stable_detach=zeros,
        retention_score=one,
        previous_best_retention=zeros,
        previous_credit_balance=zeros,
        success=false,
        terminal_failure=false,
    )
    potential_sum = sum(
        (
            REACH_PROGRESS_WEIGHT,
            GRASP_PROGRESS_WEIGHT,
            STEM_LOAD_PROGRESS_WEIGHT,
            STABLE_DETACH_WEIGHT,
            RETENTION_PROGRESS_WEIGHT,
        )
    )
    assert shaping_terms["reach_progress"].item() == REACH_PROGRESS_WEIGHT
    assert shaping_terms["grasp_progress"].item() == GRASP_PROGRESS_WEIGHT
    assert shaping_terms["stem_load_progress"].item() == STEM_LOAD_PROGRESS_WEIGHT
    assert shaping_terms["stable_detach"].item() == STABLE_DETACH_WEIGHT
    assert shaping_terms["retention_progress"].item() == RETENTION_PROGRESS_WEIGHT
    assert shaping_terms["credit_recovery"].item() == 0.0
    assert math.isclose(
        shaping.item(), potential_sum + TIME_COST_WEIGHT, rel_tol=1.0e-6
    )
    assert math.isclose(
        balance.item(), potential_sum / SHAPING_GAMMA, rel_tol=1.0e-6
    )

    failure, _, _, _, _, _, next_balance, failure_terms = tomato_pick_reward(
        reach_score=one,
        previous_best_reach=one,
        grasp_score=one,
        previous_best_grasp=one,
        stem_load_score=one,
        previous_best_stem_load=one,
        stable_detach_score=one,
        previous_best_stable_detach=one,
        retention_score=one,
        previous_best_retention=one,
        previous_credit_balance=balance,
        success=false,
        terminal_failure=torch.tensor([True]),
    )
    assert TERMINAL_FAILURE_WEIGHT == -100.0
    assert failure_terms["terminal_failure"].item() == TERMINAL_FAILURE_WEIGHT
    assert math.isclose(
        failure_terms["credit_recovery"].item(), -balance.item(), rel_tol=1.0e-6
    )
    assert failure_terms["time_cost"].item() == 0.0
    assert math.isclose(
        failure.item(), TERMINAL_FAILURE_WEIGHT - balance.item(), rel_tol=1.0e-6
    )
    assert next_balance.item() == 0.0
    # Every failed trajectory has exactly the same discounted terminal return;
    # none of the temporary milestones can be harvested by failing later.
    assert torch.allclose(
        shaping + SHAPING_GAMMA * failure,
        torch.tensor([TERMINAL_FAILURE_WEIGHT]),
    )

    delayed_balance = balance.clone()
    discounted_return = shaping.clone()
    discount = SHAPING_GAMMA
    for _ in range(7):
        wait, _, _, _, _, _, delayed_balance, _ = tomato_pick_reward(
            reach_score=one,
            previous_best_reach=one,
            grasp_score=one,
            previous_best_grasp=one,
            stem_load_score=one,
            previous_best_stem_load=one,
            stable_detach_score=one,
            previous_best_stable_detach=one,
            retention_score=one,
            previous_best_retention=one,
            previous_credit_balance=delayed_balance,
            success=false,
            terminal_failure=false,
        )
        discounted_return += discount * wait
        discount *= SHAPING_GAMMA
    delayed_failure, _, _, _, _, _, _, _ = tomato_pick_reward(
        reach_score=one,
        previous_best_reach=one,
        grasp_score=one,
        previous_best_grasp=one,
        stem_load_score=one,
        previous_best_stem_load=one,
        stable_detach_score=one,
        previous_best_stable_detach=one,
        retention_score=one,
        previous_best_retention=one,
        previous_credit_balance=delayed_balance,
        success=false,
        terminal_failure=torch.tensor([True]),
    )
    discounted_return += discount * delayed_failure
    assert torch.allclose(
        discounted_return,
        torch.tensor([TERMINAL_FAILURE_WEIGHT]),
        atol=1.0e-4,
    )

    # A no-progress timeout remains exactly -100 at any episode length.
    timeout_return = torch.zeros(1)
    timeout_balance = torch.zeros(1)
    discount = 1.0
    for _ in range(7):
        wait, _, _, _, _, _, timeout_balance, _ = tomato_pick_reward(
            reach_score=zeros,
            previous_best_reach=zeros,
            grasp_score=zeros,
            previous_best_grasp=zeros,
            stem_load_score=zeros,
            previous_best_stem_load=zeros,
            stable_detach_score=zeros,
            previous_best_stable_detach=zeros,
            retention_score=zeros,
            previous_best_retention=zeros,
            previous_credit_balance=timeout_balance,
            success=false,
            terminal_failure=false,
        )
        timeout_return += discount * wait
        discount *= SHAPING_GAMMA
    timeout_failure, _, _, _, _, _, _, _ = tomato_pick_reward(
        reach_score=zeros,
        previous_best_reach=zeros,
        grasp_score=zeros,
        previous_best_grasp=zeros,
        stem_load_score=zeros,
        previous_best_stem_load=zeros,
        stable_detach_score=zeros,
        previous_best_stable_detach=zeros,
        retention_score=zeros,
        previous_best_retention=zeros,
        previous_credit_balance=timeout_balance,
        success=false,
        terminal_failure=torch.tensor([True]),
    )
    timeout_return += discount * timeout_failure
    assert torch.allclose(
        timeout_return, torch.tensor([TERMINAL_FAILURE_WEIGHT]), atol=1.0e-4
    )


def test_downstream_rewards_require_grasp_quality() -> None:
    zeros = torch.zeros(1)
    one = torch.ones(1)
    reward, _, _, _, _, _, _, terms = tomato_pick_reward(
        reach_score=zeros,
        previous_best_reach=zeros,
        grasp_score=zeros,
        previous_best_grasp=zeros,
        stem_load_score=zeros,
        previous_best_stem_load=zeros,
        stable_detach_score=zeros,
        previous_best_stable_detach=zeros,
        retention_score=zeros,
        previous_best_retention=zeros,
        previous_credit_balance=zeros,
        success=one,
        terminal_failure=zeros,
    )
    assert terms["stable_detach"].item() == 0.0
    assert terms["retention_progress"].item() == 0.0
    assert terms["success"].item() == SUCCESS_WEIGHT
    assert reward.item() == SUCCESS_WEIGHT
