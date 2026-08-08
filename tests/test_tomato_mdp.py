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
    ALIGNED_SUCCESS_WEIGHT,
    BREAK_GRASPED_WEIGHT,
    DISCOVERY_SUCCESS_WEIGHT,
    DROP_FAILURE_WEIGHT,
    EXCESS_FORCE_FAILURE_WEIGHT,
    alignment_alpha_for_iteration,
    best_ever_progress,
    break_grasp_quality,
    grasp_contact_score,
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
        assert all(
            lo <= value <= hi
            for value, lo, hi in zip(first, HAND_LOWER_RAD, HAND_UPPER_RAD)
        )
    assert g20_synergy_to_joint_targets((-1, -1, 0, 0, 0)) != (
        g20_synergy_to_joint_targets((1, 1, 0, 0, 0))
    )
    assert g20_synergy_to_joint_targets((0, 0, 0, 0, 0)) == (
        g20_synergy_to_joint_targets(PREGRASP_SYNERGY)
    )


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


def test_each_hand_manifold_coordinate_remains_effective() -> None:
    baseline = g20_synergy_to_joint_targets((0.0,) * 5)
    for coordinate in range(5):
        action = [0.0] * 5
        action[coordinate] = 1.0
        assert g20_synergy_to_joint_targets(action) != baseline


def test_stem_coordinate_counts_pull_but_not_compression_or_lateral_force() -> None:
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


def test_stem_overload_streak_counts_only_consecutive_physics_substeps() -> None:
    streak = torch.zeros(2, dtype=torch.long)
    attached = torch.tensor([True, True])
    for _ in range(2):
        streak = update_overload_streak(
            streak, torch.tensor([True, False]), attached
        )
    assert streak.tolist() == [2, 0]
    streak = update_overload_streak(
        streak, torch.tensor([True, True]), attached
    )
    assert streak.tolist() == [3, 1]
    streak = update_overload_streak(
        streak, torch.tensor([False, True]), attached
    )
    assert streak.tolist() == [0, 2]


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


def test_grasp_score_rewards_a_second_finger_before_thumb_opposition() -> None:
    forces = torch.tensor(
        [
            [0.0, 0.6, 0.0, 0.0, 0.0],
            [0.0, 0.6, 0.3, 0.0, 0.0],
            [0.6, 0.6, 0.0, 0.0, 0.0],
        ]
    )
    score = grasp_contact_score(forces, full_contact_force_n=0.6)
    assert torch.allclose(score, torch.tensor([0.0, 0.5, 1.0]))


def _reward_step(
    *,
    reach: float = 0.0,
    best_reach: float = 0.0,
    grasp: float = 0.0,
    best_grasp: float = 0.0,
    opposed: float = 0.0,
    best_opposed: float = 0.0,
    stem_load: float = 0.0,
    best_stem_load: float = 0.0,
    stable_detach: float = 0.0,
    best_stable_detach: float = 0.0,
    retention: float = 0.0,
    best_retention: float = 0.0,
    break_grasped: float = 0.0,
    detached: bool = False,
    success: bool = False,
    dropped: bool = False,
    excess: bool = False,
    out_of_bounds: bool = False,
    timeout: bool = False,
    alpha: float = 0.0,
):
    scalar = lambda value: torch.tensor([value])
    boolean = lambda value: torch.tensor([value], dtype=torch.bool)
    return tomato_pick_reward(
        reach_score=scalar(reach),
        previous_best_reach=scalar(best_reach),
        grasp_score=scalar(grasp),
        previous_best_grasp=scalar(best_grasp),
        opposed_score=scalar(opposed),
        previous_best_opposed=scalar(best_opposed),
        stem_load_score=scalar(stem_load),
        previous_best_stem_load=scalar(best_stem_load),
        stable_detach_score=scalar(stable_detach),
        previous_best_stable_detach=scalar(best_stable_detach),
        retention_score=scalar(retention),
        previous_best_retention=scalar(best_retention),
        break_grasped_score=scalar(break_grasped),
        success=boolean(success),
        dropped=boolean(dropped),
        excess_force=boolean(excess),
        out_of_bounds=boolean(out_of_bounds),
        timed_out=boolean(timeout),
        stem_detached=boolean(detached),
        alignment_alpha=alpha,
    )


def test_discovery_endpoint_preserves_positive_hierarchy() -> None:
    result = _reward_step(
        reach=0.8,
        best_reach=0.2,
        grasp=0.7,
        best_grasp=0.3,
        opposed=1.0,
        stem_load=0.6,
        best_stem_load=0.1,
        break_grasped=1.0,
        alpha=0.0,
    )
    reward, *_, terms = result
    expected = 5.0 * 0.6 + 10.0 * 0.4 + 5.0 + 20.0 * 0.5 + 40.0 - 0.01
    assert math.isclose(reward.item(), expected, abs_tol=2.0e-6)
    assert terms["stable_detach"].item() == 0.0
    assert terms["drop_failure"].item() == 0.0


def test_discovery_failure_does_not_erase_first_success_exploration() -> None:
    reward, *_, terms = _reward_step(
        reach=1.0,
        grasp=1.0,
        opposed=1.0,
        stem_load=1.0,
        break_grasped=1.0,
        detached=True,
        dropped=True,
        alpha=0.0,
    )
    assert reward.item() > BREAK_GRASPED_WEIGHT
    assert terms["drop_failure"].item() == 0.0


def test_progress_is_bounded_and_non_repeatable() -> None:
    best = torch.zeros(1)
    total = torch.zeros(1)
    for score in (0.2, 0.8, 0.3, 0.8, 0.1):
        progress, best = best_ever_progress(torch.tensor([score]), best)
        total += progress
    assert torch.allclose(total, torch.tensor([0.8]))


def test_alignment_schedule_is_delayed_smooth_and_monotone() -> None:
    assert alignment_alpha_for_iteration(0) == 0.0
    assert alignment_alpha_for_iteration(250) == 0.0
    samples = [alignment_alpha_for_iteration(i) for i in range(0, 1501, 10)]
    assert all(left <= right for left, right in zip(samples, samples[1:]))
    assert 0.0 < alignment_alpha_for_iteration(500) < 0.35
    assert alignment_alpha_for_iteration(600) == 0.35
    assert alignment_alpha_for_iteration(1000) == 0.75
    assert alignment_alpha_for_iteration(1300) == 1.0


def test_aligned_endpoint_removes_break_and_repeated_hold_farming() -> None:
    reward, *_, terms = _reward_step(
        opposed=1.0,
        best_opposed=1.0,
        stable_detach=1.0,
        best_stable_detach=1.0,
        retention=0.95,
        best_retention=0.95,
        break_grasped=1.0,
        detached=True,
        alpha=1.0,
    )
    assert math.isclose(reward.item(), -0.1, abs_tol=1.0e-7)
    assert terms["break_grasped"].item() == 0.0


def test_aligned_success_strongly_dominates_near_success_failure() -> None:
    common = dict(
        reach=1.0,
        grasp=1.0,
        opposed=1.0,
        stem_load=1.0,
        stable_detach=1.0,
        retention=1.0,
        detached=True,
        alpha=1.0,
    )
    success, *_, success_terms = _reward_step(**common, success=True)
    drop, *_, drop_terms = _reward_step(**common, dropped=True)
    excess, *_, _ = _reward_step(**common, excess=True)
    assert success_terms["success"].item() == ALIGNED_SUCCESS_WEIGHT
    assert drop_terms["drop_failure"].item() == DROP_FAILURE_WEIGHT
    assert success.item() > drop.item() + 200.0
    assert excess.item() < drop.item()
    assert EXCESS_FORCE_FAILURE_WEIGHT < DROP_FAILURE_WEIGHT


def test_aligned_timeout_distinguishes_waiting_from_late_failure() -> None:
    waiting, *_, waiting_terms = _reward_step(timeout=True, alpha=1.0)
    late, *_, late_terms = _reward_step(
        detached=True, timeout=True, alpha=1.0
    )
    assert waiting.item() == -10.0
    assert late.item() == -30.0
    assert waiting_terms["timeout_failure"].item() > late_terms["timeout_failure"].item()


def test_success_weight_grows_without_changing_discovery_endpoint() -> None:
    discovery, *_, _ = _reward_step(success=True, alpha=0.0)
    aligned, *_, _ = _reward_step(success=True, alpha=1.0)
    assert discovery.item() == DISCOVERY_SUCCESS_WEIGHT
    assert aligned.item() == ALIGNED_SUCCESS_WEIGHT
