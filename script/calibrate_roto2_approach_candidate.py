"""Offline candidate scale check; this does not change any training reward."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def candidate_cost(quality, weight, pad_count=3):
    return -weight * (1 - np.sort(quality, axis=-1)[..., -pad_count:]).mean(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tr = np.load(args.trace)
    quality = tr["approach_quality"]
    assert np.isfinite(quality).all()
    assert quality.min() >= -2.00001 and quality.max() <= 1.00001
    active = tr["active"].astype(bool) & ~tr["done"].astype(bool)
    dt = json.loads(args.trace.with_suffix(".json").read_text())["dt"]
    time = (np.arange(len(active)) + 1) * dt
    late = active & (time[:, None] > 10)
    # With four ideal pads, releasing any one retains three ideal pads. A
    # temporary two-pad state has finite cost, not a forbidden transition.
    four_near = np.array([1., 1., 1., 1., -2.])
    one_released = four_near.copy()
    one_released[0] = -2.
    assert candidate_cost(four_near, .025) == candidate_cost(one_released, .025) == 0
    two_near = np.array([1., 1., -2., -2., -2.])
    assert np.isclose(candidate_cost(two_near, .025), -.025)
    permutation = [4, 2, 0, 3, 1]
    assert np.array_equal(candidate_cost(quality, .025),
                          candidate_cost(quality[..., permutation], .025))
    rows = []
    selections = {}
    releases = {}
    for pad_count in (3, 4):
        selected = np.zeros_like(quality, dtype=bool)
        np.put_along_axis(selected, np.argsort(quality, axis=-1)[..., -pad_count:], True, axis=-1)
        selections[str(pad_count)] = selected[late].mean(0).tolist()
        assert np.array_equal(candidate_cost(quality, .025, pad_count),
                              candidate_cost(quality[..., permutation], .025, pad_count))
        releases[str(pad_count)] = {
            "four_ideal_one_distant": float(candidate_cost(four_near, .025, pad_count)),
            "release_one_leaves_three_ideal": float(candidate_cost(one_released, .025, pad_count)),
            "two_ideal_three_distant": float(candidate_cost(two_near, .025, pad_count)),
            "all_finite_and_no_forbidden_action": True,
        }
        for weight in (.025, .05, .1):
            cost = candidate_cost(quality, weight, pad_count)
            assert np.max(cost) <= 1e-7 and np.min(cost) >= -3 * weight - 1e-7
            rows.append({
                "pad_count": pad_count,
                "weight": weight,
                "theoretical_per_step_range": [-3 * weight, 0],
                "mean_late_per_active_nonterminal_step": float(cost[late].mean()),
                "late_sum_per_initial_episode_excluding_terminal_steps": float((cost * late).sum() / active.shape[1]),
                "whole_sum_per_initial_episode_excluding_terminal_steps": float((cost * active).sum() / active.shape[1]),
            })
    result = {
        "implemented_in_training": False,
        "offline_scale_check_only": True,
        "proposal_version": 2,
        "source": str(args.trace),
        "source_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "formula": "-weight * mean(1 - topK(existing_pad_approach_quality)); compare K=3 and K=4",
        "quality_definition": "1 - clip(norm(pad_center_world + normal_world * local_fruit_radius - fruit_center_world)/0.06, max=3)",
        "provisional_first_weight_if_full_run_fails": .025,
        "provisional_first_pad_count_if_full_run_fails": 4,
        "selection_reason": "Top3 provides no incentive to recover a fourth near pad. Top4 softly encourages the redundant fourth contact while leaving any three-pad support and temporary release legal. This is a design choice, not demonstrated learning success.",
        "not_a_paper_formula": True,
        "episodes": active.shape[1],
        "dt": dt,
        "late_nonterminal_environment_steps": int(late.sum()),
        "late_selection_frequency_thumb_index_middle_ring_little_by_pad_count": selections,
        "ideal_release_checks_at_weight0025": releases,
        "finger_permutation_invariant": True,
        "top3_four_to_three_ideal_quality1_pads_has_no_extra_candidate_cost": True,
        "two_near_pads_remains_finite_cost": True,
        "rows": rows,
        "limitations": [
            "These are costs on an existing fixed trajectory, not counterfactual policy performance or learning evidence.",
            "Nonpositive shaping avoids an added positive standing-contact bonus but does not exclude a stationary local optimum.",
            "A soft topK cost does not prove feasible recontact, a learned gait, or a complete revolution.",
            "The top3 zero-extra-cost release check uses ideal equal-quality pads. Top4 pays a finite release cost, and real unequal qualities also change the cost.",
            "Existing stability, physical failure, pressure and final success gates must not be relaxed to make this candidate pass.",
            "No actor inputs, core network, PPO, actuator mapping, sensor geometry or current recipe are changed.",
        ],
        "primary_sources_rechecked": [
            "https://arxiv.org/html/2303.10880v4#S4.SS1.SSS3",
            "https://arxiv.org/html/2405.07391v2#S3.SS1",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
