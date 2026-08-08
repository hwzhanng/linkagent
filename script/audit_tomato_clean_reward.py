"""Audit Clean-v5 reward ordering, bounds, and curriculum behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.tomato_pick.reward import (  # noqa: E402
    CURRICULUM_POINTS,
    alignment_alpha_for_iteration,
    tomato_pick_reward,
)


GAMMA = 0.99
SCORE_NAMES = (
    "reach",
    "grasp",
    "opposed",
    "stem_load",
    "stable_detach",
    "retention",
)


def _flag(value: bool) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.bool)


def _value(value: float) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.float64)


def _trajectory(steps: list[dict[str, float | bool]], *, alpha: float) -> float:
    best = {name: 0.0 for name in SCORE_NAMES}
    total = 0.0
    discount = 1.0
    for step in steps:
        result = tomato_pick_reward(
            reach_score=_value(float(step.get("reach", best["reach"]))),
            previous_best_reach=_value(best["reach"]),
            grasp_score=_value(float(step.get("grasp", best["grasp"]))),
            previous_best_grasp=_value(best["grasp"]),
            opposed_score=_value(float(step.get("opposed", best["opposed"]))),
            previous_best_opposed=_value(best["opposed"]),
            stem_load_score=_value(float(step.get("stem_load", best["stem_load"]))),
            previous_best_stem_load=_value(best["stem_load"]),
            stable_detach_score=_value(
                float(step.get("stable_detach", best["stable_detach"]))
            ),
            previous_best_stable_detach=_value(best["stable_detach"]),
            retention_score=_value(float(step.get("retention", best["retention"]))),
            previous_best_retention=_value(best["retention"]),
            break_grasped_score=_value(float(step.get("break_grasped", 0.0))),
            success=_flag(bool(step.get("success", False))),
            dropped=_flag(bool(step.get("dropped", False))),
            excess_force=_flag(bool(step.get("excess", False))),
            out_of_bounds=_flag(bool(step.get("out_of_bounds", False))),
            timed_out=_flag(bool(step.get("timeout", False))),
            stem_detached=_flag(bool(step.get("detached", False))),
            alignment_alpha=alpha,
        )
        reward, *next_scores, terms = result
        if not torch.isfinite(reward).all():
            raise RuntimeError("non-finite reward in constructed trajectory")
        if not torch.allclose(reward, torch.stack(tuple(terms.values())).sum(dim=0)):
            raise RuntimeError("reward term sum mismatch")
        best.update(zip(SCORE_NAMES, (float(value.item()) for value in next_scores)))
        total += discount * float(reward.item())
        discount *= GAMMA
    return total


def _hold_steps(count: int, *, success: bool = False, excess: bool = False):
    steps: list[dict[str, float | bool]] = [
        {
            "reach": 1.0,
            "grasp": 1.0,
            "opposed": 1.0,
            "stem_load": 1.0,
            "break_grasped": 1.0,
            "detached": True,
        }
    ]
    for index in range(1, count + 1):
        steps.append(
            {
                "opposed": 1.0,
                "stable_detach": 1.0 if index >= 3 else 0.0,
                "retention": index / 20.0,
                "detached": True,
                "success": success and index == 20,
                "excess": excess and index == count,
            }
        )
    if not success and not excess:
        steps.append({"detached": True, "dropped": True})
    return steps


def _returns(alpha: float) -> dict[str, float]:
    wait = [{} for _ in range(159)] + [{"timeout": True}]
    return {
        "safe_wait_timeout": _trajectory(wait, alpha=alpha),
        "single_contact_drop": _trajectory(
            [
                {
                    "reach": 0.8,
                    "grasp": 0.2,
                    "stem_load": 0.3,
                    "detached": True,
                    "dropped": True,
                }
            ],
            alpha=alpha,
        ),
        "two_finger_break_drop": _trajectory(
            [
                {
                    "reach": 1.0,
                    "grasp": 0.6,
                    "stem_load": 0.7,
                    "break_grasped": 0.4,
                    "detached": True,
                    "dropped": True,
                }
            ],
            alpha=alpha,
        ),
        "thumb_opposed_break_drop": _trajectory(
            [
                {
                    "reach": 1.0,
                    "grasp": 1.0,
                    "opposed": 1.0,
                    "stem_load": 1.0,
                    "break_grasped": 1.0,
                    "detached": True,
                    "dropped": True,
                }
            ],
            alpha=alpha,
        ),
        "stable_hold3_drop": _trajectory(_hold_steps(3), alpha=alpha),
        "hold19_drop": _trajectory(_hold_steps(19), alpha=alpha),
        "hold19_excess": _trajectory(_hold_steps(19, excess=True), alpha=alpha),
        "hold20_success": _trajectory(_hold_steps(20, success=True), alpha=alpha),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "logs/audits/tomato_clean_v5/reward_audit.json",
    )
    args = parser.parse_args()

    discovery = _returns(0.0)
    aligned = _returns(1.0)
    schedule = {
        str(iteration): alignment_alpha_for_iteration(iteration)
        for iteration in (0, 29, 100, 250, 400, 600, 800, 1000, 1150, 1300, 1500)
    }
    checks = {
        "discovery_safe_wait_is_negative": discovery["safe_wait_timeout"] < 0.0,
        "discovery_two_finger_beats_collision": (
            discovery["two_finger_break_drop"] > discovery["single_contact_drop"]
        ),
        "discovery_thumb_beats_two_finger": (
            discovery["thumb_opposed_break_drop"]
            > discovery["two_finger_break_drop"]
        ),
        "discovery_first_success_is_best": (
            discovery["hold20_success"] > discovery["hold19_drop"] + 80.0
        ),
        "aligned_collision_drop_is_negative": aligned["single_contact_drop"] < 0.0,
        "aligned_safe_wait_is_exactly_minus_ten": abs(
            aligned["safe_wait_timeout"] + 10.0
        ) < 1.0e-6,
        "aligned_thumb_attempt_beats_waiting": (
            aligned["thumb_opposed_break_drop"] > aligned["safe_wait_timeout"]
        ),
        "aligned_stability_orders_partial_attempts": (
            aligned["hold19_drop"] > aligned["stable_hold3_drop"]
        ),
        "aligned_excess_worse_than_drop": (
            aligned["hold19_excess"] < aligned["hold19_drop"]
        ),
        "aligned_success_has_large_margin": (
            aligned["hold20_success"] > aligned["hold19_drop"] + 150.0
        ),
        "alignment_stays_zero_through_update250": (
            schedule["0"] == schedule["29"] == schedule["100"] == schedule["250"] == 0.0
        ),
        "alignment_reaches_one_by_update1300": (
            schedule["1300"] == schedule["1500"] == 1.0
        ),
    }
    report = {
        "schema_version": 2,
        "reward_version": "clean-v5",
        "passed": all(checks.values()),
        "checks": checks,
        "curriculum_points": CURRICULUM_POINTS,
        "alignment_schedule": schedule,
        "discounted_returns": {
            "discovery_alpha_0": discovery,
            "aligned_alpha_1": aligned,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
