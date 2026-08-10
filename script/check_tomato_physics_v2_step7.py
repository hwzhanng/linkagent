"""Run scripted wrench-oracle and anti-reward-hacking gates for Physics-v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from simulation.tomato_pick.physics_v1 import (
    secure_kinematic_break_credit_gate,
    secure_support_credit_gate,
)
from simulation.tomato_pick.physics_v2 import load_physics_v2_profile
from simulation.tomato_pick.physics_v2_scenarios import (
    physical_task_predicates,
    run_wrench_level_pick_scenario,
)
from simulation.tomato_pick.reward import tomato_pick_reward


def terminal_reward(*, dropped: bool = False, excess_force: bool = False, timed_out: bool = False) -> float:
    zero = torch.zeros(1)
    result = tomato_pick_reward(
        reach_score=zero,
        previous_best_reach=zero,
        grasp_score=zero,
        previous_best_grasp=zero,
        opposed_score=zero,
        previous_best_opposed=zero,
        stem_load_score=zero,
        previous_best_stem_load=zero,
        stable_detach_score=zero,
        previous_best_stable_detach=zero,
        retention_score=zero,
        previous_best_retention=zero,
        break_grasped_score=zero,
        success=torch.zeros(1, dtype=torch.bool),
        dropped=torch.tensor([dropped]),
        excess_force=torch.tensor([excess_force]),
        out_of_bounds=torch.zeros(1, dtype=torch.bool),
        timed_out=torch.tensor([timed_out]),
        stem_detached=torch.tensor([dropped]),
        alignment_alpha=1.0,
    )
    return float(result[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profile = load_physics_v2_profile()
    scripted = {
        name: run_wrench_level_pick_scenario(
            strength_quantile=quantile, seed=args.seed, sample_count=8192
        )
        for name, quantile in (("weak", 0.05), ("median", 0.50), ("strong", 0.95))
    }

    secure = torch.tensor([True])
    insecure = torch.tensor([False])
    retained_motion = torch.tensor([True])
    high_speed_motion = torch.tensor([False])
    raw_break = torch.ones(1)
    unsupported_load_credit = secure_support_credit_gate(raw_break, insecure)
    unsupported_break_credit = secure_kinematic_break_credit_gate(
        raw_break, insecure, retained_motion
    )
    high_speed_break_credit = secure_kinematic_break_credit_gate(
        raw_break, secure, high_speed_motion
    )

    same_side = physical_task_predicates(
        torch.tensor([[0.0, 4.0, 4.0, 0.0, 0.0]]),
        stem_detached=torch.ones(1, dtype=torch.bool),
        center_distance_m=torch.tensor([0.03]),
        relative_speed_m_s=torch.tensor([0.02]),
        hold_steps=torch.tensor([20]),
    )
    stale_tactile = physical_task_predicates(
        torch.zeros((1, 5)),
        stem_detached=torch.ones(1, dtype=torch.bool),
        center_distance_m=torch.tensor([0.03]),
        relative_speed_m_s=torch.tensor([0.02]),
        hold_steps=torch.tensor([20]),
    )
    detached_then_drop = physical_task_predicates(
        torch.tensor([[4.0, 4.0, 4.0, 0.0, 0.0]]),
        stem_detached=torch.ones(1, dtype=torch.bool),
        center_distance_m=torch.tensor([0.20]),
        relative_speed_m_s=torch.tensor([0.50]),
        hold_steps=torch.tensor([20]),
    )
    excessive_force = physical_task_predicates(
        torch.tensor([[20.0, 20.0, 0.0, 0.0, 0.0]]),
        stem_detached=torch.ones(1, dtype=torch.bool),
        center_distance_m=torch.tensor([0.03]),
        relative_speed_m_s=torch.tensor([0.02]),
        hold_steps=torch.tensor([20]),
    )
    checks = {
        "weak_median_strong_detach_retain_success": all(
            item["detached"] and item["retained"] and item["success"]
            for item in scripted.values()
        ),
        "one_substep_impulse_never_detaches": all(
            not item["one_substep_broken"] for item in scripted.values()
        ),
        "scripted_cases_use_full_supported_break_credit": all(
            item["break_credit"] == 1.0 for item in scripted.values()
        ),
        "scripted_gentle_contact_has_zero_damage_score": all(
            item["fruit_damage_score"] == 0.0 for item in scripted.values()
        ),
        "unsupported_pull_has_zero_load_credit": float(unsupported_load_credit) == 0.0,
        "unsupported_fracture_has_zero_break_credit": float(unsupported_break_credit) == 0.0,
        "high_speed_fracture_has_zero_break_credit": float(high_speed_break_credit) == 0.0,
        "same_side_contact_cannot_succeed": not bool(same_side["success"].item()),
        "stale_tactile_without_physical_force_cannot_succeed": not bool(stale_tactile["success"].item()),
        "detached_then_dropped_cannot_succeed": not bool(detached_then_drop["success"].item()),
        "destructive_force_cannot_succeed": not bool(excessive_force["success"].item()),
        "drop_and_excess_force_are_penalized": terminal_reward(dropped=True) < 0.0 and terminal_reward(excess_force=True) < 0.0,
        "zero_action_timeout_is_not_rewarded": terminal_reward(timed_out=True) < 0.0,
        "training_remains_blocked": profile["implementation_boundary"]["training_allowed"] is False,
        "robot_kinematic_feasibility_is_not_claimed": True,
    }
    report = {
        "schema_version": 1,
        "gate_id": "tomato_pick_physics_v2_step7_wrench_oracle",
        "scope": "pure_wrench_signal_and_reward_logic_no_robot_kinematics_no_physx",
        "passed": all(checks.values()),
        "checks": checks,
        "scripted_cases": scripted,
        "counterfactual_metrics": {
            "unsupported_load_credit": float(unsupported_load_credit),
            "unsupported_break_credit": float(unsupported_break_credit),
            "high_speed_break_credit": float(high_speed_break_credit),
            "drop_terminal_reward": terminal_reward(dropped=True),
            "excess_force_terminal_reward": terminal_reward(excess_force=True),
            "zero_action_timeout_reward": terminal_reward(timed_out=True),
        },
        "limitations": [
            "The scripted cases are wrench-level signal-chain oracles, not IK trajectories.",
            "Reachability, articulation motion, collision response, and PhysX contacts remain unverified until the exact Isaac runtime is available.",
            "Fruit damage remains a diagnostic numerical guard, not a calibrated bruise model.",
        ],
        "next_stage": "exact_runtime_integrated_gates_then_64_env_and_256x4_ppo",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
