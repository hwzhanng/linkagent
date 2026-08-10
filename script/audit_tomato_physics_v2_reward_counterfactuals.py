"""CPU-only counterfactual audit of Physics-v2 success and curriculum design."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from simulation.tomato_pick.curriculum_v2 import (
    DEFAULT_CURRICULUM_PATH,
    load_curriculum_v2,
)
from simulation.tomato_pick.physics_v2_scenarios import physical_task_predicates
from simulation.tomato_pick.reward import tomato_pick_reward


SCENARIOS: tuple[dict[str, object], ...] = (
    {
        "id": "valid_gentle_opposed_hold",
        "force_n": [4.0, 4.0, 1.0, 0.0, 0.0],
        "detached": True,
        "distance_m": 0.03,
        "speed_m_s": 0.02,
        "hold_steps": 20,
        "expected_success": True,
    },
    {
        "id": "reference_pose_without_contact",
        "force_n": [0.0, 0.0, 0.0, 0.0, 0.0],
        "detached": True,
        "distance_m": 0.03,
        "speed_m_s": 0.02,
        "hold_steps": 20,
        "expected_success": False,
    },
    {
        "id": "stale_tactile_but_zero_physical_force",
        "force_n": [0.0, 0.0, 0.0, 0.0, 0.0],
        "detached": True,
        "distance_m": 0.03,
        "speed_m_s": 0.02,
        "hold_steps": 20,
        "expected_success": False,
    },
    {
        "id": "same_side_contact_without_thumb",
        "force_n": [0.0, 4.0, 4.0, 4.0, 0.0],
        "detached": True,
        "distance_m": 0.03,
        "speed_m_s": 0.02,
        "hold_steps": 20,
        "expected_success": False,
    },
    {
        "id": "thumb_only_contact",
        "force_n": [4.0, 0.0, 0.0, 0.0, 0.0],
        "detached": True,
        "distance_m": 0.03,
        "speed_m_s": 0.02,
        "hold_steps": 20,
        "expected_success": False,
    },
    {
        "id": "supported_but_stem_attached",
        "force_n": [4.0, 4.0, 0.0, 0.0, 0.0],
        "detached": False,
        "distance_m": 0.03,
        "speed_m_s": 0.02,
        "hold_steps": 20,
        "expected_success": False,
    },
    {
        "id": "detached_but_too_far",
        "force_n": [4.0, 4.0, 0.0, 0.0, 0.0],
        "detached": True,
        "distance_m": 0.20,
        "speed_m_s": 0.02,
        "hold_steps": 20,
        "expected_success": False,
    },
    {
        "id": "detached_but_high_speed",
        "force_n": [4.0, 4.0, 0.0, 0.0, 0.0],
        "detached": True,
        "distance_m": 0.03,
        "speed_m_s": 0.50,
        "hold_steps": 20,
        "expected_success": False,
    },
    {
        "id": "insufficient_hold_duration",
        "force_n": [4.0, 4.0, 0.0, 0.0, 0.0],
        "detached": True,
        "distance_m": 0.03,
        "speed_m_s": 0.02,
        "hold_steps": 19,
        "expected_success": False,
    },
    {
        "id": "destructive_squeeze",
        "force_n": [20.0, 20.0, 20.0, 20.0, 20.0],
        "detached": True,
        "distance_m": 0.03,
        "speed_m_s": 0.02,
        "hold_steps": 20,
        "expected_success": False,
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _terminal_reward(*, success: bool = False, dropped: bool = False, excess: bool = False, timeout: bool = False) -> float:
    zero = torch.zeros(1, dtype=torch.float64)
    reward, *_ = tomato_pick_reward(
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
        success=torch.tensor([success]),
        dropped=torch.tensor([dropped]),
        excess_force=torch.tensor([excess]),
        out_of_bounds=torch.zeros(1, dtype=torch.bool),
        timed_out=torch.tensor([timeout]),
        stem_detached=torch.tensor([success or dropped]),
        alignment_alpha=1.0,
    )
    return float(reward)


def build_reward_audit() -> dict[str, object]:
    curriculum = load_curriculum_v2()
    observed = []
    for scenario in SCENARIOS:
        result = physical_task_predicates(
            torch.tensor([scenario["force_n"]], dtype=torch.float64),
            stem_detached=torch.tensor([scenario["detached"]], dtype=torch.bool),
            center_distance_m=torch.tensor([scenario["distance_m"]], dtype=torch.float64),
            relative_speed_m_s=torch.tensor([scenario["speed_m_s"]], dtype=torch.float64),
            hold_steps=torch.tensor([scenario["hold_steps"]], dtype=torch.long),
            maximum_fingertip_force_n=15.0,
        )
        success = bool(result["success"].item())
        observed.append(
            {
                **scenario,
                "observed_secure_support": bool(result["secure_support"].item()),
                "observed_kinematic_retained": bool(result["kinematic_retained"].item()),
                "observed_within_force_limit": bool(result["within_force_limit"].item()),
                "observed_success": success,
                "passed": success is scenario["expected_success"],
            }
        )
    terminal = {
        "success": _terminal_reward(success=True),
        "drop": _terminal_reward(dropped=True),
        "excess_force": _terminal_reward(excess=True),
        "timeout": _terminal_reward(timeout=True),
    }
    checks = {
        "all_counterfactual_expectations_hold": all(item["passed"] for item in observed),
        "valid_success_has_positive_terminal_reward": terminal["success"] > 0.0,
        "drop_is_penalized": terminal["drop"] < 0.0,
        "excess_force_is_penalized": terminal["excess_force"] < 0.0,
        "timeout_is_penalized": terminal["timeout"] < 0.0,
        "curriculum_is_inactive_draft": curriculum["status"] == "draft_not_runtime_active",
        "tennis_demo_is_not_a_policy_prior": curriculum["grasp_contract"]["tennis_ball_demo_used_as_prior"] is False,
        "reference_pose_is_not_required": curriculum["grasp_contract"]["reference_joint_pose_required"] is False,
        "finger_count_is_policy_selected": curriculum["grasp_contract"]["fixed_active_finger_count"] is False,
    }
    return {
        "schema_version": 1,
        "audit_id": "tomato_pick_physics_v2_reward_counterfactuals",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "pure CPU predicates and reward formulas; no Isaac, GPU, PPO, or hardware",
        "source_identity": {
            "curriculum": str(DEFAULT_CURRICULUM_PATH.relative_to(ROOT)),
            "curriculum_sha256": _sha256(DEFAULT_CURRICULUM_PATH),
        },
        "counterfactuals": observed,
        "aligned_terminal_rewards": terminal,
        "curriculum_stage_ids": [stage["id"] for stage in curriculum["stages"]],
        "checks": checks,
        "passed": all(checks.values()),
        "runtime_activation_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_reward_audit()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["passed"]}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
