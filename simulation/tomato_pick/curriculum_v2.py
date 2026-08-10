"""Validation helpers for the inactive Physics-v2 curriculum draft."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRICULUM_PATH = (
    SIMULATION_ROOT / "config/tomato_pick_physics_v2_curriculum_draft.json"
)


def load_curriculum_v2(path: Path = DEFAULT_CURRICULUM_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_curriculum_v2(value)
    return value


def validate_curriculum_v2(value: dict[str, Any]) -> None:
    if value.get("schema_version") != 1:
        raise ValueError("curriculum schema_version must be 1")
    if value.get("status") != "draft_not_runtime_active":
        raise ValueError("CPU curriculum must remain an inactive draft")
    guard = value.get("activation_guard")
    if not isinstance(guard, dict) or not guard or not all(guard.values()):
        raise ValueError("every curriculum activation guard must remain enabled")
    grasp = value.get("grasp_contract")
    if not isinstance(grasp, dict):
        raise ValueError("grasp_contract is required")
    required_false = (
        "reference_joint_pose_required",
        "tennis_ball_demo_used_as_prior",
        "fixed_active_finger_count",
    )
    if any(grasp.get(name) is not False for name in required_false):
        raise ValueError("curriculum must not hard-code the tennis demo or a grasp pose")
    if grasp.get("physical_force_not_measured_tactile_gates_success") is not True:
        raise ValueError("physical force must gate task success")
    minimum = grasp.get("minimum_supported_fingertips")
    maximum = grasp.get("maximum_supported_fingertips")
    if not isinstance(minimum, int) or not isinstance(maximum, int) or not 2 <= minimum <= maximum <= 5:
        raise ValueError("supported fingertip range must allow policy-selected enclosure")

    stages = value.get("stages")
    if not isinstance(stages, list) or len(stages) < 5:
        raise ValueError("curriculum requires at least five ordered stages")
    indices = [stage.get("index") for stage in stages]
    if indices != list(range(len(stages))):
        raise ValueError("stage indices must be contiguous from zero")
    identifiers = [stage.get("id") for stage in stages]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("stage identifiers must be unique")
    fractions = [stage.get("randomization_fraction") for stage in stages]
    if any(not isinstance(item, (int, float)) or not 0.0 <= item <= 1.0 for item in fractions):
        raise ValueError("randomization fractions must lie in [0, 1]")
    if fractions != sorted(fractions) or fractions[-1] != 1.0:
        raise ValueError("randomization must grow monotonically to one")
    for stage in stages:
        if not stage.get("single_primary_objective"):
            raise ValueError("each stage requires one primary objective")
        threshold = stage.get("promotion_threshold")
        repetitions = stage.get("consecutive_evaluations")
        if not isinstance(threshold, (int, float)) or not 0.0 < threshold <= 1.0:
            raise ValueError("promotion thresholds must lie in (0, 1]")
        if not isinstance(repetitions, int) or repetitions < 2:
            raise ValueError("promotion must require repeated evaluation")
    rules = value.get("regression_rules")
    if not isinstance(rules, dict) or rules.get("training_rollout_metrics_alone_cannot_promote") is not True:
        raise ValueError("training rollout metrics cannot promote curriculum stages")


__all__ = ["DEFAULT_CURRICULUM_PATH", "load_curriculum_v2", "validate_curriculum_v2"]
