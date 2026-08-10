"""Static validation for the inactive Physics-v2 experiment matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SIMULATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_MATRIX_PATH = (
    SIMULATION_ROOT / "config/tomato_pick_physics_v2_experiment_matrix.json"
)


def load_experiment_matrix_v2(
    path: Path = DEFAULT_EXPERIMENT_MATRIX_PATH,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_experiment_matrix_v2(value)
    return value


def validate_experiment_matrix_v2(value: dict[str, Any]) -> None:
    if value.get("schema_version") != 1:
        raise ValueError("experiment matrix schema_version must be 1")
    if value.get("status") != "inactive_cpu_authored_protocol":
        raise ValueError("experiment matrix must remain inactive")
    if value.get("task_id") != "LinkAgent-TomatoPick-Physics-v2":
        raise ValueError("unexpected experiment task_id")
    protocol = value.get("common_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("common_protocol is required")
    seeds = protocol.get("evaluation_seeds")
    if not isinstance(seeds, list) or len(set(seeds)) < 3:
        raise ValueError("independent evaluation requires at least three unique seeds")
    if protocol.get("training_metrics_can_name_best") is not False:
        raise ValueError("training metrics cannot name a best checkpoint")
    if protocol.get("real_hardware_in_scope") is not False:
        raise ValueError("real hardware must remain out of scope")

    tiers = value.get("resource_tiers")
    if not isinstance(tiers, list) or len(tiers) < 3:
        raise ValueError("smoke, numerical-probe and bounded-learning tiers are required")
    environments = [tier.get("num_environments") for tier in tiers]
    transitions = [
        tier.get("num_environments", 0)
        * tier.get("updates", 0)
        * protocol.get("policy_steps_per_environment", 0)
        for tier in tiers
    ]
    if environments[:3] != [64, 256, 256]:
        raise ValueError("resource tiers must preserve the qualified 64/256/256 layout")
    if transitions != sorted(transitions):
        raise ValueError("resource tiers must have nondecreasing transition budgets")
    if any(tier.get("authorizes_next_tier") is not False for tier in tiers):
        raise ValueError("resource results require an explicit decision between tiers")

    evaluation = value.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation protocol is required")
    required_metrics = set(evaluation.get("required_metrics", []))
    necessary = {
        "success_rate",
        "retained_rate",
        "drop_rate",
        "excess_force_rate",
        "peak_fingertip_force_n",
        "damage_proxy_mean",
        "worst_bucket_success_rate",
    }
    if not necessary.issubset(required_metrics):
        raise ValueError("evaluation omits safety or robustness metrics")
    buckets = evaluation.get("robustness_buckets")
    if not isinstance(buckets, list) or len(set(buckets)) < 6:
        raise ValueError("evaluation requires distinct physical robustness buckets")

    experiments = value.get("experiments")
    if not isinstance(experiments, list) or len(experiments) < 5:
        raise ValueError("reference plus at least four ablations are required")
    identifiers = [item.get("id") for item in experiments]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("experiment identifiers must be unique")
    references = [item for item in experiments if item.get("kind") == "reference"]
    if len(references) != 1 or references[0].get("changed_parameter_families") != []:
        raise ValueError("exactly one unchanged reference experiment is required")
    for item in experiments:
        changed = item.get("changed_parameter_families")
        if item.get("kind") == "single_family_ablation" and (
            not isinstance(changed, list) or len(changed) != 1
        ):
            raise ValueError("each ablation must change exactly one parameter family")
        if not item.get("question"):
            raise ValueError("each experiment requires one explicit question")

    guards = value.get("fail_closed_rules")
    if not isinstance(guards, dict) or not guards:
        raise ValueError("fail_closed_rules are required")
    if any(item is not False for item in guards.values()):
        raise ValueError("all experiment authorization flags must remain false")


__all__ = [
    "DEFAULT_EXPERIMENT_MATRIX_PATH",
    "load_experiment_matrix_v2",
    "validate_experiment_matrix_v2",
]
