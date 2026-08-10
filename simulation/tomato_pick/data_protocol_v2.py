"""Fail-closed validation for the offline Physics-v2 data/evaluation protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PROTOCOL_PATH = (
    ROOT / "config/tomato_pick_physics_v2_data_protocol.json"
)


def load_data_protocol_v2(
    path: Path = DEFAULT_DATA_PROTOCOL_PATH,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_data_protocol_v2(value)
    return value


def validate_data_protocol_v2(value: dict[str, Any]) -> None:
    if value.get("schema_version") != 1:
        raise ValueError("data protocol schema_version must be 1")
    if value.get("protocol_id") != "tomato_pick_physics_v2_data_protocol_v1":
        raise ValueError("unexpected Physics-v2 data protocol id")
    if value.get("status") != "cpu_authored_inactive":
        raise ValueError("data protocol must remain CPU-authored and inactive")
    if value.get("task_id") != "LinkAgent-TomatoPick-Physics-v2":
        raise ValueError("data protocol task id mismatch")

    splits = value.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("data protocol must define train, validation, and test splits")
    seed_sets: dict[str, set[int]] = {}
    for name, split in splits.items():
        seeds = split.get("seeds") if isinstance(split, dict) else None
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError(f"{name} split seeds must be unique integers")
        if split.get("checkpoint_selection_allowed") is not (name == "test"):
            raise ValueError("only the independent test split may select checkpoints")
        seed_sets[name] = set(seeds)
    if seed_sets["train"] & seed_sets["validation"]:
        raise ValueError("train and validation seeds overlap")
    if seed_sets["train"] & seed_sets["test"]:
        raise ValueError("train and test seeds overlap")
    if seed_sets["validation"] & seed_sets["test"]:
        raise ValueError("validation and test seeds overlap")
    if seed_sets["train"] != {43} or seed_sets["validation"] != {17}:
        raise ValueError("data protocol must preserve the frozen train/validation seeds")
    if len(seed_sets["test"]) < 2:
        raise ValueError("test split requires at least two independent seeds")

    derivation = value.get("seed_derivation")
    if not isinstance(derivation, dict) or derivation.get("collision_check_required") is not True:
        raise ValueError("seed derivation must require collision checking")
    if derivation.get("episode_index_origin") != 0:
        raise ValueError("episode indices must start at zero")

    record = value.get("episode_record")
    if not isinstance(record, dict):
        raise ValueError("episode_record schema is required")
    required_fields = record.get("required_fields")
    if not isinstance(required_fields, list) or len(required_fields) != len(set(required_fields)):
        raise ValueError("episode record fields must be unique")
    for field in ("episode_id", "split", "seed", "robustness_bucket", "success", "peak_palm_force_n"):
        if field not in required_fields:
            raise ValueError(f"episode record is missing {field}")
    booleans = record.get("boolean_fields")
    numerics = record.get("finite_numeric_fields")
    if not isinstance(booleans, list) or not set(booleans) <= set(required_fields):
        raise ValueError("boolean record fields are invalid")
    if not isinstance(numerics, list) or not set(numerics) <= set(required_fields):
        raise ValueError("numeric record fields are invalid")

    buckets = value.get("robustness_buckets")
    if not isinstance(buckets, list) or len(buckets) < 6 or len(set(buckets)) != len(buckets):
        raise ValueError("at least six distinct robustness buckets are required")
    aggregation = value.get("aggregation")
    if not isinstance(aggregation, dict) or aggregation.get("primary_unit") != "episode":
        raise ValueError("metrics must aggregate at episode granularity")
    metrics = aggregation.get("required_metrics")
    if not isinstance(metrics, list):
        raise ValueError("aggregation required_metrics are missing")
    required_metrics = {
        "success_rate", "retained_rate", "detach_rate", "drop_rate",
        "excess_force_rate", "timeout_rate", "peak_fingertip_force_n",
        "peak_palm_force_n", "damage_proxy_mean", "damage_proxy_peak",
        "worst_bucket_success_rate",
    }
    if not required_metrics <= set(metrics):
        raise ValueError("aggregation omits safety or robustness metrics")
    if aggregation.get("rate_denominator") != "episode_count":
        raise ValueError("rates must use episode_count as denominator")

    policy = value.get("checkpoint_policy")
    if not isinstance(policy, dict):
        raise ValueError("checkpoint policy is required")
    if policy.get("training_metrics_may_nominate") is not True:
        raise ValueError("training metrics may nominate a candidate")
    if policy.get("training_metrics_may_release") is not False:
        raise ValueError("training metrics may not release a checkpoint")
    if policy.get("release_requires_independent_test") is not True:
        raise ValueError("release must require independent test evaluation")
    if policy.get("minimum_test_seeds") != 2 or policy.get("hardware_deployment_allowed") is not False:
        raise ValueError("checkpoint release policy is not fail-closed")


__all__ = [
    "DEFAULT_DATA_PROTOCOL_PATH",
    "load_data_protocol_v2",
    "validate_data_protocol_v2",
]
