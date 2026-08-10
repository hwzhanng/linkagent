from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from script.prepare_tomato_physics_v2_gpu_resume_manifest import build_resume_manifest
from simulation.tomato_pick.experiment_v2 import (
    load_experiment_matrix_v2,
    validate_experiment_matrix_v2,
)


def test_experiment_matrix_is_single_factor_multi_seed_and_fail_closed() -> None:
    value = load_experiment_matrix_v2()
    assert len(value["common_protocol"]["evaluation_seeds"]) == 3
    assert value["common_protocol"]["training_metrics_can_name_best"] is False
    ablations = [item for item in value["experiments"] if item["kind"] == "single_family_ablation"]
    assert len(ablations) >= 4
    assert all(len(item["changed_parameter_families"]) == 1 for item in ablations)
    assert all(flag is False for flag in value["fail_closed_rules"].values())


def test_experiment_matrix_rejects_multifactor_ablation_and_metric_release() -> None:
    value = load_experiment_matrix_v2()
    multifactor = deepcopy(value)
    multifactor["experiments"][1]["changed_parameter_families"] = ["tactile", "actuator"]
    with pytest.raises(ValueError, match="exactly one"):
        validate_experiment_matrix_v2(multifactor)
    unsafe = deepcopy(value)
    unsafe["common_protocol"]["training_metrics_can_name_best"] = True
    with pytest.raises(ValueError, match="cannot name"):
        validate_experiment_matrix_v2(unsafe)


def test_resume_manifest_never_authorizes_gpu_training_without_external_checks(tmp_path: Path) -> None:
    report = build_resume_manifest()
    assert report["gpu_preflight_authorized"] is False
    assert report["ppo_training_authorized"] is False
    assert report["hardware_connection_authorized"] is False
    assert report["cpu_qualification"]["passed"] is False
    assert len(report["external_checks_pending"]) >= 6
    assert report["static_checks"]["data_protocol_is_valid_and_inactive"] is True
