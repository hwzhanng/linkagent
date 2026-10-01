from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from simulation.ppo_tomato_grasp_v3.scenario import (
    ScenarioError,
    canonical_hash,
    derive_run_scenario,
    load_scenario,
    merge_scenario_override,
    validate_scenario,
)


SCENARIO = Path(__file__).resolve().parents[1] / "scenarios" / "v3_large90_nominal.json"


def test_nominal_scenario_is_valid_and_stable() -> None:
    scenario, digest = load_scenario(SCENARIO)
    reordered = json.loads(json.dumps(scenario, sort_keys=False))
    assert scenario["name"] == "v3_large90_nominal"
    assert len(digest) == 64
    assert canonical_hash(reordered) == digest


def test_runtime_override_changes_identity() -> None:
    scenario, base_digest = load_scenario(SCENARIO)
    run_a, digest_a = derive_run_scenario(scenario, seed=1, num_envs=8, steps=700, purpose="m0")
    run_b, digest_b = derive_run_scenario(scenario, seed=2, num_envs=8, steps=700, purpose="m0")
    assert digest_a != base_digest
    assert digest_b != digest_a
    assert run_a["runtime"]["seed"] == 1


def test_hardware_and_strict_shortcuts_are_rejected() -> None:
    scenario, _ = load_scenario(SCENARIO)
    hardware = deepcopy(scenario)
    hardware["hardware_command_allowed"] = True
    with pytest.raises(ScenarioError, match="hardware_command_allowed"):
        validate_scenario(hardware)
    shortcut = deepcopy(scenario)
    shortcut["strict"]["require_current_physical_five_digit_contact"] = False
    with pytest.raises(ScenarioError, match="physical five-digit"):
        validate_scenario(shortcut)


def test_target_action_shape_is_enforced() -> None:
    scenario, _ = load_scenario(SCENARIO)
    scenario["controller"]["target_action"] = scenario["controller"]["target_action"][:-1]
    with pytest.raises(ScenarioError, match="14 values"):
        validate_scenario(scenario)


def test_collision_source_contract_rejects_mixed_proxy_state() -> None:
    scenario, _ = load_scenario(SCENARIO)
    scenario["contact_geometry"]["fingertip_proxy_physical"] = False
    with pytest.raises(ScenarioError, match="authored_proxy"):
        validate_scenario(scenario)


def test_declared_override_is_hashed_and_unknown_fields_fail() -> None:
    scenario, digest = load_scenario(SCENARIO)
    changed, changed_digest = merge_scenario_override(
        scenario, {"controller": {"thumb_contact_latch_steps": 20}}
    )
    assert changed["controller"]["thumb_contact_latch_steps"] == 20
    assert changed_digest != digest
    with pytest.raises(ScenarioError, match="unknown field"):
        merge_scenario_override(scenario, {"controller": {"invented_shortcut": True}})
