from __future__ import annotations

import json
from copy import deepcopy

import pytest

from script.audit_tomato_physics_v2_contract_parity import (
    build_contract_parity_audit,
)
from simulation.tomato_pick.config import load_object
from simulation.tomato_pick.physics_v2 import (
    PHYSICS_V2_PROFILE_PATH,
    load_physics_v2_profile,
)


def test_thresholds_and_explicit_palm_support_semantics_match() -> None:
    report = build_contract_parity_audit()
    assert report["passed_documentation_and_fail_closed_gate"] is True
    assert report["threshold_parity_achieved"] is True
    assert report["exact_success_semantic_parity_achieved"] is True
    assert report["thresholds"]["maximum_center_distance_m"] == 0.06
    assert report["thresholds"]["maximum_relative_speed_m_s"] == 0.10
    assert report["thresholds"]["maximum_fingertip_force_n"] == 8.0
    assert report["thresholds"]["minimum_palm_support_force_n"] == 0.25
    assert report["thresholds"]["palm_support_requires_detach"] is True
    assert report["formal_training_authorized"] is False


def test_profile_loader_rejects_palm_threshold_drift(tmp_path) -> None:
    profile = load_object(PHYSICS_V2_PROFILE_PATH)
    profile["success_contract_parity"]["thresholds"]["minimum_palm_support_force_n"] = 0.30
    path = tmp_path / "drifted_profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="minimum_palm_support_force_n"):
        load_physics_v2_profile(path)


def test_profile_loader_rejects_unresolved_success_contract_status(tmp_path) -> None:
    profile = load_object(PHYSICS_V2_PROFILE_PATH)
    profile = deepcopy(profile)
    profile["success_contract_parity"]["status"] = "legacy_fingertip_only"
    path = tmp_path / "legacy_profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="success contract status"):
        load_physics_v2_profile(path)
