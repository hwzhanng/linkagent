from __future__ import annotations

from script.audit_tomato_physics_v2_contract_parity import (
    build_contract_parity_audit,
)


def test_thresholds_match_but_palm_support_semantics_remain_explicitly_unresolved() -> None:
    report = build_contract_parity_audit()
    assert report["passed_documentation_and_fail_closed_gate"] is True
    assert report["threshold_parity_achieved"] is True
    assert report["exact_success_semantic_parity_achieved"] is False
    assert report["thresholds"]["maximum_center_distance_m"] == 0.06
    assert report["thresholds"]["maximum_relative_speed_m_s"] == 0.10
    assert report["thresholds"]["maximum_fingertip_force_n"] == 8.0
    assert report["formal_training_authorized"] is False
