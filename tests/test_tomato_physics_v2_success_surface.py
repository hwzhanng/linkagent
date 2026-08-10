from __future__ import annotations

from script.audit_tomato_physics_v2_success_surface import build_success_surface_audit


def test_success_surface_boundaries_are_deterministic_and_pass() -> None:
    report = build_success_surface_audit()
    assert report["passed"] is True
    assert report["formal_training_authorized"] is False
