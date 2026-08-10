from __future__ import annotations

from script.audit_tomato_physics_v2_parameter_sensitivity import (
    build_sensitivity_audit,
    render_markdown,
)


def test_cpu_sensitivity_audit_is_deterministic_and_passes() -> None:
    first = build_sensitivity_audit(samples=2048, seed=91)
    second = build_sensitivity_audit(samples=2048, seed=91)
    assert first["passed"] is True
    assert first["devices"] == ["cpu"]
    assert first["distributions"] == second["distributions"]
    assert first["screening_correlations"] == second["screening_correlations"]
    assert first["stem_mode_feasibility"] == second["stem_mode_feasibility"]
    assert first["material_bank_coverage"]["observed_bin_count"] >= 12


def test_sensitivity_report_labels_noncausal_and_uncalibrated_scope() -> None:
    report = build_sensitivity_audit(samples=2048, seed=92)
    markdown = render_markdown(report)
    assert "not tomato calibration" in markdown
    assert "not causal sensitivity indices" in markdown
    assert report["calibration_status"] == "hardware_and_cultivar_measurements_pending"
