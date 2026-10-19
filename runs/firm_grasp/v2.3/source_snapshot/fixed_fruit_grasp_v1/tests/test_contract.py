"""纯 CPU 合同测试：不导入 torch / Isaac / USD。"""

from __future__ import annotations

import math

import pytest

from research.fixed_fruit_grasp_v1 import contract as C


def test_holding_stages_do_not_require_lift_or_height() -> None:
    """holding-only 阶段不得暗含 30 mm 提举要求，也不得计入总目标。"""

    for stage in ("contact", "hold"):
        spec = C.stage_spec(stage)
        assert spec.min_height_gain_m == 0.0
        assert spec.requires_lift_program is False
        assert spec.counts_as_goal is False
        assert spec.success_requires_height_gain is False
        assert spec.goal_status == C.GOAL_STATUS_LIFT_NOT_IMPLEMENTED
        assert "lift" not in spec.success_criterion
    lift = C.stage_spec("lift")
    assert lift.min_height_gain_m == C.MIN_HEIGHT_GAIN_M
    assert lift.requires_lift_program is True
    assert lift.counts_as_goal is True
    assert lift.success_requires_height_gain is True
    assert C.lift_stage_available() is False
    assert C.LIFT_STAGE_STATUS == "NOT_IMPLEMENTED_DECLARED_ONLY"


def test_hold_window_steps_match_declared_seconds() -> None:
    hold = C.stage_spec("hold")
    assert hold.required_unsupported_hold_s == 3.0
    assert hold.required_unsupported_hold_steps == 60
    assert hold.support_release_time_s == pytest.approx(2.0)
    contact = C.stage_spec("contact")
    assert contact.required_unsupported_hold_steps == 20
    assert contact.support_release_time_s == pytest.approx(3.0)


def test_unknown_stage_and_reason_code_fail_closed() -> None:
    with pytest.raises(C.UnknownStageError):
        C.stage_spec("grasp")
    with pytest.raises(ValueError):
        C.reason_name(99)
    assert C.reason_name(1) == "success"
    assert C.reason_name(11) == "joint_effort"


def test_reason_codes_are_unique_and_terminal_codes_present() -> None:
    names = list(C.REASON_CODES.values())
    assert len(names) == len(set(names))
    for required in (
        "success",
        "timeout",
        "drop",
        "overload",
        "nonfinite",
        "penetration",
        "joint_limit",
        "self_collision",
        "instability",
        "joint_effort",
    ):
        assert required in names


def test_fruit_mass_constant_matches_declared_geometry() -> None:
    expected = (
        4.0 / 3.0 * math.pi * C.FRUIT_RADIUS_M**3 * C.FRUIT_DENSITY_KG_M3
    )
    assert C.FRUIT_MASS_KG == pytest.approx(expected, rel=1.0e-12)
    assert 2.0 * C.FRUIT_RADIUS_M == pytest.approx(0.067)


def test_lift_program_is_slow_and_does_not_overshoot() -> None:
    program = C.LIFT_PROGRAM
    assert program.speed_m_s <= 0.02
    assert program.travel_m >= C.MIN_HEIGHT_GAIN_M
    assert program.height_at_step(0) == 0.0
    assert program.height_at_step(-5) == 0.0
    assert program.height_at_step(program.duration_steps) == pytest.approx(
        program.travel_m
    )
    assert program.height_at_step(program.duration_steps + 100) == pytest.approx(
        program.travel_m
    )
    assert program.to_json()["executed"] is False


def test_summarize_records_reports_full_denominator() -> None:
    rows = [
        {
            "stage": "hold",
            "success": True,
            "goal_success": False,
            "reason": "success",
            "contact_fraction": 1.0,
            "max_unsupported_hold_s": 3.0,
            "max_height_gain_m": 0.004,
        },
        {
            "stage": "hold",
            "success": False,
            "goal_success": False,
            "reason": "drop",
            "contact_fraction": 0.0,
            "max_unsupported_hold_s": 0.1,
            "max_height_gain_m": -0.02,
        },
    ]
    summary = C.summarize_records(rows)
    assert summary["records"] == 2
    assert summary["success_count"] == 1
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["goal_success_count"] == 0
    assert summary["reason_counts"]["drop"] == 1
    assert summary["stage_counts"] == {"hold": 2}
    assert summary["best_max_unsupported_hold_s"] == pytest.approx(3.0)
    assert summary["goal_status"] == C.GOAL_STATUS_LIFT_NOT_IMPLEMENTED


def test_json_safe_rejects_tensors_and_flags_non_finite() -> None:
    assert C.json_safe(float("nan")) is None
    assert C.json_safe({"a": [1, 2.5, True]}) == {"a": [1, 2.5, True]}
    with pytest.raises(TypeError):
        C.json_safe(object())


def test_record_schema_lists_required_runtime_fields() -> None:
    fields = set(C.EPISODE_RECORD_FIELDS)
    for name in (
        "env_id",
        "episode_id",
        "success",
        "contact_fraction",
        "max_unsupported_hold_s",
        "max_height_gain_m",
        "reason",
        "goal_success",
        "goal_status",
        "finger_contact_fraction",
        "max_applied_effort_nm",
        "stage_requires_lift_program",
    ):
        assert name in fields
