from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from simulation.tomato_pick.calibration_v2 import (
    fit_calibration_proposal,
    validate_calibration_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs/tomato_pick/templates/tomato_physics_v2_calibration_dataset.template.json"


def _measured_dataset() -> dict:
    axes = [0.030, 0.029, 0.028]
    mass = 4.0 / 3.0 * math.pi * math.prod(axes) * 980.0
    specimens = []
    for index in range(20):
        surface = "dry" if index % 2 == 0 else "wet"
        slip = 30.0 if surface == "dry" else 20.0
        drop = 0.10 if index % 4 < 2 else 0.20
        stem = (
            {"mode": "axial", "peak_force_n": 6.0 + 0.05 * index, "loading_duration_s": 0.10}
            if index < 10
            else {"mode": "bending", "peak_moment_nm": 0.08 + 0.001 * index, "loading_duration_s": 0.10}
        )
        specimens.append(
            {
                "specimen_id": f"T{index:03d}",
                "cultivar": "synthetic_test_only",
                "ripeness_index_0_1": index / 19.0,
                "axes_m": axes,
                "mass_kg": mass,
                "surface_condition": surface,
                "contact_trials": [
                    {"indentation_m": depth, "force_n": 5000.0 * depth, "loading_speed_m_s": 0.001}
                    for depth in (0.001, 0.002, 0.003, 0.005)
                ],
                "slip_angle_deg": [slip],
                "rebound_trials": [
                    {"drop_height_m": drop, "rebound_height_m": 0.01 * drop}
                ],
                "stem_trials": [stem],
            }
        )
    return {
        "schema_version": 1,
        "dataset_id": "synthetic_unit_test_only",
        "status": "measured_unreviewed",
        "provenance": {
            "operator": "pytest",
            "site": "unit_test",
            "started_at": "2026-08-10T00:00:00Z",
            "instrument_ids": ["synthetic"],
            "notes": "Never use as physical evidence.",
        },
        "units": {
            "length": "m",
            "mass": "kg",
            "force": "N",
            "moment": "N_m",
            "speed": "m_s",
            "angle": "deg",
        },
        "specimens": specimens,
    }


def test_empty_template_is_valid_but_cannot_propose_parameters() -> None:
    value = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    validate_calibration_dataset(value)
    report = fit_calibration_proposal(value)
    assert report["ready_for_human_review"] is False
    assert report["parameter_proposal"] is None
    assert report["automatic_profile_activation_authorized"] is False


def test_synthetic_dataset_recovers_known_cpu_fit_but_stays_fail_closed() -> None:
    report = fit_calibration_proposal(_measured_dataset())
    assert report["ready_for_human_review"] is True
    assert all(report["evidence_gates"].values())
    assert report["summaries"]["effective_stiffness_n_m"]["median"] == pytest.approx(5000.0)
    assert report["summaries"]["density_kg_m3"]["median"] == pytest.approx(980.0)
    assert report["summaries"]["restitution"]["median"] == pytest.approx(0.1)
    assert report["parameter_proposal"]["static_friction"][0] < report["parameter_proposal"]["static_friction"][1]
    assert report["automatic_profile_activation_authorized"] is False
    assert report["ppo_training_authorized"] is False


def test_dataset_rejects_energy_gain_and_nonmonotonic_contact() -> None:
    rebound = _measured_dataset()
    rebound["specimens"][0]["rebound_trials"][0]["rebound_height_m"] = 0.5
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_calibration_dataset(rebound)
    contact = _measured_dataset()
    contact["specimens"][0]["contact_trials"][2]["force_n"] = 0.1
    with pytest.raises(ValueError, match="nondecreasing"):
        validate_calibration_dataset(contact)


def test_measured_status_cannot_be_empty() -> None:
    value = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    value["status"] = "measured_unreviewed"
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_calibration_dataset(value)
