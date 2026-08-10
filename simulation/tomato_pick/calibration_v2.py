"""Pure-CPU validation and fitting for future tomato calibration datasets.

The output is a review proposal, never an automatic profile update.  Empty
templates are valid documents but cannot produce calibration evidence.
"""

from __future__ import annotations

import math
from pathlib import Path
import statistics
from typing import Any, Iterable


ALLOWED_STATUS = {"template_no_measurements", "measured_unreviewed"}
ALLOWED_SURFACE = {"dry", "wet"}
ALLOWED_STEM_MODE = {"axial", "bending", "torsion"}


def _finite_number(value: object, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def validate_calibration_dataset(value: dict[str, Any]) -> None:
    if value.get("schema_version") != 1:
        raise ValueError("calibration dataset schema_version must be 1")
    status = value.get("status")
    if status not in ALLOWED_STATUS:
        raise ValueError("unsupported calibration dataset status")
    if not isinstance(value.get("dataset_id"), str) or not value["dataset_id"].strip():
        raise ValueError("dataset_id is required")
    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("dataset provenance is required")
    units = value.get("units")
    expected_units = {
        "length": "m",
        "mass": "kg",
        "force": "N",
        "moment": "N_m",
        "speed": "m_s",
        "angle": "deg",
    }
    if units != expected_units:
        raise ValueError("dataset units must match the fixed SI schema")
    specimens = value.get("specimens")
    if not isinstance(specimens, list):
        raise ValueError("specimens must be a list")
    if status == "template_no_measurements" and specimens:
        raise ValueError("a template cannot contain measurements")
    if status == "measured_unreviewed" and not specimens:
        raise ValueError("a measured dataset cannot be empty")

    identifiers: set[str] = set()
    for specimen in specimens:
        if not isinstance(specimen, dict):
            raise ValueError("each specimen must be an object")
        identifier = specimen.get("specimen_id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError("specimen_id values must be non-empty and unique")
        identifiers.add(identifier)
        if not isinstance(specimen.get("cultivar"), str) or not specimen["cultivar"].strip():
            raise ValueError("each specimen requires a cultivar label")
        _finite_number(
            specimen.get("ripeness_index_0_1"),
            name="ripeness_index_0_1",
            minimum=0.0,
            maximum=1.0,
        )
        axes = specimen.get("axes_m")
        if not isinstance(axes, list) or len(axes) != 3:
            raise ValueError("axes_m must contain three semi-axes")
        for axis in axes:
            _finite_number(axis, name="axis", minimum=0.01, maximum=0.10)
        _finite_number(specimen.get("mass_kg"), name="mass_kg", minimum=0.01, maximum=1.0)
        if specimen.get("surface_condition") not in ALLOWED_SURFACE:
            raise ValueError("surface_condition must be dry or wet")

        contact = specimen.get("contact_trials", [])
        if not isinstance(contact, list):
            raise ValueError("contact_trials must be a list")
        ordered_contact = []
        for trial in contact:
            if not isinstance(trial, dict):
                raise ValueError("contact trial must be an object")
            indentation = _finite_number(
                trial.get("indentation_m"), name="indentation_m", minimum=0.0001, maximum=0.02
            )
            force = _finite_number(trial.get("force_n"), name="force_n", minimum=0.0, maximum=100.0)
            _finite_number(
                trial.get("loading_speed_m_s"),
                name="loading_speed_m_s",
                minimum=0.00001,
                maximum=1.0,
            )
            ordered_contact.append((indentation, force))
        ordered_contact.sort()
        if any(right[0] <= left[0] for left, right in zip(ordered_contact, ordered_contact[1:])):
            raise ValueError("contact indentations must be unique")
        if any(right[1] < left[1] for left, right in zip(ordered_contact, ordered_contact[1:])):
            raise ValueError("contact force must be nondecreasing with indentation")

        slip = specimen.get("slip_angle_deg", [])
        if not isinstance(slip, list):
            raise ValueError("slip_angle_deg must be a list")
        for angle in slip:
            _finite_number(angle, name="slip_angle_deg", minimum=0.1, maximum=80.0)

        rebounds = specimen.get("rebound_trials", [])
        if not isinstance(rebounds, list):
            raise ValueError("rebound_trials must be a list")
        for trial in rebounds:
            if not isinstance(trial, dict):
                raise ValueError("rebound trial must be an object")
            drop = _finite_number(trial.get("drop_height_m"), name="drop_height_m", minimum=0.005, maximum=2.0)
            rebound = _finite_number(
                trial.get("rebound_height_m"), name="rebound_height_m", minimum=0.0, maximum=2.0
            )
            if rebound > drop:
                raise ValueError("rebound height cannot exceed drop height")

        stem = specimen.get("stem_trials", [])
        if not isinstance(stem, list):
            raise ValueError("stem_trials must be a list")
        for trial in stem:
            if not isinstance(trial, dict) or trial.get("mode") not in ALLOWED_STEM_MODE:
                raise ValueError("stem trial mode must be axial, bending, or torsion")
            _finite_number(
                trial.get("loading_duration_s"),
                name="loading_duration_s",
                minimum=0.001,
                maximum=60.0,
            )
            if trial["mode"] == "axial":
                _finite_number(trial.get("peak_force_n"), name="peak_force_n", minimum=0.01, maximum=500.0)
            else:
                _finite_number(
                    trial.get("peak_moment_nm"), name="peak_moment_nm", minimum=0.0001, maximum=10.0
                )


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot summarize an empty sample")
    position = fraction * (len(sorted_values) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return sorted_values[left]
    alpha = position - left
    return sorted_values[left] * (1.0 - alpha) + sorted_values[right] * alpha


def _summary(values: Iterable[float]) -> dict[str, float | int] | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return {
        "sample_count": len(ordered),
        "minimum": ordered[0],
        "p05": _quantile(ordered, 0.05),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": _quantile(ordered, 0.95),
        "maximum": ordered[-1],
    }


def fit_calibration_proposal(value: dict[str, Any]) -> dict[str, object]:
    validate_calibration_dataset(value)
    specimens = value["specimens"]
    axes_by_dimension = [[], [], []]
    masses: list[float] = []
    densities: list[float] = []
    stiffness: list[float] = []
    friction: dict[str, list[float]] = {"dry": [], "wet": []}
    restitution: list[float] = []
    drop_heights: set[float] = set()
    stem: dict[str, list[float]] = {"axial": [], "bending": [], "torsion": []}
    contact_specimens = 0
    for specimen in specimens:
        axes = [float(item) for item in specimen["axes_m"]]
        mass = float(specimen["mass_kg"])
        for index, axis in enumerate(axes):
            axes_by_dimension[index].append(axis)
        masses.append(mass)
        densities.append(mass / (4.0 / 3.0 * math.pi * math.prod(axes)))
        trials = specimen.get("contact_trials", [])
        if len(trials) >= 2:
            numerator = sum(float(item["indentation_m"]) * float(item["force_n"]) for item in trials)
            denominator = sum(float(item["indentation_m"]) ** 2 for item in trials)
            stiffness.append(numerator / denominator)
            contact_specimens += 1
        condition = specimen["surface_condition"]
        friction[condition].extend(math.tan(math.radians(float(angle))) for angle in specimen.get("slip_angle_deg", []))
        for trial in specimen.get("rebound_trials", []):
            drop = float(trial["drop_height_m"])
            rebound = float(trial["rebound_height_m"])
            restitution.append(math.sqrt(rebound / drop))
            drop_heights.add(round(drop, 6))
        for trial in specimen.get("stem_trials", []):
            if trial["mode"] == "axial":
                stem["axial"].append(float(trial["peak_force_n"]))
            else:
                stem[trial["mode"]].append(float(trial["peak_moment_nm"]))

    summaries = {
        "axis_x_m": _summary(axes_by_dimension[0]),
        "axis_y_m": _summary(axes_by_dimension[1]),
        "axis_z_m": _summary(axes_by_dimension[2]),
        "mass_kg": _summary(masses),
        "density_kg_m3": _summary(densities),
        "effective_stiffness_n_m": _summary(stiffness),
        "static_friction_dry": _summary(friction["dry"]),
        "static_friction_wet": _summary(friction["wet"]),
        "restitution": _summary(restitution),
        "stem_axial_force_n": _summary(stem["axial"]),
        "stem_bending_moment_nm": _summary(stem["bending"]),
        "stem_torsional_moment_nm": _summary(stem["torsion"]),
    }
    gates = {
        "geometry_at_least_10_specimens": len(specimens) >= 10,
        "contact_at_least_10_specimens": contact_specimens >= 10,
        "dry_slip_at_least_10_trials": len(friction["dry"]) >= 10,
        "wet_slip_at_least_10_trials": len(friction["wet"]) >= 10,
        "rebound_at_least_20_trials_two_heights": len(restitution) >= 20 and len(drop_heights) >= 2,
        "stem_axial_at_least_10_trials": len(stem["axial"]) >= 10,
        "stem_bending_at_least_10_trials": len(stem["bending"]) >= 10,
    }
    ready = all(gates.values())
    proposal = {
        "fruit_equatorial_radius_m": (
            [summaries["axis_x_m"]["p05"], summaries["axis_x_m"]["p95"]]
            if summaries["axis_x_m"] else None
        ),
        "fruit_density_kg_m3": (
            [summaries["density_kg_m3"]["p05"], summaries["density_kg_m3"]["p95"]]
            if summaries["density_kg_m3"] else None
        ),
        "effective_stiffness_n_m": (
            [summaries["effective_stiffness_n_m"]["p05"], summaries["effective_stiffness_n_m"]["p95"]]
            if summaries["effective_stiffness_n_m"] else None
        ),
        "static_friction": (
            [summaries["static_friction_wet"]["p05"], summaries["static_friction_dry"]["p95"]]
            if summaries["static_friction_wet"] and summaries["static_friction_dry"] else None
        ),
        "restitution": summaries["restitution"]["median"] if summaries["restitution"] else None,
        "stem_axial_force_n": (
            [summaries["stem_axial_force_n"]["p05"], summaries["stem_axial_force_n"]["p95"]]
            if summaries["stem_axial_force_n"] else None
        ),
        "stem_bending_moment_nm": (
            [summaries["stem_bending_moment_nm"]["p05"], summaries["stem_bending_moment_nm"]["p95"]]
            if summaries["stem_bending_moment_nm"] else None
        ),
    }
    return {
        "schema_version": 1,
        "dataset_id": value["dataset_id"],
        "dataset_status": value["status"],
        "scope": "empirical CPU fit proposal; no simulation run and no automatic profile mutation",
        "specimen_count": len(specimens),
        "summaries": summaries,
        "evidence_gates": gates,
        "ready_for_human_review": ready,
        "parameter_proposal": proposal if ready else None,
        "fit_assumptions": [
            "fruit volume is an ellipsoid from measured semi-axes",
            "contact stiffness is a zero-intercept linear least-squares fit per specimen",
            "static friction equals tan(incline slip angle)",
            "restitution equals sqrt(rebound height divided by drop height)",
            "empirical p05-p95 intervals are proposals, not uncertainty confidence bounds",
        ],
        "automatic_profile_activation_authorized": False,
        "ppo_training_authorized": False,
    }


__all__ = ["fit_calibration_proposal", "validate_calibration_dataset"]
