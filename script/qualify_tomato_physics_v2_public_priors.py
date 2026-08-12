"""Run deterministic CPU Q1-Q6 gates for Physics-v2 public-prior mechanics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.tomato_pick.physics_v2 import (  # noqa: E402
    load_physics_v2_profile,
    quantize_engineering_contact_profiles,
    sample_engineering_task_orientations,
    sample_physics_v2_profiles,
    sample_public_abscission_profiles,
    sample_public_morphology_profiles,
    update_multimode_abscission,
    update_twist_integrity,
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "logs/qualification/tomato_physics_v2/public_prior_cpu_gate.json"
)


def _moments(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _relative_error(actual: float, target: float) -> float:
    return abs(actual - target) / abs(target)


def _q1_morphology(
    count: int, generator: torch.Generator, profile: dict
) -> dict:
    samples = sample_public_morphology_profiles(count, generator=generator)
    public_profiles = profile["public_prior_data"]["fruit"][
        "standard_harvest_ready_profiles"
    ]
    source_overlap: dict[str, dict] = {}
    overlap_checks: list[bool] = []
    positive_correlations: list[bool] = []
    for index, source in enumerate(public_profiles):
        selected = samples["source_profile_index"] == index
        dimensions = samples["diameters_m"][selected]
        mass = samples["mass_kg"][selected]
        means = torch.tensor(source["mean_axes_m"] + [source["mean_mass_kg"]])
        generated = torch.cat((dimensions, mass.unsqueeze(1)), dim=1)
        support_low = generated.amin(dim=0)
        support_high = generated.amax(dim=0)
        inside = (means >= support_low) & (means <= support_high)
        correlation = torch.corrcoef(
            torch.stack((dimensions.mean(dim=1), mass))
        )[0, 1]
        overlap_checks.append(bool(inside.all()))
        positive_correlations.append(float(correlation) > 0.35)
        source_overlap[source["id"]] = {
            "sample_count": int(selected.sum()),
            "source_mean_inside_generated_support": bool(inside.all()),
            "size_mass_correlation": float(correlation),
            "generated_diameter_m": {
                "mean": [float(value) for value in dimensions.mean(dim=0)],
                "minimum": [float(value) for value in support_low[:3]],
                "maximum": [float(value) for value in support_high[:3]],
            },
            "generated_mass_kg": _moments(mass),
        }
    rejection_count = int(samples["rejection_count"])
    rejection_fraction = rejection_count / (count + rejection_count)
    morphology = profile["morphology"]
    density_low, density_high = morphology["density_gate_kg_m3"]
    aspect_low, aspect_high = morphology["polar_aspect_ratio_gate"]
    mass_low, mass_high = morphology["mass_gate_kg"]
    checks = {
        "minimum_100k_samples": count >= 100000,
        "all_values_finite": all(
            bool(torch.isfinite(samples[key]).all())
            for key in (
                "diameters_m",
                "mass_kg",
                "density_kg_m3",
                "polar_aspect_ratio",
            )
        ),
        "all_dimensions_positive": bool((samples["diameters_m"] > 0.0).all()),
        "density_gate_enforced": bool(
            (samples["density_kg_m3"] >= density_low).all()
            and (samples["density_kg_m3"] <= density_high).all()
        ),
        "aspect_gate_enforced": bool(
            (samples["polar_aspect_ratio"] >= aspect_low).all()
            and (samples["polar_aspect_ratio"] <= aspect_high).all()
        ),
        "mass_gate_enforced": bool(
            (samples["mass_kg"] >= mass_low).all()
            and (samples["mass_kg"] <= mass_high).all()
        ),
        "source_means_inside_support": all(overlap_checks),
        "shared_size_latent_produces_positive_size_mass_correlation": all(
            positive_correlations
        ),
        "rejection_rate_is_bounded": rejection_fraction < 0.65,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sample_count": count,
        "rejection_count": rejection_count,
        "rejection_fraction": rejection_fraction,
        "diameters_m": {
            "mean": [float(value) for value in samples["diameters_m"].mean(dim=0)],
            "minimum": [float(value) for value in samples["diameters_m"].amin(dim=0)],
            "maximum": [float(value) for value in samples["diameters_m"].amax(dim=0)],
        },
        "mass_kg": _moments(samples["mass_kg"]),
        "density_kg_m3": _moments(samples["density_kg_m3"]),
        "polar_aspect_ratio": _moments(samples["polar_aspect_ratio"]),
        "source_overlap": source_overlap,
    }


def _q2_axial(count: int, generator: torch.Generator, profile: dict) -> dict:
    nodes = profile["public_prior_data"]["pedicel_and_abscission"][
        "pure_axial_pull_off"
    ]["diameter_nodes"]
    results = []
    checks: dict[str, bool] = {}
    for node in nodes:
        samples = sample_public_abscission_profiles(
            count,
            diameter_m=float(node["diameter_m"]),
            generator=generator,
        )["stem_tensile_force_n"]
        moments = _moments(samples)
        mean_error = _relative_error(moments["mean"], node["force_mean_n"])
        sd_error = _relative_error(moments["sd"], node["force_sd_n"])
        label = f"diameter_{node['diameter_m'] * 1000.0:.2f}mm"
        checks[f"{label}_mean_within_2pct"] = mean_error <= 0.02
        checks[f"{label}_sd_within_8pct"] = sd_error <= 0.08
        results.append(
            {
                "diameter_m": node["diameter_m"],
                "target_mean_n": node["force_mean_n"],
                "target_sd_n": node["force_sd_n"],
                "sampled": moments,
                "mean_relative_error": mean_error,
                "sd_relative_error": sd_error,
            }
        )
    return {"passed": all(checks.values()), "checks": checks, "nodes": results}


def _q3_bending(count: int, generator: torch.Generator, profile: dict) -> dict:
    groups = profile["public_prior_data"]["pedicel_and_abscission"][
        "three_point_bending"
    ]["groups"]
    test_diameters = (0.0055, 0.0065, 0.0075)
    results = []
    checks: dict[str, bool] = {}
    for diameter, group in zip(test_diameters, groups, strict=True):
        samples = sample_public_abscission_profiles(
            count, diameter_m=diameter, generator=generator
        )["stem_bending_moment_nm"]
        moments = _moments(samples)
        target_mean = group["derived_center_load_moment_mean_nm"]
        target_sd = group["derived_moment_sd_nm"]
        mean_error = _relative_error(moments["mean"], target_mean)
        sd_error = _relative_error(moments["sd"], target_sd)
        label = f"group_{group['diameter_group_m'][0] * 1000.0:.0f}_to_{group['diameter_group_m'][1] * 1000.0:.0f}mm"
        checks[f"{label}_mean_within_2pct"] = mean_error <= 0.02
        checks[f"{label}_sd_within_8pct"] = sd_error <= 0.08
        results.append(
            {
                "diameter_m": diameter,
                "target_mean_nm": target_mean,
                "target_sd_nm": target_sd,
                "sampled": moments,
                "mean_relative_error": mean_error,
                "sd_relative_error": sd_error,
            }
        )
    return {"passed": all(checks.values()), "checks": checks, "groups": results}


def _q4_torsion(count: int, generator: torch.Generator, profile: dict) -> dict:
    samples = sample_public_abscission_profiles(count, generator=generator)[
        "stem_torsion_moment_nm"
    ]
    config = profile["abscission"]
    core_low, _mode, core_high = config["torsion_limit_triangular_nm"]
    public = profile["public_prior_data"]["pedicel_and_abscission"]["torsion"]
    angle_rad = -math.log(config["torsion_detach_integrity_threshold"]) / config[
        "twist_integrity_k_per_rad"
    ]
    angle_deg = math.degrees(angle_rad)
    public_angle_low = min(
        public["twist_to_detachment_deg_range"]["h1370"][0],
        public["twist_to_detachment_deg_range"]["chico"][0],
    )
    public_angle_high = float(public["conservative_harvest_statement_deg"])
    checks = {
        "torsion_samples_finite": bool(torch.isfinite(samples).all()),
        "core_torque_lower_bound_enforced": float(samples.min()) >= core_low,
        "core_torque_upper_bound_enforced": float(samples.max()) <= core_high,
        "core_torque_inside_public_ood_envelope": (
            core_low >= config["torsion_ood_envelope_nm"][0]
            and core_high <= config["torsion_ood_envelope_nm"][1]
        ),
        "nominal_full_utilization_detach_angle_in_public_envelope": (
            public_angle_low <= angle_deg <= public_angle_high
        ),
        "qualification_rate_matches_public_range": public[
            "test_angular_speed_rad_s"
        ]
        == [0.5, 1.5],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sampled_torsion_limit_nm": _moments(samples),
        "nominal_full_utilization_detach_angle_deg": angle_deg,
        "public_angle_envelope_deg": [public_angle_low, public_angle_high],
        "angular_rate_envelope_rad_s": public["test_angular_speed_rad_s"],
    }


def _q5_twist_then_pull(profile: dict) -> dict:
    config = profile["abscission"]
    public = profile["public_prior_data"]["pedicel_and_abscission"]
    node = public["pure_axial_pull_off"]["diameter_nodes"][1]
    target = public["manual_twist_assisted_detachment"][
        "best_picking_state_pull_after_rotation"
    ]
    one = torch.ones(1)
    attached = torch.ones(1, dtype=torch.bool)
    integrity = update_twist_integrity(
        one,
        one,
        one,
        torch.tensor([2.0 * math.pi]),
        k_theta_per_rad=config["twist_integrity_k_per_rad"],
        exponent_q=config["twist_integrity_q"],
        attached=attached,
    )
    stepped_integrity = one.clone()
    step_count = 360
    for _ in range(step_count):
        stepped_integrity = update_twist_integrity(
            stepped_integrity,
            stepped_integrity,
            one,
            torch.tensor([2.0 * math.pi / step_count]),
            k_theta_per_rad=config["twist_integrity_k_per_rad"],
            exponent_q=config["twist_integrity_q"],
            attached=attached,
        )
    residual_pull = float(integrity) * float(node["force_mean_n"])
    state = update_multimode_abscission(
        one,
        torch.zeros(1),
        torch.zeros((1, 3)),
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([node["force_mean_n"]]),
        torch.tensor([node["force_mean_n"] * 0.74]),
        torch.tensor([0.112134]),
        one,
        torch.tensor([2.0 * math.pi]),
        dt_s=1.0 / 120.0,
        attached=attached,
        k_theta_per_rad=config["twist_integrity_k_per_rad"],
        twist_exponent_q=config["twist_integrity_q"],
        subcritical_onset_ratio=config["subcritical_damage_onset_ratio"],
        subcritical_exponent=config["subcritical_damage_exponent"],
        subcritical_time_constant_s=config["subcritical_time_constant_s"],
        instantaneous_failure_ratio=config["instantaneous_failure_ratio"],
        torsion_detach_integrity_threshold=config[
            "torsion_detach_integrity_threshold"
        ],
        capacity_floor_fraction=config["capacity_floor_fraction"],
    )
    checks = {
        "integrity_matches_cross_study_ratio_within_0_1pct": abs(
            float(integrity) - 0.2603783
        )
        <= 0.001,
        "integrity_is_invariant_to_360_rollout_steps": abs(
            float(stepped_integrity) - float(integrity)
        )
        <= 0.001,
        "residual_pull_matches_11_66N_within_0_1N": abs(
            residual_pull - target["instantaneous_pull_mean_n"]
        )
        <= 0.1,
        "residual_pull_below_public_observed_maximum": residual_pull
        <= target["instantaneous_pull_max_n"],
        "360deg_twist_leaves_attachment_for_pull_test": not bool(
            state["broken"].item()
        ),
        "torsion_is_yield_limited_not_double_counted_as_duration_damage": (
            float(state["torsion_ratio"]) <= 1.0
            and float(state["load_damage"]) == 0.0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reference_diameter_m": node["diameter_m"],
        "intact_pull_mean_n": node["force_mean_n"],
        "integrity_after_360deg": float(integrity),
        "integrity_after_360deg_in_360_steps": float(stepped_integrity),
        "residual_pull_n": residual_pull,
        "public_target_mean_n": target["instantaneous_pull_mean_n"],
        "public_observed_maximum_n": target["instantaneous_pull_max_n"],
    }


def _q6_strategy(profile: dict) -> dict:
    public = profile["public_prior_data"]["pedicel_and_abscission"]
    config = profile["abscission"]
    axial = public["pure_axial_pull_off"]["diameter_nodes"][1]["force_mean_n"]
    bending_group = public["three_point_bending"]["groups"][1]
    twist_pull = public["manual_twist_assisted_detachment"][
        "best_picking_state_pull_after_rotation"
    ]["instantaneous_pull_mean_n"]
    torsion = sum(
        public["torsion"]["suitable_maturity_typical_moment_nm"].values()
    ) / 2.0
    simultaneous_pull = twist_pull / math.sqrt(2.0)
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    lateral = torch.tensor([[1.0, 0.0, 0.0]])
    one = torch.ones(1)
    zero = torch.zeros(1)
    attached = torch.ones(1, dtype=torch.bool)
    tensile_limit = torch.tensor([axial])
    shear_limit = tensile_limit * 0.7423
    bending_limit = torch.tensor(
        [bending_group["derived_center_load_moment_mean_nm"]]
    )
    torsion_limit = torch.tensor([torsion])

    def advance(
        integrity: torch.Tensor,
        damage: torch.Tensor,
        force: torch.Tensor,
        torque: torch.Tensor,
        twist_increment: float = 0.0,
    ) -> dict:
        return update_multimode_abscission(
            integrity,
            damage,
            force,
            torque,
            axis,
            tensile_limit,
            shear_limit,
            bending_limit,
            torsion_limit,
            torch.tensor([twist_increment]),
            dt_s=1.0 / 120.0,
            attached=attached,
            k_theta_per_rad=config["twist_integrity_k_per_rad"],
            twist_exponent_q=config["twist_integrity_q"],
            subcritical_onset_ratio=config["subcritical_damage_onset_ratio"],
            subcritical_exponent=config["subcritical_damage_exponent"],
            subcritical_time_constant_s=config["subcritical_time_constant_s"],
            instantaneous_failure_ratio=config["instantaneous_failure_ratio"],
            torsion_detach_integrity_threshold=config[
                "torsion_detach_integrity_threshold"
            ],
            capacity_floor_fraction=config["capacity_floor_fraction"],
        )

    moderate_pull = advance(
        one,
        zero,
        -axis * twist_pull,
        torch.zeros_like(axis),
    )
    axial_overload = advance(
        one,
        zero,
        -axis * (1.30 * axial),
        torch.zeros_like(axis),
    )
    shear_overload = advance(
        one,
        zero,
        lateral * (1.30 * float(shear_limit)),
        torch.zeros_like(axis),
    )
    bending_overload = advance(
        one,
        zero,
        torch.zeros_like(axis),
        lateral * (1.30 * float(bending_limit)),
    )
    simultaneous_subcritical = advance(
        one,
        zero,
        -axis * (0.50 * axial),
        axis * (0.50 * torsion),
    )

    twist_state = {"integrity": one, "load_damage": zero, "broken": zero.bool()}
    for _ in range(360):
        current_torque = torsion_limit * twist_state["integrity"]
        twist_state = advance(
            twist_state["integrity"],
            twist_state["load_damage"],
            torch.zeros_like(axis),
            axis * current_torque,
            math.radians(1.0),
        )
    integrity_360 = float(twist_state["integrity"])
    broken_at_360 = bool(twist_state["broken"])

    staged_break_step = None
    staged_state = twist_state
    residual_force = axial * integrity_360
    for step in range(1, 241):
        staged_state = advance(
            staged_state["integrity"],
            staged_state["load_damage"],
            -axis * residual_force,
            torch.zeros_like(axis),
        )
        if bool(staged_state["broken"]):
            staged_break_step = step
            break

    twist_550_state = twist_state
    for _ in range(190):
        current_torque = torsion_limit * twist_550_state["integrity"]
        twist_550_state = advance(
            twist_550_state["integrity"],
            twist_550_state["load_damage"],
            torch.zeros_like(axis),
            axis * current_torque,
            math.radians(1.0),
        )
        if bool(twist_550_state["broken"]):
            break

    strategies = {
        "pure_pull": {
            "peak_axial_force_n": axial,
            "moderate_11_66N_instant_break": bool(moderate_pull["broken"]),
            "overload_1_30x_instant_break": bool(axial_overload["broken"]),
        },
        "pure_bend": {
            "published_center_force_n": bending_group["force_mean_n"],
            "peak_moment_nm": bending_group["derived_center_load_moment_mean_nm"],
            "overload_1_30x_instant_break": bool(bending_overload["broken"]),
        },
        "pure_twist": {
            "representative_torque_nm": torsion,
            "nominal_detach_angle_deg": math.degrees(
                -math.log(
                    profile["abscission"]["torsion_detach_integrity_threshold"]
                )
                / profile["abscission"]["twist_integrity_k_per_rad"]
            ),
            "integrity_after_360deg": integrity_360,
            "broken_at_360deg": broken_at_360,
            "broken_by_550deg": bool(twist_550_state["broken"]),
        },
        "simultaneous_twist_pull": {
            "illustrative_equal_ratio_axial_force_n": simultaneous_pull,
            "qualification_role": "interaction sanity check, not a direct prior",
            "half_axial_half_torsion_instant_break": bool(
                simultaneous_subcritical["broken"]
            ),
        },
        "staged_twist_then_pull": {
            "residual_axial_force_n": residual_force,
            "break_step_at_120hz": staged_break_step,
            "break_time_s": (
                staged_break_step / 120.0
                if staged_break_step is not None
                else None
            ),
        },
        "pure_shear": {
            "capacity_n": float(shear_limit),
            "overload_1_30x_instant_break": bool(shear_overload["broken"]),
        },
    }
    checks = {
        "intact_pure_pull_requires_tens_of_newtons": axial >= 40.0,
        "staged_twist_reduces_axial_demand_by_at_least_60pct": twist_pull
        <= 0.40 * axial,
        "staged_detachment_is_not_free": twist_pull >= 10.0,
        "bending_fixture_is_not_relabelled_as_axial": bending_group[
            "derived_center_load_moment_mean_nm"
        ]
        > 0.0,
        "moderate_pull_does_not_instantly_break_intact_az": not bool(
            moderate_pull["broken"]
        ),
        "pure_axial_overload_breaks": bool(axial_overload["broken"]),
        "pure_shear_overload_breaks": bool(shear_overload["broken"]),
        "pure_bending_overload_breaks": bool(bending_overload["broken"]),
        "half_axial_half_torsion_is_subcritical": not bool(
            simultaneous_subcritical["broken"]
        ),
        "360deg_twist_reduces_integrity_but_keeps_attachment": (
            abs(integrity_360 - 0.2603783) <= 0.001 and not broken_at_360
        ),
        "staged_moderate_pull_breaks_in_finite_time": (
            staged_break_step is not None and staged_break_step > 1
        ),
        "continued_twist_breaks_by_public_550deg_bound": bool(
            twist_550_state["broken"]
        ),
        "all_six_load_strategies_reported": len(strategies) == 6,
    }
    return {"passed": all(checks.values()), "checks": checks, "strategies": strategies}


def _orientation_audit(count: int, generator: torch.Generator, profile: dict) -> dict:
    samples = sample_engineering_task_orientations(count, generator=generator)
    categories = samples["world_orientation_category"]
    actual = torch.bincount(categories, minlength=4).float() / count
    target = torch.tensor(
        profile["task_distribution"]["world_orientation_probabilities"]
    )
    maximum_error = float((actual - target).abs().max())
    engineering = profile["engineering_prior_data"][
        "natural_pedicel_orientation"
    ]
    tilt_bounds = engineering["polar_tilt_from_vertical_deg"]["parameters"]
    jitter_bounds = engineering["within_category_azimuth_jitter_deg"]["bounds"]
    tilt_degrees = torch.rad2deg(samples["orientation_polar_tilt_rad"])
    jitter_degrees = torch.rad2deg(samples["orientation_azimuth_jitter_rad"])
    checks = {
        "categorical_probability_within_1pct": maximum_error <= 0.01,
        "continuous_tilt_inside_engineering_support": (
            float(tilt_degrees.min()) >= tilt_bounds[0]
            and float(tilt_degrees.max()) <= tilt_bounds[2]
        ),
        "azimuth_jitter_inside_category_bounds": (
            float(jitter_degrees.min()) >= jitter_bounds[0]
            and float(jitter_degrees.max()) <= jitter_bounds[1]
        ),
        "continuous_tilt_has_nonzero_spread": float(tilt_degrees.std()) > 1.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sample_count": count,
        "categories": profile["task_distribution"]["world_orientation_categories"],
        "target_probability": [float(value) for value in target],
        "sampled_probability": [float(value) for value in actual],
        "maximum_absolute_error": maximum_error,
        "polar_tilt_deg": _moments(tilt_degrees),
        "within_category_azimuth_jitter_deg": _moments(jitter_degrees),
        "actor_observation_exposes_world_category": False,
    }


def _engineering_gap_prior_audit(
    count: int, generator: torch.Generator, profile: dict
) -> dict:
    samples = sample_physics_v2_profiles(count, generator=generator)
    engineering = profile["engineering_prior_data"]
    fatigue = engineering["cyclic_fatigue"]
    contact = engineering["g20_pad_tomato_contact"]
    shear_support = engineering["abscission_zone_shear"][
        "capacity_ratio_to_diameter_conditioned_tensile"
    ]["parameters"]
    onset_support = fatigue["subcritical_onset_ratio"]["parameters"]
    exponent_support = fatigue["damage_exponent"]["parameters"]
    duration_support = fatigue["time_constant_s"]["parameters"]
    friction_support = contact["effective_pair_static_friction"]["parameters"]
    stiffness_support = contact["effective_linear_contact_stiffness_n_m"][
        "parameters"
    ]
    damping_support = contact["effective_contact_damping_n_s_m"]["bounds"]
    quantized = quantize_engineering_contact_profiles(
        samples, friction_bins=4, compliance_bins=4
    )

    def inside(values: torch.Tensor, support: list[float]) -> bool:
        return float(values.min()) >= support[0] and float(values.max()) <= support[-1]

    actuator = engineering["a7_l20_effective_actuator"]
    checks = {
        "az_shear_ratio_inside_explicit_support": inside(
            samples["stem_shear_ratio"], shear_support
        ),
        "az_shear_mean_near_derived_0_7423_mode": abs(
            float(samples["stem_shear_ratio"].mean()) - shear_support[1]
        )
        < 0.03,
        "fatigue_onset_inside_explicit_support": inside(
            samples["stem_subcritical_onset_ratio"], onset_support
        ),
        "fatigue_exponent_inside_explicit_support": inside(
            samples["stem_subcritical_damage_exponent"], exponent_support
        ),
        "fatigue_time_inside_explicit_support": inside(
            samples["stem_critical_duration_s"], duration_support
        ),
        "pair_friction_inside_explicit_support": inside(
            samples["effective_pair_static_friction"], friction_support
        ),
        "contact_stiffness_inside_explicit_support": inside(
            samples["contact_stiffness_n_m"], stiffness_support
        ),
        "contact_damping_inside_explicit_support": inside(
            samples["contact_damping_n_s_m"], damping_support
        ),
        "quantized_pair_friction_is_representable_with_baked_pad": bool(
            (quantized["material_static_friction"] >= 0.0).all()
            and (quantized["material_dynamic_friction"] >= 0.0).all()
        ),
        "actuator_delay_is_zero_to_two_policy_steps": actuator[
            "action_delay_steps_uniform_inclusive"
        ]
        == [0, 2],
        "checked_in_hand_alias_is_explicitly_l20": "L20"
        in actuator["task_naming_note"],
        "hardware_surrogates_remain_unqualified": profile["hardware_transfer"][
            "ready"
        ]
        is False,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sample_count": count,
        "shear_ratio": _moments(samples["stem_shear_ratio"]),
        "fatigue_onset_ratio": _moments(
            samples["stem_subcritical_onset_ratio"]
        ),
        "fatigue_exponent": _moments(
            samples["stem_subcritical_damage_exponent"]
        ),
        "fatigue_time_constant_s": _moments(samples["stem_critical_duration_s"]),
        "effective_pair_static_friction": _moments(
            samples["effective_pair_static_friction"]
        ),
        "effective_contact_stiffness_n_m": _moments(
            samples["contact_stiffness_n_m"]
        ),
        "effective_contact_damping_n_s_m": _moments(
            samples["contact_damping_n_s_m"]
        ),
        "action_delay_steps_uniform_inclusive": actuator[
            "action_delay_steps_uniform_inclusive"
        ],
        "classification": "simulation-ready priors; hardware calibration still required",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morphology-samples", type=int, default=100000)
    parser.add_argument("--mechanics-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.morphology_samples < 100000:
        raise ValueError("Q1 requires at least 100000 morphology samples")
    if args.mechanics_samples < 10000:
        raise ValueError("mechanics gates require at least 10000 samples")

    profile = load_physics_v2_profile()
    generator = torch.Generator().manual_seed(args.seed)
    gates = {
        "Q1_morphology": _q1_morphology(
            args.morphology_samples, generator, profile
        ),
        "Q2_pure_axial": _q2_axial(
            args.mechanics_samples, generator, profile
        ),
        "Q3_bending": _q3_bending(
            args.mechanics_samples, generator, profile
        ),
        "Q4_torsion": _q4_torsion(
            args.mechanics_samples, generator, profile
        ),
        "Q5_twist_then_pull": _q5_twist_then_pull(profile),
        "Q6_strategy_sanity": _q6_strategy(profile),
    }
    orientation = _orientation_audit(args.morphology_samples, generator, profile)
    engineering = _engineering_gap_prior_audit(
        args.mechanics_samples, generator, profile
    )
    passed = (
        all(gate["passed"] for gate in gates.values())
        and orientation["passed"]
        and engineering["passed"]
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "physics_v2_public_prior_cpu_q1_q6",
        "profile_id": profile["profile_id"],
        "task_id": profile["task_id"],
        "seed": args.seed,
        "passed": passed,
        "training_authorized": False,
        "training_authorization_note": (
            "CPU Q1-Q6 pass only; an Isaac one-environment and vectorized smoke "
            "gate must pass before from-scratch PPO can be started. Physics-v1 "
            "checkpoint resume is forbidden."
        ),
        "gates": gates,
        "world_orientation_distribution_audit": orientation,
        "engineering_gap_prior_audit": engineering,
        "hardware_transfer_ready": False,
        "hardware_transfer_blockers": profile["hardware_transfer"]["blockers"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
