from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

from script.gate_tomato_physics_v1_first_success import _tail_trend_summary
from simulation.tomato_pick.physics_v1 import (
    combined_stem_load_ratio,
    conservative_tangential_capacity,
    load_physics_v1_profile,
    quantize_contact_profiles,
    sample_correlated_fruit_profiles,
    secure_kinematic_break_credit_gate,
    secure_support_credit_gate,
    stem_load_damage_progress_score,
    update_stem_damage,
)
from simulation.tomato_pick.reset_v1 import (
    sample_bounded_joint_ellipsoid_offsets,
    sample_bounded_joint_workspace_offsets,
    sample_hanging_fruit_quaternions,
    sample_joint_ellipsoid_offsets,
    sample_pregrasp_fruit_offsets,
)


def test_physics_v1_profile_is_independent_and_formal_training_ready() -> None:
    profile = load_physics_v1_profile()
    assert profile["profile_id"] == "tomato_pick_physics_v1"
    assert profile["parent_evidence"]["task_id"] == "LinkAgent-TomatoPick-Clean-v5"
    assert profile["training_gate"]["ready"] is True
    assert profile["training_gate"]["formal_training_started"] is False
    assert profile["training_gate"]["qualification"] == "literature_prior_provisional"
    assert profile["training_gate"]["blockers"] == []
    assert profile["hand"]["active_joint_count"] == 16
    assert profile["hand"]["passive_follower_count"] == 5
    assert profile["hand"]["active_drive_armature_kg_m2"] == 0.002
    assert (
        profile["hand"]["passive_mimic_driver_velocity_safety_factor"]
        == 0.95
    )
    assert profile["hand"]["passive_mimic_max_tracking_error_rad"] == 0.35
    assert profile["secure_success"]["is_training_success"] is True
    assert profile["secure_success"]["minimum_supported_fingertips"] >= 2
    assert profile["secure_success"]["thumb_opposition_required"] is True
    assert "not_user_calibrated" in profile["contact"]["status"]
    assert "not_user_calibrated" in profile["stem_prior"]["status"]
    protocol = profile["calibration_protocol"]
    assert set(protocol) == {
        "hand_free_motion",
        "arm_free_motion",
        "fruit_contact",
        "stem",
    }
    assert protocol["hand_free_motion"]["minimum_repetitions_per_trajectory"] >= 5
    assert protocol["fruit_contact"]["minimum_samples_per_condition"] >= 10
    assert protocol["stem"]["minimum_samples_per_mode"] >= 20
    reset = profile["reset_workspace"]
    assert reset["version"] == 3
    assert reset["sampling_space"].startswith("seven_joint_continuous_ellipsoid")
    assert len(reset["arm_joint_ellipsoid_radius_deg_in_a7_order"]) == 7
    assert reset["arm_joint_ellipsoid_radius_deg_in_a7_order"] == [
        42.0,
        50.0,
        58.0,
        34.0,
        48.0,
        46.0,
        52.0,
    ]
    assert len(reset["arm_joint_correlated_modes_deg_in_a7_order"]) == 1
    assert reset["arm_joint_correlated_mode_coefficient_ranges"] == [[-1.0, 0.30]]
    assert reset["fruit_approach_distance_m"] == [0.045, 0.060]
    assert reset["fruit_sampling_max_attempts"] == 64
    assert reset["maximum_closure_translation_norm_m"] == 0.065
    assert reset["minimum_closure_translation_headroom_fraction"] == 0.25
    assert reset["minimum_closure_joint_limit_margin_fraction"] == 0.03


def test_physics_v1_derived_robot_asset_is_frozen_and_independent() -> None:
    profile = load_physics_v1_profile()
    root = Path(__file__).resolve().parents[1]
    asset = profile["robot_asset"]
    source = root / asset["source_usd"]
    derived = root / asset["derived_usd"]
    assert source != derived
    assert source.is_file() and derived.is_file()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == asset[
        "source_usd_sha256"
    ]
    assert hashlib.sha256(derived.read_bytes()).hexdigest() == asset[
        "derived_usd_sha256"
    ]
    assert asset["collision_matrix_baked"] is True
    assert asset["self_collision_representation"] == (
        "distal_envelope_filtered_pairs_outside_solve_contact_disabled_v2"
    )
    assert asset["stored_active_hand_filtered_pair_count"] == 7
    assert asset["outside_solve_contact_disabled_body_count"] == 59
    assert asset["runtime_authored_collision_pairs"] == 0
    assert asset["passive_mimics_baked"] is True
    assert asset["runtime_authored_passive_mimics"] == 0
    assert asset["passive_mimic_count"] == 5
    assert len(asset["passive_mimic_sha256"]) == 64
    assert asset["hand_material_baked"] is True
    assert asset["runtime_authored_hand_material_bindings"] == 0
    assert asset["hand_material_body_count"] == 54


def test_correlated_fruit_profiles_are_bounded_and_mass_follows_geometry() -> None:
    generator = torch.Generator().manual_seed(42)
    values = sample_correlated_fruit_profiles(4096, generator=generator)
    axes = values["axes_m"]
    mass = values["mass_kg"]
    density = values["density_kg_m3"]
    expected = 4.0 / 3.0 * math.pi * axes.prod(dim=1) * density
    assert torch.allclose(mass, expected, atol=1.0e-8)
    assert float(axes.min()) >= 0.021
    assert float(axes.max()) <= 0.037
    assert 0.05 <= float(mass.min())
    assert float(mass.max()) <= 0.22
    assert 4.0 <= float(values["stem_tensile_force_n"].min())
    assert float(values["stem_tensile_force_n"].max()) <= 8.0
    assert float(values["stem_intact_tensile_force_n"].min()) > 15.0
    assert float(values["stem_intact_tensile_force_n"].max()) < 90.0
    assert bool(
        (
            values["stem_tensile_force_n"]
            <= values["stem_safe_tangential_capacity_n"]
        ).all()
    )
    size_proxy = axes.mean(dim=1)
    correlation = torch.corrcoef(
        torch.stack((size_proxy, values["stem_diameter_m"]))
    )[0, 1]
    assert float(correlation) > 0.65


def test_abscission_envelope_has_three_finger_safe_wrench_margin() -> None:
    values = sample_correlated_fruit_profiles(
        20000, generator=torch.Generator().manual_seed(91)
    )
    capacity = conservative_tangential_capacity(
        values["static_friction"],
        maximum_fingertip_force_n=8.0,
        supported_fingertips=3,
        utilization_fraction=0.75,
    )
    assert torch.allclose(capacity, values["stem_safe_tangential_capacity_n"])
    assert bool((values["stem_tensile_force_n"] <= capacity).all())
    required_normal_per_finger = values["stem_tensile_force_n"] / (
        3.0 * values["static_friction"]
    )
    assert float(required_normal_per_finger.max()) < 8.0


def test_weak_median_strong_abscission_profiles_break_by_duration_not_spike() -> None:
    profile = load_physics_v1_profile()
    values = sample_correlated_fruit_profiles(
        20000, generator=torch.Generator().manual_seed(37)
    )
    indices = torch.argsort(values["stem_tensile_force_n"])[
        torch.tensor([0, len(values["stem_tensile_force_n"]) // 2, -1])
    ]
    critical = values["stem_critical_duration_s"][indices]
    attached = torch.ones(3, dtype=torch.bool)
    damage = torch.zeros(3)
    dt = 1.0 / 120.0
    damage, broken = update_stem_damage(
        damage,
        torch.ones(3),
        dt_s=dt,
        critical_duration_s=critical,
        onset_ratio=profile["stem_prior"]["damage_onset_ratio"],
        recovery_time_s=profile["stem_prior"]["damage_recovery_time_s"],
        attached=attached,
    )
    assert not bool(broken.any())
    for _ in range(16):
        damage, broken = update_stem_damage(
            damage,
            torch.ones(3),
            dt_s=dt,
            critical_duration_s=critical,
            onset_ratio=profile["stem_prior"]["damage_onset_ratio"],
            recovery_time_s=profile["stem_prior"]["damage_recovery_time_s"],
            attached=attached,
        )
    assert bool(broken.all())


def test_stem_load_ratio_combines_tension_shear_and_bending() -> None:
    axis = torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1)
    tensile = torch.full((4,), 10.0)
    shear = torch.full((4,), 5.0)
    bending = torch.full((4,), 0.2)
    force = torch.tensor(
        [[0.0, 0.0, -10.0], [5.0, 0.0, 0.0], [0.0, 0.0, 10.0], [3.0, 0.0, -8.0]]
    )
    torque = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.12, 0.0, 0.0]]
    )
    ratio = combined_stem_load_ratio(
        force, torque, axis, tensile, shear, bending
    )
    assert torch.allclose(ratio[:3], torch.ones(3), atol=1.0e-6)
    assert float(ratio[3]) > 1.0


def test_stem_progress_preserves_load_gradient_and_requires_sustained_damage() -> None:
    load = torch.tensor([0.0, 0.5, 1.0, 1.5])
    damage = torch.tensor([0.0, 0.0, 0.0, 1.0])
    opposed = torch.ones(4)
    score = stem_load_damage_progress_score(load, damage, opposed)
    assert torch.allclose(score, torch.tensor([0.0, 0.375, 0.75, 1.0]))
    # An instantaneous overload cannot consume the full bridge; the final
    # quarter is available only after the physical damage integrator advances.
    assert float(score[2]) < 1.0
    assert float(score[3]) == 1.0


def test_stem_progress_is_bounded_and_opposition_gates_every_component() -> None:
    score = stem_load_damage_progress_score(
        torch.tensor([-2.0, 0.8, 4.0]),
        torch.tensor([3.0, 0.6, -1.0]),
        torch.tensor([0.0, 0.5, 2.0]),
    )
    assert torch.allclose(score, torch.tensor([0.0, 0.375, 0.75]))
    assert bool(((score >= 0.0) & (score <= 1.0)).all())


def test_secure_support_gate_blocks_unsupported_downstream_credit() -> None:
    score = torch.tensor([0.0, 0.25, 0.75, 1.0])
    support = torch.tensor([False, True, False, True])
    assert torch.equal(
        secure_support_credit_gate(score, support),
        torch.tensor([0.0, 0.25, 0.0, 1.0]),
    )


def test_secure_support_gate_rejects_non_boolean_or_mismatched_support() -> None:
    score = torch.ones(2)
    for support in (torch.ones(2), torch.ones(3, dtype=torch.bool)):
        try:
            secure_support_credit_gate(score, support)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid secure-support credit gate input accepted")


def test_break_credit_requires_secure_and_kinematically_retained_fracture() -> None:
    score = torch.ones(4)
    secure = torch.tensor([False, True, True, True])
    kinematic = torch.tensor([True, False, True, False])
    assert torch.equal(
        secure_kinematic_break_credit_gate(score, secure, kinematic),
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
    )


def test_break_credit_gate_rejects_invalid_kinematic_mask() -> None:
    score = torch.ones(2)
    secure = torch.ones(2, dtype=torch.bool)
    for kinematic in (torch.ones(2), torch.ones(3, dtype=torch.bool)):
        try:
            secure_kinematic_break_credit_gate(score, secure, kinematic)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid kinematic break-credit mask accepted")


def test_first_success_gate_rejects_a_single_rare_non_tail_success() -> None:
    records = [
        {
            "iteration": iteration,
            "detach_rate": 0.6,
            "drop_rate": 0.57,
            "retained_rate": 0.0002 if iteration == 28 else 0.0,
            "success_rate": 0.0002 if iteration == 28 else 0.0,
        }
        for iteration in range(50)
    ]
    _, checks = _tail_trend_summary(records)
    assert checks["tail_window_complete"] is True
    assert checks["tail_success_repeated"] is False
    assert checks["tail_retention_persistent"] is False


def test_first_success_gate_accepts_persistent_improving_tail() -> None:
    records = []
    for iteration in range(50):
        tail = iteration >= 40
        records.append(
            {
                "iteration": iteration,
                "detach_rate": 0.6,
                "drop_rate": 0.48 if tail else 0.54,
                "retained_rate": 0.001 if tail and iteration % 2 == 0 else 0.0,
                "success_rate": 0.0002 if tail and iteration in {44, 48} else 0.0,
            }
        )
    summary, checks = _tail_trend_summary(records)
    assert all(checks.values())
    assert summary["tail_success_positive_iterations"] == 2
    assert summary["tail_retained_positive_iterations"] == 5
    assert summary["tail_drop_detach_ratio"] < summary["previous_drop_detach_ratio"]


def test_stem_progress_rejects_invalid_weight_or_shape() -> None:
    values = torch.ones(2)
    for invalid in (0.0, 1.0, -0.1, 1.1):
        try:
            stem_load_damage_progress_score(
                values, values, values, instantaneous_load_fraction=invalid
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid instantaneous fraction was accepted")
    try:
        stem_load_damage_progress_score(values, torch.ones(3), values)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched reward inputs were accepted")


def test_correlated_contact_profiles_use_a_bounded_ordered_material_bank() -> None:
    profiles = sample_correlated_fruit_profiles(
        4096, generator=torch.Generator().manual_seed(7)
    )
    quantized = quantize_contact_profiles(
        profiles, friction_bins=4, compliance_bins=4
    )
    material_bin = quantized["material_bin"]
    assert int(material_bin.min()) >= 0
    assert int(material_bin.max()) < 16
    assert int(torch.unique(material_bin).numel()) <= 16
    assert torch.all(
        quantized["material_dynamic_friction"]
        <= quantized["material_static_friction"]
    )
    friction_correlation = torch.corrcoef(
        torch.stack(
            (
                profiles["static_friction"],
                quantized["material_static_friction"],
            )
        )
    )[0, 1]
    stiffness_correlation = torch.corrcoef(
        torch.stack(
            (
                profiles["contact_stiffness_n_m"],
                quantized["material_stiffness_n_m"],
            )
        )
    )[0, 1]
    assert float(friction_correlation) > 0.95
    assert float(stiffness_correlation) > 0.95


def test_stem_damage_rejects_one_substep_spike_and_breaks_after_duration() -> None:
    dt = 1.0 / 120.0
    critical = torch.tensor([0.05])
    attached = torch.tensor([True])
    damage = torch.zeros(1)
    damage, broken = update_stem_damage(
        damage,
        torch.ones(1),
        dt_s=dt,
        critical_duration_s=critical,
        onset_ratio=0.55,
        recovery_time_s=0.12,
        attached=attached,
    )
    assert not bool(broken.item())
    assert 0.0 < float(damage.item()) < 1.0
    for _ in range(5):
        damage, broken = update_stem_damage(
            damage,
            torch.ones(1),
            dt_s=dt,
            critical_duration_s=critical,
            onset_ratio=0.55,
            recovery_time_s=0.12,
            attached=attached,
        )
    assert bool(broken.item())


def test_stem_damage_recovers_below_onset() -> None:
    damage, broken = update_stem_damage(
        torch.tensor([0.5]),
        torch.tensor([0.1]),
        dt_s=1.0 / 120.0,
        critical_duration_s=torch.tensor([0.05]),
        onset_ratio=0.55,
        recovery_time_s=0.12,
        attached=torch.tensor([True]),
    )
    assert not bool(broken.item())
    assert 0.0 <= float(damage.item()) < 0.5


def test_joint_reset_offsets_fill_but_never_leave_ellipsoid() -> None:
    radii = torch.deg2rad(torch.tensor([14.0, 18.0, 20.0, 18.0, 22.0, 20.0, 28.0]))
    offsets = sample_joint_ellipsoid_offsets(
        8192, radii, generator=torch.Generator().manual_seed(42)
    )
    normalized_radius = torch.linalg.vector_norm(offsets / radii, dim=1)
    assert float(normalized_radius.max()) <= 1.0 + 1.0e-6
    assert float(torch.quantile(normalized_radius, 0.95)) > 0.90
    assert torch.all(offsets.amax(dim=0) > 0.60 * radii)
    assert torch.all(offsets.amin(dim=0) < -0.60 * radii)


def test_bounded_joint_reset_rejects_limits_instead_of_clipping() -> None:
    radii = torch.deg2rad(
        torch.tensor([42.0, 50.0, 58.0, 34.0, 48.0, 46.0, 52.0])
    )
    lower = torch.deg2rad(torch.tensor([-35.0, -45.0, -40.0, -18.0, -44.0, -38.0, -27.0]))
    upper = torch.deg2rad(torch.tensor([38.0, 43.0, 46.0, 29.0, 41.0, 39.0, 31.0]))
    offsets = sample_bounded_joint_ellipsoid_offsets(
        8192,
        radii,
        lower_offset_rad=lower,
        upper_offset_rad=upper,
        generator=torch.Generator().manual_seed(43),
    )
    normalized_radius = torch.linalg.vector_norm(offsets / radii, dim=1)
    assert float(normalized_radius.max()) <= 1.0 + 1.0e-6
    assert bool((offsets > lower).all())
    assert bool((offsets < upper).all())
    # Rejection keeps a continuous distribution; no sample is projected onto
    # either bound as a clamp would do.
    assert not bool(torch.isclose(offsets, lower, atol=1.0e-7, rtol=0.0).any())
    assert not bool(torch.isclose(offsets, upper, atol=1.0e-7, rtol=0.0).any())


def test_bounded_joint_reset_validates_bounds() -> None:
    radii = torch.ones(2)
    try:
        sample_bounded_joint_ellipsoid_offsets(
            4,
            radii,
            lower_offset_rad=torch.tensor([0.0, 1.0]),
            upper_offset_rad=torch.tensor([0.0, 2.0]),
        )
    except ValueError as error:
        assert "lower joint offset bound" in str(error)
    else:
        raise AssertionError("invalid joint bounds were accepted")


def test_correlated_workspace_mode_is_continuous_and_bounded() -> None:
    radii = torch.deg2rad(torch.tensor([42.0, 50.0, 58.0, 34.0, 48.0, 46.0, 52.0]))
    modes = torch.deg2rad(
        torch.tensor([[8.4, 17.9, 3.7, 22.0, 0.9, 2.6, -8.7]])
    )
    lower = torch.full((7,), -1.2)
    upper = torch.full((7,), 1.2)
    values = sample_bounded_joint_workspace_offsets(
        8192,
        radii,
        correlated_modes_rad=modes,
        mode_coefficient_ranges=torch.tensor([[-1.0, 0.30]]),
        lower_offset_rad=lower,
        upper_offset_rad=upper,
        generator=torch.Generator().manual_seed(44),
    )
    offsets = values["offset_rad"]
    base = values["base_ellipsoid_offset_rad"]
    coefficient = values["mode_coefficients"]
    assert bool((offsets > lower).all() and (offsets < upper).all())
    assert float(torch.linalg.vector_norm(base / radii, dim=1).max()) <= 1.0 + 1.0e-6
    assert float(coefficient.min()) >= -1.0
    assert float(coefficient.max()) <= 0.30
    assert torch.allclose(offsets, base + coefficient @ modes, atol=1.0e-7)
    assert float(coefficient.std()) > 0.35


def test_pregrasp_fruit_offsets_form_a_bounded_forward_cone() -> None:
    center = torch.tensor([0.148, -0.0607, 0.0535])
    values = sample_pregrasp_fruit_offsets(
        4096,
        closed_grasp_center_m=center,
        distance_range_m=(0.05, 0.09),
        yaw_range_rad=(math.radians(-28.0), math.radians(28.0)),
        pitch_range_rad=(math.radians(-18.0), math.radians(18.0)),
        generator=torch.Generator().manual_seed(7),
    )
    direction = values["direction_local"]
    distance = torch.linalg.vector_norm(values["local_position_m"] - center, dim=1)
    assert bool((direction[:, 0] > 0.0).all())
    assert 0.05 <= float(distance.min())
    assert float(distance.max()) <= 0.09 + 1.0e-6
    assert float(values["yaw_rad"].abs().max()) <= math.radians(28.0)
    assert float(values["pitch_rad"].abs().max()) <= math.radians(18.0)


def test_hanging_fruit_orientation_has_full_yaw_but_bounded_tilt() -> None:
    values = sample_hanging_fruit_quaternions(
        4096,
        maximum_tilt_rad=math.radians(15.0),
        yaw_range_rad=(-math.pi, math.pi),
        device="cpu",
        generator=torch.Generator().manual_seed(11),
    )
    quaternion = values["quaternion_wxyz"]
    assert torch.allclose(
        torch.linalg.vector_norm(quaternion, dim=1), torch.ones(4096), atol=1.0e-6
    )
    assert float(values["tilt_rad"].max()) <= math.radians(15.0) + 1.0e-6
    assert float(values["yaw_rad"].min()) < -3.0
    assert float(values["yaw_rad"].max()) > 3.0
