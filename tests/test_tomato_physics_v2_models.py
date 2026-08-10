from __future__ import annotations

import math

import torch

from simulation.tomato_pick.physics_v2 import load_physics_v2_profile
from simulation.tomato_pick.physics_v2_models import (
    apply_joint_target_uncertainty_v2,
    fixed_fruit_contact_cases,
    initial_fruit_damage_state,
    initial_tactile_uncertainty_state,
    linear_compliant_contact_force,
    quantize_contact_material_bank,
    sample_correlated_fruit_contact_profiles,
    sample_stem_profiles_v2,
    sample_uncertainty_profiles_v2,
    stem_load_components_v2,
    update_fruit_damage_diagnostics,
    update_tactile_uncertainty_v2,
)
from simulation.tomato_pick.physics_v2_isolated import (
    incline_will_slide,
    restitution_energy_fraction,
    semi_implicit_free_fall,
    simulate_damped_compression,
)
from simulation.tomato_pick.physics_v2_scenarios import (
    physical_task_predicates,
    run_wrench_level_pick_scenario,
)


def test_v2_isolated_free_fall_has_correct_acceleration_without_energy_gain() -> None:
    result = semi_implicit_free_fall(
        height_m=1.0, mass_kg=0.12, dt_s=1.0 / 120.0, steps=30
    )
    acceleration = torch.diff(result["velocity_m_s"]) / (1.0 / 120.0)
    assert torch.allclose(acceleration, torch.full_like(acceleration, -9.81))
    assert bool((torch.diff(result["energy_j"]) <= 1.0e-12).all())
    relative_loss = float(
        (result["energy_j"][0] - result["energy_j"][-1])
        / result["energy_j"][0]
    )
    assert relative_loss < 0.011


def test_v2_isolated_incline_and_restitution_cases_are_ordered() -> None:
    friction = torch.tensor([0.50, 0.55, 0.60], dtype=torch.float64)
    assert not bool(incline_will_slide(friction, 20.0).any())
    assert bool(incline_will_slide(friction, 35.0).all())
    assert restitution_energy_fraction(0.01) == 0.0001


def test_v2_isolated_damped_contact_does_not_gain_energy() -> None:
    result = simulate_damped_compression(
        mass_kg=0.10,
        stiffness_n_m=8000.0,
        damping_n_s_m=35.0,
        initial_indentation_m=0.002,
        initial_closing_speed_m_s=0.05,
        dt_s=1.0 / 20000.0,
        steps=1000,
    )
    assert bool((torch.diff(result["energy_j"]) <= 1.0e-10).all())
    assert float(result["energy_j"][-1]) < float(result["energy_j"][0])


def test_v2_wrench_level_weak_median_strong_oracles_complete() -> None:
    for quantile in (0.05, 0.50, 0.95):
        result = run_wrench_level_pick_scenario(
            strength_quantile=quantile, seed=42, sample_count=2048
        )
        assert result["detached"] is True
        assert result["retained"] is True
        assert result["success"] is True
        assert result["one_substep_broken"] is False
        assert result["break_credit"] == 1.0
        assert result["fruit_damage_score"] == 0.0
        assert result["aligned_return"] > 200.0


def test_v2_success_uses_physical_force_not_stale_tactile_measurement() -> None:
    measured_force = torch.full((1, 5), 4.0)
    physical_force = torch.zeros_like(measured_force)
    result = physical_task_predicates(
        physical_force,
        stem_detached=torch.ones(1, dtype=torch.bool),
        center_distance_m=torch.tensor([0.03]),
        relative_speed_m_s=torch.tensor([0.02]),
        hold_steps=torch.tensor([20]),
    )
    assert bool((measured_force > 0.25).all())
    assert not bool(result["secure_support"].item())
    assert not bool(result["success"].item())


def test_v2_same_side_contact_and_high_speed_cannot_succeed() -> None:
    same_side = physical_task_predicates(
        torch.tensor([[0.0, 4.0, 4.0, 0.0, 0.0]]),
        stem_detached=torch.ones(1, dtype=torch.bool),
        center_distance_m=torch.tensor([0.03]),
        relative_speed_m_s=torch.tensor([0.02]),
        hold_steps=torch.tensor([20]),
    )
    high_speed = physical_task_predicates(
        torch.tensor([[4.0, 4.0, 4.0, 0.0, 0.0]]),
        stem_detached=torch.ones(1, dtype=torch.bool),
        center_distance_m=torch.tensor([0.03]),
        relative_speed_m_s=torch.tensor([0.50]),
        hold_steps=torch.tensor([20]),
    )
    assert not bool(same_side["success"].item())
    assert not bool(high_speed["success"].item())


def test_v2_excessive_physical_force_cannot_succeed() -> None:
    result = physical_task_predicates(
        torch.tensor([[20.0, 20.0, 0.0, 0.0, 0.0]]),
        stem_detached=torch.ones(1, dtype=torch.bool),
        center_distance_m=torch.tensor([0.03]),
        relative_speed_m_s=torch.tensor([0.02]),
        hold_steps=torch.tensor([20]),
        maximum_fingertip_force_n=8.0,
    )
    assert not bool(result["within_force_limit"].item())
    assert not bool(result["success"].item())


def test_v2_palm_contact_can_retain_only_after_finger_qualified_detach() -> None:
    palm_hold = physical_task_predicates(
        torch.zeros((1, 5)),
        physical_palm_force_n=torch.tensor([2.0]),
        stem_detached=torch.ones(1, dtype=torch.bool),
        center_distance_m=torch.tensor([0.03]),
        relative_speed_m_s=torch.tensor([0.02]),
        hold_steps=torch.tensor([20]),
    )
    attached = physical_task_predicates(
        torch.zeros((1, 5)),
        physical_palm_force_n=torch.tensor([2.0]),
        stem_detached=torch.zeros(1, dtype=torch.bool),
        center_distance_m=torch.tensor([0.03]),
        relative_speed_m_s=torch.tensor([0.02]),
        hold_steps=torch.tensor([20]),
    )
    overloaded = physical_task_predicates(
        torch.zeros((1, 5)),
        physical_palm_force_n=torch.tensor([9.0]),
        stem_detached=torch.ones(1, dtype=torch.bool),
        center_distance_m=torch.tensor([0.03]),
        relative_speed_m_s=torch.tensor([0.02]),
        hold_steps=torch.tensor([20]),
    )
    assert bool(palm_hold["palm_support"].item())
    assert bool(palm_hold["success"].item())
    assert not bool(attached["palm_support"].item())
    assert not bool(attached["success"].item())
    assert not bool(overloaded["within_force_limit"].item())
    assert not bool(overloaded["success"].item())


def test_v2_fruit_profiles_are_deterministic_correlated_and_mass_consistent() -> None:
    first = sample_correlated_fruit_contact_profiles(
        8192, generator=torch.Generator().manual_seed(42)
    )
    second = sample_correlated_fruit_contact_profiles(
        8192, generator=torch.Generator().manual_seed(42)
    )
    for key in first:
        assert torch.equal(first[key], second[key])
    expected_mass = 4.0 / 3.0 * math.pi * first["axes_m"].prod(dim=1) * first[
        "density_kg_m3"
    ]
    assert torch.allclose(first["mass_kg"], expected_mass, atol=1.0e-8)
    assert 0.045 < float(first["mass_kg"].min())
    assert float(first["mass_kg"].max()) < 0.22
    stiffness_ripeness = torch.corrcoef(
        torch.stack((first["contact_stiffness_n_m"], first["ripeness"]))
    )[0, 1]
    friction_wetness = torch.corrcoef(
        torch.stack((first["static_friction"], first["wetness"]))
    )[0, 1]
    assert float(stiffness_ripeness) < -0.70
    assert float(friction_wetness) < -0.60
    assert bool((first["dynamic_friction"] < first["static_friction"]).all())


def test_v2_fixed_contact_cases_preserve_soft_nominal_firm_order() -> None:
    cases = fixed_fruit_contact_cases()
    for key in ("effective_stiffness_n_m", "effective_damping_n_s_m", "mass_kg"):
        assert cases["soft"][key] < cases["nominal"][key] < cases["firm"][key]
    assert cases["soft"]["dynamic_friction"] < cases["soft"]["static_friction"]


def test_v2_linear_contact_is_unilateral_and_monotonic() -> None:
    indentation = torch.tensor([-0.001, 0.0, 0.001, 0.002, 0.003])
    speed = torch.zeros_like(indentation)
    stiffness = torch.full_like(indentation, 5000.0)
    damping = torch.full_like(indentation, 25.0)
    force = linear_compliant_contact_force(indentation, speed, stiffness, damping)
    assert torch.equal(force[:2], torch.zeros(2))
    assert torch.all(force[2:] == torch.tensor([5.0, 10.0, 15.0]))
    closing = linear_compliant_contact_force(
        torch.full((2,), 0.001),
        torch.tensor([0.0, 0.1]),
        torch.full((2,), 5000.0),
        torch.full((2,), 25.0),
    )
    assert closing[1] > closing[0]


def test_v2_contact_material_bank_is_bounded_and_deterministic() -> None:
    profiles = sample_correlated_fruit_contact_profiles(
        4096, generator=torch.Generator().manual_seed(7)
    )
    result = quantize_contact_material_bank(profiles, friction_bins=4, compliance_bins=4)
    assert int(result["material_bin"].min()) >= 0
    assert int(result["material_bin"].max()) < 16
    assert len(torch.unique(result["material_bin"])) >= 12
    assert bool((result["material_dynamic_friction"] < result["material_static_friction"]).all())


def test_v2_damage_diagnostics_are_monotonic_and_do_not_change_mdp() -> None:
    cfg = load_physics_v2_profile()["fruit_damage_proxy_v2"]
    assert cfg["affects_reward"] is False
    assert cfg["affects_termination"] is False
    state = initial_fruit_damage_state(torch.zeros((2, 5)))
    previous_score = state["damage_score"].clone()
    for step in range(120):
        safe_force = min(4.0, step * 0.2)
        force = torch.stack(
            (torch.full((5,), safe_force), torch.full((5,), 7.0))
        )
        indentation = torch.stack(
            (torch.full((5,), safe_force / 5500.0), torch.full((5,), 0.006))
        )
        state = update_fruit_damage_diagnostics(
            state, force, indentation, dt_s=1.0 / 120.0, config=cfg
        )
        assert bool((state["damage_score"] >= previous_score).all())
        previous_score = state["damage_score"].clone()
    assert float(state["damage_score"][0].max()) == 0.0
    assert float(state["damage_score"][1].min()) > 0.50
    assert float(state["damage_score"].max()) <= 1.0


def test_v2_stem_modes_are_explicit_and_do_not_clamp_literature_to_hand() -> None:
    fruit = sample_correlated_fruit_contact_profiles(
        20000, generator=torch.Generator().manual_seed(19)
    )
    learn = sample_stem_profiles_v2(fruit, mode="learnability")
    literature = sample_stem_profiles_v2(fruit, mode="literature_axial")
    mixed = sample_stem_profiles_v2(fruit, mode="bend_twist_prior")
    assert bool((learn["stem_tensile_force_n"] <= 0.90 * learn["stem_safe_tangential_capacity_n"] + 1.0e-6).all())
    assert float(learn["stem_translation_feasible"].float().mean()) == 1.0
    assert float(literature["stem_translation_feasible"].float().mean()) == 0.0
    assert float(literature["stem_tensile_force_n"].min()) >= 20.0
    assert float(literature["stem_tensile_force_n"].max()) > 70.0
    assert float(mixed["stem_bending_feasible"].float().mean()) == 1.0
    assert float(mixed["stem_torsion_feasible"].float().mean()) == 1.0
    assert torch.equal(torch.unique(learn["stem_mode_code"]), torch.tensor([0]))
    assert torch.equal(torch.unique(literature["stem_mode_code"]), torch.tensor([1]))
    assert torch.equal(torch.unique(mixed["stem_mode_code"]), torch.tensor([2]))


def test_v2_stem_interaction_includes_tension_shear_bending_and_torsion() -> None:
    axis = torch.tensor([[0.0, 0.0, 1.0]]).repeat(5, 1)
    force = torch.tensor(
        [[0.0, 0.0, -10.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.5, 0.0, -5.0]]
    )
    torque = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.1], [0.1, 0.0, 0.05]]
    )
    values = stem_load_components_v2(
        force,
        torque,
        axis,
        torch.full((5,), 10.0),
        torch.full((5,), 5.0),
        torch.full((5,), 0.2),
        torch.full((5,), 0.1),
    )
    assert torch.allclose(values["combined"][:4], torch.ones(4), atol=1.0e-6)
    assert torch.allclose(values["combined"][4], torch.tensor(1.0), atol=1.0e-6)
    assert values["torsion"][3] == 1.0


def test_v2_uncertainty_sampling_is_deterministic_and_bounded() -> None:
    first = sample_uncertainty_profiles_v2(
        4096, generator=torch.Generator().manual_seed(123)
    )
    second = sample_uncertainty_profiles_v2(
        4096, generator=torch.Generator().manual_seed(123)
    )
    for key in first:
        assert torch.equal(first[key], second[key])
    cfg = load_physics_v2_profile()["uncertainty_v2"]["episode_fixed"]
    mapping = {
        "hand_response_scale": "hand_response_scale",
        "hand_target_bias_rad": "hand_target_bias_rad",
        "hand_deadband_rad": "hand_deadband_rad",
        "hand_backlash_rad": "hand_backlash_rad",
        "arm_response_scale": "arm_response_scale",
        "tactile_gain": "tactile_gain",
        "tactile_bias_n": "tactile_bias_n",
        "tactile_drift_rate_n_s": "tactile_drift_rate_n_s",
        "tactile_saturation_n": "tactile_saturation_n",
        "tactile_release_time_constant_s": "tactile_release_time_constant_s",
        "tactile_dropout_probability": "tactile_dropout_probability",
    }
    for output_key, config_key in mapping.items():
        low, high = cfg[config_key]
        assert float(first[output_key].min()) >= low
        assert float(first[output_key].max()) <= high
    assert int(first["tactile_delay_policy_steps"].min()) >= 0
    assert int(first["tactile_delay_policy_steps"].max()) <= 2


def test_v2_joint_uncertainty_respects_deadband_speed_and_position_limits() -> None:
    current = torch.tensor([[0.0, 0.5, -0.9]])
    desired = torch.tensor([[0.001, 1.5, -2.0]])
    result = apply_joint_target_uncertainty_v2(
        desired,
        current,
        torch.full_like(current, -1.0),
        torch.full_like(current, 1.0),
        torch.full_like(current, 0.1),
        torch.tensor([[1.15, 0.85, 1.10]]),
        torch.tensor([[0.005, 0.005, -0.005]]),
        torch.tensor([[0.002, 0.002, 0.002]]),
        torch.tensor([[0.01, 0.01, 0.01]]),
    )
    assert result[0, 0] == current[0, 0]
    assert bool((torch.abs(result - current) <= 0.100001).all())
    assert bool(((result >= -1.0) & (result <= 1.0)).all())


def test_v2_tactile_delay_release_dropout_and_saturation_are_explicit() -> None:
    physical = torch.zeros((1, 1))
    profiles = {
        "tactile_delay_policy_steps": torch.tensor([2]),
        "tactile_release_time_constant_s": torch.tensor([[0.05]]),
        "tactile_drift_rate_n_s": torch.zeros((1, 1)),
        "tactile_gain": torch.ones((1, 1)),
        "tactile_bias_n": torch.zeros((1, 1)),
        "tactile_saturation_n": torch.tensor([[5.0]]),
        "tactile_dropout_probability": torch.zeros((1, 1)),
    }
    state = initial_tactile_uncertainty_state(physical, maximum_delay_steps=2)
    samples = []
    for value in (10.0, 0.0, 0.0, 0.0, 0.0, 0.0):
        state = update_tactile_uncertainty_v2(
            state,
            torch.tensor([[value]]),
            profiles,
            dt_s=0.05,
            generator=torch.Generator().manual_seed(1),
        )
        samples.append(float(state["measured_n"][0, 0]))
    assert samples[0] == 0.0 and samples[1] == 0.0
    assert 0.0 < samples[2] <= 5.0
    assert samples[3] < samples[2]
    assert samples[4] < samples[3]

    profiles["tactile_delay_policy_steps"] = torch.tensor([0])
    profiles["tactile_dropout_probability"] = torch.ones((1, 1))
    previous = state["measured_n"].clone()
    state = update_tactile_uncertainty_v2(
        state, torch.tensor([[10.0]]), profiles, dt_s=0.05
    )
    assert torch.equal(state["measured_n"], previous)
