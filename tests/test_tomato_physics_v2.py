"""CPU invariants for Physics-v2 public-prior crop mechanics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from simulation.tomato_pick.physics_v2 import (
    load_physics_v2_profile,
    quantize_engineering_contact_profiles,
    sample_engineering_task_orientations,
    sample_physics_v2_profiles,
    sample_public_abscission_profiles,
    sample_public_morphology_profiles,
    sample_public_task_orientation_categories,
    update_multimode_abscission,
    update_twist_integrity,
)


def test_profile_is_hash_pinned_and_forbids_v1_resume() -> None:
    profile = load_physics_v2_profile()
    assert profile["task_id"] == "LinkAgent-TomatoPick-Physics-v2-PublicPrior-v1"
    assert profile["training"]["from_scratch_required"] is True
    assert profile["training"]["physics_v1_checkpoint_resume_allowed"] is False
    assert profile["training"]["same_profile_full_state_resume_allowed"] is True
    assert profile["qualification"]["passed"] is True
    assert profile["status"] == (
        "qualified_ready_for_external_audit_training_not_started"
    )
    assert profile["training"]["segment_updates"] == 400
    assert profile["training"]["fixed_total_iterations"] is None
    assert profile["paper_strategy_audit"]["diagnostic_only_not_rewarded"] is True
    assert profile["paper_strategy_audit"]["final_axial_pull_n"] == [1.0, 15.29]
    assert (
        profile["engineering_prior_data"]["profile_id"]
        == "tomato_pick_physics_v2_engineering_prior_v1"
    )
    assert "4-8 N effective detachment-force prior" in profile["frozen_parent"][
        "do_not_reuse"
    ]


def test_morphology_sampler_is_correlated_and_rejects_unphysical_samples() -> None:
    generator = torch.Generator().manual_seed(17)
    samples = sample_public_morphology_profiles(20000, generator=generator)
    assert torch.isfinite(samples["diameters_m"]).all()
    assert (samples["diameters_m"] > 0.0).all()
    assert (samples["mass_kg"] >= 0.05).all()
    assert (samples["mass_kg"] <= 0.40).all()
    assert (samples["density_kg_m3"] >= 800.0).all()
    assert (samples["density_kg_m3"] <= 1200.0).all()
    assert (samples["polar_aspect_ratio"] >= 0.65).all()
    assert (samples["polar_aspect_ratio"] <= 1.20).all()
    for index in range(3):
        selected = samples["source_profile_index"] == index
        size = samples["diameters_m"][selected].mean(dim=1)
        mass = samples["mass_kg"][selected]
        assert torch.corrcoef(torch.stack((size, mass)))[0, 1] > 0.35


def test_abscission_axial_capacity_depends_on_diameter() -> None:
    generator = torch.Generator().manual_seed(42)
    small = sample_public_abscission_profiles(
        30000, diameter_m=0.00552, generator=generator
    )
    large = sample_public_abscission_profiles(
        30000, diameter_m=0.00775, generator=generator
    )
    assert abs(float(small["stem_tensile_force_n"].mean()) - 40.262) < 1.0
    assert abs(float(large["stem_tensile_force_n"].mean()) - 72.003) < 1.5
    assert float(large["stem_tensile_force_n"].mean()) > float(
        small["stem_tensile_force_n"].mean()
    )
    assert float(small["stem_tensile_force_n"].mean()) > 8.0


def test_360_degree_twist_matches_public_residual_pull_target() -> None:
    profile = load_physics_v2_profile()
    config = profile["abscission"]
    attached = torch.ones(1, dtype=torch.bool)
    integrity = update_twist_integrity(
        torch.ones(1),
        torch.ones(1),
        torch.ones(1),
        torch.tensor([2.0 * math.pi]),
        k_theta_per_rad=config["twist_integrity_k_per_rad"],
        exponent_q=config["twist_integrity_q"],
        attached=attached,
    )
    assert abs(float(integrity) - 0.2603783) < 0.001
    assert abs(float(integrity) * 44.781 - 11.66) < 0.1


def test_360_degree_twist_is_invariant_to_rollout_step_size() -> None:
    profile = load_physics_v2_profile()
    config = profile["abscission"]
    attached = torch.ones(1, dtype=torch.bool)
    integrity = torch.ones(1)
    intact_torsion_limit = torch.ones(1)
    step_count = 360
    for _ in range(step_count):
        current_yield_torque = intact_torsion_limit * integrity
        integrity = update_twist_integrity(
            integrity,
            current_yield_torque,
            intact_torsion_limit,
            torch.tensor([2.0 * math.pi / step_count]),
            k_theta_per_rad=config["twist_integrity_k_per_rad"],
            exponent_q=config["twist_integrity_q"],
            attached=attached,
        )
    assert abs(float(integrity) - 0.2603783) < 0.001


def test_torsion_yield_does_not_double_count_duration_damage() -> None:
    profile = load_physics_v2_profile()
    config = profile["abscission"]
    attached = torch.ones(1, dtype=torch.bool)
    state = update_multimode_abscission(
        torch.ones(1),
        torch.zeros(1),
        torch.zeros((1, 3)),
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([44.781]),
        torch.tensor([33.13794]),
        torch.tensor([0.112134]),
        torch.ones(1),
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
    assert float(state["torsion_ratio"]) <= 1.0
    assert float(state["load_damage"]) == 0.0
    assert not bool(state["broken"])


def test_world_orientation_sampler_matches_public_distribution() -> None:
    generator = torch.Generator().manual_seed(2026)
    categories = sample_public_task_orientation_categories(
        100000, generator=generator
    )
    actual = torch.bincount(categories, minlength=4).float() / categories.numel()
    target = torch.tensor([0.119, 0.111, 0.063, 0.706])
    assert (actual - target).abs().max() < 0.01


def test_continuous_orientation_prior_preserves_categories_and_bounds() -> None:
    generator = torch.Generator().manual_seed(17)
    samples = sample_engineering_task_orientations(100000, generator=generator)
    categories = samples["world_orientation_category"]
    actual = torch.bincount(categories, minlength=4).float() / categories.numel()
    assert (actual - torch.tensor([0.119, 0.111, 0.063, 0.706])).abs().max() < 0.01
    tilt_deg = torch.rad2deg(samples["orientation_polar_tilt_rad"])
    jitter_deg = torch.rad2deg(samples["orientation_azimuth_jitter_rad"])
    assert float(tilt_deg.min()) >= 0.0
    assert float(tilt_deg.max()) <= 15.0
    assert float(tilt_deg.std()) > 1.0
    assert float(jitter_deg.min()) >= -45.0
    assert float(jitter_deg.max()) <= 45.0


def test_engineering_gap_priors_are_bounded_and_contact_mapping_is_exact() -> None:
    generator = torch.Generator().manual_seed(42)
    samples = sample_physics_v2_profiles(30000, generator=generator)
    assert float(samples["stem_shear_ratio"].min()) >= 0.60
    assert float(samples["stem_shear_ratio"].max()) <= 0.90
    assert abs(float(samples["stem_shear_ratio"].mean()) - 0.7423) < 0.03
    assert float(samples["stem_subcritical_onset_ratio"].min()) >= 0.45
    assert float(samples["stem_subcritical_onset_ratio"].max()) <= 0.75
    assert float(samples["stem_subcritical_damage_exponent"].min()) >= 1.5
    assert float(samples["stem_subcritical_damage_exponent"].max()) <= 3.0
    assert float(samples["stem_critical_duration_s"].min()) >= 0.20
    assert float(samples["stem_critical_duration_s"].max()) <= 1.20
    assert float(samples["effective_pair_static_friction"].min()) >= 0.30
    assert float(samples["effective_pair_static_friction"].max()) <= 0.70
    assert float(samples["contact_stiffness_n_m"].min()) >= 3000.0
    assert float(samples["contact_stiffness_n_m"].max()) <= 12000.0
    materials = quantize_engineering_contact_profiles(
        samples, friction_bins=4, compliance_bins=4
    )
    effective_static = 0.5 * (materials["material_static_friction"] + 0.55)
    effective_dynamic = 0.5 * (materials["material_dynamic_friction"] + 0.45)
    assert torch.allclose(
        effective_static, materials["effective_pair_static_friction"]
    )
    assert torch.allclose(
        effective_dynamic, materials["effective_pair_dynamic_friction"]
    )


def test_vectorized_fatigue_priors_change_damage_rate_per_environment() -> None:
    attached = torch.ones(2, dtype=torch.bool)
    state = update_multimode_abscission(
        torch.ones(2),
        torch.zeros(2),
        torch.tensor([[0.0, 0.0, -8.0], [0.0, 0.0, -8.0]]),
        torch.zeros((2, 3)),
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        torch.full((2,), 10.0),
        torch.full((2,), 8.0),
        torch.full((2,), 0.1),
        torch.full((2,), 0.1),
        torch.zeros(2),
        dt_s=0.05,
        attached=attached,
        k_theta_per_rad=0.214162,
        twist_exponent_q=1.0,
        subcritical_onset_ratio=torch.tensor([0.45, 0.75]),
        subcritical_exponent=torch.tensor([1.5, 3.0]),
        subcritical_time_constant_s=torch.tensor([0.20, 1.20]),
        instantaneous_failure_ratio=1.25,
        torsion_detach_integrity_threshold=0.15,
        capacity_floor_fraction=0.001,
    )
    assert float(state["load_damage"][0]) > float(state["load_damage"][1])
    assert torch.isfinite(state["load_damage"]).all()


def test_curve_controller_has_explicit_segments_and_no_hidden_cap() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "script/run_tomato_physics_v2_curve_controller.sh").read_text()
    profile = json.loads(
        (
            root
            / "simulation/config/tomato_pick_physics_v2_public_prior_v1.json"
        ).read_text()
    )
    runner_source = (
        root
        / "simulation/tomato_pick/isaac/physics_v2/agents/rsl_rl_ppo_cfg.py"
    ).read_text()
    assert 'segment_updates=400' in source
    assert '--max_iterations "${segment_updates}"' in source
    assert 'hard_iteration_cap:null' in source
    assert 'float(\'${current_rate}\') >= .90' in source
    assert 'int(\'${plateau_streak}\') >= 2' in source
    assert profile["training"]["curve_controller"] == (
        "script/run_tomato_physics_v2_curve_controller.sh"
    )
    assert "max_iterations = 400" in runner_source
    assert "max_iterations = 1800" not in runner_source


def test_paper_strategy_requires_physical_retention_and_final_success() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "simulation/tomato_pick/isaac/physics_v2/env.py"
    ).read_text()
    assert "(twist_assisted | bend_assisted)" in source
    assert "& moderate_pull" in source
    assert "& substep_secure" in source
    assert "& self._break_substep_kinematic_retained" in source
    assert "self._stem_effective_twist_rad" in source
    assert "self.cfg.paper_strategy_final_axial_pull_n" in source
    assert "PhysicsV2/paper_strategy_success_rate" in source
    assert "& self._success[env_ids]" in source
