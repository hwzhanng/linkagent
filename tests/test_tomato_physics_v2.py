"""CPU invariants for Physics-v2 public-prior crop mechanics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from script.physics_v2_audit_inputs import (
    input_sha256,
    require_current_input_sha256,
)
from simulation.tomato_pick.physics_v2 import (
    discovery_capacity_scale,
    issued_command_history_observation,
    kinematic_retention_margin_score,
    load_bearing_opposition_score,
    load_bearing_opposition_target_n,
    load_physics_v2_profile,
    normalized_controller_tracking_error,
    quantize_engineering_contact_profiles,
    relative_position_history_observation,
    sample_engineering_task_orientations,
    sample_physics_v2_profiles,
    sample_public_abscission_profiles,
    sample_public_morphology_profiles,
    sample_public_task_orientation_categories,
    retention_ready_conditioned_capacity_scale,
    update_multimode_abscission,
    update_twist_integrity,
    v2_outcome_alignment_alpha_for_iteration,
)


def test_qualification_input_fingerprint_rejects_stale_reports(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "candidate.txt"
    input_path.write_text("candidate-v1\n", encoding="utf-8")
    report = {"input_sha256": input_sha256(tmp_path, ("candidate.txt",))}
    require_current_input_sha256(
        report,
        tmp_path,
        ("candidate.txt",),
        report_name="smoke.json",
    )
    input_path.write_text("candidate-v2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="qualification report is stale"):
        require_current_input_sha256(
            report,
            tmp_path,
            ("candidate.txt",),
            report_name="smoke.json",
        )


def test_discovery_capacity_curriculum_reaches_full_public_prior_before_tail() -> None:
    values = [
        discovery_capacity_scale(
            iteration,
            initial_scale=0.10,
            hold_initial_scale_iterations=0.0,
            full_capacity_iteration=40.0,
        )
        for iteration in (0.0, 10.0, 20.0, 25.0, 30.0, 35.0, 37.5, 40.0, 49.0)
    ]
    assert math.isclose(values[0], 0.10)
    assert math.isclose(values[1], 0.240625)
    assert math.isclose(values[2], 0.55)
    assert math.isclose(values[3], 0.715234375)
    assert math.isclose(values[4], 0.859375)
    assert math.isclose(values[5], 0.961328125)
    assert math.isclose(values[6], 0.989892578125)
    assert values[7:] == [1.0, 1.0]


def test_load_bearing_opposition_uses_mass_and_actual_pair_friction() -> None:
    target = load_bearing_opposition_target_n(
        torch.tensor([0.16, 0.01]),
        torch.tensor([0.40, 1.00]),
        full_contact_force_n=0.60,
        safety_factor=1.25,
    )
    assert torch.allclose(target, torch.tensor([2.4525, 0.60]), atol=1.0e-6)

    forces = torch.zeros((4, 5))
    forces[0, :2] = target[0]
    forces[1, :2] = 0.5 * target[0]
    forces[2, 0] = target[0]
    forces[2, 3] = 0.25 * target[0]
    forces[3, :2] = 0.60
    score = load_bearing_opposition_score(
        forces,
        target[0].expand(forces.shape[0]),
        full_contact_force_n=0.60,
    )
    expected_half = 0.5 * (1.0 + 0.25)
    expected_quarter = 0.5 * (1.0 + 0.25)
    expected_contact = 0.5 * (1.0 + float((0.60 / target[0]) ** 2))
    assert torch.allclose(
        score,
        torch.tensor(
            [1.0, expected_half, expected_quarter, expected_contact]
        ),
    )


def test_bending_torsion_assistance_requires_current_retention_readiness() -> None:
    retention_ready = torch.tensor([True, False, True, False])
    bending_torsion_scale = retention_ready_conditioned_capacity_scale(
        0.10, retention_ready
    )
    assert torch.equal(
        bending_torsion_scale, torch.tensor([0.10, 1.0, 0.10, 1.0])
    )
    assert torch.equal(
        retention_ready_conditioned_capacity_scale(1.0, retention_ready),
        torch.ones(4),
    )


def test_kinematic_retention_margin_is_zero_at_strict_boundaries() -> None:
    score = kinematic_retention_margin_score(
        torch.tensor([0.0, 0.06, 0.12, 0.13]),
        torch.tensor([0.0, 0.125, 0.25, 0.10]),
        maximum_distance_m=0.12,
        maximum_relative_speed_m_s=0.25,
    )
    assert torch.allclose(score, torch.tensor([1.0, 0.5, 0.0, 0.0]))


def test_issued_command_history_is_newest_first_and_deployable() -> None:
    history = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    encoded = issued_command_history_observation(history, history_steps=2)
    assert encoded.shape == (2, 8)
    assert torch.equal(encoded[:, :4], history[0])
    assert torch.equal(encoded[:, 4:], history[1])
    with pytest.raises(ValueError):
        issued_command_history_observation(history, history_steps=4)


def test_relative_position_history_is_reset_safe_and_deployable() -> None:
    current = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    previous = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    encoded = relative_position_history_observation(
        current, previous, torch.tensor([False, True])
    )
    assert torch.equal(encoded[0], torch.tensor([1.0, 2.0, 3.0, 0.0, 1.0, 2.0]))
    assert torch.equal(encoded[1], torch.tensor([4.0, 5.0, 6.0, 4.0, 5.0, 6.0]))
    with pytest.raises(ValueError):
        relative_position_history_observation(
            current[:, :2], previous[:, :2], torch.tensor([False, True])
        )


def test_controller_tracking_error_uses_local_action_step_units() -> None:
    position_error_w = torch.tensor(
        [[0.004, -0.008, 0.012], [-0.002, 0.001, 0.003]]
    )
    rotation_error_w = torch.tensor(
        [[0.020, -0.080, 0.030], [0.010, 0.020, -0.030]]
    )
    grasp_basis_w = torch.eye(3).repeat(2, 1, 1)
    grasp_basis_w[1] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    action_scales = torch.tensor([0.004, 0.004, 0.006, 0.04, 0.04, 0.06])
    encoded = normalized_controller_tracking_error(
        position_error_w,
        rotation_error_w,
        grasp_basis_w,
        action_scales,
    )
    assert torch.allclose(
        encoded,
        torch.tensor(
            [
                [1.0, -1.0, 1.0, 0.5, -1.0, 0.5],
                [0.25, 0.5, 0.5, 0.5, -0.25, -0.5],
            ]
        ),
    )
    with pytest.raises(ValueError):
        normalized_controller_tracking_error(
            position_error_w[:, :2],
            rotation_error_w[:, :2],
            grasp_basis_w,
            action_scales,
        )
    with pytest.raises(ValueError):
        normalized_controller_tracking_error(
            position_error_w,
            rotation_error_w,
            grasp_basis_w,
            torch.zeros(6),
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
    assert profile["training_curriculum"] == {
        "classification": (
            "training-only mechanics curriculum with public-capacity and "
            "strict-retention-margin reward normalization; not a crop prior "
            "or acceptance relaxation"
        ),
        "enabled_by_environment_variable": (
            "TOMATO_V2_DISCOVERY_CAPACITY_CURRICULUM=1"
        ),
        "initial_tensile_shear_capacity_scale": 1.0,
        "initial_bending_torsion_capacity_scale": 0.10,
        "hold_initial_scale_iterations": 0,
        "full_public_prior_iteration": 40,
        "bending_torsion_assistance_requires_secure_support": True,
        "bending_torsion_assistance_requires_kinematic_retention": True,
        "stem_load_progress_uses_public_capacity": True,
        "stem_load_progress_requires_kinematic_retention": True,
        "stem_load_progress_uses_retention_margin": True,
        "retention_margin_score": (
            "geometric mean of normalized center-distance and relative-speed "
            "margins; exactly zero at either unchanged strict retention boundary"
        ),
        "capacity_scale_curve": "smoothstep from the first update",
        "applies_to": {
            "tensile_shear": "100% throughout",
            "bending_torsion": (
                "10% at update 0 then smoothstep to 100% at update 40, "
                "but only when the current 120 Hz state satisfies both strict "
                "thumb-plus-other-finger secure support and the unchanged retained "
                "distance/speed predicates; every other state remains 100%"
            ),
        },
        "basis": profile["training_curriculum"]["basis"],
        "reward_policy": (
            "reward weights and topology are unchanged; instantaneous stem-load "
            "progress is normalized to fixed public capacities and, during assisted "
            "training, credited only in a current strict-support and kinematically "
            "retained state, then modulated inside that strict envelope by the "
            "centered-and-slow retention margin"
        ),
        "first_success_tail_requires_full_public_prior": True,
        "evaluation_capacity_scale": 1.0,
    }
    assert profile["training_curriculum"][
        "initial_tensile_shear_capacity_scale"
    ] == 1.0
    assert profile["training_curriculum"][
        "initial_bending_torsion_capacity_scale"
    ] == 0.10
    assert profile["training_observation"] == {
        "classification": (
            "isolated Physics-v2 actor observability repair; no privileged "
            "simulator state and no change to the frozen Clean-v5/V1 interface"
        ),
        "base_actor_dimension": 66,
        "physics_v2_actor_dimension": 103,
        "base_critic_dimension": 83,
        "physics_v2_critic_dimension": 120,
        "added_actor_blocks": [
            "previous normalized perceived fruit-center position relative to "
            "the palm (3D)",
            "current and previous normalized perceived fruit-center position "
            "relative to the fingertip centroid in the palm frame (6D)",
            "current normalized controller command-to-actual end-effector "
            "tracking error in local one-action-step units (6D)",
            "two most recently issued bounded policy commands, newest first "
            "(22D)",
        ],
        "current_actor_block": (
            "current normalized perceived fruit-center position relative to "
            "the palm (3D, inherited first block)"
        ),
        "inherited_action_block": (
            "previous command actually applied by the delayed actuator "
            "surrogate (11D)"
        ),
        "issued_command_history_steps": 2,
        "maximum_sampled_action_delay_steps": 2,
        "reset_semantics": (
            "both previous perceived relative-position samples equal their "
            "current samples and both issued-command history slots are zero "
            "on the first observation after reset"
        ),
        "controller_tracking_error_exposed_to_actor": True,
        "controller_tracking_error_history_steps": 1,
        "privileged_velocity_exposed_to_actor": False,
        "privileged_action_delay_exposed_to_actor": False,
        "privileged_stem_wrench_exposed_to_actor": False,
        "privileged_detachment_exposed_to_actor": False,
        "basis": profile["training_observation"]["basis"],
    }
    assert profile["training_policy"] == {
        "classification": profile["training_policy"]["classification"],
        "actor_model": (
            "single-layer GRU followed by the inherited 512-256-128 ELU "
            "MLP head"
        ),
        "actor_recurrent_type": "gru",
        "actor_recurrent_hidden_dimension": 128,
        "actor_recurrent_layers": 1,
        "actor_observation_dimension": 103,
        "critic_model": "inherited feed-forward 512-256-128 ELU MLP",
        "critic_observation_dimension": 120,
        "truncated_backpropagation_steps": 32,
        "basis": profile["training_policy"]["basis"],
    }
    assert profile["training_objective"] == {
        "classification": profile["training_objective"]["classification"],
        "discovery_endpoint_through_iteration": 10,
        "fully_outcome_aligned_iteration": 40,
        "interpolation": "smoothstep",
        "strict_joint_ready_break_credit_scale": 1.0,
        "strict_joint_ready_break_credit_weight": (
            "unchanged BREAK_GRASPED_WEIGHT"
        ),
        "time_cost_alignment_scale": 0.0,
        "time_cost_weight": "unchanged DISCOVERY_TIME_COST_WEIGHT",
        "failure_and_stable_detach_alignment_uses_schedule": True,
        "unchanged_reward_term_set": True,
        "unchanged_individual_reward_weights": True,
        "clean_v5_and_physics_v1_behavior_unchanged": True,
        "basis": profile["training_objective"]["basis"],
    }
    opposition = profile["training_shaping"]["load_bearing_opposition"]
    assert opposition["enabled"] is True
    assert opposition["safety_factor"] == 1.25
    assert opposition["reward_weight"] == (
        "unchanged inherited OPPOSED_PROGRESS_WEIGHT"
    )
    assert "mass*gravity/(2*effective_pair_static_friction)" in opposition[
        "equation"
    ]
    assert profile["training"]["first_success_gate"] == {
        "script": "script/gate_tomato_physics_v2_first_success.py",
        "candidate_id": "physics_v2_retention_objective_v1",
        "num_envs": 8192,
        "updates": 50,
        "seed": 42,
        "output": "logs/qualification/tomato_physics_v2/first_success_8192x50_gate.json",
        "required_before_curve_controller": True,
    }
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
    assert ".training.first_success_gate.output" in source
    assert "first-success gate did not pass" in source
    assert "configured_gate_spec.candidate_id" in source
    assert "script/physics_v2_audit_inputs.py" in source
    assert "TOMATO_V2_DISCOVERY_CAPACITY_CURRICULUM=1" in source
    assert "tail_full_public_prior_capacity" in source
    assert profile["training"]["curve_controller"] == (
        "script/run_tomato_physics_v2_curve_controller.sh"
    )
    assert "max_iterations = 400" in runner_source
    assert "max_iterations = 1800" not in runner_source


def test_first_success_budget_is_profile_driven_and_never_auto_extended() -> None:
    root = Path(__file__).resolve().parents[1]
    gate_source = (
        root / "script/gate_tomato_physics_v2_first_success.py"
    ).read_text()
    train_source = (root / "script/train_tomato_physics_v2.py").read_text()
    profile = json.loads(
        (
            root
            / "simulation/config/tomato_pick_physics_v2_public_prior_v1.json"
        ).read_text()
    )
    assert profile["training"]["first_success_gate"]["updates"] == 50
    assert 'default=configured_updates' in gate_source
    assert '"automatic_extension": False' in gate_source
    assert '"requires_user_approval": True' in gate_source
    assert '"eligible_to_request_extension"' in gate_source
    assert 'configured_updates - 1' in gate_source
    assert 'int(gate_spec["updates"])' in train_source
    assert '"hold_initial_scale_iterations"' in gate_source
    assert '"Metrics/ever_retained_rate": "ever_retained_rate"' in gate_source
    assert '"training_objective": profile["training_objective"]' in gate_source
    assert "require_current_input_sha256" in gate_source
    assert 'report.get("candidate_id")' in gate_source
    assert '"--preflight-only"' in gate_source
    assert '"training_started": False' in gate_source
    assert '"hardware_connected": False' in gate_source
    assert 'positive(tail, "ever_retained_rate") >= 3' in gate_source
    assert 'float(record.get("ever_retained_rate", 0.0)) > 0.0' in gate_source


def test_v2_actor_uses_bounded_recurrent_memory_without_recurrent_critic() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "simulation/tomato_pick/isaac/physics_v2/agents/rsl_rl_ppo_cfg.py"
    ).read_text()
    assert "RslRlRNNModelCfg" in source
    assert 'rnn_type="gru"' in source
    assert "rnn_hidden_dim=128" in source
    assert "rnn_num_layers=1" in source
    assert source.count("actor = RslRlRNNModelCfg(") == 1
    assert "critic = RslRlRNNModelCfg(" not in source


def test_v2_logs_episode_level_retention_latches_without_reward_change() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "simulation/tomato_pick/isaac/physics_v2/env.py"
    ).read_text()
    assert "self._ever_retained |= self._retained" in source
    assert "self._maximum_hold_streak = torch.maximum(" in source
    assert '"Metrics/ever_retained_rate"' in source
    assert '"Metrics/mean_max_hold_steps"' in source
    assert "self._ever_retained[env_ids] = False" in source
    assert "self._maximum_hold_streak[env_ids] = 0" in source


def test_v2_alignment_schedule_fits_the_first_success_gate() -> None:
    assert v2_outcome_alignment_alpha_for_iteration(0) == 0.0
    assert v2_outcome_alignment_alpha_for_iteration(10) == 0.0
    assert v2_outcome_alignment_alpha_for_iteration(25) == 0.5
    assert v2_outcome_alignment_alpha_for_iteration(40) == 1.0
    assert v2_outcome_alignment_alpha_for_iteration(50) == 1.0
    samples = [
        v2_outcome_alignment_alpha_for_iteration(i) for i in range(51)
    ]
    assert all(left <= right for left, right in zip(samples, samples[1:]))


def test_v2_preserves_only_the_strict_joint_ready_break_bridge() -> None:
    root = Path(__file__).resolve().parents[1]
    base_source = (root / "simulation/tomato_pick/isaac/env.py").read_text()
    v2_source = (
        root / "simulation/tomato_pick/isaac/physics_v2/env.py"
    ).read_text()
    reward_source = (root / "simulation/tomato_pick/reward.py").read_text()
    assert "def _break_grasped_reward_scale" in base_source
    assert "return 1.0 - float(alignment_alpha)" in base_source
    assert "break_grasped_alignment_scale" in reward_source
    override = v2_source.split(
        "def _break_grasped_reward_scale", 1
    )[1].split("def _apply_public_orientation", 1)[0]
    assert "return 1.0" in override
    assert "self._break_substep_grasp_quality" in v2_source
    assert "self._break_substep_secure_support" in v2_source
    assert "self._break_substep_kinematic_retained" in v2_source


def test_v2_keeps_dense_time_cost_small_while_terminal_alignment_progresses() -> None:
    root = Path(__file__).resolve().parents[1]
    base_source = (root / "simulation/tomato_pick/isaac/env.py").read_text()
    v2_source = (
        root / "simulation/tomato_pick/isaac/physics_v2/env.py"
    ).read_text()
    reward_source = (root / "simulation/tomato_pick/reward.py").read_text()
    assert "def _time_cost_alignment_scale" in base_source
    assert "return float(alignment_alpha)" in base_source
    assert "time_cost_alignment_scale" in reward_source
    override = v2_source.split(
        "def _time_cost_alignment_scale", 1
    )[1].split("def _apply_public_orientation", 1)[0]
    assert "return 0.0" in override


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


def test_v2_keeps_v1_reward_hierarchy_and_uses_mode_aware_training_mechanics() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "simulation/tomato_pick/isaac/physics_v2/env.py"
    ).read_text()
    stem_progress = source.split("def _stem_progress_reward_score", 1)[1].split(
        "def _apply_public_orientation", 1
    )[0]
    assert "stem_load_damage_progress_score" in stem_progress
    assert "secure_support_credit_gate" in stem_progress
    assert "self._shadow_secure_support" in stem_progress
    assert "def _get_rewards" not in source
    assert "def _break_grasped_reward_score" in source
    break_credit = source.split("def _break_grasped_reward_score", 1)[1].split(
        "def _apply_public_orientation", 1
    )[0]
    assert "self._break_substep_grasp_quality" in break_credit
    assert "self._break_substep_secure_support" in break_credit
    assert "self._break_substep_kinematic_retained" in break_credit
    assert "fruit_in_hand.float()" not in break_credit
    assert "def _retention_reward_score" not in source
    assert "controlled_break" not in source
    assert "retention_readiness" not in source
    assert "scaled_stem_progress" not in source
    assert "discovery_load_ratio_reward_scale" not in source
    assert "discovery_front_end_progress_scale" not in source
    assert "mean_break_relative_speed_m_s" in source
    assert "break_kinematic_retained_rate" in source
    assert "break_joint_ready_rate" in source
    assert "PhysicsV2/discovery_capacity_scale" in source
    assert "discovery_capacity_scale(" in source
    assert "tensile_limit_n = self._stem_tensile_limit_n * tensile_shear_scale" in source
    assert (
        "bending_limit_nm = self._stem_bending_limit_nm * bending_torsion_scale"
        in source
    )
    assert (
        "torsion_limit_nm = self._stem_torsion_limit_nm * bending_torsion_scale"
        in source
    )
    assert "PhysicsV2/tensile_shear_capacity_scale" in source
    assert "PhysicsV2/bending_torsion_capacity_scale" in source
    assert "retention_ready_conditioned_capacity_scale" in source
    assert "_stem_public_combined_load_ratio" in source
    assert "_stem_load_progress_ready" in source
    assert "_stem_load_progress_retention_margin" in source
    assert "kinematic_retention_margin_score" in source
    assert "combined_abscission_load_ratio" in source
    assert "self._stem_tensile_limit_n" in source
    assert "credit_ready" in source
    assert "physical_force = self._physical_finger_force()" in source
    assert "curriculum_secure_support" in source
    assert "curriculum_kinematic_retention" in source
    assert "curriculum_retention_ready" in source
    assert "PhysicsV2/bending_torsion_curriculum_assisted_break_rate" in source
    assert "def _get_observations" in source
    assert "previous_relative_position" in source
    assert "issued_command_history_observation" in source
    assert "relative_position_history_observation" in source
    assert "fingertip_center" in source
    assert "tip_center_position_history" in source
    assert "normalized_controller_tracking_error" in source
    assert "_break_substep_grasp_quality" in source
    assert "compute_pose_error" in source
    assert "controller_tracking_error" in source
    assert "previous_controller_tracking_error" not in source
    assert "history_steps=2" in source
    assert "_previous_perceived_relative_position_reset_pending" in source
    assert "Exact simulator fruit" in source
    assert "velocity and sampled delay remain hidden" in source
    cfg_source = (
        root / "simulation/tomato_pick/isaac/physics_v2/env_cfg.py"
    ).read_text()
    assert (
        "PHYSICS_V2_OBSERVATION_DIM = OBSERVATION_DIM + 3 + 6 + 6 + 2 * ACTION_DIM"
        in cfg_source
    )
    assert (
        "PHYSICS_V2_CRITIC_STATE_DIM = CRITIC_STATE_DIM + 3 + 6 + 6 + 2 * ACTION_DIM"
        in cfg_source
    )
