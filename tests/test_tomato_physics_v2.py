from __future__ import annotations

import ast
from pathlib import Path

from simulation.tomato_pick.physics_v2 import (
    load_physics_v2_profile,
    physics_v2_step1_report,
    verify_parent_physics_v1_baseline,
    verify_physics_v2_robot_asset,
)


ROOT = Path(__file__).resolve().parents[1]


def _class_methods(path: Path, class_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
    raise AssertionError(f"class not found: {class_name}")


def test_physics_v2_step1_is_independent_and_training_blocked() -> None:
    profile = load_physics_v2_profile()
    assert profile["profile_id"] == "tomato_pick_physics_v2"
    assert profile["parent_baseline"]["profile_id"] == "tomato_pick_physics_v1"
    issues = profile["parent_baseline"]["known_issues"]
    assert [issue["id"] for issue in issues] == [
        "physics_v1_derived_usd_declared_hash_mismatch"
    ]
    assert issues[0]["status"] == "confirmed_parent_issue_preserved"
    changes = profile["implementation_boundary"]["semantic_changes"]
    assert changes[0] == "independent_physics_v2_robot_asset_identity_without_parameter_change"
    assert "diagnostic_only_fruit_damage_exposure_telemetry" in changes
    assert profile["implementation_boundary"]["training_allowed"] is False
    assert profile["stage_gate"]["ready"] is False
    assert profile["stage_gate"]["formal_training_started"] is False
    assert profile["stage_gate"]["blockers"]


def test_physics_v2_preserves_learning_contract() -> None:
    contract = load_physics_v2_profile()["learning_contract"]
    assert contract["actor_observation_dim"] == 66
    assert contract["critic_state_dim"] == 83
    assert contract["action_dim"] == 11
    assert contract["physics_rate_hz"] == 120
    assert contract["policy_rate_hz"] == 20


def test_physics_v2_parent_baseline_hashes_match() -> None:
    result = verify_parent_physics_v1_baseline()
    assert result["passed"] is True
    assert len(result["files"]) >= 6
    assert all(record["matches"] for record in result["files"].values())


def test_physics_v2_parameter_evidence_is_complete() -> None:
    profile = load_physics_v2_profile()
    allowed = set(profile["parameter_evidence_schema"]["allowed_statuses"])
    families = profile["parameter_families"]
    assert {
        "robot_geometry_and_collision",
        "hand_mass_inertia_and_actuation",
        "arm_mass_inertia_and_actuation",
        "fruit_geometry_mass_and_density",
        "finger_pad_fruit_contact",
        "fruit_damage_proxy",
        "stem_detachment",
        "tactile_measurement",
    } <= set(families)
    for family in families.values():
        assert family["evidence_status"] in allowed
        assert family["uncertainty"]
        assert family["next_stage"] >= 2


def test_physics_v2_preserves_parent_and_learning_interface_methods() -> None:
    methods = _class_methods(
        ROOT / "simulation/tomato_pick/isaac/physics_v2/env.py",
        "TomatoPickPhysicsV2Env",
    )
    assert set(methods) == {
        "__init__",
            "_apply_action",
            "_apply_attachment_wrench",
            "_apply_workspace_reset",
            "_additional_excess_force_mask",
            "_get_dones",
            "_measured_finger_pressure",
            "_move_stem_visual",
            "_pre_physics_step",
            "_physical_palm_force",
            "_retention_reference",
            "_retention_support_diagnostics",
            "_retention_support_mask",
            "_reset_idx",
    }
    assert "_get_observations" not in methods
    assert "_get_rewards" not in methods


def test_physics_v2_stage2_has_independent_composition_asset() -> None:
    profile = load_physics_v2_profile()
    assert profile["stage_gate"]["current_stage"] >= 2
    assert profile["robot_asset_v2"]["physical_parameter_changes"] == []
    result = verify_physics_v2_robot_asset(profile)
    assert result["passed"] is True
    assert all(result["checks"].values())


def test_physics_v2_env_config_selects_independent_asset() -> None:
    source = (
        ROOT / "simulation/tomato_pick/isaac/physics_v2/env_cfg.py"
    ).read_text(encoding="utf-8")
    assert "PHYSICS_V2_ROBOT_USD_PATH" in source
    assert "robot.spawn.usd_path" in source
    assert _class_methods(
        ROOT / "simulation/tomato_pick/isaac/physics_v2/env_cfg.py",
        "TomatoPickPhysicsV2EnvCfg",
    ) == []


def test_physics_v2_step1_report_passes() -> None:
    report = physics_v2_step1_report()
    assert report["passed"] is True
    assert report["training_allowed"] is False
    assert report["checks"]["parent_integrity_discrepancy_documented"] is True
    assert report["next_stage"] == "static_robot_asset_audit"
