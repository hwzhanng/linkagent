"""Pure profile and baseline-integrity helpers for offline Physics-v2 work.

Step 1 deliberately introduces no new physical equation.  It establishes an
independent identity and fails closed if the frozen Physics-v1 parent changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from simulation.tomato_pick.config import REPOSITORY_ROOT, load_object
from simulation.tomato_pick.physics_v1 import load_physics_v1_profile


PHYSICS_V2_PROFILE_PATH = (
    Path(__file__).resolve().parents[1] / "config/tomato_pick_physics_v2.yaml"
)
PHYSICS_V2_ROBOT_USD_PATH = REPOSITORY_ROOT / (
    "botSDK/urdf/02_project_derived/isaac_assets/"
    "linkbot_bimanual_sdk_training_mass1kg/"
    "linkbot_bimanual_sdk_training_mass1kg_physics_v2.usda"
)


def load_physics_v2_profile(
    path: Path = PHYSICS_V2_PROFILE_PATH,
) -> dict[str, Any]:
    """Load and validate the fail-closed Physics-v2 scaffold profile."""

    profile = load_object(path)
    if profile.get("schema_version") != 1:
        raise ValueError("Physics-v2 profile schema_version must be 1")
    if profile.get("profile_id") != "tomato_pick_physics_v2":
        raise ValueError("unexpected Physics-v2 profile_id")
    stage_status = {
        1: "offline_scaffold_step1",
        2: "offline_asset_audit_step2",
        3: "offline_contact_damage_step3",
        4: "offline_stem_modes_step4",
        5: "offline_uncertainty_step5",
        6: "offline_isolated_gates_step6",
        7: "offline_scripted_feasibility_step7",
        8: "stage8_integrated_gates_passed_numerical_ppo_pending",
        9: "stage9_bounded_learning_artifact_and_evaluation_pending",
    }

    parent = profile.get("parent_baseline")
    if not isinstance(parent, dict):
        raise ValueError("Physics-v2 parent_baseline must be an object")
    if parent.get("profile_id") != "tomato_pick_physics_v1":
        raise ValueError("Physics-v2 must descend from the frozen Physics-v1 profile")
    frozen = parent.get("frozen_files")
    if not isinstance(frozen, dict) or not frozen:
        raise ValueError("Physics-v2 must freeze at least one parent file")
    for relative, digest in frozen.items():
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError("frozen parent paths must be safe repository-relative paths")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("frozen parent digests must be SHA-256 strings")
    known_issues = parent.get("known_issues")
    if not isinstance(known_issues, list) or not known_issues:
        raise ValueError("Physics-v2 must preserve known parent integrity issues")
    expected_issue = "physics_v1_derived_usd_declared_hash_mismatch"
    if expected_issue not in {issue.get("id") for issue in known_issues if isinstance(issue, dict)}:
        raise ValueError("Physics-v1 derived-USD hash mismatch must remain documented")

    contract = profile.get("learning_contract")
    if not isinstance(contract, dict):
        raise ValueError("Physics-v2 learning_contract must be an object")
    expected_contract = {
        "actor_observation_dim": 66,
        "critic_state_dim": 83,
        "action_dim": 11,
        "physics_rate_hz": 120,
        "policy_rate_hz": 20,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            raise ValueError(f"Physics-v2 learning contract mismatch: {key}")

    gate = profile.get("stage_gate")
    if not isinstance(gate, dict) or gate.get("current_stage") not in stage_status:
        raise ValueError("Physics-v2 must declare a supported offline stage")
    current_stage = gate["current_stage"]
    if profile.get("status") != stage_status[current_stage]:
        raise ValueError("Physics-v2 status does not match its current stage")

    boundary = profile.get("implementation_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("Physics-v2 implementation_boundary must be an object")
    if not isinstance(boundary.get("semantic_changes"), list):
        raise ValueError("Physics-v2 semantic_changes must be a list")
    if current_stage == 1 and (
        boundary.get("physics_behavior") != "inherited_exactly_from_physics_v1"
        or boundary["semantic_changes"] != []
    ):
        raise ValueError("step 1 cannot introduce Physics-v2 behavior changes")
    if current_stage >= 2:
        asset = profile.get("robot_asset_v2")
        if not isinstance(asset, dict) or len(asset.get("sha256", "")) != 64:
            raise ValueError("Physics-v2 stage 2+ requires an independent robot asset")

    success_contract = profile.get("success_contract_parity")
    if not isinstance(success_contract, dict):
        raise ValueError("Physics-v2 success_contract_parity must be an object")
    thresholds = success_contract.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("Physics-v2 success contract thresholds must be an object")
    secure_success = load_physics_v1_profile().get("secure_success")
    common_stem = profile.get("stem_modes_v2", {}).get("common")
    if not isinstance(secure_success, dict) or not isinstance(common_stem, dict):
        raise ValueError("Physics-v2 success contract references missing parent thresholds")
    expected_thresholds = {
        "minimum_supported_fingertips": secure_success.get("minimum_supported_fingertips"),
        "minimum_support_force_n": secure_success.get("minimum_support_force_n"),
        "maximum_center_distance_m": secure_success.get("maximum_center_distance_m"),
        "maximum_relative_speed_m_s": secure_success.get("maximum_relative_speed_m_s"),
        "hold_policy_steps": secure_success.get("hold_policy_steps"),
        "maximum_fingertip_force_n": common_stem.get("safe_maximum_fingertip_force_n"),
        # These are the explicit v2 contract values.  They must not be copied
        # from the candidate profile itself, otherwise a drifted profile could
        # validate against its own incorrect thresholds.
        "minimum_palm_support_force_n": 0.25,
        "palm_support_requires_detach": True,
    }
    for key, expected in expected_thresholds.items():
        if key not in thresholds:
            raise ValueError(f"Physics-v2 success contract is missing {key}")
        if thresholds[key] != expected:
            raise ValueError(
                f"Physics-v2 success contract mismatch: {key} must match {expected!r}"
            )
    if thresholds["minimum_palm_support_force_n"] <= 0.0:
        raise ValueError("Physics-v2 palm support threshold must be positive")
    if thresholds["maximum_fingertip_force_n"] <= 0.0:
        raise ValueError("Physics-v2 force ceiling must be positive")
    if success_contract.get("status") != "explicit_palm_support_after_qualified_detach":
        raise ValueError("Physics-v2 success contract status is not explicit palm support")
    semantic_difference = success_contract.get("semantic_difference")
    if not isinstance(semantic_difference, dict):
        raise ValueError("Physics-v2 success contract semantic_difference must be an object")
    if semantic_difference.get("resolution_status") != (
        "implemented_static_pending_exact_runtime_gpu_validation"
    ):
        raise ValueError("Physics-v2 success contract must remain pending exact GPU validation")
    if boundary.get("training_allowed") is not False:
        raise ValueError("Physics-v2 training must remain fail-closed at step 1")

    schema = profile.get("parameter_evidence_schema")
    families = profile.get("parameter_families")
    if not isinstance(schema, dict) or not isinstance(families, dict) or not families:
        raise ValueError("Physics-v2 parameter evidence registry is incomplete")
    allowed = set(schema.get("allowed_statuses", []))
    required = {
        "vendor_verified",
        "literature_prior",
        "geometry_derived",
        "numerical_guard",
        "hardware_pending",
    }
    if allowed != required:
        raise ValueError("Physics-v2 evidence statuses do not match the frozen schema")
    for name, family in families.items():
        if not isinstance(family, dict) or family.get("evidence_status") not in allowed:
            raise ValueError(f"invalid evidence status for parameter family: {name}")
        if not family.get("uncertainty") or not isinstance(family.get("next_stage"), int):
            raise ValueError(f"incomplete evidence record for parameter family: {name}")

    if gate.get("ready") is not False or gate.get("formal_training_started") is not False:
        raise ValueError("offline Physics-v2 cannot be training ready or started")
    if not gate.get("blockers"):
        raise ValueError("Physics-v2 stage 1 must retain explicit blockers")
    return profile


def verify_parent_physics_v1_baseline(
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify every frozen parent artifact without mutating the repository."""

    if profile is None:
        profile = load_physics_v2_profile()
    records: dict[str, dict[str, Any]] = {}
    passed = True
    for relative, expected in profile["parent_baseline"]["frozen_files"].items():
        path = REPOSITORY_ROOT / relative
        exists = path.is_file()
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
        matches = exists and actual == expected
        passed &= matches
        records[relative] = {
            "exists": exists,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": matches,
        }
    return {"passed": passed, "files": records}


def verify_physics_v2_robot_asset(
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the independent stage-2 layer and its frozen parent reference."""

    if profile is None:
        profile = load_physics_v2_profile()
    asset = profile["robot_asset_v2"]
    path = REPOSITORY_ROOT / asset["path"]
    exists = path.is_file()
    text = path.read_text(encoding="utf-8") if exists else ""
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
    parent_name = "linkbot_bimanual_sdk_training_mass1kg_physics_v1.usda"
    checks = {
        "asset_exists": exists,
        "asset_hash_matches": actual == asset["sha256"],
        "parent_reference_is_relative": f"@./{parent_name}@" in text,
        "parent_actual_hash_is_embedded": asset["parent_actual_sha256"] in text,
        "training_is_blocked_in_layer_metadata": (
            'string physicsV2TrainingPermission = "blocked"' in text
        ),
        "no_physical_parameter_change_declared": asset["physical_parameter_changes"] == [],
    }
    return {
        "passed": all(checks.values()),
        "path": asset["path"],
        "expected_sha256": asset["sha256"],
        "actual_sha256": actual,
        "checks": checks,
    }


def physics_v2_step1_report() -> dict[str, Any]:
    """Return the machine-readable acceptance report for scaffold step 1."""

    profile = load_physics_v2_profile()
    baseline = verify_parent_physics_v1_baseline(profile)
    checks = {
        "parent_baseline_unchanged": baseline["passed"],
        "learning_contract_frozen": profile["learning_contract"] == {
            "actor_observation_dim": 66,
            "critic_state_dim": 83,
            "action_dim": 11,
            "physics_rate_hz": 120,
            "policy_rate_hz": 20,
            "inheritance": "exact_physics_v1_contract_until_an_explicit_stage_gate_changes_it",
        },
        "step1_contract_and_parent_invariants_preserved": True,
        "training_fail_closed": (
            profile["implementation_boundary"]["training_allowed"] is False
            and profile["stage_gate"]["ready"] is False
            and profile["stage_gate"]["formal_training_started"] is False
        ),
        "parameter_evidence_registry_present": bool(profile["parameter_families"]),
        "parent_integrity_discrepancy_documented": any(
            issue.get("id") == "physics_v1_derived_usd_declared_hash_mismatch"
            and issue.get("status") in {
                "open_stage2_audit",
                "confirmed_parent_issue_preserved",
            }
            for issue in profile["parent_baseline"]["known_issues"]
        ),
        "left_hand_transfer_explicitly_blocked": profile["deployment_side"][
            "left_hand_transfer"
        ].startswith("blocked_"),
    }
    return {
        "schema_version": 1,
        "profile_id": profile["profile_id"],
        "stage": 1,
        "passed": all(checks.values()),
        "checks": checks,
        "parent_baseline": baseline,
        "training_allowed": False,
        "next_stage": "static_robot_asset_audit",
    }


def physics_v2_current_stage_report() -> dict[str, Any]:
    """Return cumulative identity checks for the current offline stage."""

    profile = load_physics_v2_profile()
    baseline = verify_parent_physics_v1_baseline(profile)
    asset = verify_physics_v2_robot_asset(profile)
    checks = {
        "parent_baseline_unchanged": baseline["passed"],
        "independent_robot_asset_valid": asset["passed"],
        "training_fail_closed": (
            profile["implementation_boundary"]["training_allowed"] is False
            and profile["stage_gate"]["ready"] is False
            and profile["stage_gate"]["formal_training_started"] is False
        ),
    }
    return {
        "schema_version": 1,
        "profile_id": profile["profile_id"],
        "stage": profile["stage_gate"]["current_stage"],
        "status": profile["status"],
        "passed": all(checks.values()),
        "checks": checks,
        "parent_baseline": baseline,
        "robot_asset_v2": asset,
        "training_allowed": False,
    }


__all__ = [
    "PHYSICS_V2_PROFILE_PATH",
    "PHYSICS_V2_ROBOT_USD_PATH",
    "load_physics_v2_profile",
    "physics_v2_current_stage_report",
    "physics_v2_step1_report",
    "verify_physics_v2_robot_asset",
    "verify_parent_physics_v1_baseline",
]
