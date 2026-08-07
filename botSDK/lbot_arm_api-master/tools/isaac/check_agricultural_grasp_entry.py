#!/usr/bin/env python3
"""Validate the C2-0 tomato-pick simulation contract without starting Isaac."""

from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[4]
BOTSDK_ROOT = REPO_ROOT / "botSDK"
DERIVED_ROOT = BOTSDK_ROOT / "urdf/02_project_derived"
PROFILE = DERIVED_ROOT / "simulation_baselines/tomato_pick_v1.yaml"
LEGACY_PROFILE = (
    DERIVED_ROOT
    / "simulation_baselines/agricultural_grasp_no_equipment_proxy.yaml"
)
INTEGRATION_MANIFEST = BOTSDK_ROOT / "INTEGRATION_MANIFEST.json"
MOUNTS = (
    DERIVED_ROOT
    / "ros2_packages/linkbot_bimanual_sdk_description/config/mounts.json"
)
URDF = (
    DERIVED_ROOT
    / "ros2_packages/linkbot_bimanual_sdk_description/urdf/"
    "linkbot_bimanual_sdk_description.urdf"
)
MASS_MANIFEST = (
    DERIVED_ROOT
    / "isaac_assets/linkbot_bimanual_sdk_training_mass1kg/"
    "mass_correction_manifest.json"
)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _close(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    try:
        left_values = tuple(float(value) for value in left)  # type: ignore[arg-type]
        right_values = tuple(float(value) for value in right)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return len(left_values) == len(right_values) and all(
        math.isclose(a, b, abs_tol=tolerance, rel_tol=0.0)
        for a, b in zip(left_values, right_values)
    )


def _urdf_mounts() -> dict[str, tuple[float, ...]]:
    root = ET.parse(URDF).getroot()
    result: dict[str, tuple[float, ...]] = {}
    for side in ("left", "right"):
        joint = root.find(f"./joint[@name='{side}_hand_mount_joint']")
        if joint is None:
            raise RuntimeError(f"missing {side}_hand_mount_joint")
        origin = joint.find("origin")
        if origin is None or "rpy" not in origin.attrib:
            raise RuntimeError(f"missing rpy on {side}_hand_mount_joint")
        result[side] = tuple(float(value) for value in origin.attrib["rpy"].split())
    return result


def main() -> int:
    required = (
        PROFILE,
        LEGACY_PROFILE,
        INTEGRATION_MANIFEST,
        MOUNTS,
        URDF,
        MASS_MANIFEST,
    )
    files_present = {
        str(path.relative_to(REPO_ROOT)): path.is_file() for path in required
    }
    if not all(files_present.values()):
        print(json.dumps({"passed": False, "files_present": files_present}, indent=2))
        return 2

    profile = _json(PROFILE)
    legacy = _json(LEGACY_PROFILE)
    integration = _json(INTEGRATION_MANIFEST)
    mounts = _json(MOUNTS)
    mass_manifest = _json(MASS_MANIFEST)
    urdf_mounts = _urdf_mounts()

    source_paths = profile["source_of_truth"]
    if not isinstance(source_paths, dict):
        raise TypeError("source_of_truth must be an object")
    source_files_present = {
        relative: (REPO_ROOT / relative).exists()
        for relative in source_paths.values()
        if isinstance(relative, str) and "/" in relative
    }

    policy = profile["policy"]
    tactile = profile["tactile"]
    task = profile["task"]
    robot = profile["robot"]
    gates = profile["entry_gates"]
    environment = profile["environment"]
    if not all(isinstance(value, dict) for value in (policy, tactile, task, robot, gates, environment)):
        raise TypeError("profile sections must be objects")

    profile_mounts = robot["mounts_rpy_rad"]
    if not isinstance(profile_mounts, dict):
        raise TypeError("robot.mounts_rpy_rad must be an object")
    mount_checks = {}
    for side in ("left", "right"):
        mount_record = mounts[f"{side}_hand"]
        if not isinstance(mount_record, dict):
            raise TypeError(f"mounts.{side}_hand must be an object")
        mount_checks[side] = (
            _close(profile_mounts[side], mount_record["rpy"])
            and _close(profile_mounts[side], urdf_mounts[side])
        )

    hardware = robot["hardware_validation"]
    verified_matrix = tactile["verified_hardware_matrix"]
    calibrated_force = tactile["calibrated_resultant_force_n"]
    simulation_proxy = tactile["simulation_proxy"]
    stem = task["stem"]
    success = task["success"]
    if not all(
        isinstance(value, dict)
        for value in (
            hardware,
            verified_matrix,
            calibrated_force,
            simulation_proxy,
            stem,
            success,
        )
    ):
        raise TypeError("nested profile sections must be objects")

    observed_topology = integration["observed_topology"]
    if not isinstance(observed_topology, dict):
        raise TypeError("observed_topology must be an object")
    right_g20 = observed_topology["right_g20"]
    if not isinstance(right_g20, dict):
        raise TypeError("observed_topology.right_g20 must be an object")

    path_values = (
        mass_manifest.get("source_asset"),
        mass_manifest.get("generated_asset"),
    )
    checks = {
        "profile_is_current_tomato_pick_v1": (
            profile.get("spec_id") == "tomato_pick_v1"
            and profile.get("status") == "ready_for_c2_environment_implementation"
        ),
        "all_source_of_truth_paths_exist": all(source_files_present.values()),
        "policy_contract_is_113d_22d_20hz": (
            policy.get("observation_dim") == 113
            and policy.get("action_dim") == 22
            and policy.get("policy_rate_hz") == 20
            and policy.get("phases") == ["contact", "grasp", "detach", "hold"]
            and environment.get("physics_rate_hz") == 120
            and environment.get("policy_decimation") == 6
        ),
        "mounts_match_profile_manifest_and_urdf": all(mount_checks.values()),
        "right_g20_remains_pending": (
            hardware.get("right_g20") == "pending_direct_control_and_tactile"
            and right_g20.get("validated") is False
        ),
        "matrix_raw_is_not_declared_newtons": (
            verified_matrix.get("shape") == [5, 12, 6]
            and verified_matrix.get("range") == [0, 255]
            and verified_matrix.get("unit") is None
            and verified_matrix.get("semantics")
            == "relative_spatial_weight_not_newtons"
        ),
        "uncalibrated_hardware_force_is_invalid": (
            calibrated_force.get("hardware_status")
            == "blocked_until_per_finger_calibration"
            and calibrated_force.get("before_calibration")
            == "force_valid_false_and_force_values_null"
            and calibrated_force.get("raw_to_newton_linear_conversion_allowed")
            is False
        ),
        "simulated_force_is_labeled_proxy": (
            simulation_proxy.get("resultant_unit") == "N"
            and simulation_proxy.get("must_be_marked_simulated") is True
            and simulation_proxy.get("may_be_claimed_as_hardware_calibration")
            is False
        ),
        "task_requires_detachment_and_retention": (
            stem.get("enabled") is True
            and stem.get("detachment_event")
            == "constraint_transitions_from_intact_to_broken_once"
            and success
            == {
                "stem_detached": True,
                "fruit_retained_without_slip": True,
                "dropped": False,
                "verifier_valid": True,
            }
        ),
        "ppo_entry_uses_physics_reward_and_optimizer_gates": (
            gates.get("c2_environment_implementation") == "allowed"
            and gates.get("rule_policy_acceptance")
            == "diagnostic_only_not_a_training_gate"
            and gates.get("ppo_training")
            == "allowed_after_113d_22d_physics_reward_and_optimizer_gates"
            and isinstance(gates.get("ppo_acceptance_requires"), list)
            and len(gates["ppo_acceptance_requires"]) >= 6
        ),
        "reward_and_termination_do_not_leak_privileged_state": (
            isinstance(environment.get("reward"), dict)
            and environment["reward"].get("sim_ground_truth_in_actor") is False
            and isinstance(environment.get("termination"), dict)
            and environment["termination"].get("success")
            == "detached_and_retained_without_slip_and_not_dropped"
        ),
        "legacy_proxy_is_not_a_training_entry": (
            legacy.get("status") == "legacy_reference_only"
            and legacy.get("current_training_entry") is False
        ),
        "mass_manifest_paths_are_repo_relative": all(
            isinstance(value, str) and not Path(value).is_absolute()
            for value in path_values
        ),
    }
    passed = all(checks.values())
    result = {
        "schema_version": 2,
        "stage": "c2_0",
        "passed": passed,
        "checks": checks,
        "mount_checks": mount_checks,
        "source_files_present": source_files_present,
        "next_gate": "start_fresh_ppo_and_monitor_full_task_success",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
