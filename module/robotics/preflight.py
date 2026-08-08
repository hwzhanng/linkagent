"""Fail-closed deployment inspection for tomato-pick skill packages."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from module.skills.tomato_pick import (
    ACTION_LAYOUT_ID,
    INTERFACE_VERSION,
    OBSERVATION_LAYOUT_ID,
    load_skill_profile,
    validate_skill_manifest,
)

from .deployment import BimanualDeploymentConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class DeploymentPreflightReport:
    mode: str
    checks: tuple[PreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "mode": self.mode,
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
        }

    def require_passed(self) -> None:
        failed = [check.name for check in self.checks if not check.passed]
        if failed:
            raise RuntimeError("deployment preflight failed: " + ", ".join(failed))


def run_deployment_preflight(
    *,
    skill_profile: str | Path,
    mode: str,
    robot_deployment: str | Path | None = None,
    tactile_calibration: str | Path | None = None,
    verifier_thresholds: str | Path | None = None,
    minimum_success_rate: float = 0.80,
) -> DeploymentPreflightReport:
    if mode not in {"simulation", "shadow", "hardware"}:
        raise ValueError("mode must be simulation, shadow, or hardware")
    checks: list[PreflightCheck] = []
    try:
        skill_config = load_skill_profile(skill_profile)
        manifest_path = skill_config.get("manifest_path")
        manifest_file, manifest, policy = validate_skill_manifest(manifest_path)
        checks.append(PreflightCheck("skill_manifest_and_hashes", True, str(manifest_file)))
    except BaseException as exc:
        checks.append(PreflightCheck("skill_manifest_and_hashes", False, f"{type(exc).__name__}: {exc}"))
        return DeploymentPreflightReport(mode, tuple(checks))

    interface_ok = (
        policy.interface_version == INTERFACE_VERSION
        and policy.observation_layout_id == OBSERVATION_LAYOUT_ID
        and policy.action_layout_id == ACTION_LAYOUT_ID
    )
    checks.append(
        PreflightCheck(
            "policy_interface_version",
            interface_ok,
            f"{policy.interface_version}/{policy.observation_layout_id}/{policy.action_layout_id}",
        )
    )
    evaluation = manifest.get("evaluation")
    eval_ok = (
        isinstance(evaluation, dict)
        and evaluation.get("selection_status") == "selected_best"
        and evaluation.get("fixed_deterministic") is True
        and isinstance(evaluation.get("seeds"), list)
        and len(set(evaluation["seeds"])) >= 3
        and int(evaluation.get("completed_episodes", 0)) > 0
        and float(evaluation.get("success_rate", -1.0)) >= float(minimum_success_rate)
    )
    checks.append(
        PreflightCheck(
            "fixed_multiseed_evaluation",
            bool(eval_ok),
            "selected deterministic best with >=3 seeds" if eval_ok else "missing, pending, or below success gate",
        )
    )
    fixed_manifest = manifest.get("artifacts", {}).get("fixed_evaluation_manifest")
    try:
        if not isinstance(fixed_manifest, dict):
            raise ValueError("skill package has no fixed evaluation manifest")
        release_path = (manifest_file.parent / str(fixed_manifest["path"])).resolve()
        release = json.loads(release_path.read_text(encoding="utf-8"))
        release_best = release.get("best_summary")
        selection_gate = release.get("selection_gate")
        release_ok = (
            release.get("schema_version") == 1
            and isinstance(selection_gate, dict)
            and selection_gate.get("passed") is True
            and isinstance(release_best, dict)
            and release.get("best_checkpoint_sha256") == policy.checkpoint_sha256
            and release_best.get("checkpoint_sha256") == policy.checkpoint_sha256
            and int(release_best.get("completed_episodes", 0))
                == int(evaluation.get("completed_episodes", -1))
            and abs(
                float(release_best.get("success_rate", -1.0))
                - float(evaluation.get("success_rate", -2.0))
            ) <= 1.0e-12
            and _sha256(release_path) == fixed_manifest.get("sha256")
        )
        checks.append(
            PreflightCheck(
                "evaluation_selection_provenance",
                bool(release_ok),
                "hardened selection gate and selected checkpoint are bound to this skill",
            )
        )
    except BaseException as exc:
        checks.append(
            PreflightCheck(
                "evaluation_selection_provenance",
                False,
                f"{type(exc).__name__}: {exc}",
            )
        )
    deployment = manifest.get("deployment", {})
    if mode == "hardware":
        hardware_approved = (
            isinstance(deployment, dict)
            and deployment.get("hardware_approved") is True
        )
        checks.append(
            PreflightCheck(
                "skill_hardware_approval",
                hardware_approved,
                "explicit graded release required",
            )
        )
    independent_ok = isinstance(deployment, dict) and deployment.get("requires_independent_pick_verifier") is True
    checks.append(PreflightCheck("independent_verifier_required", independent_ok, "manifest requirement"))

    if robot_deployment is not None:
        try:
            robot_path = Path(robot_deployment).expanduser().resolve()
            robot = BimanualDeploymentConfig.from_mapping(
                json.loads(robot_path.read_text(encoding="utf-8"))
            )
            ready = robot.hardware_ready if mode == "hardware" else True
            checks.append(PreflightCheck("robot_deployment", ready, str(robot_path)))
        except BaseException as exc:
            checks.append(PreflightCheck("robot_deployment", False, f"{type(exc).__name__}: {exc}"))
    elif mode in {"shadow", "hardware"}:
        checks.append(PreflightCheck("robot_deployment", False, "robot deployment config is required"))

    calibration_payload: dict[str, Any] | None = None
    if tactile_calibration is not None:
        try:
            calibration_path = Path(tactile_calibration).expanduser().resolve()
            calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
            valid = (
                calibration_payload.get("schema_version") == 1
                and bool(calibration_payload.get("calibration_id"))
                and (mode != "hardware" or calibration_payload.get("hardware_validated") is True)
            )
            checks.append(PreflightCheck("tactile_calibration", valid, str(calibration_path)))
        except BaseException as exc:
            checks.append(PreflightCheck("tactile_calibration", False, f"{type(exc).__name__}: {exc}"))
    elif mode == "hardware":
        checks.append(PreflightCheck("tactile_calibration", False, "real tactile calibration is required"))

    if verifier_thresholds is not None:
        try:
            verifier_path = Path(verifier_thresholds).expanduser().resolve()
            verifier = json.loads(verifier_path.read_text(encoding="utf-8"))
            calibration_matches = (
                calibration_payload is None
                or verifier.get("tactile_calibration_id") == calibration_payload.get("calibration_id")
            )
            valid = (
                verifier.get("schema_version") == 1
                and bool(verifier.get("profile_id"))
                and calibration_matches
                and (mode != "hardware" or verifier.get("hardware_validated") is True)
            )
            checks.append(PreflightCheck("verifier_thresholds", valid, str(verifier_path)))
        except BaseException as exc:
            checks.append(PreflightCheck("verifier_thresholds", False, f"{type(exc).__name__}: {exc}"))
    elif mode == "hardware":
        checks.append(PreflightCheck("verifier_thresholds", False, "real-data verifier thresholds are required"))

    if mode == "hardware":
        try:
            sdk_manifest = json.loads(
                (Path(__file__).resolve().parents[2] / "botSDK/INTEGRATION_MANIFEST.json").read_text(encoding="utf-8")
            )
            complete = sdk_manifest.get("hardware_validation", {}).get("complete") is True
            checks.append(PreflightCheck("vendor_hardware_validation", complete, "botSDK integration manifest"))
        except BaseException as exc:
            checks.append(PreflightCheck("vendor_hardware_validation", False, f"{type(exc).__name__}: {exc}"))
    return DeploymentPreflightReport(mode, tuple(checks))


__all__ = [
    "DeploymentPreflightReport",
    "PreflightCheck",
    "run_deployment_preflight",
]
