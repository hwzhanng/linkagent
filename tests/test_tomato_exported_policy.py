from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from module.skills.tomato_pick import (
    ACTION_LAYOUT_ID,
    OBSERVATION_LAYOUT_ID,
    TorchScriptTomatoPickPolicy,
    TorchExportTomatoPickPolicy,
    TomatoPickObservation,
    build_policy,
    load_skill_profile,
    validate_skill_manifest,
)


_CANDIDATE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/simulation/tomato_pick_clean_v5/model_1600_candidate/skill_manifest.json"
)
_BEST = (
    Path(__file__).resolve().parents[1]
    / "artifacts/simulation/tomato_pick_clean_v5/model_1350_best/skill_manifest.json"
)


@pytest.mark.parametrize(
    ("manifest", "expected_status"),
    (
        (_BEST, "best_fixed_multiseed_evaluated"),
        (_CANDIDATE, "candidate_pending_fixed_multiseed_evaluation"),
    ),
)
def test_packaged_skill_manifest_artifacts_are_hash_bound(
    manifest: Path,
    expected_status: str,
) -> None:
    if not manifest.is_file():
        pytest.skip(f"skill package has not been generated: {manifest.parent.name}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == expected_status
    assert payload["deployment"]["hardware_approved"] is False
    assert payload["policy_manifest"]["observation_layout_id"] == OBSERVATION_LAYOUT_ID
    assert payload["policy_manifest"]["action_layout_id"] == ACTION_LAYOUT_ID
    for artifact in payload["artifacts"].values():
        path = manifest.parent / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    _, _, contract = validate_skill_manifest(manifest)
    assert contract.policy_id == payload["skill_id"]


def test_candidate_skill_manifest_is_bound_and_hardware_blocked() -> None:
    if not _CANDIDATE.is_file():
        pytest.skip("candidate skill package has not been generated")
    payload = json.loads(_CANDIDATE.read_text(encoding="utf-8"))
    assert payload["status"] == "candidate_pending_fixed_multiseed_evaluation"
    assert payload["deployment"]["hardware_approved"] is False
    assert payload["policy_manifest"]["observation_layout_id"] == OBSERVATION_LAYOUT_ID
    assert payload["policy_manifest"]["action_layout_id"] == ACTION_LAYOUT_ID
    for artifact in payload["artifacts"].values():
        path = _CANDIDATE.parent / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    _, _, contract = validate_skill_manifest(_CANDIDATE)
    assert contract.policy_id == payload["skill_id"]


@pytest.mark.parametrize(
    "manifest",
    (_BEST, _CANDIDATE),
)
def test_packaged_torchscript_policy_matches_saved_canaries(manifest: Path) -> None:
    pytest.importorskip("torch")
    if not manifest.is_file():
        pytest.skip(f"skill package has not been generated: {manifest.parent.name}")
    policy = TorchScriptTomatoPickPolicy(manifest_path=manifest)
    canaries = json.loads(
        (manifest.parent / "inference_canaries.json").read_text(encoding="utf-8")
    )["cases"]
    for case in canaries:
        actual = policy.act(TomatoPickObservation(tuple(case["observation"])))
        assert actual.values == pytest.approx(case["clipped_action"], abs=1.0e-6)
    with pytest.raises(RuntimeError, match="not approved for hardware"):
        TorchScriptTomatoPickPolicy(
            manifest_path=manifest,
            allow_hardware=True,
        )


def test_candidate_torchscript_policy_matches_saved_canaries() -> None:
    pytest.importorskip("torch")
    if not _CANDIDATE.is_file():
        pytest.skip("candidate skill package has not been generated")
    policy = TorchScriptTomatoPickPolicy(manifest_path=_CANDIDATE)
    canaries = json.loads(
        (_CANDIDATE.parent / "inference_canaries.json").read_text(encoding="utf-8")
    )["cases"]
    for case in canaries:
        actual = policy.act(TomatoPickObservation(tuple(case["observation"])))
        assert actual.values == pytest.approx(case["clipped_action"], abs=1.0e-6)
    with pytest.raises(RuntimeError, match="not approved for hardware"):
        TorchScriptTomatoPickPolicy(
            manifest_path=_CANDIDATE,
            allow_hardware=True,
        )


@pytest.mark.parametrize(
    ("manifest", "profile_name"),
    (
        (_BEST, "tomato_pick_clean_v5_model1350_best.json"),
        (_CANDIDATE, "tomato_pick_clean_v5_model1600_candidate.json"),
    ),
)
def test_packaged_torch_export_and_profile_match_saved_canaries(
    manifest: Path,
    profile_name: str,
) -> None:
    pytest.importorskip("torch")
    if not manifest.is_file():
        pytest.skip(f"skill package has not been generated: {manifest.parent.name}")
    profile = (
        Path(__file__).resolve().parents[1]
        / "config/skills"
        / profile_name
    )
    values = load_skill_profile(profile)
    policy = build_policy(values)
    assert isinstance(policy, TorchExportTomatoPickPolicy)
    canaries = json.loads(
        (manifest.parent / "inference_canaries.json").read_text(encoding="utf-8")
    )["cases"]
    for case in canaries:
        actual = policy.act(TomatoPickObservation(tuple(case["observation"])))
        assert actual.values == pytest.approx(case["clipped_action"], abs=1.0e-6)
