from __future__ import annotations

import hashlib
import json
from pathlib import Path

from script.audit_tomato_physics_v2_training_artifacts import (
    build_audit,
    parse_training_log,
    render_markdown,
)


def _block(iteration: int, *, success: float, retained: float, drop: float, reward: float) -> str:
    return f"""
 Learning iteration {iteration}/3
 Steps per second: 120
 Mean value loss: 0.5
 Mean surrogate loss: -0.02
 Mean entropy loss: 1.0
 Mean reward: {reward}
 Mean action std: 0.6
 Metrics/success_rate: {success}
 Episode_Reward/contact: 0.5
 Metrics/drop_rate: {drop}
 Metrics/detach_rate: 0.4
 Metrics/retained_rate: {retained}
 Episode_Reward/excess_force_failure: 0.0
 Iteration time: 1.0s
"""


def test_parser_ignores_incomplete_iteration_and_extracts_finite_metrics() -> None:
    text = _block(0, success=0.0, retained=0.1, drop=0.8, reward=1.0)
    text += "Learning iteration 1/3\nMean reward: 99\n"
    result = parse_training_log(text)
    assert len(result.iterations) == 1
    assert result.iterations[0]["iteration"] == 0
    assert result.iterations[0]["mean_reward"] == 1.0
    assert result.requested_last_iteration == 3
    assert result.errors == []


def test_audit_hashes_artifacts_and_nominates_physical_metrics_before_reward(
    tmp_path: Path,
) -> None:
    log = tmp_path / "training.log"
    log.write_text(
        _block(0, success=0.0, retained=0.0, drop=0.1, reward=100.0)
        + _block(1, success=0.2, retained=0.3, drop=0.2, reward=2.0)
        + _block(2, success=0.1, retained=0.5, drop=0.0, reward=3.0)
        + "SimulationContext cleared\n",
        encoding="utf-8",
    )
    run = tmp_path / "run"
    run.mkdir()
    for iteration in range(3):
        (run / f"model_{iteration}.pt").write_bytes(bytes([iteration + 1]) * 64)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"max_updates": 3}), encoding="utf-8")

    audit = build_audit(
        log_path=log,
        run_directory=run,
        manifest_path=manifest,
        candidate_count=2,
    )

    assert audit["passed_artifact_integrity"] is True
    assert audit["training"]["reached_expected_updates"] is True
    assert audit["training"]["simulation_context_cleared"] is True
    candidates = audit["nominated_for_independent_gpu_evaluation"]
    assert [item["iteration"] for item in candidates] == [1, 2]
    expected = hashlib.sha256(bytes([2]) * 64).hexdigest()
    assert candidates[0]["sha256"] == expected
    assert "multi-seed Isaac evaluation" in render_markdown(audit)


def test_audit_fails_integrity_when_log_contains_runtime_error(tmp_path: Path) -> None:
    log = tmp_path / "training.log"
    log.write_text(
        _block(0, success=0.0, retained=0.0, drop=0.0, reward=0.0)
        + "Traceback (most recent call last):\n",
        encoding="utf-8",
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "model_0.pt").write_bytes(b"checkpoint")
    audit = build_audit(log_path=log, run_directory=run)
    assert audit["passed_artifact_integrity"] is False
    assert audit["training"]["clean_finite_log"] is False
