"""Shared source fingerprint contract for Physics-v2 qualification gates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping


AUDITED_RUNTIME_INPUTS = (
    "script/physics_v2_audit_inputs.py",
    "simulation/config/tomato_pick_physics_v2_public_prior_v1.json",
    "simulation/tomato_pick/physics_v2.py",
    "simulation/tomato_pick/isaac/env.py",
    "simulation/tomato_pick/isaac/physics_v2/env_cfg.py",
    "simulation/tomato_pick/isaac/physics_v2/env.py",
    "simulation/tomato_pick/isaac/physics_v2/agents/rsl_rl_ppo_cfg.py",
    "simulation/tomato_pick/reward.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_sha256(
    project_root: Path, relative_paths: Iterable[str]
) -> dict[str, str]:
    return {
        relative_path: sha256_file(project_root / relative_path)
        for relative_path in relative_paths
    }


def require_current_input_sha256(
    report: Mapping[str, object],
    project_root: Path,
    relative_paths: Iterable[str],
    *,
    report_name: str,
) -> None:
    recorded = report.get("input_sha256")
    if not isinstance(recorded, Mapping):
        raise RuntimeError(
            f"qualification report has no input fingerprint: {report_name}"
        )
    for relative_path in relative_paths:
        expected = recorded.get(relative_path)
        actual = sha256_file(project_root / relative_path)
        if not isinstance(expected, str) or expected != actual:
            raise RuntimeError(
                "qualification report is stale for "
                f"{relative_path}: {report_name}"
            )


__all__ = [
    "AUDITED_RUNTIME_INPUTS",
    "input_sha256",
    "require_current_input_sha256",
    "sha256_file",
]
