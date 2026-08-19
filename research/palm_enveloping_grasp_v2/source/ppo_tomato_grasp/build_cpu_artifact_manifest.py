#!/usr/bin/env python3
"""Build an integrity-checked manifest for transferring the CPU candidate bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_manifest(
    checkpoint_path: Path,
    actor_path: Path,
    report_paths: list[Path],
) -> dict[str, Any]:
    paths = [checkpoint_path, actor_path, *report_paths]
    if len(paths) != len(set(path.resolve() for path in paths)):
        raise ValueError("artifact paths must be unique")
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("every artifact must be a regular non-symlink file")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    candidate_path = str(checkpoint_path.resolve())
    candidate_update = int(checkpoint["update"])
    environment_contract = checkpoint.get("environment_contract")
    tensor_contract = checkpoint.get("contract")
    if not isinstance(environment_contract, dict) or not isinstance(tensor_contract, dict):
        raise ValueError("checkpoint lacks frozen contracts")
    if tensor_contract.get("simulation_only") is not True:
        raise ValueError("checkpoint is not explicitly simulation-only")
    json_reports = {path.name: _json(path) for path in report_paths}
    matching_evidence = 0
    for report in json_reports.values():
        report_environment = report.get("environment_contract")
        if report_environment is not None and report_environment != environment_contract:
            raise ValueError("report environment contract does not match checkpoint")
        path = report.get("checkpoint", report.get("source_checkpoint"))
        update = report.get("checkpoint_update", report.get("source_update"))
        if path is not None or update is not None:
            if path != candidate_path or int(update) != candidate_update:
                raise ValueError("report evidence lineage does not match checkpoint")
            matching_evidence += 1
        if report.get("hardware_release_allowed") is True or report.get("hardware_command_allowed") is True:
            raise ValueError("artifact attempts to authorize hardware")
        if report.get("isaac_physx_release_allowed") is True:
            raise ValueError("artifact attempts to authorize Isaac/PhysX release")
    actor_metadata_path = actor_path.with_suffix(actor_path.suffix + ".json")
    if actor_metadata_path not in report_paths:
        raise ValueError("TorchScript metadata must be included among reports")
    actor_metadata = json_reports[actor_metadata_path.name]
    if actor_metadata.get("environment_contract") != environment_contract:
        raise ValueError("actor environment contract mismatch")
    if actor_metadata.get("tensor_contract") != tensor_contract:
        raise ValueError("actor tensor contract mismatch")
    entries = []
    for path in paths:
        entries.append({
            "name": path.name,
            "absolute_source_path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return {
        "schema_version": 1,
        "status": "cpu_candidate_transfer_bundle_integrity_checked_simulation_only",
        "checkpoint_update": candidate_update,
        "tensor_contract": tensor_contract,
        "environment_contract": environment_contract,
        "matching_lineage_evidence_files": matching_evidence,
        "artifacts": entries,
        "formal_training_allowed": False,
        "isaac_physx_release_allowed": False,
        "hardware_release_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.checkpoint, args.actor, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
