"""Build the P0 content-addressed baseline manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from simulation.ppo_tomato_grasp_v3.provenance import artifact_records, atomic_json_write, machine_snapshot
from simulation.ppo_tomato_grasp_v3.scenario import load_scenario


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO = Path(__file__).resolve().parent / "scenarios" / "v3_large90_nominal.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenario, scenario_hash = load_scenario(args.scenario)
    critical = [
        args.scenario,
        PROJECT / "PALM_ENVELOPING_GRASP_V3_AUDIT_AND_EXECUTION_PLAN_20260820.md",
        PROJECT / scenario["evaluation"]["checkpoint"],
        PROJECT / scenario["evaluation"]["legacy_reference_report"],
        PROJECT / "simulation/ppo_tomato_grasp/enveloping_verifier.py",
        PROJECT / "simulation/ppo_tomato_grasp_v2",
        PROJECT / "simulation/ppo_tomato_grasp_v3",
        PROJECT / "archives/pinch_small_fruit_band4_20260815",
    ]
    manifest = {
        "version": "v3_p0_freeze_manifest_1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT),
        "scenario_name": scenario["name"],
        "scenario_sha256": scenario_hash,
        "hardware_command_allowed": False,
        "assets": artifact_records(critical, relative_to=PROJECT),
        "machine": machine_snapshot(PROJECT),
    }
    missing = [record["path"] for record in manifest["assets"] if not record["exists"]]
    if missing:
        raise FileNotFoundError(f"cannot freeze baseline; missing assets: {missing}")
    atomic_json_write(args.output, manifest)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
