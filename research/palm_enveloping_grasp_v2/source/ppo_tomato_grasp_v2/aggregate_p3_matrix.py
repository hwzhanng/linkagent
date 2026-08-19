"""Aggregate P3 contact-geometry screening reports without hiding sparse resets.

The screening matrix intentionally uses the same checkpoint and action course
while varying only proxy geometry, PhysX offsets, and mass/density ranges.  A
cell is not allowed to pass a gate merely because it completed few episodes;
the report therefore exposes completion counts and applies an explicit
minimum-sample flag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["source_file"] = path.name
    return data


def _screen_gate(d: dict[str, Any], minimum_episodes: int) -> dict[str, Any]:
    drop = float(d.get("drop_rate", 1.0))
    overload = float(d.get("overload_rate", 1.0))
    palm = float(d.get("palm_supported_rate", 0.0))
    enough = int(d.get("completed_episodes", 0)) >= minimum_episodes
    return {
        "enough_episodes": enough,
        "t0_a_pass_if_sufficient": enough and drop <= 0.10 and overload <= 0.02 and palm >= 0.80,
        "drop_le_10": drop <= 0.10,
        "overload_le_2": overload <= 0.02,
        "palm_ge_80": palm >= 0.80,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-episodes", type=int, default=32)
    args = parser.parse_args()

    reports = []
    for path in sorted(args.input.glob("*.json")):
        d = _load(path)
        gate = _screen_gate(d, args.minimum_episodes)
        geom = d.get("resolved_contact_geometry", {})
        reports.append({
            "source_file": path.name,
            "completed_episodes": int(d.get("completed_episodes", 0)),
            "success_rate": float(d.get("success_rate", 0.0)),
            "drop_rate": float(d.get("drop_rate", 1.0)),
            "overload_rate": float(d.get("overload_rate", 1.0)),
            "palm_supported_rate": float(d.get("palm_supported_rate", 0.0)),
            "measured_palm_supported_rate": float(d.get("measured_palm_supported_rate", 0.0)),
            "distal_contact_all_rate": float(d.get("distal_contact_all_rate", 0.0)),
            "measured_distal_contact_all_rate": float(d.get("measured_distal_contact_all_rate", 0.0)),
            "measured_distal_contact_steps_mean": d.get("measured_distal_contact_steps_mean", []),
            "measured_distal_max_streak_mean": d.get("measured_distal_max_streak_mean", []),
            "measured_link4_force_integral_n_s_mean": d.get("measured_link4_force_integral_n_s_mean", []),
            "resolved_contact_geometry": geom,
            "gate": gate,
        })

    # Rank stable cells first, but do not call the ranking a success criterion.
    # The sparse-episode flag remains visible in every row.
    reports.sort(key=lambda x: (
        bool(x["gate"]["t0_a_pass_if_sufficient"]),
        x["gate"]["enough_episodes"],
        x["palm_supported_rate"] - 2.0 * x["drop_rate"] - 1.5 * x["overload_rate"],
    ), reverse=True)

    summary = {
        "version": "p3_geometry_matrix_screen_v1",
        "minimum_episodes_for_gate": args.minimum_episodes,
        "cell_count": len(reports),
        "t0_a_pass_cells": sum(r["gate"]["t0_a_pass_if_sufficient"] for r in reports),
        "sufficient_cells": sum(r["gate"]["enough_episodes"] for r in reports),
        "note": "Screening cells are diagnostic only; full T0-A requires >=256 independent completed episodes.",
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
