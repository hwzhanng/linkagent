#!/usr/bin/env python3
"""Harden a completed Clean-v5 sweep with seed gates and failure taxonomy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    d = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / d
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return center - radius, center + radius


def _iteration(result: dict[str, object]) -> int:
    return int(Path(str(result["checkpoint"])).stem.split("_")[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--minimum-seed-success", type=float, default=0.80)
    parser.add_argument("--maximum-seed-spread", type=float, default=0.05)
    args = parser.parse_args()
    root = args.evaluation_dir.expanduser().resolve()
    source_manifest_path = root / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError("evaluation sweep manifest is not complete")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    settings = source.get("evaluation", {})
    expected_seeds = tuple(int(seed) for seed in settings.get("seeds", ()))
    expected_iterations = tuple(int(item["iteration"]) for item in source.get("candidate_summaries", ()))
    if len(set(expected_seeds)) < 3 or not expected_iterations:
        raise ValueError("release selection requires >=3 seeds and candidates")

    candidates: list[dict[str, object]] = []
    detail: dict[str, object] = {}
    for iteration in expected_iterations:
        results = []
        for seed in expected_seeds:
            path = root / f"model_{iteration}_seed_{seed}.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            if _iteration(result) != iteration or int(result["seed"]) != seed:
                raise ValueError(f"identity mismatch in {path.name}")
            if result.get("deterministic_policy") is not True:
                raise ValueError(f"{path.name} is not deterministic")
            checkpoint = Path(str(result["checkpoint"]))
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != result.get("checkpoint_sha256"):
                raise ValueError(f"checkpoint hash mismatch for {path.name}")
            results.append(result)
        episodes = sum(int(row["completed_episodes"]) for row in results)
        successes = sum(int(row["success_count"]) for row in results)
        categories = {
            "success": successes,
            "drop": sum(int(row.get("drop_count", 0)) for row in results),
            "excess_force": sum(int(row.get("excess_force_count", 0)) for row in results),
            "out_of_bounds": sum(int(row.get("out_of_bounds_count", 0)) for row in results),
            "timeout": sum(int(row.get("timeout_count", 0)) for row in results),
        }
        categories["unclassified_failure"] = max(0, episodes - sum(categories.values()))
        per_seed = [
            {
                "seed": int(row["seed"]),
                "episodes": int(row["completed_episodes"]),
                "success_rate": float(row["success_rate"]),
                "drop_rate": float(row["drop_rate"]),
                "excess_force_rate": int(row.get("excess_force_count", 0)) / int(row["completed_episodes"]),
                "out_of_bounds_rate": int(row.get("out_of_bounds_count", 0)) / int(row["completed_episodes"]),
                "timeout_rate": int(row.get("timeout_count", 0)) / int(row["completed_episodes"]),
            }
            for row in sorted(results, key=lambda value: int(value["seed"]))
        ]
        seed_success = [float(row["success_rate"]) for row in per_seed]
        low, high = _wilson(successes, episodes)
        checkpoint = Path(str(results[0]["checkpoint"]))
        candidate = {
            "iteration": iteration,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": str(results[0]["checkpoint_sha256"]),
            "completed_episodes": episodes,
            "success_count": successes,
            "success_rate": successes / episodes,
            "success_wilson_95_low": low,
            "success_wilson_95_high": high,
            "drop_count": categories["drop"],
            "drop_rate": categories["drop"] / episodes,
            "detach_rate": sum(float(row["detach_count"]) for row in results) / episodes,
            "retained_rate": sum(float(row["retained_count"]) for row in results) / episodes,
            "minimum_seed_success_rate": min(seed_success),
            "maximum_seed_success_rate": max(seed_success),
            "seed_success_spread": max(seed_success) - min(seed_success),
            "seed_gate_passed": (
                min(seed_success) >= args.minimum_seed_success
                and max(seed_success) - min(seed_success) <= args.maximum_seed_spread
            ),
        }
        candidates.append(candidate)
        detail[str(iteration)] = {
            "per_seed": per_seed,
            "terminal_counts": categories,
            "terminal_rates": {name: count / episodes for name, count in categories.items()},
        }

    eligible = [item for item in candidates if item["seed_gate_passed"]]
    ranking = sorted(
        eligible,
        key=lambda item: (float(item["success_rate"]), -float(item["drop_rate"])),
        reverse=True,
    )
    if not ranking:
        raise RuntimeError("no checkpoint passes the fixed multi-seed release gate")
    best = ranking[0]
    release = {
        **source,
        "evaluation": {
            **settings,
            "minimum_seed_success_rate": args.minimum_seed_success,
            "maximum_seed_success_spread": args.maximum_seed_spread,
        },
        "candidate_summaries": candidates,
        "ranking": [int(item["iteration"]) for item in ranking],
        "best_checkpoint": best["checkpoint"],
        "best_checkpoint_sha256": best["checkpoint_sha256"],
        "best_summary": best,
        "selection_gate": {
            "passed": True,
            "deterministic": True,
            "minimum_distinct_seeds": 3,
            "failure_classification": "failure_summary.json",
        },
    }
    (root / "release_manifest.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failure = {
        "schema_version": 1,
        "task": source.get("task"),
        "best_iteration": best["iteration"],
        "candidates": detail,
    }
    (root / "failure_summary.json").write_text(
        json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Clean-v5 fixed evaluation failure classification",
        "",
        f"Selected best: `model_{best['iteration']}.pt` ({best['success_rate']:.2%} success).",
        "",
        "| iteration | success | worst seed | spread | drop | seed gate |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in sorted(candidates, key=lambda value: int(value["iteration"])):
        lines.append(
            f"| {item['iteration']} | {item['success_rate']:.2%} | "
            f"{item['minimum_seed_success_rate']:.2%} | {item['seed_success_spread']:.2%} | "
            f"{item['drop_rate']:.2%} | {'yes' if item['seed_gate_passed'] else 'no'} |"
        )
    best_rates = detail[str(best["iteration"])]["terminal_rates"]
    lines.extend(["", "Best-checkpoint terminal distribution:", ""])
    for name, rate in best_rates.items():
        lines.append(f"- {name}: {rate:.2%}")
    (root / "failure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "release_manifest.json")


if __name__ == "__main__":
    main()
