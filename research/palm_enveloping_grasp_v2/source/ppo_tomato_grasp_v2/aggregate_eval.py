"""Aggregate independent JSON evaluations with Wilson confidence bounds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--pattern", default="p2_*.json")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - radius), min(1.0, center + radius)


reports = [json.loads(path.read_text()) for path in sorted(args.directory.glob(args.pattern))]
groups: dict[str, list[dict]] = {}
for report in reports:
    groups.setdefault(report["profile"], []).append(report)
aggregate = {"version": "v2_eval_aggregate_1", "reports": len(reports), "profiles": {}}
for profile, values in groups.items():
    n = sum(int(item.get("completed_episodes", 0)) for item in values)
    successes = sum(round(float(item.get("success_rate", 0.0)) * int(item.get("completed_episodes", 0))) for item in values)
    drops = sum(round(float(item.get("drop_rate", 0.0)) * int(item.get("completed_episodes", 0))) for item in values)
    overloads = sum(round(float(item.get("overload_rate", 0.0)) * int(item.get("completed_episodes", 0))) for item in values)
    aggregate["profiles"][profile] = {
        "seeds": [item["seed"] for item in values], "completed_episodes": n,
        "success_rate": successes / max(n, 1), "success_wilson_95": wilson(successes, n),
        "drop_rate": drops / max(n, 1), "overload_rate": overloads / max(n, 1),
        "overload_wilson_95": wilson(overloads, n),
    }
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
print(json.dumps(aggregate, indent=2))
