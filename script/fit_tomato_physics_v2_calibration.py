"""Validate a measured tomato dataset and emit a fail-closed fit proposal."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.tomato_pick.calibration_v2 import fit_calibration_proposal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-review-ready", action="store_true")
    args = parser.parse_args(argv)
    dataset_path = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    output = args.output if args.output.is_absolute() else ROOT / args.output
    encoded = dataset_path.read_bytes()
    dataset = json.loads(encoded)
    report = fit_calibration_proposal(dataset)
    report["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    report["source_dataset"] = str(dataset_path.resolve())
    report["source_dataset_sha256"] = hashlib.sha256(encoded).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "ready_for_human_review": report["ready_for_human_review"]}))
    if args.require_review_ready and not report["ready_for_human_review"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
