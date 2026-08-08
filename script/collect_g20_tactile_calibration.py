#!/usr/bin/env python3
"""Normalize externally recorded G20 calibration samples without opening SDK/CAN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


FINGERS = ("thumb", "index", "middle", "ring", "little")


def _record(value: object, line_number: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"line {line_number}: sample must be an object")
    finger = value.get("finger")
    if finger not in FINGERS:
        raise ValueError(f"line {line_number}: finger must be one of {FINGERS}")
    raw = value.get("raw_sum")
    load = value.get("reference_force_n")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw < 0:
        raise ValueError(f"line {line_number}: raw_sum must be >= 0")
    if isinstance(load, bool) or not isinstance(load, (int, float)) or load < 0:
        raise ValueError(f"line {line_number}: reference_force_n must be >= 0")
    region = value.get("contact_region", "center")
    if not isinstance(region, str) or not region.strip():
        raise ValueError(f"line {line_number}: contact_region must be non-empty")
    return {
        "finger": finger,
        "raw_sum": float(raw),
        "reference_force_n": float(load),
        "contact_region": region.strip(),
        "temperature_c": value.get("temperature_c"),
        "source_sequence": value.get("source_sequence"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, help="recorder output; stdin if omitted")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--hand-serial", required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    args = parser.parse_args()
    text = (
        args.source_jsonl.expanduser().read_text(encoding="utf-8")
        if args.source_jsonl is not None
        else sys.stdin.read()
    )
    samples = [
        _record(json.loads(line), index)
        for index, line in enumerate(text.splitlines(), 1)
        if line.strip()
    ]
    if not samples:
        raise ValueError("no calibration samples were provided")
    payload = {
        "schema_version": 1,
        "session_id": args.session_id,
        "hand_serial": args.hand_serial,
        "side": args.side,
        "capture_mode": "external_recorder_no_sdk_access",
        "samples": samples,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
