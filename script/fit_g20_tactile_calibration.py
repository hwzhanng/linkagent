#!/usr/bin/env python3
"""Fit per-finger affine G20 force calibration from known-load samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


FINGERS = ("thumb", "index", "middle", "ring", "little")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fit(rows: list[dict[str, object]]) -> dict[str, float]:
    if len(rows) < 3:
        raise ValueError("each finger requires at least three samples")
    x = [float(row["raw_sum"]) for row in rows]
    y = [float(row["reference_force_n"]) for row in rows]
    if min(y) > 0.05 or max(y) < 1.0:
        raise ValueError("each finger requires both a near-zero and >=1 N sample")
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    variance = sum((item - mean_x) ** 2 for item in x)
    if variance <= 0.0:
        raise ValueError("raw calibration values have zero variance")
    gain = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / variance
    offset = mean_y - gain * mean_x
    if gain <= 0.0:
        raise ValueError("calibration gain must be positive")
    predictions = [gain * item + offset for item in x]
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(predictions, y)) / len(y))
    total = sum((item - mean_y) ** 2 for item in y)
    residual = sum((a - b) ** 2 for a, b in zip(predictions, y))
    r_squared = 1.0 - residual / total if total > 0.0 else 0.0
    return {
        "gain_n_per_raw_sum": gain,
        "offset_n": offset,
        "rmse_n": rmse,
        "r_squared": r_squared,
        "sample_count": float(len(rows)),
        "minimum_reference_n": min(y),
        "maximum_reference_n": max(y),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument("--operator-approval", help="required to mark real data validated")
    args = parser.parse_args()
    source = args.input.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("samples"), list):
        raise ValueError("input is not a schema_version 1 calibration capture")
    grouped = {
        finger: [row for row in payload["samples"] if row.get("finger") == finger]
        for finger in FINGERS
    }
    fitted = {finger: _fit(rows) for finger, rows in grouped.items()}
    result = {
        "schema_version": 1,
        "calibration_id": args.calibration_id,
        "hand_serial": payload.get("hand_serial"),
        "side": payload.get("side"),
        "source_capture": str(source),
        "source_capture_sha256": _sha(source),
        "model": "normal_force_n = gain * matrix_raw_sum + offset",
        "fingers": fitted,
        "hardware_validated": bool(args.operator_approval),
        "operator_approval": args.operator_approval,
        "release_note": (
            "Threshold selection still requires retained/slip/drop trials on the real fruit."
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
