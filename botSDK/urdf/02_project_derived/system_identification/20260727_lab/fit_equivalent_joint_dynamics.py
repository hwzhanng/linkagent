#!/usr/bin/env python3
"""Fit a first-pass equivalent joint friction model from a laboratory CSV.

This tool intentionally reports current-domain coefficients when calibrated
torque is unavailable.  It never relabels raw current as N*m.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


TARGET_COLUMNS = (
    ("joint_torque_feedback_nm", "N*m"),
    ("estimated_joint_torque_nm", "N*m"),
    ("motor_current_feedback_a", "A"),
    ("motor_current_a", "A"),
    ("motor_current_raw", "raw_current_unit"),
)
VELOCITY_COLUMNS = (
    "qd_feedback_rad_s",
    "feedback_speed_rad_s",
)
POSITION_COLUMNS = (
    "q_feedback_rad",
    "feedback_position_rad",
)


def numeric(row: dict[str, str], name: str) -> float | None:
    text = (row.get(name) or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def first_populated(rows: list[dict[str, str]], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if any(numeric(row, name) is not None for row in rows):
            return name
    return None


def truthy(text: str) -> bool:
    return text.strip().lower() not in {"0", "false", "no", "invalid"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--joint", required=True, help="Exact joint_name to fit")
    parser.add_argument("--velocity-min", type=float, default=0.01)
    parser.add_argument(
        "--no-position-basis",
        action="store_true",
        help="Do not add sin(q)/cos(q) nuisance terms for gravity/load variation",
    )
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if (row.get("joint_name") or "") == args.joint
            and truthy(row.get("valid") or "true")
        ]
    if not rows:
        raise SystemExit(f"No valid rows found for joint_name={args.joint!r}")

    target_name = first_populated(rows, tuple(name for name, _ in TARGET_COLUMNS))
    velocity_name = first_populated(rows, VELOCITY_COLUMNS)
    position_name = first_populated(rows, POSITION_COLUMNS)
    if target_name is None or velocity_name is None:
        raise SystemExit("A populated torque/current column and rad/s velocity column are required")
    target_unit = dict(TARGET_COLUMNS)[target_name]

    samples: list[tuple[float, float, float | None]] = []
    for row in rows:
        velocity = numeric(row, velocity_name)
        target = numeric(row, target_name)
        position = numeric(row, position_name) if position_name else None
        if velocity is None or target is None or abs(velocity) < args.velocity_min:
            continue
        samples.append((velocity, target, position))
    if len(samples) < 12:
        raise SystemExit(f"Only {len(samples)} usable moving samples; at least 12 are required")

    use_position = (
        not args.no_position_basis
        and position_name is not None
        and all(position is not None for _, _, position in samples)
    )
    design = []
    target_values = []
    for velocity, target, position in samples:
        row = [velocity, math.copysign(1.0, velocity), 1.0]
        if use_position:
            assert position is not None
            row.extend([math.sin(position), math.cos(position)])
        design.append(row)
        target_values.append(target)
    matrix = np.asarray(design, dtype=float)
    values = np.asarray(target_values, dtype=float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(matrix, values, rcond=None)
    prediction = matrix @ coefficients
    residual = values - prediction
    ss_res = float(residual @ residual)
    centered = values - values.mean()
    ss_total = float(centered @ centered)
    r_squared = 1.0 - ss_res / ss_total if ss_total > 0 else None

    result = {
        "joint_name": args.joint,
        "model": (
            "y=b*qd+coulomb*sign(qd)+bias+g_sin*sin(q)+g_cos*cos(q)"
            if use_position
            else "y=b*qd+coulomb*sign(qd)+bias"
        ),
        "target_column": target_name,
        "target_unit": target_unit,
        "velocity_column": velocity_name,
        "position_column": position_name if use_position else None,
        "sample_count": len(samples),
        "positive_velocity_samples": sum(v > 0 for v, _, _ in samples),
        "negative_velocity_samples": sum(v < 0 for v, _, _ in samples),
        "rank": int(rank),
        "condition_number": float(singular_values[0] / singular_values[-1]),
        "r_squared": r_squared,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "coefficients": {
            "viscous": {
                "value": float(coefficients[0]),
                "unit": f"{target_unit}/(rad/s)",
            },
            "coulomb": {"value": float(coefficients[1]), "unit": target_unit},
            "bias": {"value": float(coefficients[2]), "unit": target_unit},
        },
        "safe_interpretation": (
            "Physical torque-domain estimate; still validate sensor bias and gravity model."
            if target_unit == "N*m"
            else "Equivalent current-domain coefficient only; obtain current-to-torque calibration before writing URDF damping."
        ),
    }
    if use_position:
        result["coefficients"]["position_sin"] = {
            "value": float(coefficients[3]),
            "unit": target_unit,
        }
        result["coefficients"]["position_cos"] = {
            "value": float(coefficients[4]),
            "unit": target_unit,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

