#!/usr/bin/env python3
"""Read one five-fingertip pressure-matrix snapshot without motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from read_only_connect import close_bus_only, load_config, require_can_up


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = WORKSPACE / "config/hand_left.yaml"
DEFAULT_OUTPUT = WORKSPACE / "logs/latest_left_g20_pressure_snapshot.json"
MATRIX_ATTRIBUTES = {
    "thumb": "thumb_matrix",
    "index": "index_matrix",
    "middle": "middle_matrix",
    "ring": "ring_matrix",
    "little": "little_matrix",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = load_config(args.config.expanduser().resolve())
    touch_type = config["hand"].get("touch_type")
    if touch_type not in {"matrix", "matrix_v2"}:
        raise RuntimeError("configured hand is not a matrix-touch device")
    interface = config["transport"]["can_interface"]
    require_can_up(interface)

    from linker_hand_ros2_sdk.LinkerHand.linker_hand_api import LinkerHandApi

    api = None
    try:
        api = LinkerHandApi(
            hand_type=config["hand"]["side"],
            hand_joint=config["hand"]["model"],
            modbus="None",
            can=interface,
        )
        time.sleep(0.2)
        read_matrix = (
            api.get_matrix_touch_v2
            if touch_type == "matrix_v2"
            else api.get_matrix_touch
        )
        for _ in range(2):
            read_matrix()
            time.sleep(0.35)

        matrices = {
            finger: getattr(api.hand, attribute).astype(int).tolist()
            for finger, attribute in MATRIX_ATTRIBUTES.items()
        }
    finally:
        if api is not None:
            close_bus_only(api)

    summary = {}
    for finger, matrix in matrices.items():
        flat = [value for row in matrix for value in row]
        summary[finger] = {
            "shape": [len(matrix), len(matrix[0]) if matrix else 0],
            "minimum": min(flat),
            "maximum": max(flat),
            "sum": sum(flat),
            "nonzero_cells": sum(value > 0 for value in flat),
        }

    result = {
        "model": config["hand"]["model"],
        "side": config["hand"]["side"],
        "can_interface": interface,
        "summary": summary,
        "matrices": matrices,
        "actuator_commands_sent": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
