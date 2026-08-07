#!/usr/bin/env python3
"""Connect to one LinkerHand without issuing any actuator command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import yaml


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = WORKSPACE / "config/hand_left.yaml"
DEFAULT_OUTPUT = WORKSPACE / "logs/latest_left_g20_connection.json"
SUPPORTED_MODELS = {"O6", "L6", "L7", "L10", "L20", "G20", "L21", "L25"}


def load_config(path: Path, *, require_disarmed: bool = True) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    if require_disarmed and data.get("armed") is not False:
        raise ValueError("read-only connection requires armed: false")

    hand = data.get("hand", {})
    transport = data.get("transport", {})
    model = hand.get("model")
    side = hand.get("side")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported hand model: {model!r}")
    if side not in {"left", "right"}:
        raise ValueError(f"invalid hand side: {side!r}")
    if transport.get("type") != "can":
        raise ValueError("this read-only connector currently requires transport.type: can")
    if transport.get("bitrate") != 1_000_000:
        raise ValueError("LinkerHand CAN bitrate must be 1000000")
    return data


def require_can_up(interface: str) -> None:
    operstate = Path(f"/sys/class/net/{interface}/operstate")
    if not operstate.is_file():
        raise RuntimeError(f"CAN interface does not exist: {interface}")
    state = operstate.read_text(encoding="utf-8").strip()
    if state != "up":
        raise RuntimeError(f"CAN interface {interface} is not up (state={state})")


def classify_touch(api) -> dict:
    hand = api.hand
    observed_codes: list[int | None] = []

    for _ in range(4):
        observed_codes.append(api.get_touch_type())
        time.sleep(0.15)

    raw_b0 = list(getattr(hand, "xb0", []))
    normal_force = list(getattr(hand, "normal_force", []))

    if raw_b0 == [2] or 2 in observed_codes:
        code = 2
        kind = "matrix"
    elif raw_b0 == [1] or 1 in observed_codes:
        code = 1
        kind = "pressure"
    elif any(
        isinstance(value, (int, float)) and value != -1
        for value in normal_force
    ):
        code = 1
        kind = "pressure_legacy"
    else:
        code = -1
        kind = "none"

    return {
        "code": code,
        "kind": kind,
        "raw_b0": raw_b0,
        "normal_force": normal_force,
        "observed_codes": observed_codes,
    }


def read_state(api) -> list[int]:
    # G20 SDK 3.1.1 returns the state from get_state(), while its
    # get_state_for_pub() wrapper currently forgets to return a value.
    for _ in range(4):
        state = api.get_state()
        time.sleep(0.08)
        if state is not None:
            values = list(state)
            if values:
                return [int(value) for value in values]

        state = api.get_state_for_pub()
        time.sleep(0.08)
        if state is not None:
            values = list(state)
            if values:
                return [int(value) for value in values]
    return []


def read_faults(api, model: str, attempts: int = 3) -> list[int]:
    """Try both known read-only L20 fault-query forms.

    Some L20 firmware does not answer either query.  An empty result therefore
    means unsupported telemetry, never "no fault".  A payload containing ones
    would clear faults and is deliberately never sent here.
    """
    if model == "G20":
        # G20 has per-joint, read-only fault-query frames. Do not reuse the
        # legacy L20 0x07 command on this device.
        faults: list[int] = []
        for _ in range(attempts):
            result = api.get_fault()
            time.sleep(0.12)
            if result is not None:
                faults = [int(value) for value in result]
                if len(faults) == 20:
                    return faults
        return faults

    hand = api.hand
    hand.x07 = []
    for _ in range(attempts):
        hand.send_command(0x07, [], sleep=0.03)
        time.sleep(0.15)
        faults = [int(value) for value in hand.x07]
        if len(faults) == 5:
            return faults
        hand.send_command(0x07, [0, 0, 0, 0, 0], sleep=0.03)
        time.sleep(0.15)
        faults = [int(value) for value in hand.x07]
        if len(faults) == 5:
            return faults
    return [int(value) for value in hand.x07]


def close_bus_only(api) -> None:
    hand = getattr(api, "hand", None)
    if hand is None:
        return
    if hasattr(hand, "running"):
        hand.running = False
    receive_thread = getattr(hand, "receive_thread", None)
    if receive_thread is not None and receive_thread.is_alive():
        receive_thread.join(timeout=1.2)
    bus = getattr(hand, "bus", None)
    if bus is not None:
        bus.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = load_config(args.config.expanduser().resolve())
    hand_config = config["hand"]
    transport = config["transport"]
    interface = transport["can_interface"]
    require_can_up(interface)

    # Import only after all safety checks pass. Construction performs read/query
    # traffic but this script never calls position, speed, current, or torque APIs.
    from linker_hand_ros2_sdk.LinkerHand.linker_hand_api import LinkerHandApi

    api = None
    try:
        api = LinkerHandApi(
            hand_type=hand_config["side"],
            hand_joint=hand_config["model"],
            modbus="None",
            can=interface,
        )
        time.sleep(0.2)
        touch = classify_touch(api)
        state = read_state(api)
        faults = read_faults(api, hand_config["model"])
        expected_length = 20 if hand_config["model"] in {"L20", "G20"} else None
        state_valid = (
            expected_length is not None
            and len(state) == expected_length
            and any(value >= 0 for value in state)
        )

        result = {
            "model": hand_config["model"],
            "side": hand_config["side"],
            "can_interface": interface,
            "can_id": "0x27" if hand_config["side"] == "right" else "0x28",
            "state": state,
            "state_valid": state_valid,
            "faults": faults,
            "faults_valid": len(faults) == (20 if hand_config["model"] == "G20" else 5),
            "touch": touch,
            "actuator_commands_sent": False,
        }
    finally:
        if api is not None:
            close_bus_only(api)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["state_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
