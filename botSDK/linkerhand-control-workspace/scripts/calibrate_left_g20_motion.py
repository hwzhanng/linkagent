#!/usr/bin/env python3
"""Run a guarded motion-characterisation session on the mounted left G20.

The script covers the 16 feedback-bearing G20 logical channels. Indices 11-14
are vendor-reserved and are held at zero. Every movement is monitored using
position feedback, aggregate fingertip pressure, CAN-backed fault queries and
temperature checkpoints. The hand returns to its initial pose and its original
per-joint speed/torque settings are restored even when a trial fails.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable


WORKSPACE = Path(__file__).resolve().parents[1]
SDK_PACKAGE = (
    WORKSPACE.parent
    / "linkerhand-ros2-sdk-main"
    / "linker_hand_ros2_sdk"
    / "linker_hand_ros2_sdk"
)
DEFAULT_OUTPUT_DIR = WORKSPACE / "logs" / "motion_calibration"
ACTIVE_INDICES = tuple(range(11)) + tuple(range(15, 20))
RESERVED_INDICES = (11, 12, 13, 14)
JOINT_NAMES = (
    "thumb_base",
    "index_base",
    "middle_base",
    "ring_base",
    "little_base",
    "thumb_yaw",
    "index_yaw",
    "middle_yaw",
    "ring_yaw",
    "little_yaw",
    "thumb_roll",
    "reserved_1",
    "reserved_2",
    "reserved_3",
    "reserved_4",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "little_tip",
)

# Vendor G20 GUI "open" preset, with the four non-feedback reserved values
# replaced by zero. The remaining values are unchanged.
SAFE_OPEN = [
    255, 255, 255, 255, 255,
    255, 193, 148, 105, 42,
    245, 0, 0, 0, 0,
    255, 255, 255, 255, 255,
]

# Independently safe envelope after the mounted-hand trial on 2026-07-26.
# The vendor pose library demonstrates much wider yaw values, but those values
# rely on coordinated neighbouring-finger avoidance. A single yaw joint at
# those extrema can strike another finger, so independent yaw/roll tests are
# deliberately restricted to a small neighbourhood around the open pose.
INDEPENDENT_SAFE_ENVELOPE = {
    0: (38, 255),
    1: (0, 255),
    2: (0, 255),
    3: (0, 255),
    4: (0, 255),
    5: (247, 255),
    6: (185, 200),
    7: (140, 156),
    8: (97, 113),
    9: (42, 50),
    10: (235, 242),
    15: (64, 255),
    16: (0, 255),
    17: (0, 255),
    18: (0, 255),
    19: (0, 255),
}

SPEED_LEVELS = (26, 51, 102)       # about 10%, 20%, 40% of 255
TORQUE_LEVELS = (38, 64, 102)      # about 15%, 25%, 40% of 255
PRESSURE_ABORT_RAW = 60
TEMPERATURE_ABORT_C = 60
POSITION_TOLERANCE_RAW = 2
SAMPLE_PERIOD_SECONDS = 0.035


class CalibrationAbort(RuntimeError):
    pass


def require_can_up(interface: str) -> None:
    state_file = Path(f"/sys/class/net/{interface}/operstate")
    if not state_file.is_file():
        raise CalibrationAbort(f"CAN interface missing: {interface}")
    if state_file.read_text(encoding="utf-8").strip() != "up":
        raise CalibrationAbort(f"CAN interface is not up: {interface}")


def close_bus(api) -> None:
    hand = getattr(api, "hand", None)
    if hand is None:
        return
    hand.running = False
    receive_thread = getattr(hand, "receive_thread", None)
    if receive_thread is not None and receive_thread.is_alive():
        receive_thread.join(timeout=1.2)
    bus = getattr(hand, "bus", None)
    if bus is not None:
        bus.shutdown()


def valid_state(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 20
        and all(isinstance(item, (int, float)) and 0 <= item <= 255 for item in value)
    )


def read_state(api, attempts: int = 4) -> list[int]:
    for _ in range(attempts):
        state = api.get_state()
        if valid_state(state):
            return [int(value) for value in state]
        time.sleep(0.04)
    raise CalibrationAbort("invalid G20 position feedback")


def query_raw_groups(raw, kind: str) -> list[list[int]]:
    maps: dict[str, tuple[tuple[Callable, ...], tuple[str, ...]]] = {
        "speed": (
            (
                raw.get_thumb_speed,
                raw.get_index_speed,
                raw.get_middle_speed,
                raw.get_ring_speed,
                raw.get_little_speed,
            ),
            ("x49", "x4A", "x4B", "x4C", "x4D"),
        ),
        "torque": (
            (
                raw.get_thumb_torque,
                raw.get_index_torque,
                raw.get_middle_torque,
                raw.get_ring_torque,
                raw.get_little_torque,
            ),
            ("x51", "x52", "x53", "x54", "x55"),
        ),
        "temperature": (
            (
                raw.get_thumb_temperature,
                raw.get_index_temperature,
                raw.get_middle_temperature,
                raw.get_ring_temperature,
                raw.get_little_temperature,
            ),
            ("x61", "x62", "x63", "x64", "x65"),
        ),
        "fault": (
            (
                raw.get_thumb_fault,
                raw.get_index_fault,
                raw.get_middle_fault,
                raw.get_ring_fault,
                raw.get_little_fault,
            ),
            ("x59", "x5A", "x5B", "x5C", "x5D"),
        ),
    }
    functions, attributes = maps[kind]
    for function in functions:
        function()
    time.sleep(0.12)
    return [[int(item) for item in getattr(raw, attr)] for attr in attributes]


def restore_raw_groups(raw, speeds: list[list[int]], torques: list[list[int]]) -> None:
    for function, values in zip(
        (
            raw.set_thumb_speed,
            raw.set_index_speed,
            raw.set_middle_speed,
            raw.set_ring_speed,
            raw.set_little_speed,
        ),
        speeds,
    ):
        if len(values) == 6:
            function(values)
    for function, values in zip(
        (
            raw.set_thumb_torque,
            raw.set_index_torque,
            raw.set_middle_torque,
            raw.set_ring_torque,
            raw.set_little_torque,
        ),
        torques,
    ):
        if len(values) == 6:
            function(values)


def maximum_active_temperature(groups: list[list[int]]) -> int:
    # Physical G20 axes occupy raw slots 0, 1, 2 and 5. Zero-valued absent
    # motor slots are deliberately excluded.
    values = [
        group[index]
        for group in groups
        for index in (0, 1, 2, 5)
        if len(group) == 6 and group[index] > 0
    ]
    return max(values, default=0)


def has_fault(groups: list[list[int]]) -> bool:
    return any(value != 0 for group in groups for value in group)


def movement_target(anchor: list[int], index: int, magnitude: int) -> list[int]:
    low, high = INDEPENDENT_SAFE_ENVELOPE[index]
    value = anchor[index]
    if value - magnitude >= low:
        destination = value - magnitude
    elif value + magnitude <= high:
        destination = value + magnitude
    else:
        destination = low if abs(value - low) >= abs(value - high) else high
    target = list(anchor)
    target[index] = int(destination)
    for reserved in RESERVED_INDICES:
        target[reserved] = 0
    return target


def trial_timeout(start: int, target: int, speed_raw: int) -> float:
    distance = abs(target - start)
    return min(14.0, max(1.6, 1.2 + distance / max(speed_raw * 0.45, 4.5)))


def summarize_samples(
    samples: list[dict], start_value: int, target_value: int, command_time: float,
) -> dict:
    first_motion = next(
        (
            sample["monotonic_s"] - command_time
            for sample in samples
            if abs(sample["position_raw"] - start_value) >= 1
        ),
        None,
    )
    settled = next(
        (
            samples[index]["monotonic_s"] - command_time
            for index in range(3, len(samples))
            if all(
                abs(samples[inner]["position_raw"] - target_value)
                <= POSITION_TOLERANCE_RAW
                for inner in range(index - 3, index + 1)
            )
        ),
        None,
    )
    velocities = []
    for previous, current in zip(samples, samples[1:]):
        delta_t = current["monotonic_s"] - previous["monotonic_s"]
        if delta_t > 0:
            velocities.append(
                (current["position_raw"] - previous["position_raw"]) / delta_t
            )
    final = samples[-1]["position_raw"] if samples else start_value
    return {
        "first_motion_seconds": (
            round(first_motion, 5) if first_motion is not None else None
        ),
        "settle_seconds": round(settled, 5) if settled is not None else None,
        "peak_feedback_speed_raw_s": (
            round(max((abs(value) for value in velocities), default=0.0), 3)
        ),
        "final_position_raw": final,
        "final_error_raw": target_value - final,
        "achieved_delta_raw": final - start_value,
        "success": abs(final - target_value) <= POSITION_TOLERANCE_RAW,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-mounted", action="store_true")
    parser.add_argument("--speed-repeats", type=int, default=2)
    args = parser.parse_args()
    if not args.execute or not args.confirm_mounted:
        raise SystemExit(
            "Motion is locked. Supply --execute --confirm-mounted after checking the fixture."
        )
    if not 1 <= args.speed_repeats <= 5:
        raise SystemExit("--speed-repeats must be between 1 and 5")
    require_can_up(args.can)
    if not SDK_PACKAGE.is_dir():
        raise SystemExit(f"SDK package missing: {SDK_PACKAGE}")
    sys.path.insert(0, str(SDK_PACKAGE))
    from LinkerHand.linker_hand_api import LinkerHandApi

    timestamp = datetime.now().astimezone()
    session_id = timestamp.strftime("left_g20_motion_%Y%m%d_%H%M%S")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{session_id}.json"
    csv_path = output_dir / f"{session_id}_summary.csv"

    api = None
    initial: list[int] = []
    original_speeds: list[list[int]] = []
    original_torques: list[list[int]] = []
    trials: list[dict] = []
    control = {"speed": None, "torque": None}
    result = {
        "session_id": session_id,
        "timestamp": timestamp.isoformat(),
        "model": "G20",
        "side": "left",
        "can_interface": args.can,
        "active_indices": list(ACTIVE_INDICES),
        "reserved_indices": list(RESERVED_INDICES),
        "joint_names": list(JOINT_NAMES),
        "independent_safe_envelope": {
            str(index): list(bounds)
            for index, bounds in INDEPENDENT_SAFE_ENVELOPE.items()
        },
        "yaw_collision_policy": (
            "Yaw/roll extrema from vendor coordinated presets are forbidden "
            "for isolated-joint tests after observed finger-to-finger contact."
        ),
        "pressure_abort_raw": PRESSURE_ABORT_RAW,
        "temperature_abort_c": TEMPERATURE_ABORT_C,
        "actuator_commands_sent": False,
        "completed": False,
        "abort_reason": "",
        "trials": trials,
    }

    def checkpoint() -> None:
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def set_control(speed_raw: int, torque_raw: int) -> None:
        if control["speed"] != speed_raw:
            api.set_speed([speed_raw] * 5)
            control["speed"] = speed_raw
            time.sleep(0.06)
        if control["torque"] != torque_raw:
            api.set_torque([torque_raw] * 5)
            control["torque"] = torque_raw
            time.sleep(0.06)

    def run_move(
        *,
        phase: str,
        joint_index: int | None,
        target: list[int],
        speed_raw: int,
        torque_raw: int,
        repeat: int = 1,
    ) -> dict:
        set_control(speed_raw, torque_raw)
        start_pose = read_state(api)
        observed_index = (
            joint_index
            if joint_index is not None
            else max(
                ACTIVE_INDICES,
                key=lambda index: abs(target[index] - start_pose[index]),
            )
        )
        start_value = start_pose[observed_index]
        target_value = target[observed_index]
        timeout = trial_timeout(start_value, target_value, speed_raw)
        command_time = time.monotonic()
        api.finger_move(target)
        result["actuator_commands_sent"] = True
        samples = []
        settled_count = 0
        while time.monotonic() - command_time < timeout:
            state = read_state(api)
            raw = api.hand
            raw.get_normal_force()
            time.sleep(0.012)
            normal = [int(value) for value in raw.x90]
            peak_pressure = max(normal, default=0)
            sample = {
                "monotonic_s": time.monotonic(),
                "position_raw": state[observed_index],
                "pose": state,
                "normal_force_raw": normal,
            }
            samples.append(sample)
            if peak_pressure >= PRESSURE_ABORT_RAW:
                raise CalibrationAbort(
                    f"pressure safety stop: {peak_pressure} >= {PRESSURE_ABORT_RAW}"
                )
            if (
                target_value < start_value
                and state[observed_index] < target_value - 12
            ):
                raise CalibrationAbort(
                    f"negative overshoot safety stop on "
                    f"{JOINT_NAMES[observed_index]}: target={target_value}, "
                    f"feedback={state[observed_index]}"
                )
            if (
                target_value > start_value
                and state[observed_index] > target_value + 12
            ):
                raise CalibrationAbort(
                    f"positive overshoot safety stop on "
                    f"{JOINT_NAMES[observed_index]}: target={target_value}, "
                    f"feedback={state[observed_index]}"
                )
            if abs(state[observed_index] - target_value) <= POSITION_TOLERANCE_RAW:
                settled_count += 1
                if settled_count >= 4:
                    break
            else:
                settled_count = 0
            remaining = SAMPLE_PERIOD_SECONDS - (
                time.monotonic() - sample["monotonic_s"]
            )
            if remaining > 0:
                time.sleep(remaining)
        summary = summarize_samples(
            samples, start_value, target_value, command_time
        )
        trial = {
            "trial_number": len(trials) + 1,
            "phase": phase,
            "repeat": repeat,
            "joint_index": observed_index,
            "joint_name": JOINT_NAMES[observed_index],
            "speed_raw": speed_raw,
            "speed_percent": round(speed_raw / 255 * 100, 2),
            "torque_limit_raw": torque_raw,
            "torque_limit_percent": round(torque_raw / 255 * 100, 2),
            "command_time_monotonic_s": command_time,
            "start_position_raw": start_value,
            "target_position_raw": target_value,
            "timeout_seconds": timeout,
            "maximum_normal_force_raw": max(
                (
                    max(sample["normal_force_raw"], default=0)
                    for sample in samples
                ),
                default=0,
            ),
            **summary,
            "samples": samples,
        }
        trials.append(trial)
        checkpoint()
        return trial

    def move_to_anchor(
        phase: str, anchor: list[int], speed: int = 51, torque: int = 64,
    ) -> None:
        current = read_state(api)
        maximum_delta = max(
            abs(anchor[index] - current[index]) for index in ACTIVE_INDICES
        )
        steps = max(1, math.ceil(maximum_delta / 16))
        for step in range(1, steps + 1):
            waypoint = list(current)
            for index in ACTIVE_INDICES:
                waypoint[index] = round(
                    current[index]
                    + (anchor[index] - current[index]) * step / steps
                )
            for reserved in RESERVED_INDICES:
                waypoint[reserved] = 0
            run_move(
                phase=phase,
                joint_index=None,
                target=waypoint,
                speed_raw=speed,
                torque_raw=torque,
                repeat=step,
            )

    try:
        api = LinkerHandApi(
            hand_type="left", hand_joint="G20", modbus="None", can=args.can
        )
        time.sleep(0.2)
        raw = api.hand
        initial = read_state(api)
        original_speeds = query_raw_groups(raw, "speed")
        original_torques = query_raw_groups(raw, "torque")
        temperatures_before = query_raw_groups(raw, "temperature")
        faults_before = query_raw_groups(raw, "fault")
        if has_fault(faults_before):
            raise CalibrationAbort(f"pre-existing motor fault: {faults_before}")
        if maximum_active_temperature(temperatures_before) >= TEMPERATURE_ABORT_C:
            raise CalibrationAbort(
                f"temperature too high: {temperatures_before}"
            )
        result.update(
            {
                "serial_number": str(getattr(api, "serial_number", "")),
                "initial_pose": initial,
                "original_speed_raw_per_finger": original_speeds,
                "original_torque_raw_per_finger": original_torques,
                "temperature_before_raw_per_finger": temperatures_before,
                "fault_before_raw_per_finger": faults_before,
            }
        )
        checkpoint()

        print("[1/6] moving to vendor safe-open anchor", flush=True)
        move_to_anchor("safe_open_transition", SAFE_OPEN)

        print("[2/6] probing all 16 feedback channels", flush=True)
        for sequence, index in enumerate(ACTIVE_INDICES, 1):
            print(
                f"  probe {sequence:02d}/16 {JOINT_NAMES[index]}",
                flush=True,
            )
            target = movement_target(SAFE_OPEN, index, 8)
            run_move(
                phase="channel_probe",
                joint_index=index,
                target=target,
                speed_raw=26,
                torque_raw=38,
            )
            run_move(
                phase="channel_probe_return",
                joint_index=index,
                target=SAFE_OPEN,
                speed_raw=26,
                torque_raw=38,
            )

        print("[3/6] scanning vendor-demonstrated position envelope", flush=True)
        for sequence, index in enumerate(ACTIVE_INDICES, 1):
            print(
                f"  range {sequence:02d}/16 {JOINT_NAMES[index]}",
                flush=True,
            )
            low, high = INDEPENDENT_SAFE_ENVELOPE[index]
            for label, value in (("low", low), ("high", high)):
                if value == SAFE_OPEN[index]:
                    continue
                target = list(SAFE_OPEN)
                target[index] = value
                run_move(
                    phase=f"range_{label}",
                    joint_index=index,
                    target=target,
                    speed_raw=51,
                    torque_raw=64,
                )
                run_move(
                    phase=f"range_{label}_return",
                    joint_index=index,
                    target=SAFE_OPEN,
                    speed_raw=51,
                    torque_raw=64,
                )
            if sequence % 4 == 0:
                temperatures = query_raw_groups(raw, "temperature")
                if maximum_active_temperature(temperatures) >= TEMPERATURE_ABORT_C:
                    raise CalibrationAbort(
                        f"temperature safety stop: {temperatures}"
                    )

        print("[4/6] running three-speed repeated response tests", flush=True)
        for sequence, index in enumerate(ACTIVE_INDICES, 1):
            print(
                f"  speed {sequence:02d}/16 {JOINT_NAMES[index]}",
                flush=True,
            )
            target = movement_target(SAFE_OPEN, index, 24)
            for speed in SPEED_LEVELS:
                for repeat in range(1, args.speed_repeats + 1):
                    run_move(
                        phase="speed_response",
                        joint_index=index,
                        target=target,
                        speed_raw=speed,
                        torque_raw=64,
                        repeat=repeat,
                    )
                    run_move(
                        phase="speed_response_return",
                        joint_index=index,
                        target=SAFE_OPEN,
                        speed_raw=speed,
                        torque_raw=64,
                        repeat=repeat,
                    )

        print("[5/6] running three-level torque-limit response tests", flush=True)
        for sequence, index in enumerate(ACTIVE_INDICES, 1):
            print(
                f"  torque {sequence:02d}/16 {JOINT_NAMES[index]}",
                flush=True,
            )
            target = movement_target(SAFE_OPEN, index, 24)
            for torque in TORQUE_LEVELS:
                run_move(
                    phase="torque_limit_response",
                    joint_index=index,
                    target=target,
                    speed_raw=51,
                    torque_raw=torque,
                )
                run_move(
                    phase="torque_limit_response_return",
                    joint_index=index,
                    target=SAFE_OPEN,
                    speed_raw=51,
                    torque_raw=torque,
                )

        print("[6/6] measuring minimum command/deadband and restoring", flush=True)
        for sequence, index in enumerate(ACTIVE_INDICES, 1):
            print(
                f"  deadband {sequence:02d}/16 {JOINT_NAMES[index]}",
                flush=True,
            )
            for magnitude in (1, 2, 3):
                target = movement_target(SAFE_OPEN, index, magnitude)
                run_move(
                    phase="deadband",
                    joint_index=index,
                    target=target,
                    speed_raw=26,
                    torque_raw=64,
                    repeat=magnitude,
                )
                run_move(
                    phase="deadband_return",
                    joint_index=index,
                    target=SAFE_OPEN,
                    speed_raw=26,
                    torque_raw=64,
                    repeat=magnitude,
                )

        move_to_anchor("restore_initial", initial)
        temperatures_after = query_raw_groups(raw, "temperature")
        faults_after = query_raw_groups(raw, "fault")
        final_pose = read_state(api)
        result.update(
            {
                "temperature_after_raw_per_finger": temperatures_after,
                "fault_after_raw_per_finger": faults_after,
                "final_pose_before_control_restore": final_pose,
                "maximum_final_pose_error_raw": max(
                    abs(final_pose[index] - initial[index])
                    for index in ACTIVE_INDICES
                ),
                "trial_count": len(trials),
                "successful_trial_count": sum(
                    bool(trial["success"]) for trial in trials
                ),
                "completed": not has_fault(faults_after),
            }
        )
    except BaseException as exc:
        result["abort_reason"] = f"{type(exc).__name__}: {exc}"
        print(f"CALIBRATION ABORT: {result['abort_reason']}", flush=True)
        if api is not None and initial:
            try:
                move_to_anchor("abort_restore_initial", initial)
            except BaseException as restore_exc:
                result["restore_error"] = (
                    f"{type(restore_exc).__name__}: {restore_exc}"
                )
    finally:
        if api is not None:
            try:
                if original_speeds and original_torques:
                    restore_raw_groups(
                        api.hand, original_speeds, original_torques
                    )
                    result["original_control_settings_restored"] = True
                result["final_pose_after_control_restore"] = read_state(api)
            except BaseException as final_exc:
                result["finalization_error"] = (
                    f"{type(final_exc).__name__}: {final_exc}"
                )
            close_bus(api)
        result["trial_count"] = len(trials)
        result["successful_trial_count"] = sum(
            bool(trial["success"]) for trial in trials
        )
        checkpoint()

        summary_fields = [
            "trial_number",
            "phase",
            "repeat",
            "joint_index",
            "joint_name",
            "speed_raw",
            "speed_percent",
            "torque_limit_raw",
            "torque_limit_percent",
            "start_position_raw",
            "target_position_raw",
            "first_motion_seconds",
            "settle_seconds",
            "peak_feedback_speed_raw_s",
            "final_position_raw",
            "final_error_raw",
            "achieved_delta_raw",
            "maximum_normal_force_raw",
            "success",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_fields)
            writer.writeheader()
            for trial in trials:
                writer.writerow(
                    {field: trial.get(field) for field in summary_fields}
                )
        print(f"JSON: {json_path}", flush=True)
        print(f"CSV:  {csv_path}", flush=True)
        print(
            f"Trials: {len(trials)}, successful: "
            f"{sum(bool(trial['success']) for trial in trials)}",
            flush=True,
        )

    return 0 if result["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
