#!/usr/bin/env python3
"""Characterise large-amplitude high-speed motion on the mounted left G20.

Large sweeps are restricted to the ten flexion/tip channels that cannot strike
neighbouring fingers while the hand is otherwise open. Yaw and thumb-roll
channels are exercised only within an 8-raw-unit neighbourhood of the open
pose. This policy follows an observed finger-to-finger yaw collision during
the 2026-07-26 mounted calibration.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time


WORKSPACE = Path(__file__).resolve().parents[1]
SDK_PACKAGE = (
    WORKSPACE.parent
    / "linkerhand-ros2-sdk-main"
    / "linker_hand_ros2_sdk"
    / "linker_hand_ros2_sdk"
)
DEFAULT_MAIN_LOG = (
    WORKSPACE
    / "logs"
    / "motion_calibration"
    / "left_g20_motion_20260726_213025.json"
)
DEFAULT_OUTPUT_DIR = WORKSPACE / "logs" / "motion_calibration"
SAFE_OPEN = [
    255, 255, 255, 255, 255,
    255, 193, 148, 105, 42,
    245, 0, 0, 0, 0,
    255, 255, 255, 255, 255,
]
JOINT_NAMES = (
    "thumb_base", "index_base", "middle_base", "ring_base", "little_base",
    "thumb_yaw", "index_yaw", "middle_yaw", "ring_yaw", "little_yaw",
    "thumb_roll", "reserved_1", "reserved_2", "reserved_3", "reserved_4",
    "thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip",
)
FLEXION_INDICES = (0, 1, 2, 3, 4, 15, 16, 17, 18, 19)
YAW_ROLL_INDICES = (5, 6, 7, 8, 9, 10)
ACTIVE_INDICES = FLEXION_INDICES + YAW_ROLL_INDICES
FLEXION_TARGETS = {
    0: 38,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    15: 64,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
}
YAW_ROLL_TARGETS = {
    5: 247,
    6: 185,
    7: 140,
    8: 97,
    9: 50,
    10: 237,
}
SPEED_LEVELS = (153, 204, 255)  # 60%, 80%, 100%
TORQUE_LIMIT_RAW = 153           # 60%; enough for unloaded speed tests
PRESSURE_ABORT_RAW = 45
TEMPERATURE_ABORT_C = 60


class Abort(RuntimeError):
    pass


def state(api) -> list[int]:
    for _ in range(4):
        value = api.get_state()
        if (
            isinstance(value, list)
            and len(value) == 20
            and all(0 <= item <= 255 for item in value)
        ):
            return [int(item) for item in value]
        time.sleep(0.03)
    raise Abort("invalid position feedback")


def close(api) -> None:
    api.hand.running = False
    api.hand.receive_thread.join(timeout=1.2)
    api.hand.bus.shutdown()


def restore_controls(raw, speeds: list[list[int]], torques: list[list[int]]) -> None:
    speed_functions = (
        raw.set_thumb_speed,
        raw.set_index_speed,
        raw.set_middle_speed,
        raw.set_ring_speed,
        raw.set_little_speed,
    )
    torque_functions = (
        raw.set_thumb_torque,
        raw.set_index_torque,
        raw.set_middle_torque,
        raw.set_ring_torque,
        raw.set_little_torque,
    )
    for function, values in zip(speed_functions, speeds):
        function(values)
    for function, values in zip(torque_functions, torques):
        function(values)


def query_temperature(raw) -> list[list[int]]:
    for function in (
        raw.get_thumb_temperature,
        raw.get_index_temperature,
        raw.get_middle_temperature,
        raw.get_ring_temperature,
        raw.get_little_temperature,
    ):
        function()
    time.sleep(0.12)
    return [
        list(value)
        for value in (raw.x61, raw.x62, raw.x63, raw.x64, raw.x65)
    ]


def query_faults(raw) -> list[list[int]]:
    for function in (
        raw.get_thumb_fault,
        raw.get_index_fault,
        raw.get_middle_fault,
        raw.get_ring_fault,
        raw.get_little_fault,
    ):
        function()
    time.sleep(0.12)
    return [
        list(value)
        for value in (raw.x59, raw.x5A, raw.x5B, raw.x5C, raw.x5D)
    ]


def max_temperature(groups: list[list[int]]) -> int:
    values = [
        group[index]
        for group in groups
        for index in (0, 1, 2, 5)
        if len(group) == 6 and group[index] > 0
    ]
    return max(values, default=0)


def move(api, target: list[int], index: int, timeout: float = 5.0) -> dict:
    before = state(api)
    start_value = before[index]
    target_value = target[index]
    command_time = time.monotonic()
    api.finger_move(target)
    samples = []
    settled = 0
    while time.monotonic() - command_time < timeout:
        pose = state(api)
        api.hand.get_normal_force()
        time.sleep(0.008)
        normal = [int(value) for value in api.hand.x90]
        now = time.monotonic()
        samples.append(
            {
                "elapsed_s": round(now - command_time, 5),
                "position_raw": pose[index],
                "pose": pose,
                "normal_force_raw": normal,
            }
        )
        if max(normal, default=0) >= PRESSURE_ABORT_RAW:
            raise Abort(f"pressure safety stop: {normal}")
        # A position controller that runs materially past its requested target
        # is not allowed to continue toward a mechanical endpoint.
        if target_value < start_value and pose[index] < target_value - 12:
            raise Abort(
                f"negative overshoot safety stop on {JOINT_NAMES[index]}: "
                f"target={target_value}, feedback={pose[index]}"
            )
        if target_value > start_value and pose[index] > target_value + 12:
            raise Abort(
                f"positive overshoot safety stop on {JOINT_NAMES[index]}: "
                f"target={target_value}, feedback={pose[index]}"
            )
        if abs(pose[index] - target_value) <= 3:
            settled += 1
            if settled >= 3:
                break
        else:
            settled = 0

    first_motion = next(
        (
            sample["elapsed_s"]
            for sample in samples
            if abs(sample["position_raw"] - start_value) >= 1
        ),
        None,
    )
    velocities = []
    for previous, current in zip(samples, samples[1:]):
        delta_t = current["elapsed_s"] - previous["elapsed_s"]
        if delta_t > 0:
            velocities.append(
                (current["position_raw"] - previous["position_raw"]) / delta_t
            )
    final = samples[-1]["position_raw"]
    return {
        "start_position_raw": start_value,
        "target_position_raw": target_value,
        "final_position_raw": final,
        "final_error_raw": target_value - final,
        "first_motion_seconds": first_motion,
        "settle_seconds": (
            samples[-1]["elapsed_s"] if abs(final - target_value) <= 3 else None
        ),
        "peak_feedback_speed_raw_s": round(
            max((abs(value) for value in velocities), default=0.0), 3
        ),
        "maximum_normal_force_raw": max(
            (
                max(sample["normal_force_raw"], default=0)
                for sample in samples
            ),
            default=0,
        ),
        "success": abs(final - target_value) <= 3,
        "samples": samples,
    }


def ramp_to_open(api, start: list[int]) -> None:
    maximum = max(abs(SAFE_OPEN[index] - start[index]) for index in ACTIVE_INDICES)
    steps = max(1, (maximum + 7) // 8)
    for step in range(1, steps + 1):
        waypoint = list(start)
        for index in ACTIVE_INDICES:
            waypoint[index] = round(
                start[index]
                + (SAFE_OPEN[index] - start[index]) * step / steps
            )
        waypoint[11:15] = [0, 0, 0, 0]
        api.finger_move(waypoint)
        time.sleep(0.12)
    # Interpolated commands can advance faster than a low-speed actuator.
    # Do not disable/restore torque until feedback has actually reached open.
    api.finger_move(SAFE_OPEN)
    deadline = time.monotonic() + 15.0
    settled = 0
    while time.monotonic() < deadline:
        feedback = state(api)
        api.hand.get_normal_force()
        time.sleep(0.015)
        if max(list(api.hand.x90), default=0) >= PRESSURE_ABORT_RAW:
            raise Abort("pressure safety stop while restoring open pose")
        if all(
            abs(feedback[index] - SAFE_OPEN[index]) <= 4
            for index in ACTIVE_INDICES
        ):
            settled += 1
            if settled >= 5:
                return
        else:
            settled = 0
    raise Abort(f"open-pose restore timeout: {state(api)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--main-log", type=Path, default=DEFAULT_MAIN_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-mounted", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not args.execute or not args.confirm_mounted:
        raise SystemExit("motion locked: use --execute --confirm-mounted")
    if not 1 <= args.repeats <= 4:
        raise SystemExit("--repeats must be 1..4")
    if Path(f"/sys/class/net/{args.can}/operstate").read_text().strip() != "up":
        raise SystemExit(f"{args.can} is not up")

    main_log = json.loads(args.main_log.read_text(encoding="utf-8"))
    original_speeds = main_log["original_speed_raw_per_finger"]
    original_torques = main_log["original_torque_raw_per_finger"]
    timestamp = datetime.now().astimezone()
    output = args.output_dir.resolve() / (
        timestamp.strftime("left_g20_high_speed_%Y%m%d_%H%M%S.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": timestamp.isoformat(),
        "model": "G20",
        "side": "left",
        "speed_levels_raw": list(SPEED_LEVELS),
        "speed_levels_percent": [
            round(value / 255 * 100, 2) for value in SPEED_LEVELS
        ],
        "torque_limit_raw": TORQUE_LIMIT_RAW,
        "torque_limit_percent": round(TORQUE_LIMIT_RAW / 255 * 100, 2),
        "large_amplitude_indices": list(FLEXION_INDICES),
        "yaw_roll_small_amplitude_indices": list(YAW_ROLL_INDICES),
        "collision_policy": (
            "No large-amplitude yaw/roll. Large motion is restricted to "
            "flexion/tip axes while all neighbouring fingers remain open."
        ),
        "actuator_commands_sent": False,
        "completed": False,
        "abort_reason": "",
        "trials": [],
    }

    def save() -> None:
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    api = None
    try:
        sys.path.insert(0, str(SDK_PACKAGE))
        from LinkerHand.linker_hand_api import LinkerHandApi

        api = LinkerHandApi(
            hand_type="left", hand_joint="G20", modbus="None", can=args.can
        )
        time.sleep(0.2)
        result["initial_pose"] = state(api)
        before_temperature = query_temperature(api.hand)
        result["temperature_before_raw_per_finger"] = before_temperature
        if max_temperature(before_temperature) >= TEMPERATURE_ABORT_C:
            raise Abort(f"temperature too high: {before_temperature}")
        api.set_torque([TORQUE_LIMIT_RAW] * 5)
        api.set_speed([51] * 5)
        result["actuator_commands_sent"] = True
        ramp_to_open(api, state(api))

        for family, indices, targets in (
            ("large_flexion", FLEXION_INDICES, FLEXION_TARGETS),
            ("small_yaw_roll", YAW_ROLL_INDICES, YAW_ROLL_TARGETS),
        ):
            for sequence, index in enumerate(indices, 1):
                print(
                    f"{family} {sequence:02d}/{len(indices):02d} "
                    f"{JOINT_NAMES[index]}",
                    flush=True,
                )
                target = list(SAFE_OPEN)
                target[index] = targets[index]
                for speed in SPEED_LEVELS:
                    api.set_speed([speed] * 5)
                    time.sleep(0.04)
                    for repeat in range(1, args.repeats + 1):
                        outbound = move(api, target, index)
                        result["trials"].append(
                            {
                                "trial_number": len(result["trials"]) + 1,
                                "family": family,
                                "direction": "outbound",
                                "joint_index": index,
                                "joint_name": JOINT_NAMES[index],
                                "speed_raw": speed,
                                "speed_percent": round(speed / 255 * 100, 2),
                                "repeat": repeat,
                                **outbound,
                            }
                        )
                        inbound = move(api, SAFE_OPEN, index)
                        result["trials"].append(
                            {
                                "trial_number": len(result["trials"]) + 1,
                                "family": family,
                                "direction": "return",
                                "joint_index": index,
                                "joint_name": JOINT_NAMES[index],
                                "speed_raw": speed,
                                "speed_percent": round(speed / 255 * 100, 2),
                                "repeat": repeat,
                                **inbound,
                            }
                        )
                        save()
                temperature = query_temperature(api.hand)
                if max_temperature(temperature) >= TEMPERATURE_ABORT_C:
                    raise Abort(f"temperature safety stop: {temperature}")

        result["temperature_after_raw_per_finger"] = query_temperature(api.hand)
        result["fault_after_raw_per_finger"] = query_faults(api.hand)
        result["final_open_pose"] = state(api)
        result["trial_count"] = len(result["trials"])
        result["successful_trial_count"] = sum(
            bool(trial["success"]) for trial in result["trials"]
        )
        result["completed"] = not any(
            value
            for group in result["fault_after_raw_per_finger"]
            for value in group
        )
    except BaseException as exc:
        result["abort_reason"] = f"{type(exc).__name__}: {exc}"
        print(f"HIGH-SPEED ABORT: {result['abort_reason']}", flush=True)
    finally:
        if api is not None:
            try:
                api.set_speed([26] * 5)
                api.set_torque([38] * 5)
                ramp_to_open(api, state(api))
                restore_controls(api.hand, original_speeds, original_torques)
                result["original_control_settings_restored"] = True
                result["final_pose_after_control_restore"] = state(api)
            except BaseException as exc:
                result["finalization_error"] = f"{type(exc).__name__}: {exc}"
            close(api)
        result["trial_count"] = len(result["trials"])
        result["successful_trial_count"] = sum(
            bool(trial["success"]) for trial in result["trials"]
        )
        save()
        print(output, flush=True)
        print(
            f"Trials: {result['trial_count']}, successful: "
            f"{result['successful_trial_count']}",
            flush=True,
        )
    return 0 if result["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
