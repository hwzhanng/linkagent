#!/usr/bin/env python3
"""Fit a conservative G20 motion overlay from mounted-hand calibration logs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics

import yaml


WORKSPACE = Path(__file__).resolve().parents[1]
LOG_DIR = WORKSPACE / "logs" / "motion_calibration"
DEFAULT_MAIN = LOG_DIR / "left_g20_motion_20260726_213025.json"
DEFAULT_HIGH = LOG_DIR / "left_g20_high_speed_20260726_214203.json"
DEFAULT_ABORTED_MAX = LOG_DIR / "left_g20_max_speed_reliability_20260726.json"
DEFAULT_RELIABILITY_80 = LOG_DIR / "left_g20_80pct_reliability_20260726.json"
DEFAULT_MID_DEADBAND = LOG_DIR / "left_g20_midrange_deadband_20260726.json"
DEFAULT_YAML = LOG_DIR / "left_g20_motion_overlay_20260726.yaml"
DEFAULT_REPORT = LOG_DIR / "LEFT_G20_MOTION_CALIBRATION_20260726.md"
ACTIVE_INDICES = tuple(range(11)) + tuple(range(15, 20))
FLEXION_INDICES = (0, 1, 2, 3, 4, 15, 16, 17, 18, 19)
YAW_ROLL_INDICES = (5, 6, 7, 8, 9, 10)
JOINT_SPECS = {
    0: ("thumb_base", "thumb_joint2", 0.0, 0.84, True),
    1: ("index_base", "index_joint1", 0.0, 1.66, True),
    2: ("middle_base", "middle_joint1", 0.0, 1.66, True),
    3: ("ring_base", "ring_joint1", 0.0, 1.66, True),
    4: ("little_base", "little_joint1", 0.0, 1.66, True),
    # Yaw/roll radian signs use the delivered left URDF limits plus the vendor
    # L20 left-axis inversion convention. G20 SDK 3.1.1 does not provide its
    # own raw-to-radian conversion, so these signs remain an inferred mapping.
    5: ("thumb_yaw", "thumb_joint0", -1.05, 0.49, True),
    6: ("index_yaw", "index_joint0", -0.26, 0.26, True),
    7: ("middle_yaw", "middle_joint0", -0.26, 0.26, True),
    8: ("ring_yaw", "ring_joint0", -0.26, 0.26, True),
    9: ("little_yaw", "little_joint0", -0.26, 0.26, True),
    10: ("thumb_roll", "thumb_joint1", 0.0, 1.45, True),
    15: ("thumb_tip", "thumb_joint3", 0.0, 1.32, True),
    16: ("index_tip", "index_joint2", 0.0, 1.36, True),
    17: ("middle_tip", "middle_joint2", 0.0, 1.36, True),
    18: ("ring_tip", "ring_joint2", 0.0, 1.36, True),
    19: ("little_tip", "little_joint2", 0.0, 1.36, True),
}

# Margins are based on measured feedback endpoints, not the nominal 0..255
# protocol range. Yaw/roll are restricted by the observed self-collision.
TRAINING_RAW_LIMITS = {
    0: (50, 250),
    1: (15, 250),
    2: (5, 250),
    3: (5, 250),
    4: (15, 250),
    5: (247, 255),
    6: (185, 200),
    7: (140, 156),
    8: (97, 113),
    9: (42, 50),
    10: (235, 242),
    15: (75, 250),
    16: (5, 250),
    17: (5, 250),
    18: (10, 250),
    19: (5, 250),
}
YAW_ROLL_VELOCITY_CAP_RAD_S = {
    5: 0.50,
    6: 0.20,
    7: 0.20,
    8: 0.20,
    9: 0.20,
    10: 0.40,
}


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def raw_to_rad(index: int, raw: float) -> float:
    _, _, lower, upper, inverted = JOINT_SPECS[index]
    ratio = max(0.0, min(255.0, raw)) / 255.0
    return (
        upper + (lower - upper) * ratio
        if inverted
        else lower + (upper - lower) * ratio
    )


def effective_velocity(trial: dict, high_speed: bool) -> float | None:
    first = trial.get("first_motion_seconds")
    if first is None:
        return None
    start = trial["start_position_raw"]
    final = trial["final_position_raw"]
    previous = start
    last_change = None
    for sample in trial["samples"]:
        position = sample["position_raw"]
        if position != previous:
            last_change = (
                sample["elapsed_s"]
                if high_speed
                else sample["monotonic_s"]
                - trial["command_time_monotonic_s"]
            )
            previous = position
    if last_change is None or last_change <= first:
        return None
    return abs(final - start) / (last_change - first)


def measured_deadband(
    main: dict, mid_deadband: dict, index: int,
) -> int | None:
    if index in FLEXION_INDICES:
        directional = []
        for direction in (-1, 1):
            candidates = [
                int(trial["delta_command_raw"])
                for trial in mid_deadband["trials"]
                if trial["joint_index"] == index
                and trial["direction"] == direction
                and abs(trial["feedback_delta_raw"]) >= 1
            ]
            if not candidates:
                return None
            directional.append(min(candidates))
        # Require a command step effective in both directions.
        return max(directional)

    candidates = []
    for trial in main["trials"]:
        if trial["phase"] != "deadband" or trial["joint_index"] != index:
            continue
        if abs(trial["achieved_delta_raw"]) >= 1:
            candidates.append(int(trial["repeat"]))
    return min(candidates, default=None)


def observed_endpoint(high: dict, index: int) -> tuple[int, int]:
    values = []
    for trial in high["trials"]:
        if trial["joint_index"] != index:
            continue
        values.extend(
            int(sample["position_raw"]) for sample in trial["samples"]
        )
    return (min(values), max(values)) if values else (0, 255)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-log", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--high-speed-log", type=Path, default=DEFAULT_HIGH)
    parser.add_argument("--aborted-max-log", type=Path, default=DEFAULT_ABORTED_MAX)
    parser.add_argument(
        "--reliability-80-log", type=Path, default=DEFAULT_RELIABILITY_80
    )
    parser.add_argument(
        "--mid-deadband-log", type=Path, default=DEFAULT_MID_DEADBAND
    )
    parser.add_argument("--yaml-output", type=Path, default=DEFAULT_YAML)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    main_log = json.loads(args.main_log.read_text(encoding="utf-8"))
    high_log = json.loads(args.high_speed_log.read_text(encoding="utf-8"))
    aborted_max = (
        json.loads(args.aborted_max_log.read_text(encoding="utf-8"))
        if args.aborted_max_log.is_file()
        else {}
    )
    reliability_80 = json.loads(
        args.reliability_80_log.read_text(encoding="utf-8")
    )
    mid_deadband = json.loads(
        args.mid_deadband_log.read_text(encoding="utf-8")
    )

    main_velocity: dict[tuple[int, int], list[float]] = defaultdict(list)
    for trial in main_log["trials"]:
        if trial["phase"] not in {
            "speed_response",
            "speed_response_return",
        }:
            continue
        value = effective_velocity(trial, high_speed=False)
        if value is not None:
            main_velocity[(trial["joint_index"], trial["speed_raw"])].append(value)

    high_velocity: dict[tuple[int, int], list[float]] = defaultdict(list)
    for trial in high_log["trials"]:
        value = effective_velocity(trial, high_speed=True)
        if value is not None:
            high_velocity[(trial["joint_index"], trial["speed_raw"])].append(value)

    channels = {}
    report_rows = []
    for index in ACTIVE_INDICES:
        logical_name, urdf_joint, lower, upper, inverted = JOINT_SPECS[index]
        raw_low, raw_high = TRAINING_RAW_LIMITS[index]
        angular_values = (raw_to_rad(index, raw_low), raw_to_rad(index, raw_high))
        safe_lower_rad, safe_upper_rad = sorted(angular_values)
        deadband = measured_deadband(main_log, mid_deadband, index)

        speed_table = {}
        for speed in (26, 51, 102):
            value = median(main_velocity[(index, speed)])
            speed_table[str(speed)] = (
                round(value, 3) if value is not None else None
            )
        for speed in (153, 204, 255):
            # Large-amplitude high-speed values are valid only for flexion.
            value = (
                median(high_velocity[(index, speed)])
                if index in FLEXION_INDICES
                else None
            )
            speed_table[str(speed)] = (
                round(value, 3) if value is not None else None
            )

        velocity_80_raw = speed_table["204"]
        angular_range = upper - lower
        velocity_80_rad = (
            velocity_80_raw * angular_range / 255.0
            if velocity_80_raw is not None
            else None
        )
        if index in FLEXION_INDICES and velocity_80_rad is not None:
            # 15% reserve below the measured unloaded median.
            recommended_velocity = velocity_80_rad * 0.85
        else:
            recommended_velocity = YAW_ROLL_VELOCITY_CAP_RAD_S[index]

        if index in FLEXION_INDICES:
            endpoint_low, endpoint_high = observed_endpoint(high_log, index)
            note = "large-amplitude 60/80/100% measured"
        else:
            endpoint_low, endpoint_high = raw_low, raw_high
            note = "large sweep forbidden: observed finger-to-finger collision"
        if index == 10:
            note += "; stable 3-4 raw command/feedback offset"
        if index in YAW_ROLL_INDICES:
            note += "; radian sign is inferred, not optically measured"

        channel = {
            "logical_index": index,
            "logical_name": logical_name,
            "urdf_joint": urdf_joint,
            "inverted_raw_mapping": inverted,
            "urdf_full_range_rad": [lower, upper],
            "training_command_limit_raw": [raw_low, raw_high],
            "training_limit_rad": [
                round(safe_lower_rad, 6),
                round(safe_upper_rad, 6),
            ],
            "observed_feedback_envelope_raw": [endpoint_low, endpoint_high],
            "minimum_observed_command_step_raw": deadband,
            "effective_speed_raw_s_by_command": speed_table,
            "measured_80_percent_velocity_rad_s": (
                round(velocity_80_rad, 4)
                if velocity_80_rad is not None
                else None
            ),
            "recommended_isaac_velocity_limit_rad_s": round(
                recommended_velocity, 4
            ),
            "notes": note,
        }
        channels[str(index)] = channel
        report_rows.append(channel)

    valid_low_speed_latency = [
        trial["first_motion_seconds"]
        for trial in main_log["trials"]
        if trial["phase"] in {"speed_response", "speed_response_return"}
        and trial["first_motion_seconds"] is not None
        and trial["joint_index"] != 10
    ]
    valid_high_speed_latency = [
        trial["first_motion_seconds"]
        for trial in high_log["trials"]
        if trial["family"] == "large_flexion"
        and trial["first_motion_seconds"] is not None
    ]
    maximum_temperature = max(
        value
        for group in high_log["temperature_after_raw_per_finger"]
        for value in group
    )
    overlay = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "device": {
            "model": "G20",
            "side": "left",
            "serial_number": main_log.get("serial_number"),
            "firmware": "1.0.10",
            "sdk": "3.1.1",
            "can_id": "0x28",
            "can_bitrate": 1_000_000,
        },
        "evidence": {
            "main_motion_log": str(args.main_log.resolve()),
            "high_speed_log": str(args.high_speed_log.resolve()),
            "aborted_max_reliability_log": (
                str(args.aborted_max_log.resolve())
                if args.aborted_max_log.is_file()
                else None
            ),
            "reliability_80_log": str(args.reliability_80_log.resolve()),
            "mid_deadband_log": str(args.mid_deadband_log.resolve()),
            "main_trial_count": main_log["trial_count"],
            "high_speed_trial_count": high_log["trial_count"],
            "reliability_80_trial_count": reliability_80["trial_count"],
            "reliability_80_successful_trial_count": reliability_80[
                "successful_trial_count"
            ],
            "mid_deadband_trial_count": mid_deadband["trial_count"],
            "manual_observation": (
                "An isolated large yaw sweep struck a neighbouring finger. "
                "All isolated yaw/roll endpoint sweeps are invalid for limits."
            ),
        },
        "global_policy": {
            "continuous_speed_cap_raw": 204,
            "continuous_speed_cap_percent": 80.0,
            "continuous_speed_validation": (
                "passed at enabled thermal steady state: 100/100 "
                "large-stroke flexion/tip moves, 5 round trips per axis"
            ),
            "speed_255_status": (
                "experimental_only_not_continuous: repeated hot full-stroke "
                "thumb-base test overshot command 38 to feedback 0 and "
                "temporarily rejected its return command"
            ),
            "high_speed_torque_limit_tested_raw": 153,
            "torque_interpretation": (
                "SDK value is a maximum limit, not measured output torque"
            ),
            "measured_command_to_motion_detection_median_s": {
                "low_mid": round(statistics.median(valid_low_speed_latency), 5),
                "high": round(statistics.median(valid_high_speed_latency), 5),
            },
            "maximum_session_temperature_c": max(
                maximum_temperature,
                max(
                    value
                    for group in reliability_80["temperature_after"]
                    for value in group
                ),
                max(
                    value
                    for group in mid_deadband["temperature_after"]
                    for value in group
                ),
            ),
            "temperature_stop_c": 60,
            "yaw_roll_policy": (
                "Use narrow independent limits below; wider vendor poses "
                "require an explicit coordinated collision-free trajectory."
            ),
            "pressure_during_unloaded_motion_max_raw": max(
                max(trial["maximum_normal_force_raw"] for trial in main_log["trials"]),
                max(trial["maximum_normal_force_raw"] for trial in high_log["trials"]),
            ),
        },
        "channels": channels,
        "unidentified": {
            "absolute_effort_nm": "requires force gauge and known lever arm",
            "viscous_damping_nms_per_rad": (
                "not identifiable: G20 has no measured motor current/torque"
            ),
            "coulomb_friction_nm": (
                "not identifiable: G20 has no measured motor current/torque"
            ),
            "joint_inertia_kg_m2": "requires physical pendulum/CAD/load test",
        },
    }

    args.yaml_output.parent.mkdir(parents=True, exist_ok=True)
    args.yaml_output.write_text(
        yaml.safe_dump(overlay, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    lines = [
        "# 左手 G20 实机运动标定报告",
        "",
        f"- 设备：`{overlay['device']['serial_number']}`，固件 1.0.10，SDK 3.1.1",
        f"- 基础/低中速试验：{main_log['trial_count']} 条",
        f"- 大幅高速试验：{high_log['trial_count']} 条",
        (
            f"- 80% 热稳态可靠性：{reliability_80['successful_trial_count']}/"
            f"{reliability_80['trial_count']} 条到位"
        ),
        f"- 中位双向死区：{mid_deadband['trial_count']} 条",
        "- CAN 全程 ERROR-ACTIVE，TX/RX error 为 0；运动后故障码全 0。",
        (
            f"- 最高记录温度：{overlay['global_policy']['maximum_session_temperature_c']}°C；"
            "会话安全停止阈值为 60°C。"
        ),
        "",
        "## 核心结论",
        "",
        (
            "- 连续运行速度硬上限为官方上限的 **80%（raw 204）**，"
            "已在 53–54°C 热稳态完成 100/100 次大行程验证。"
            "raw 255 只保留为短时实验档，禁止连续全行程。"
        ),
        (
            "- 100% 热态重复试验中，拇指根部命令 38 时反馈越过到 0，"
            "并暂时拒绝立即回程；该组已中止并安全恢复。"
        ),
        (
            "- 单指大范围侧摆会打到相邻手指。侧摆/横摆只能使用下表窄范围；"
            "更宽动作必须采用带自碰撞检查的多指协同轨迹。"
        ),
        (
            "- G20 不回传实际电流；SDK 扭矩值是最大扭矩限制。"
            "因此本次得到速度、延迟、死区和等效位置响应，不能得到真实 N·m 阻尼/摩擦。"
        ),
        "",
        "## Isaac/URDF 建议覆盖值",
        "",
        "|通道|关节|训练 raw 范围|80%实测 rad/s|建议 velocity rad/s|死区 raw|说明|",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in report_rows:
        raw_limits = "–".join(str(value) for value in row["training_command_limit_raw"])
        measured = row["measured_80_percent_velocity_rad_s"]
        deadband = row["minimum_observed_command_step_raw"]
        lines.append(
            f"|{row['logical_index']}|`{row['urdf_joint']}`|{raw_limits}|"
            f"{measured if measured is not None else '未作大幅测试'}|"
            f"{row['recommended_isaac_velocity_limit_rad_s']}|"
            f"{deadband if deadband is not None else '待中位补测'}|"
            f"{row['notes']}|"
        )
    lines.extend(
        [
            "",
            "## 不可由本次连接独立识别",
            "",
            "- 绝对输出力矩、粘性阻尼和库仑摩擦：需要推拉力计、已知力臂和同步载荷。",
            "- 连杆质量、质心和惯量：需要称重、重心和摆实验或原厂 CAD。",
            "- 原始触觉到 N 的映射：需要标准法向/切向载荷。",
            "",
            "机器可读覆盖层："
            f"`{args.yaml_output.resolve()}`",
            "",
            "原始失败状态被保留，不按“成功数据”改写。人工确认发生自碰撞的侧摆端点"
            "数据明确排除在训练限位之外。",
            "",
        ]
    )
    args.report_output.write_text("\n".join(lines), encoding="utf-8")
    print(args.yaml_output.resolve())
    print(args.report_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
