#!/usr/bin/env python3
"""独立重算 F3 G2 逐子步证据；只读 eval output，不启动仿真。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


COMBINATIONS = ((0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4))
DIGITS = ("thumb", "index", "middle", "ring", "little")
HARD_FAULTS = {
    "overload", "nonfinite", "penetration", "joint_limit",
    "self_collision", "instability", "joint_effort",
}
G = 9.81
REQUIRED_STEPS = 360
PHYSICS_DT = 1.0 / 120.0


class RejectWindow(ValueError):
    """一段候选窗口未满足独立 G2 条件。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_trace(eval_output: Path) -> Path:
    if eval_output.is_file():
        if eval_output.name != "pad_trace.jsonl":
            raise FileNotFoundError(f"期望 pad_trace.jsonl：{eval_output}")
        return eval_output.resolve()
    options = (
        eval_output / "pad_trace.jsonl",
        eval_output / "output" / "pad_trace.jsonl",
    )
    for path in options:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"未找到 {eval_output}/pad_trace.jsonl 或 {eval_output}/output/pad_trace.jsonl"
    )


def read_trace(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                outer = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"trace 第 {line_number} 行不是有效 JSON：{exc}") from exc
            if not isinstance(outer, dict):
                raise ValueError(f"trace 第 {line_number} 行外层必须是对象")
            record = outer.get("record", outer)
            if not isinstance(record, dict):
                raise ValueError(f"trace 第 {line_number} 行 record 必须是对象")
            rows.append(record)
            lines.append(raw if raw.endswith("\n") else raw + "\n")
    if not rows:
        raise ValueError("pad_trace.jsonl 为空")
    return rows, lines


def load_metadata(eval_output: Path, trace_path: Path) -> tuple[Path, dict[str, Any]]:
    output_dir = eval_output.parent if eval_output.is_file() else eval_output
    candidates = [
        output_dir / "run_metadata.json",
        output_dir / "output" / "run_metadata.json",
        trace_path.parent / "run_metadata.json",
    ]
    for path in dict.fromkeys(candidates):
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("run_metadata.json 顶层必须是对象")
            return path.resolve(), payload
    raise FileNotFoundError("评估目录缺少 run_metadata.json")


def get_protocol(metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float, float]:
    identity = metadata.get("task_identity")
    if not isinstance(identity, dict):
        raise ValueError("run_metadata.task_identity 缺失")
    protocol = identity.get("f3_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("task_identity.f3_protocol 缺失")
    if protocol.get("goal") != "g2":
        raise ValueError("本工具只接受 goal='g2' 的 G2 评估身份")
    expected = {
        "physics_hz": 120,
        "required_substeps": REQUIRED_STEPS,
        "strict_start_max_substep": 60,
        "pulse_start_substep": 180,
        "pulse_duration_substeps": 12,
        "recovery_deadline_substep": 252,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"f3_protocol.{key} 必须为 {value!r}，得到 {protocol.get(key)!r}")
    if not math.isclose(float(protocol.get("recovery_max_speed_m_s", math.nan)), 0.02, abs_tol=1e-12):
        raise ValueError("f3_protocol.recovery_max_speed_m_s 必须为 0.02")

    fruit = identity.get("fruit")
    if not isinstance(fruit, dict):
        raise ValueError("task_identity.fruit 缺失，不能读取实际质量/半径")
    mass = float(fruit.get("computed_mass_kg", math.nan))
    axes = fruit.get("axes_m")
    if not isinstance(axes, list) or not axes:
        raise ValueError("task_identity.fruit.axes_m 缺失")
    radius = float(axes[0])
    if not math.isfinite(mass) or mass <= 0 or not math.isfinite(radius) or radius <= 0:
        raise ValueError("identity 中实际质量/半径必须为正有限数")
    if abs(mass - 0.141731) > 1e-6 or abs(radius - 0.0335) > 1e-6:
        raise ValueError("identity 的实际质量或半径偏离冻结球果实值 0.141731 kg / 0.0335 m")
    mode = protocol.get("disturbance_mode")
    if mode not in ("quiet", "force", "torque"):
        raise ValueError(f"未知 disturbance_mode：{mode!r}")
    if mode != "quiet":
        axis = protocol.get("disturbance_axis")
        sign = protocol.get("disturbance_sign")
        if isinstance(axis, bool) or axis not in (0, 1, 2):
            raise ValueError("disturbance_axis 必须为 0、1 或 2")
        if sign not in (-1.0, 1.0):
            raise ValueError("disturbance_sign 必须为 -1 或 1")
        ratio_key = "force_ratio" if mode == "force" else "torque_ratio"
        ratio = float(protocol.get(ratio_key, math.nan))
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"{ratio_key} 必须为正有限数")
    return identity, protocol, mass, radius


def get_evaluation_matrix(identity: dict[str, Any]) -> list[Any] | None:
    """接受 F4 manifest 正式字段和兼容别名。"""
    matrix = identity.get("f4_matrix")
    cases = matrix.get("cases") if isinstance(matrix, dict) else None
    if cases is None:
        cases = identity.get("evaluation_matrix")
    if cases is None:
        return None
    if not isinstance(cases, list) or not cases:
        raise ValueError("task_identity.f4_matrix.cases/evaluation_matrix 必须为非空列表")
    return cases


def streak_matrix(row: dict[str, Any]) -> list[list[int]]:
    value = row.get("g2_streak")
    if not isinstance(value, list) or not value:
        raise ValueError("trace 缺少 [env,6] g2_streak")
    if any(not isinstance(env_row, list) or len(env_row) != 6 for env_row in value):
        raise ValueError("g2_streak 形状必须为 [env,6]")
    result = []
    for env_row in value:
        parsed = []
        for item in env_row:
            if isinstance(item, bool) or int(item) != item or int(item) < 0:
                raise ValueError("g2_streak 必须为非负整数")
            parsed.append(int(item))
        result.append(parsed)
    return result


def candidates_from_streak(rows: list[dict[str, Any]]) -> list[tuple[int, int, int]]:
    """用 streak>=360 找到其真实值为 1 的起点，再截取最初 360 行。"""
    matrices = [streak_matrix(row) for row in rows]
    env_count = len(matrices[0])
    if any(len(matrix) != env_count for matrix in matrices):
        raise ValueError("g2_streak 环境数在 trace 中发生变化")
    result: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for end_index, matrix in enumerate(matrices):
        for env_id in range(env_count):
            for combo_id, value in enumerate(matrix[env_id]):
                if value < REQUIRED_STEPS:
                    continue
                cursor = end_index
                expected = value
                while expected > 1 and cursor > 0:
                    current_step = rows[cursor].get("sim_step")
                    previous_step = rows[cursor - 1].get("sim_step")
                    if (matrices[cursor - 1][env_id][combo_id] != expected - 1
                            or int(current_step) != int(previous_step) + 1):
                        break
                    cursor -= 1
                    expected -= 1
                if expected != 1:
                    candidate = (-1, env_id, combo_id)
                    if candidate not in seen:
                        seen.add(candidate)
                        result.append(candidate)
                    continue
                start = cursor
                candidate = (start, env_id, combo_id)
                if start + REQUIRED_STEPS <= len(rows) and candidate not in seen:
                    seen.add(candidate)
                    result.append(candidate)
    result.sort()
    return result


def _finite_vector(value: Any, length: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise RejectWindow(f"{field} 必须为长度 {length} 的数组")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise RejectWindow(f"{field} 含非有限值")
    return result


def _env_value(row: dict[str, Any], key: str, env_id: int) -> Any:
    value = row.get(key)
    if not isinstance(value, list) or env_id >= len(value):
        raise RejectWindow(f"trace 缺少 {key}[{env_id}]")
    return value[env_id]


def matrix_scenario_for_env(
    row: dict[str, Any], env_id: int, protocol: dict[str, Any],
    evaluation_matrix: list[Any] | None,
) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    """Resolve per-environment matrix tags; never infer a tagged case as quiet."""
    has_case_id = "matrix_case_id" in row
    has_scenario = "matrix_scenario" in row
    if evaluation_matrix is None:
        if has_case_id or has_scenario:
            raise RejectWindow("trace 含 matrix 标签但 task_identity 缺 evaluation_matrix")
        scenario = {
            "mode": protocol["disturbance_mode"],
            "axis": protocol.get("disturbance_axis", 0),
            "sign": protocol.get("disturbance_sign", 1.0),
        }
        return None, scenario, None
    if not has_case_id or not has_scenario:
        raise RejectWindow("evaluation_matrix 存在时每个 trace row 都必须记录 matrix_case_id 和 matrix_scenario")
    case_id = _env_value(row, "matrix_case_id", env_id)
    tagged = _env_value(row, "matrix_scenario", env_id)
    if isinstance(case_id, bool) or not isinstance(case_id, (int, str)):
        raise RejectWindow("matrix_case_id 必须是整数或字符串")
    if not isinstance(tagged, dict):
        raise RejectWindow("matrix_scenario[env] 必须是对象且不能为 null")
    if "mode" not in tagged:
        raise RejectWindow("matrix_scenario 缺少 mode")
    scenario = {
        "mode": tagged["mode"],
        "axis": tagged.get("axis", None if tagged["mode"] == "quiet" else 0),
        "sign": tagged.get("sign", None if tagged["mode"] == "quiet" else 1.0),
    }
    if scenario["mode"] not in ("quiet", "force", "torque"):
        raise RejectWindow(f"matrix_scenario.mode 无效：{scenario['mode']!r}")
    if scenario["mode"] == "quiet":
        if (scenario["axis"] is None) != (scenario["sign"] is None):
            raise RejectWindow("quiet matrix_scenario 的 axis/sign 必须同时为 null 或同时给定")
        if scenario["axis"] is not None:
            if isinstance(scenario["axis"], bool) or scenario["axis"] not in (0, 1, 2):
                raise RejectWindow("quiet matrix_scenario.axis 必须为 null 或 0、1、2")
            if isinstance(scenario["sign"], bool) or scenario["sign"] not in (-1, -1.0, 1, 1.0):
                raise RejectWindow("quiet matrix_scenario.sign 必须为 null 或 -1、1")
    else:
        if isinstance(scenario["axis"], bool) or scenario["axis"] not in (0, 1, 2):
            raise RejectWindow("matrix_scenario.axis 必须为 0、1 或 2")
        if isinstance(scenario["sign"], bool) or scenario["sign"] not in (-1, -1.0, 1, 1.0):
            raise RejectWindow("matrix_scenario.sign 必须为 -1 或 1")

    matching_entry = None
    for index, entry in enumerate(evaluation_matrix):
        identifiers = [index]
        if isinstance(entry, dict):
            identifiers.extend(
                entry[key]
                for key in ("matrix_case_id", "case_id", "id", "name")
                if key in entry
            )
        elif isinstance(entry, (str, int)) and not isinstance(entry, bool):
            identifiers.append(entry)
        if any(case_id == identifier for identifier in identifiers):
            matching_entry = entry
            break
    if matching_entry is None:
        raise RejectWindow(f"matrix_case_id={case_id!r} 不在 task_identity.evaluation_matrix 中")

    # If metadata carries scenario fields, bind the trace tag to them as well.
    descriptor = (
        matching_entry.get("matrix_scenario", matching_entry.get("scenario", matching_entry))
        if isinstance(matching_entry, dict) else None
    )
    if isinstance(descriptor, dict) and "mode" in descriptor:
        expected = {
            "mode": descriptor["mode"],
            "axis": descriptor.get("axis", None if descriptor["mode"] == "quiet" else 0),
            "sign": descriptor.get("sign", None if descriptor["mode"] == "quiet" else 1.0),
        }
        if scenario != expected:
            raise RejectWindow(f"matrix_scenario 与 evaluation_matrix case 不一致：{scenario!r} != {expected!r}")
    return case_id, scenario, matching_entry if isinstance(matching_entry, dict) else None


def _quat_normalize(q: list[float], field: str) -> list[float]:
    norm = math.sqrt(sum(x * x for x in q))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RejectWindow(f"{field} 四元数范数无效")
    return [x / norm for x in q]


def quat_multiply_xyzw(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def quat_rotate_xyzw(q: list[float], vector: list[float]) -> list[float]:
    q = _quat_normalize(q, "hand_pose_w")
    qv = q[:3]
    # v' = v + 2 * (w * cross(qv,v) + cross(qv,cross(qv,v)))
    cross1 = [qv[1] * vector[2] - qv[2] * vector[1],
              qv[2] * vector[0] - qv[0] * vector[2],
              qv[0] * vector[1] - qv[1] * vector[0]]
    cross2 = [qv[1] * cross1[2] - qv[2] * cross1[1],
              qv[2] * cross1[0] - qv[0] * cross1[2],
              qv[0] * cross1[1] - qv[1] * cross1[0]]
    return [vector[i] + 2.0 * (q[3] * cross1[i] + cross2[i]) for i in range(3)]


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(item * item for item in vector))


def distance(a: list[float], b: list[float]) -> float:
    return norm([x - y for x, y in zip(a, b)])


def check_faults_and_invalid(row: dict[str, Any], env_id: int) -> None:
    faults = row.get("hard_faults")
    if not isinstance(faults, dict) or not HARD_FAULTS.issubset(faults):
        raise RejectWindow("七类 hard_faults 字段不完整")
    for name in HARD_FAULTS:
        values = faults[name]
        if not isinstance(values, list) or env_id >= len(values) or bool(values[env_id]):
            raise RejectWindow(f"hard_faults.{name} 非零或缺失")
    invalid = row.get("invalid_contact_channels")
    if not isinstance(invalid, list):
        raise RejectWindow("invalid_contact_channels 缺失")
    for channel in invalid:
        if not isinstance(channel, dict) or "env_ids" not in channel:
            raise RejectWindow("invalid_contact_channels 条目无法按环境审计")
        if env_id in channel["env_ids"]:
            raise RejectWindow("候选窗出现 invalid contact channel")


def contacts_centroids(
    row: dict[str, Any], env_id: int, combo: tuple[int, int, int], pad_force: list[float], fruit_center: list[float]
) -> tuple[list[list[float]], float, float]:
    contacts = row.get("contacts")
    if not isinstance(contacts, list):
        raise RejectWindow("contacts 缺失")
    accum = {digit: [0.0, 0.0, 0.0, 0.0] for digit in combo}
    for contact in contacts:
        if not isinstance(contact, dict):
            raise RejectWindow("contacts 条目不是对象")
        body = contact.get("body")
        if not isinstance(body, str):
            raise RejectWindow("contact.body 缺失")
        digit_name = body.split("_", 1)[0]
        if digit_name not in DIGITS:
            continue
        digit_id = DIGITS.index(digit_name)
        if digit_id not in combo:
            continue
        env_ids = contact.get("env_ids")
        labels = contact.get("labels")
        forces = contact.get("forces_n")
        points = contact.get("points_w")
        if not all(isinstance(x, list) for x in (env_ids, labels, forces, points)):
            raise RejectWindow("contact 缺 env_ids/labels/forces_n/points_w")
        if not (len(env_ids) == len(labels) == len(forces) == len(points)):
            raise RejectWindow("contact 点字段长度不一致")
        for index, contact_env in enumerate(env_ids):
            if int(contact_env) != env_id or labels[index] != 1:
                continue
            force = float(forces[index])
            point = _finite_vector(points[index], 3, "contacts.points_w")
            if not math.isfinite(force) or force <= 0:
                raise RejectWindow("PAD label=1 接触力必须为正有限数")
            entry = accum[digit_id]
            entry[0] += force
            for axis in range(3):
                entry[axis + 1] += force * point[axis]

    centers: list[list[float]] = []
    for digit_id in combo:
        weight, *weighted_position = accum[digit_id]
        if weight <= 0:
            raise RejectWindow(f"{DIGITS[digit_id]} 缺少 label=1 PAD 接触点")
        tolerance = max(1e-5, 1e-4 * pad_force[digit_id])
        if abs(weight - pad_force[digit_id]) > tolerance:
            raise RejectWindow(f"{DIGITS[digit_id]} contacts label=1 力与 truth PAD 力不一致")
        centers.append([value / weight for value in weighted_position])

    radial: list[list[float]] = []
    for center in centers:
        direction = [center[i] - fruit_center[i] for i in range(3)]
        length = norm(direction)
        if not math.isfinite(length) or length <= 1e-12:
            raise RejectWindow("接触中心与果心重合，无法计算布局")
        radial.append([value / length for value in direction])
    nonthumb_sum = [radial[1][i] + radial[2][i] for i in range(3)]
    sum_norm = norm(nonthumb_sum)
    if sum_norm <= 1e-12:
        raise RejectWindow("两个非拇指径向和退化")
    opposite_dot = sum(radial[0][i] * nonthumb_sum[i] / sum_norm for i in range(3))
    separation = distance(centers[1], centers[2])
    return centers, separation, opposite_dot


def verify_candidate(
    rows: list[dict[str, Any]],
    start: int,
    env_id: int,
    combo_id: int,
    protocol: dict[str, Any],
    mass: float,
    radius: float,
    evaluation_matrix: list[Any] | None = None,
) -> dict[str, Any]:
    window = rows[start : start + REQUIRED_STEPS]
    if len(window) != REQUIRED_STEPS:
        raise RejectWindow("候选窗不足 360 行")
    combo = COMBINATIONS[combo_id]
    speeds: list[float] = []
    drifts: list[float] = []
    angles: list[float] = []
    angular_speeds: list[float] = []
    clearances: list[float] = []
    minimum_forces = [math.inf] * 5
    minimum_shares = [math.inf] * 5
    minimum_separation = math.inf
    maximum_opposition = -math.inf
    max_force_error = 0.0
    max_torque_error = 0.0
    relative_anchor: list[float] | None = None
    position_anchor: list[float] | None = None

    start_substep = None
    previous_sim_step = None
    mode: str | None = None
    axis = 0
    sign = 1.0
    matrix_case_id = None
    selected_scenario: dict[str, Any] | None = None
    force_ratio: float | None = None
    torque_ratio: float | None = None
    force_magnitude = 0.0
    torque_magnitude = 0.0
    max_rotation = math.radians(float(protocol.get("max_rotation_deg", 10.0)))
    max_angular_speed = float(protocol.get("max_relative_angular_speed_rad_s", 0.5))
    min_separation = float(protocol.get("min_nonthumb_separation_m", 0.006))
    max_opposition = float(protocol.get("max_opposition_dot", -0.3))

    for offset, row in enumerate(window):
        case_id, scenario, case_entry = matrix_scenario_for_env(
            row, env_id, protocol, evaluation_matrix
        )
        if mode is None:
            mode = scenario["mode"]
            if mode != "quiet":
                axis = int(scenario["axis"])
                sign = float(scenario["sign"])
            matrix_case_id = case_id
            selected_scenario = dict(scenario)
            force_ratio_source = (
                case_entry.get("force_ratio")
                if case_entry is not None and "force_ratio" in case_entry
                else protocol.get("force_ratio")
            )
            torque_ratio_source = (
                case_entry.get("torque_ratio")
                if case_entry is not None and "torque_ratio" in case_entry
                else protocol.get("torque_ratio")
            )
            try:
                force_ratio = None if force_ratio_source is None else float(force_ratio_source)
                torque_ratio = None if torque_ratio_source is None else float(torque_ratio_source)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RejectWindow("case force_ratio/torque_ratio 必须为数值或 null") from exc
            if force_ratio is not None and (not math.isfinite(force_ratio) or force_ratio < 0):
                raise RejectWindow("case force_ratio 必须为非负有限数")
            if torque_ratio is not None and (not math.isfinite(torque_ratio) or torque_ratio < 0):
                raise RejectWindow("case torque_ratio 必须为非负有限数")
            if mode == "force" and (force_ratio is None or force_ratio <= 0):
                raise RejectWindow("force scenario 缺少正 force_ratio")
            if mode == "torque" and (torque_ratio is None or torque_ratio <= 0):
                raise RejectWindow("torque scenario 缺少正 torque_ratio")
            force_magnitude = (force_ratio or 0.0) * mass * G
            torque_magnitude = (torque_ratio or 0.0) * mass * G * radius
        elif scenario != selected_scenario or case_id != matrix_case_id:
            raise RejectWindow("候选窗中 matrix_case_id/matrix_scenario 发生变化")
        expected_streak = offset + 1
        if streak_matrix(row)[env_id][combo_id] != expected_streak:
            raise RejectWindow("候选不是从 g2_streak=1 起连续增长的前 360 子步")
        sim_step = row.get("sim_step")
        dt = float(row.get("dt_s", math.nan))
        if isinstance(sim_step, bool) or not isinstance(sim_step, int):
            raise RejectWindow("sim_step 缺失或非整数")
        if previous_sim_step is not None and sim_step != previous_sim_step + 1:
            raise RejectWindow("sim_step 不连续")
        previous_sim_step = sim_step
        if not math.isfinite(dt) or abs(dt - PHYSICS_DT) > 1e-12:
            raise RejectWindow("dt_s 不等于 1/120 秒")

        substep = _env_value(row, "post_release_substep", env_id)
        if isinstance(substep, bool) or int(substep) != substep:
            raise RejectWindow("post_release_substep 非整数")
        substep = int(substep)
        if start_substep is None:
            start_substep = substep
            if start_substep < 0 or start_substep > int(protocol["strict_start_max_substep"]):
                raise RejectWindow("三指窗口起点晚于释放后 60 子步")
        elif substep != start_substep + offset:
            raise RejectWindow("post_release_substep 在候选窗内不连续")
        if _env_value(row, "released", env_id) is not True:
            raise RejectWindow("released 非 true")
        for key in ("support_force_w", "support_torque_w"):
            wrench = _finite_vector(_env_value(row, key, env_id), 3, key)
            if any(component != 0.0 for component in wrench):
                raise RejectWindow(f"{key} 非逐分量零")

        check_faults_and_invalid(row, env_id)

        truth = row.get("truth")
        if not isinstance(truth, dict):
            raise RejectWindow("truth 缺失")
        pad = _finite_vector(_env_value(truth, "pad_force_n", env_id), 5, "truth.pad_force_n")
        nonpad = _finite_vector(_env_value(truth, "nonpad_force_n", env_id), 5, "truth.nonpad_force_n")
        unknown = _env_value(truth, "unknown", env_id)
        if not isinstance(unknown, list) or len(unknown) != 5 or any(bool(value) for value in unknown):
            raise RejectWindow("truth.unknown 非零或缺失")
        if any(value < 0 for value in pad + nonpad):
            raise RejectWindow("truth PAD/NON_PAD 力含负值")
        palm = float(_env_value(row, "palm_force_n", env_id))
        if not math.isfinite(palm) or palm < 0 or sum(nonpad) + palm != 0.0:
            raise RejectWindow("五指 NON_PAD 力之和加 palm_force_n 不为零")
        total_pad = sum(pad)
        if not math.isfinite(total_pad) or total_pad <= 0:
            raise RejectWindow("五指 PAD 总力无效")
        for digit_id in combo:
            share = pad[digit_id] / total_pad
            if pad[digit_id] < 0.15 or share < 0.05:
                raise RejectWindow(f"{DIGITS[digit_id]} PAD 力或总 PAD 份额不足")
        for digit_id in range(5):
            minimum_forces[digit_id] = min(minimum_forces[digit_id], pad[digit_id])
            minimum_shares[digit_id] = min(minimum_shares[digit_id], pad[digit_id] / total_pad)

        pose = _finite_vector(_env_value(row, "fruit_pose_w", env_id), 7, "fruit_pose_w")
        fruit_center = pose[:3]
        clearance = pose[2] - radius
        if not math.isfinite(clearance) or clearance <= 0.005:
            raise RejectWindow("fruit_bottom <= 0.005 m")
        clearances.append(clearance)

        position = _finite_vector(_env_value(row, "fruit_in_hand_m", env_id), 3, "fruit_in_hand_m")
        if position_anchor is None:
            position_anchor = position
        drift = distance(position, position_anchor)
        drifts.append(drift)
        if drift > 0.01:
            raise RejectWindow("相对手位置漂移超过 0.01 m")
        speed = float(_env_value(row, "relative_speed_m_s", env_id))
        if not math.isfinite(speed) or speed < 0 or speed > 0.05:
            raise RejectWindow("relative_speed_m_s 超过 0.05 m/s 或无效")
        speeds.append(speed)

        hand_pose = _finite_vector(_env_value(row, "hand_pose_w", env_id), 7, "hand_pose_w")
        hand_q = _quat_normalize(hand_pose[3:7], "hand_pose_w")
        fruit_q = _quat_normalize(pose[3:7], "fruit_pose_w")
        relative_q = _quat_normalize(
            quat_multiply_xyzw([-hand_q[0], -hand_q[1], -hand_q[2], hand_q[3]], fruit_q),
            "relative pose",
        )
        recorded_relative = _quat_normalize(
            _finite_vector(_env_value(row, "relative_quat_xyzw", env_id), 4, "relative_quat_xyzw"),
            "relative_quat_xyzw",
        )
        consistency = abs(sum(a * b for a, b in zip(relative_q, recorded_relative)))
        if consistency < 1.0 - 1e-5:
            raise RejectWindow("relative_quat_xyzw 与 hand/fruit 原始 pose 不一致")
        if relative_anchor is None:
            relative_anchor = relative_q
        dot = max(0.0, min(1.0, abs(sum(a * b for a, b in zip(relative_anchor, relative_q)))))
        angle = 2.0 * math.acos(dot)
        angles.append(angle)
        if angle > max_rotation + 1e-7:
            raise RejectWindow("相对姿态偏转超过协议角度上限")

        fruit_angular = _finite_vector(_env_value(row, "fruit_angular_velocity_w", env_id), 3,
                                       "fruit_angular_velocity_w")
        hand_angular = _finite_vector(_env_value(row, "hand_angular_velocity_w", env_id), 3,
                                      "hand_angular_velocity_w")
        relative_angular = norm([a - b for a, b in zip(fruit_angular, hand_angular)])
        recorded_angular = float(_env_value(row, "relative_angular_speed_rad_s", env_id))
        if not math.isfinite(recorded_angular) or abs(recorded_angular - relative_angular) > 1e-5:
            raise RejectWindow("relative_angular_speed_rad_s 与原始角速度不一致")
        if relative_angular > max_angular_speed:
            raise RejectWindow("相对角速度超过协议上限")
        angular_speeds.append(relative_angular)

        _, separation, opposition = contacts_centroids(row, env_id, combo, pad, fruit_center)
        minimum_separation = min(minimum_separation, separation)
        maximum_opposition = max(maximum_opposition, opposition)
        if separation < min_separation:
            raise RejectWindow("两非拇指 label=1 力加权中心间距不足 6 mm")
        if opposition > max_opposition:
            raise RejectWindow("label=1 接触中心的重算 opposition_dot 超限")

        test_force = _finite_vector(_env_value(row, "test_disturbance_force_w", env_id), 3,
                                    "test_disturbance_force_w")
        test_torque = _finite_vector(_env_value(row, "test_disturbance_torque_w", env_id), 3,
                                     "test_disturbance_torque_w")
        active = 180 <= substep < 192 and mode != "quiet"
        hand_axis = [0.0, 0.0, 0.0]
        hand_axis[axis] = sign
        if active and mode == "force":
            expected_force = [value * force_magnitude for value in quat_rotate_xyzw(hand_q, hand_axis)]
            expected_torque = [0.0, 0.0, 0.0]
        elif active and mode == "torque":
            expected_force = [0.0, 0.0, 0.0]
            expected_torque = [value * torque_magnitude for value in quat_rotate_xyzw(hand_q, hand_axis)]
        else:
            expected_force = [0.0, 0.0, 0.0]
            expected_torque = [0.0, 0.0, 0.0]
        force_error = max(abs(a - b) for a, b in zip(test_force, expected_force))
        torque_error = max(abs(a - b) for a, b in zip(test_torque, expected_torque))
        max_force_error = max(max_force_error, force_error)
        max_torque_error = max(max_torque_error, torque_error)
        if force_error > 2e-6 or torque_error > 2e-8:
            raise RejectWindow(f"test wrench 与 metadata/world 变换重算不符，substep={substep}")
        if mode != "quiet" and substep >= int(protocol["recovery_deadline_substep"]):
            if speed > float(protocol["recovery_max_speed_m_s"]):
                raise RejectWindow("252 子步后未恢复到 0.02 m/s 并保持到窗结束")

    if mode is None or selected_scenario is None:
        raise RejectWindow("候选窗未解析到扰动场景")
    expected_pulse_substeps = list(range(180, 192)) if mode != "quiet" else []
    if mode != "quiet":
        actual_pulse_substeps = [
            int(_env_value(row, "post_release_substep", env_id))
            for row in window
            if 180 <= int(_env_value(row, "post_release_substep", env_id)) < 192
        ]
        if actual_pulse_substeps != expected_pulse_substeps or len(actual_pulse_substeps) != 12:
            raise RejectWindow("候选窗必须包含 180..191 恰好 12 个脉冲子步")
    return {
        "env_id": env_id,
        "combo_index": combo_id,
        "digits": [DIGITS[index] for index in combo],
        "sim_step_start": window[0]["sim_step"],
        "sim_step_end": window[-1]["sim_step"],
        "post_release_start_substep": start_substep,
        "substeps": REQUIRED_STEPS,
        "dt_sum_s": sum(float(row["dt_s"]) for row in window),
        "min_pad_force_n_by_digit": dict(zip(DIGITS, minimum_forces)),
        "min_pad_share_by_digit": dict(zip(DIGITS, minimum_shares)),
        "max_relative_speed_m_s": max(speeds),
        "max_drift_from_window_anchor_m": max(drifts),
        "min_fruit_bottom_clearance_m": min(clearances),
        "max_relative_rotation_rad": max(angles),
        "max_relative_angular_speed_rad_s": max(angular_speeds),
        "min_nonthumb_centroid_separation_m": minimum_separation,
        "max_opposition_dot": maximum_opposition,
        "disturbance_mode": mode,
        "matrix_case_id": matrix_case_id,
        "matrix_scenario": selected_scenario,
        "force_ratio": force_ratio,
        "torque_ratio": torque_ratio,
        "pulse_substeps": expected_pulse_substeps,
        "identity_mass_kg": mass,
        "identity_radius_m": radius,
        "max_force_recompute_error_n": max_force_error,
        "max_torque_recompute_error_nm": max_torque_error,
        "all_support_wrenches_zero": True,
        "all_faults_invalid_unknown_and_nonpad_loads_zero": True,
        "original_full_mesh_reclassification": "not_performed; contact-label numerical geometry only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_output", type=Path, help="评估输出目录或 pad_trace.jsonl 文件")
    parser.add_argument("--output", type=Path, required=True, help="报告 JSON 输出路径")
    args = parser.parse_args()
    trace_path = resolve_trace(args.eval_output)
    metadata_path, metadata = load_metadata(args.eval_output, trace_path)
    identity, protocol, mass, radius = get_protocol(metadata)
    evaluation_matrix = get_evaluation_matrix(identity)
    rows, raw_lines = read_trace(trace_path)
    candidates = candidates_from_streak(rows)

    report_path = args.output.resolve()
    window_path = report_path.with_name("window.jsonl")
    if report_path.exists() or window_path.exists():
        raise FileExistsError(f"输出已存在：{report_path} 或 {window_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    selected: tuple[int, int, int] | None = None
    passed: dict[str, Any] | None = None
    for start, env_id, combo_id in candidates:
        if start < 0:
            results.append({
                "status": "REJECTED",
                "env_id": env_id,
                "combo_index": combo_id,
                "reason": "trace 中出现 g2_streak>=360，但找不到对应 g2_streak=1 起点，无法重算窗口 anchor",
            })
            continue
        try:
            result = verify_candidate(
                rows, start, env_id, combo_id, protocol, mass, radius, evaluation_matrix
            )
            result["status"] = "PASS_NUMERIC_G2_WINDOW"
            if passed is None:
                selected = (start, env_id, combo_id)
                passed = result
        except (RejectWindow, KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            result = {
                "status": "REJECTED",
                "env_id": env_id,
                "combo_index": combo_id,
                "streak_start_trace_index": start,
                "reason": str(exc),
            }
        results.append(result)

    if selected is not None:
        start, _, _ = selected
        with window_path.open("x", encoding="utf-8") as stream:
            stream.writelines(raw_lines[start : start + REQUIRED_STEPS])
        status = "PASS_G2_NUMERIC_WINDOW"
    else:
        window_path.write_text("", encoding="utf-8")
        status = "NO_G2_CANDIDATE" if not candidates else "NO_VALID_G2_WINDOW"

    report = {
        "schema": "verify_g2_numeric_window.v1",
        "status": status,
        "eval_output": str(args.eval_output.resolve()),
        "trace": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "run_metadata": str(metadata_path),
        "task_version": identity.get("task_version"),
        "protocol": protocol,
        "identity_mass_kg": mass,
        "identity_radius_m": radius,
        "candidate_count": len(candidates),
        "candidate_results": results,
        "selected_window": passed,
        "window_file": str(window_path),
        "window_sha256": sha256_file(window_path),
        "window_lines": REQUIRED_STEPS if selected is not None else 0,
        "verifier": str(Path(__file__).resolve()),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "scope_note": "Independent numerical recomputation from truth, contacts, and raw state; does not trust g2qualified or cached g2_angle. Original full-mesh triangle reclassification was not performed.",
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "candidates": len(candidates), "report": str(report_path),
                      "window": str(window_path)}))
    return 0 if status == "PASS_G2_NUMERIC_WINDOW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
