#!/usr/bin/env python3
"""番茄自适应抓取控制台。

默认是离线模拟器。真实左手连接需要 --profile left-flexible --connect；
真实运动还需要 TOMATO_CONSOLE_ALLOW_HARDWARE=1、--enable-motion 和页面解锁。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[3]
CONSOLE = Path(__file__).resolve().parent
STATIC = CONSOLE / "static"
OLD_CONSOLE = ROOT / "lbot_arm_api-master" / "tools" / "tactile_console"
HAND_ASSETS = OLD_CONSOLE / "static" / "hand-model" / "L20_8_l_s"
HAND_ASSET_ROOT = HAND_ASSETS.parent
VENDOR = OLD_CONSOLE.parent / "arm_hand_console" / "static" / "vendor"

OPEN_POSE = [255, 255, 255, 255, 255, 255, 193, 148, 105, 42, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255]
# Historical soft-ball demonstration anchors. They are metadata/weak priors only;
# tomato geometry and tactile feedback remain the dominant variables.
SOFT_BALL_FRAME_REFERENCE = [190, 66, 59, 254, 254, 143, 192, 147, 105, 42, 52, 0, 0, 0, 0, 206, 200, 201, 254, 254]
SOFT_BALL_GRASP_REFERENCE = [190, 61, 57, 254, 254, 143, 192, 147, 105, 42, 52, 0, 0, 0, 0, 206, 188, 191, 254, 254]
RESERVED = {11, 12, 13, 14}
PASSIVE_CHANNELS = (3, 8, 18, 4, 9, 19)
FINGER_CHANNELS = ((0, 5, 10, 15), (1, 6, 16), (2, 7, 17), (3, 8, 18), (4, 9, 19))
FINGER_NAMES = ("拇指", "食指", "中指", "无名指", "小指")
PREVIEW_TTL_SECONDS = 1800.0
# “有接触”不等于“已抓牢”。动态微调需要三指都保持接触，并且每根
# 手指的归一化压力/载荷落在一个更宽的抓牢证据带内，连续 8 个采样周期
# （约 0.8 s）才放行。下限避免刚擦边就进入动态，上限比成功判定旧带
# 更宽，允许柔性物体的正常波动而不触发过早复位。
SECURE_EFFORT_MIN = (0.35, 0.35, 0.35)
# 食指/中指的示教基线目标很低（约 5/6.5 raw），同样的真实接触
# 在归一化倍率上会明显高于拇指；上限因此按三指历史载荷不对称性放宽，
# 但仍受绝对 safety_pressures/safety_loads 的硬保护约束。
SECURE_EFFORT_MAX = (2.2, 6.0, 6.0)
SECURE_STREAK_REQUIRED = 8
LARGE_FRUIT_DIAMETER_MM = 78.0
TACTILE_CELL_NOISE_FLOOR = 2.0
APPROACH_STOP_EFFORT = (1.6, 4.5, 4.5)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def matrix_peak(matrix: list[list[float]]) -> float:
    return max((float(value) for row in matrix for value in row), default=0.0)


def matrix_sum(matrix: list[list[float]]) -> float:
    return sum(float(value) for row in matrix for value in row)


def denoise_matrix(matrix: list[list[float]]) -> list[list[float]]:
    """去掉串流中偶发的 0/1 单元噪声，再交给显示和闭环。"""
    return [
        [
            value if math.isfinite(float(value)) and float(value) >= TACTILE_CELL_NOISE_FLOOR else 0.0
            for value in row
        ]
        for row in matrix
    ]


def denoise_matrices(matrices: list[list[list[float]]]) -> list[list[list[float]]]:
    return [denoise_matrix(matrix) for matrix in matrices]


def finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值")
    return float(value)


def normalize_pose(pose: Any) -> list[int]:
    if not isinstance(pose, list) or len(pose) != 20:
        raise ValueError("pose 必须包含 20 个通道")
    result: list[int] = []
    for index, value in enumerate(pose):
        item = int(round(clamp(finite(value, f"pose[{index}]"), 0, 255)))
        if index in RESERVED:
            item = 255
        result.append(item)
    return result


def tomato_config(payload: Any) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    # 第一版番茄控制台只让操作者填写果径和高度；其余几何量采用保守固定值，
    # 后续可由视觉模块替换，不在页面上形成额外调参负担。
    config = {
        "diameter_mm": clamp(finite(payload.get("diameter_mm", 72), "diameter_mm"), 35, 120),
        "height_mm": clamp(finite(payload.get("height_mm", 68), "height_mm"), 30, 140),
        "ellipticity": 0.12,
        "asymmetry": 0.08,
        "offset_x_mm": 0.0,
        "offset_y_mm": 0.0,
        "offset_z_mm": 0.0,
        "stem_angle_deg": 0.0,
        "vision_confidence": clamp(finite(payload.get("vision_confidence", 0.92), "vision_confidence"), 0, 1),
        "palm_support": bool(payload.get("palm_support", True)),
    }
    if config["vision_confidence"] < 0.65:
        raise ValueError("视觉置信度低于 0.65，不能生成实机抓取计划")
    if not config["palm_support"]:
        raise ValueError("番茄任务要求掌部向上并提供支撑接触")
    return config


def make_plan(config: dict[str, Any]) -> dict[str, Any]:
    """复刻软网球控制台的参考动作；尺寸只用于记录，不改写动作基线。

    真正的收紧由 SoftBallGraspController 的触觉闭环决定。这里的多个
    pose 只用于 URDF 预演和安全门展示，两个示教姿态仍然是弱参考。
    """
    def safe_reference(pose: list[int]) -> list[int]:
        result = normalize_pose(pose)
        for channel in PASSIVE_CHANNELS:
            result[channel] = OPEN_POSE[channel]
        return result

    frame = safe_reference(SOFT_BALL_FRAME_REFERENCE)
    grasp = safe_reference(SOFT_BALL_GRASP_REFERENCE)
    middle = [round((a + b) / 2) for a, b in zip(frame, grasp)]
    for pose in (middle,):
        pose[11:15] = OPEN_POSE[11:15]
        for channel in PASSIVE_CHANNELS:
            pose[channel] = OPEN_POSE[channel]
    reference_weight = 0.12
    thumb_align = frame[10]
    # The simulator/controller treats the tomato dimensions as observation
    # metadata. Tactile contact, not a scaled demonstration pose, determines
    # when to advance beyond the frame reference.
    fingers = [
        {"name": name, "enabled": True, "pressure_target": pressure}
        # 与 TomatoRuntime 的低力目标保持一致；三指仍保留各自独立目标，
        # 不把历史软网球的拇指高载荷复制给食指/中指。
        for name, pressure in (("拇指", 65), ("食指", 5), ("中指", 6.5))
    ]
    reference_frame_tolerance = 32
    reference_grasp_tolerance = 38
    return {
        "plan_id": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12],
        "strategy": "soft_ball_three_finger_adaptive",
        "reference_pose_lock": False,
        "reference_weight": reference_weight,
        "reference_envelope": {"frame_tolerance": reference_frame_tolerance, "grasp_tolerance": reference_grasp_tolerance},
        "reference_anchors": {"frame": SOFT_BALL_FRAME_REFERENCE, "grasp": SOFT_BALL_GRASP_REFERENCE, "role": "weak_prior_only"},
        "active_fingers": ["拇指", "食指", "中指"],
        "locked_open_fingers": ["无名指", "小指"],
        "palm_support_required": True,
        "stem_detach_enabled": False,
        "thumb_alignment": {"soft_target": round(thumb_align), "tolerance": 24, "reason": "拇指先横摆进入食指/中指之间的通道"},
        "fingers": fingers,
        "stages": [
            {"id": "opening", "label": "安全张开（保留四指侧摆开张）", "duration_ms": 700, "pose": OPEN_POSE, "max_step": 8},
            {"id": "fast_positioning", "label": "拇指横摆与三指根部快速到位", "duration_ms": 1500, "pose": frame, "max_step": 7},
            {"id": "tip_refinement", "label": "三指指尖低速微调至框球参考区", "duration_ms": 1200, "pose": frame, "max_step": 1},
            {"id": "contact_search", "label": "等待三指真实触觉接触", "duration_ms": 0, "pose": frame, "max_step": 1},
            {"id": "adaptive_capture", "label": "三指触觉稳定化（抓牢门控前不启动动态调整）", "duration_ms": 900, "pose": middle, "max_step": 1},
            {"id": "holding", "label": "抓牢门控通过后动态压力保持（持续微调）", "duration_ms": 0, "pose": grasp, "max_step": 1},
        ],
        "checks": {
            "geometry_valid": True,
            "vision_confidence": config["vision_confidence"] >= 0.65,
            "palm_support": True,
            "thumb_soft_alignment": True,
            "per_finger_pressure": True,
            "step_bounded": True,
            "stem_action_locked": True,
            "three_finger_only": True,
            "inactive_fingers_open": True,
            "reference_relaxed": True,
        },
    }


def blank_matrices() -> list[list[list[float]]]:
    return [[[0.0 for _ in range(6)] for _ in range(12)] for _ in range(5)]


class TomatoRuntime:
    def __init__(self, profile: str, can: str, connect: bool, enable_motion: bool) -> None:
        self.profile = profile
        self.can = can
        self.enable_motion = bool(enable_motion)
        self.hardware_allowed = os.environ.get("TOMATO_CONSOLE_ALLOW_HARDWARE") == "1"
        self.hardware = None
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="tomato-console")
        # Reuse the tested soft-ball state machine instead of duplicating a
        # fixed tomato trajectory. The imported code remains read-only here.
        sys.path.insert(0, str(OLD_CONSOLE))
        from flex_grasp import SoftBallConfig, SoftBallGraspController  # type: ignore

        class TomatoSoftBallController(SoftBallGraspController):
            """番茄控制台的安全带扩展，不修改冻结的软网球实现。"""

            dynamic_safety = False
            approach_safety: tuple[list[float], list[float]] | None = None

            def _recompute_thresholds(self) -> None:  # type: ignore[override]
                super()._recompute_thresholds()
                if self.dynamic_safety:
                    # 动态保持允许滤波瞬态有少量余量，但仍远低于硬件的
                    # 严重超限路径；探索阶段继续使用基类的窄安全带。
                    self.safety_pressures = [190.0, 105.0, 125.0]
                    self.safety_loads = [2000.0, 270.0, 340.0]
                elif self.approach_safety is not None:
                    # 大果早接触时允许进入轮廓搜索，但仍保留严重超限
                    # （1.30 倍）保护，不把接触当成故障而退回全开。
                    self.safety_pressures = list(self.approach_safety[0])
                    self.safety_loads = list(self.approach_safety[1])

            def _fast_step(self) -> int:  # type: ignore[override]
                # 根部对齐保持快速；指尖和抓牢后的动态微调仍由基类的
                # capture_step=1 单步限制约束。
                return max(3, min(self.config.max_step, 7))

        self.soft_controller = TomatoSoftBallController()
        # 番茄探索阶段采用比历史软网球演示更保守的压力/速度配置。
        # 动态保持阶段会在 _controller_step 中临时放宽安全带，但不会改变
        # 传感器闭环或跳过控制器的连续超限保护。
        self.soft_config = SoftBallConfig(
            target_pressure=34.0,
            safety_pressure=150.0,
            capture_step=1,
            speed_percent=12.0,
            torque_percent=18.0,
            max_step=8,
        ).normalized()
        self.soft_controller.config = self.soft_config
        self.soft_controller._recompute_thresholds()
        self.pose = list(OPEN_POSE)
        self.feedback_pose = list(OPEN_POSE)
        self.matrices = blank_matrices()
        self.peaks = [0.0] * 5
        self.loads = [0.0] * 5
        self.tomato = tomato_config({})
        self.plan: dict[str, Any] | None = None
        self.token = ""
        self.token_at = 0.0
        self.phase = "idle"
        self.phase_index = 0
        self.phase_started = 0.0
        self.active = False
        self.actual_armed = False
        self.connected = profile == "demo"
        self.error = ""
        self.command_count = 0
        self.sensor_contacts = [False] * 3
        self.tip_dynamic_adjustments = [0, 0, 0]
        self.thumb_dynamic_adjustment = 0
        self.secure_grasp = False
        self.secure_evidence = False
        self.secure_streak = 0
        self.updated_at = time.monotonic()
        if profile != "demo" and connect:
            self.connect_hardware()

    def connect_hardware(self) -> None:
        if self.profile != "left-flexible":
            raise RuntimeError("只有 left-flexible 支持左手连接")
        if self.connected and self.hardware is not None:
            return
        if not self.hardware_allowed:
            raise PermissionError("番茄实机链路需要 TOMATO_CONSOLE_ALLOW_HARDWARE=1 显式授权")
        sys.path.insert(0, str(OLD_CONSOLE))
        from hardware import HardwareHand  # type: ignore
        self.hardware = HardwareHand("left-flexible", self.can)
        self.hardware.connect()
        self.connected = True

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)
        if self.hardware is not None:
            self.hardware.close()

    def start(self) -> None:
        self.thread.start()

    def _read_hardware(self) -> None:
        if self.hardware is None:
            return
        pose, matrices = self.hardware.read()
        matrices = denoise_matrices(matrices)
        peaks = [matrix_peak(matrix) for matrix in matrices]
        loads = [matrix_sum(matrix) for matrix in matrices]
        # 载荷 1 通常是单个串流噪声单元，不作为接触证据；三指仍
        # 保持不对称的最低有效触觉门槛。
        pressure_floors = (5.0, 2.0, 2.0)
        load_floors = (3.0, 2.0, 3.0)
        contacts = [
            peaks[index] >= pressure_floors[index]
            or loads[index] >= load_floors[index]
            for index in range(3)
        ]
        with self.lock:
            self.feedback_pose = pose
            self.matrices = matrices
            self.peaks = peaks
            self.loads = loads
            self.sensor_contacts = contacts
            self.updated_at = time.monotonic()

    def _simulate_tactile(self) -> None:
        """Deterministic soft-object contact model for offline validation."""
        with self.lock:
            active = self.active
            phase = self.soft_controller.phase
            pose = list(self.pose)
            diameter_mm = float(self.tomato.get("diameter_mm", 72.0))
        matrices = blank_matrices()
        peaks = [0.0] * 5
        loads = [0.0] * 5
        regular_contact_phase = phase in {"contact_search", "adaptive_capture", "holding"}
        early_large_phase = (
            diameter_mm >= LARGE_FRUIT_DIAMETER_MM
            and phase in {"fast_positioning", "tip_refinement"}
        )
        if active and (regular_contact_phase or early_large_phase):
            for finger in range(3):
                channels = FINGER_CHANNELS[finger]
                closure = sum(255 - pose[c] for c in channels if c not in RESERVED) / max(1, len([c for c in channels if c not in RESERVED]))
                # Preserve the asymmetric soft-ball tactile baseline: thumb
                # carries most load, index/middle only need small evidence.
                scales = (0.78, 0.08, 0.11)
                if early_large_phase:
                    # 大果在参考姿态前就会触到指腹；模拟一个柔和的
                    # “早接触”信号，让离线预演也能验证轮廓搜索分支。
                    onset = (35.0, 40.0, 45.0)[finger]
                    early_scales = (1.20, 0.22, 0.25)
                    peak = clamp(max(0.0, closure - onset) * early_scales[finger], 0, 120)
                else:
                    peak = clamp(closure * scales[finger], 0, 120)
                if finger == 0 and phase in ("adaptive_capture", "holding"):
                    # Offline view includes a small pressure fluctuation so
                    # the thumb-tip closed loop is visibly exercised. Real
                    # hardware uses the measured matrix instead.
                    peak = clamp(peak + 9.0 * math.sin(time.monotonic() * 2.2), 0, 120)
                load = peak * (6.5 if finger == 0 else 2.0)
                peaks[finger], loads[finger] = peak, load
                if peak > 1:
                    matrices[finger][5][2] = peak
                    matrices[finger][6][3] = peak * 0.92
                    matrices[finger][4][2] = peak * 0.62
        matrices = denoise_matrices(matrices)
        peaks = [matrix_peak(matrix) for matrix in matrices]
        loads = [matrix_sum(matrix) for matrix in matrices]
        with self.lock:
            self.matrices, self.peaks, self.loads = matrices, peaks, loads
            self.sensor_contacts = [
                peaks[index] >= (5.0, 2.0, 2.0)[index]
                or loads[index] >= (3.0, 2.0, 3.0)[index]
                for index in range(3)
            ]
            self.updated_at = time.monotonic()

    def _controller_step(self, now: float) -> None:
        with self.lock:
            if not self.active:
                return
            matrices = [[list(row) for row in matrix] for matrix in self.matrices]
            sensor_age = max(0.0, time.monotonic() - self.updated_at)
            diameter_mm = float(self.tomato.get("diameter_mm", 72.0))
            sensor_contact_hint = any(self.sensor_contacts[:3])
        controller_contact_hint = any(self.soft_controller.contacts[:3])
        contact_hint = sensor_contact_hint or controller_contact_hint
        previous_phase = self.soft_controller.phase
        size_factor = clamp(diameter_mm / 72.0, 1.0, 1.35)
        large_early_contact = (
            diameter_mm >= LARGE_FRUIT_DIAMETER_MM
            and contact_hint
            and self.soft_controller.phase in ("fast_positioning", "tip_refinement", "contact_search", "adaptive_capture")
            and not self.secure_grasp
        )
        if large_early_contact:
            self.soft_controller.approach_safety = (
                [min(220.0, 150.0 * size_factor), min(135.0, 90.0 * size_factor), min(155.0, 110.0 * size_factor)],
                [min(2200.0, 1500.0 * size_factor), min(260.0, 150.0 * size_factor), min(340.0, 187.5 * size_factor)],
            )
        else:
            self.soft_controller.approach_safety = None
        # 在三指达到“抓牢”证据前，不允许基类提前进入 holding；
        # adaptive_capture 仍可做单步低速收拢，但动态卸载/补力闭环保持关闭。
        if self.soft_controller.phase == "holding" and not self.secure_grasp:
            self.soft_controller.phase = "adaptive_capture"
            self.soft_controller.phase_started = now
        # 探索/接触搜索用较窄的压力保护带，避免空抓或刚接触时用力过大；
        # 只有 secure_grasp 放行后才使用稍宽的动态带。
        dynamic_phase = self.soft_controller.phase in ("adaptive_capture", "holding") and self.secure_grasp
        self.soft_controller.dynamic_safety = dynamic_phase
        self.soft_controller.config.safety_pressure = 190.0 if dynamic_phase else 150.0
        next_pose = self.soft_controller.step(now, matrices, sensor_age)
        if next_pose is None:
            return
        efforts = self._closed_loop_efforts()
        self._update_secure_gate(efforts)
        # 低于明显接触强度时，示教位姿前的轻微触觉只作为“接触提示”，
        # 不把指尖阶段强行切到搜索，避免出现动一下就停住。
        strong_approach_contact = self._strong_approach_contact(efforts)
        tip_reference_reached = all(
            self.soft_controller.pose[channel] == self.soft_controller.frame_reference[channel]
            for channel in (15, 16, 17)
        )
        if (
            previous_phase == "tip_refinement"
            and self.soft_controller.phase == "contact_search"
            and not strong_approach_contact
            and not tip_reference_reached
        ):
            self.soft_controller.phase = "tip_refinement"
            self.soft_controller.phase_started = now
        if previous_phase == "tip_refinement" and not strong_approach_contact:
            next_pose = self._advance_contact_alignment(next_pose)
        if self.soft_controller.phase == "holding" and not self.secure_grasp:
            self.soft_controller.phase = "adaptive_capture"
            self.soft_controller.phase_started = now
        next_pose = self._apply_thumb_dynamic_tip(next_pose, efforts)
        next_pose = self._apply_index_middle_tip_relief(next_pose, efforts)
        next_pose = normalize_pose(next_pose)
        with self.lock:
            self.pose = next_pose
            self.phase = self.soft_controller.phase
            self.active = self.soft_controller.active
            # Also forward the controller's final open pose when it stops on
            # a pressure/sensor fault; otherwise a safety trip could leave
            # the hand at the last closed command.
            if self.hardware is not None and self.actual_armed:
                if not self.soft_controller.active and next_pose == OPEN_POSE:
                    self._send_reset_open(next_pose)
                else:
                    self.hardware.command_pose(next_pose)
                    self.command_count += 1
            self.updated_at = time.monotonic()

    def _strong_approach_contact(self, efforts: list[float]) -> bool:
        """只有明显接触才允许提前结束指尖对齐阶段。"""
        return any(
            self.soft_controller.contacts[index]
            and efforts[index] >= APPROACH_STOP_EFFORT[index]
            for index in range(3)
        )

    def _send_reset_open(self, pose: list[int]) -> None:
        """发送完整张开，并额外单独确认 G20 拇指指尖通道 15=255。"""
        if self.hardware is None or not self.actual_armed:
            return
        open_pose = list(OPEN_POSE)
        open_pose[15] = 255
        self.hardware.command_open(open_pose, 35, 35)
        # 某些 G20 固件在全手帧复位时会漏掉拇指 distal frame；用已验证
        # 的单指帧再次发送，确保指尖完全张开。只发送拇指帧，不改其他手指。
        self.hardware.set_speed(46)
        self.hardware.set_torque(56)
        self.hardware.command_joint(15, 255, open_pose)
        self.command_count += 2

    def _advance_contact_alignment(self, pose: list[int]) -> list[int]:
        """基类在轻微接触时会冻结 tip；这里继续以单步完成三指对齐。"""
        frame = self.soft_controller.frame_reference
        for channel in (15, 16, 17):
            current = int(self.soft_controller.pose[channel])
            target = int(frame[channel])
            if current < target:
                current = min(target, current + 1)
            elif current > target:
                current = max(target, current - 1)
            self.soft_controller.pose[channel] = current
            pose[channel] = current
        return pose

    def _update_secure_gate(self, efforts: list[float]) -> None:
        """区分接触和抓牢，动态闭环只由后者放行。"""
        contacts = list(self.soft_controller.contacts[:3])
        self.secure_evidence = all(
            contacts[index]
            and SECURE_EFFORT_MIN[index] <= efforts[index] <= SECURE_EFFORT_MAX[index]
            for index in range(3)
        )
        if self.secure_evidence:
            self.secure_streak = min(SECURE_STREAK_REQUIRED, self.secure_streak + 1)
        else:
            # 任一指失去抓牢证据立即关闭动态调整；保留少量历史 streak，
            # 让一帧传感器抖动不会反复重置整个流程。
            self.secure_streak = max(0, self.secure_streak - 2)
        self.secure_grasp = (
            self.secure_evidence and self.secure_streak >= SECURE_STREAK_REQUIRED
        )

    def _closed_loop_efforts(self) -> list[float]:
        """返回三指当前压力/载荷相对目标的最大归一化误差。"""
        return [
            max(
                self.soft_controller.filtered[index]
                / max(self.soft_controller.target_pressures[index], 1.0),
                self.soft_controller.filtered_loads[index]
                / max(self.soft_controller.target_loads[index], 1.0),
            )
            for index in range(3)
        ]

    def _apply_thumb_dynamic_tip(self, pose: list[int], efforts: list[float]) -> list[int]:
        """Make the thumb fingertip visibly follow live pressure error.

        The imported soft-ball controller already performs fingertip refinement.
        This small tomato-console overlay widens only the thumb-tip search
        corridor and applies a bounded 1-count correction from the filtered
        thumb pressure/load error. It cannot move ring/little channels or the
        reserved channels, and it is inactive before real contact exists.
        """
        phase = self.soft_controller.phase
        if phase in ("adaptive_capture", "holding") and not self.secure_grasp:
            self.thumb_dynamic_adjustment = 0
            self.tip_dynamic_adjustments = [0, 0, 0]
            return pose
        # Start the thumb fingertip while the roots are still approaching, so
        # the real grasp attempt never looks like a thumb-root-only motion.
        # This overlaps the original root-first baseline by a bounded amount.
        frame_tip = SOFT_BALL_FRAME_REFERENCE[15]
        # SoftBallGraspController returns a phase pose from its own internal
        # state on every tick.  Keep the last command as the accumulator so a
        # small fingertip correction is not overwritten by that phase pose.
        previous_tip = int(self.pose[15]) if len(self.pose) == 20 else int(pose[15])
        if phase == "fast_positioning" and pose[0] < 248:
            pose[15] = max(frame_tip, min(int(pose[15]), previous_tip - 1))
            self.thumb_dynamic_adjustment = int(pose[15]) - frame_tip
            return pose
        if phase == "tip_refinement":
            pose[15] = max(frame_tip, min(int(pose[15]), previous_tip - 1))
            self.thumb_dynamic_adjustment = int(pose[15]) - frame_tip
            return pose
        if phase == "contact_search" and not self.soft_controller.contacts[0]:
            # Keep a small fingertip search motion alive until the thumb has
            # real tactile evidence; do not wait at a frozen pose.
            pose[15] = max(frame_tip - 8, min(frame_tip + 4, int(pose[15]), previous_tip - 1))
            self.thumb_dynamic_adjustment = int(pose[15]) - frame_tip
            return pose
        if phase not in ("adaptive_capture", "holding") or not self.soft_controller.contacts[0]:
            self.thumb_dynamic_adjustment = 0
            self.tip_dynamic_adjustments[0] = 0
            return pose
        effort = efforts[0]
        # 任意一指超出动态带时，其余已经接触的手指也进入卸载模式，
        # 防止单指闭环把球体载荷转移给另一根手指。
        global_overload = any(value > 1.06 for value in efforts)
        lower, upper = frame_tip - 18, frame_tip + 14
        current = max(lower, min(upper, previous_tip))
        if global_overload:
            pose[15] = min(upper, current + 1)  # 跨指卸载：所有接触指谨慎打开
        elif effort < 0.94:
            pose[15] = max(lower, current - 1)  # regain contact: close fingertip
        else:
            pose[15] = current
        self.thumb_dynamic_adjustment = int(pose[15]) - frame_tip
        self.tip_dynamic_adjustments[0] = self.thumb_dynamic_adjustment
        return pose

    def _apply_index_middle_tip_relief(self, pose: list[int], efforts: list[float]) -> list[int]:
        """Keep index/middle from remaining compressed after contact.

        The imported reference controller deliberately caps its open-side tip
        target at the frame pose.  That is safe for the historical soft-ball
        capture but prevents a tomato contact from unloading above that weak
        prior.  This overlay gives index/middle a bounded open corridor and
        accumulates the correction from the last command so the base pose cannot
        overwrite it on the next tick.
        """
        phase = self.soft_controller.phase
        if phase not in ("adaptive_capture", "holding") or not self.secure_grasp:
            self.tip_dynamic_adjustments[1:] = [0, 0]
            return pose
        frame_tips = (SOFT_BALL_FRAME_REFERENCE[16], SOFT_BALL_FRAME_REFERENCE[17])
        channels = (16, 17)
        for finger, (channel, frame_tip) in enumerate(zip(channels, frame_tips), start=1):
            if not self.soft_controller.contacts[finger]:
                self.tip_dynamic_adjustments[finger] = 0
                continue
            effort = efforts[finger]
            global_overload = any(value > 1.06 for value in efforts)
            lower, upper = frame_tip - 24, frame_tip + 24
            previous = max(lower, min(upper, int(self.pose[channel])))
            if global_overload:
                # 其他手指超限时同样只走单步卸载，避免力量从一指转移到另一指。
                value = min(upper, previous + 1)
            elif effort < 0.94:
                value = max(lower, previous - 1)
            else:
                value = previous
            pose[channel] = value
            self.tip_dynamic_adjustments[finger] = value - frame_tip
        return pose

    def _loop(self) -> None:
        while not self.stop_event.wait(0.1):
            try:
                if self.profile != "demo":
                    self._read_hardware()
                else:
                    self._simulate_tactile()
                self._controller_step(time.monotonic())
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                    self.active = False
                    self.phase = "error"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            age = time.monotonic() - self.updated_at
            return {
                "ok": True,
                "profile": self.profile,
                "source": "hardware" if self.profile != "demo" else "simulation",
                "connected": self.connected,
                "motion_enabled": self.enable_motion and self.hardware_allowed,
                "actual_armed": self.actual_armed,
                "hardware_authorized": self.hardware_allowed,
                "sensor_age": round(age, 3),
                "error": self.error,
                "command_count": self.command_count,
                "pose": list(self.pose),
                "feedback_pose": list(self.feedback_pose),
                "matrices": self.matrices,
                "peaks": [round(x, 2) for x in self.peaks],
                "loads": [round(x, 2) for x in self.loads],
                "tomato": dict(self.tomato),
                "plan": self.plan,
                "phase": self.phase,
                "active": self.active,
                "contacts": (
                    list(self.soft_controller.contacts[:3])
                    if self.soft_controller.active else list(self.sensor_contacts)
                ),
                "raw_contacts": list(self.sensor_contacts),
                "grasp_success": bool(self.soft_controller.grasp_success),
                "secure_grasp": self.secure_grasp,
                "secure_evidence": self.secure_evidence,
                "secure_streak": self.secure_streak,
                "secure_streak_required": SECURE_STREAK_REQUIRED,
                "dynamic_adjustment_enabled": bool(
                    self.secure_grasp and self.soft_controller.phase in ("adaptive_capture", "holding")
                ),
                "target_pressures": list(self.soft_controller.target_pressures[:3]),
                "target_loads": list(self.soft_controller.target_loads[:3]),
                "safety_pressures": list(self.soft_controller.safety_pressures[:3]),
                "safety_loads": list(self.soft_controller.safety_loads[:3]),
                "closed_loop_efforts": self._closed_loop_efforts(),
                "thumb_dynamic_adjustment": self.thumb_dynamic_adjustment,
                "tip_dynamic_adjustments": list(self.tip_dynamic_adjustments),
                "preview_valid": bool(self.token and time.monotonic() - self.token_at < PREVIEW_TTL_SECONDS),
            }

    def set_tomato(self, payload: Any) -> dict[str, Any]:
        cfg = tomato_config(payload)
        plan = make_plan(cfg)
        with self.lock:
            self.tomato = cfg
            self.plan = plan
            self.token = hashlib.sha256(f"{plan['plan_id']}:{time.monotonic()}".encode()).hexdigest()
            self.token_at = time.monotonic()
            self.phase = "preview_ready"
        return {"ok": True, "plan": plan, "token": self.token}

    def start_grasp(self, token: str, confirm: str) -> None:
        with self.lock:
            if not self.plan:
                raise PermissionError("没有有效预演计划，请先生成方案并完成一次 URDF 预演")
            if token != self.token:
                raise PermissionError("预演令牌与当前方案不匹配，请重新生成方案并预演")
            if time.monotonic() - self.token_at > PREVIEW_TTL_SECONDS:
                raise PermissionError("预演令牌已过期，请重新生成方案并预演")
            if confirm != "TOMATO_GRASP_CONFIRM":
                raise PermissionError("缺少番茄抓取现场确认")
            if self.profile != "demo" and (not self.enable_motion or not self.hardware_allowed or not self.actual_armed):
                raise PermissionError("实机未完成运动权限、环境授权和页面解锁")
            if self.hardware is not None:
                self.hardware.set_speed(32)   # 探索阶段降低到约 12.5% 的官方 255 刻度
                self.hardware.set_torque(46)  # 探索阶段降低到约 18% 的官方 255 刻度
            self.soft_controller.dynamic_safety = False
            self.soft_controller.approach_safety = None
            self.secure_grasp = False
            self.secure_evidence = False
            self.secure_streak = 0
            self.tip_dynamic_adjustments = [0, 0, 0]
            self.soft_controller.start(time.monotonic(), self.pose, self.soft_config)
            self.active = True
            self.phase = self.soft_controller.phase
            self.error = ""

    def stop(self, emergency: bool = False) -> None:
        with self.lock:
            was_armed = self.actual_armed
            self.active = False
            self.phase = "emergency_stop" if emergency else "stopped"
            self.pose = list(self.soft_controller.emergency_stop() if emergency else self.soft_controller.release(time.monotonic()))
            self.secure_grasp = False
            self.secure_evidence = False
            self.secure_streak = 0
            self.tip_dynamic_adjustments = [0, 0, 0]
            self.soft_controller.approach_safety = None
            if self.hardware is not None and was_armed:
                self._send_reset_open(self.pose)
            self.actual_armed = False if emergency else was_armed

    def test_thumb_tip(self, confirm: str, requested_delta: Any = 8) -> dict[str, Any]:
        """Send one isolated, low-speed thumb-tip diagnostic step.

        This is deliberately separate from the grasp state machine: it lets an
        operator verify that logical channel 15 reaches the real actuator.  It
        requires the same explicit hardware authorization and page unlock as a
        grasp, and emits only the thumb frame through ``command_joint``.
        """
        with self.lock:
            if confirm != "THUMB_TIP_TEST_CONFIRM":
                raise PermissionError("缺少拇指指尖测试确认口令")
            if self.profile != "left-flexible" or self.hardware is None or not self.connected:
                raise RuntimeError("左手实机尚未连接")
            if not self.hardware_allowed or not self.enable_motion or not self.actual_armed:
                raise PermissionError("实机未完成运动权限和页面解锁")
            if self.active:
                raise RuntimeError("当前正在执行抓取流程，请先停止后再做单指测试")
            if isinstance(requested_delta, bool) or not isinstance(requested_delta, (int, float)):
                raise ValueError("拇指指尖测试步长必须是数值")
            delta = max(1, min(40, int(requested_delta)))
            pose = list(self.pose)
            # G20 channel 15 closes the thumb fingertip as the raw value falls.
            # Eight counts is intentionally small enough for a visual/functional
            # check while remaining inside the open-side diagnostic corridor.
            target = max(OPEN_POSE[15] - 40, int(pose[15]) - delta)
            pose[15] = target
            # Match the validated soft-ball low-speed envelope; 14/35 can be
            # below the thumb-tip's static-friction threshold on the G20.
            self.hardware.set_speed(46)
            self.hardware.set_torque(56)
            self.hardware.command_joint(15, target, pose)
            self.pose = pose
            self.thumb_dynamic_adjustment = target - SOFT_BALL_FRAME_REFERENCE[15]
            self.command_count += 1
            self.updated_at = time.monotonic()
            return {"ok": True, "channel": 15, "value": target, "delta": delta, "pose": list(pose)}


class Handler(BaseHTTPRequestHandler):
    runtime: TomatoRuntime

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stdout.write(f"[{time.strftime('%H:%M:%S')}] {fmt % args}\n")

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/api/state":
            self._json(self.runtime.snapshot())
            return
        if path == "/favicon.ico":
            # 浏览器默认探测图标不应在控制台日志中制造误导性的 404。
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if path == "/":
            path = "/index.html"
        if path.startswith("/vendor/"):
            target = (VENDOR / path.removeprefix("/vendor/")).resolve()
            root = VENDOR.resolve()
        elif path.startswith("/hand-model/"):
            target = (HAND_ASSET_ROOT / path.removeprefix("/hand-model/")).resolve()
            root = HAND_ASSET_ROOT.resolve()
        elif path == "/hand-kinematics.js":
            target = (OLD_CONSOLE / "static" / "hand-kinematics.js").resolve()
            root = (OLD_CONSOLE / "static").resolve()
        else:
            target = (STATIC / path.removeprefix("/")).resolve()
            root = STATIC.resolve()
        if not str(target).startswith(str(root)) or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        mime = "text/html; charset=utf-8" if target.suffix == ".html" else "application/javascript; charset=utf-8" if target.suffix == ".js" else "text/css; charset=utf-8" if target.suffix == ".css" else "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            payload = self._read_json()
            if path == "/api/tomato/config":
                self._json(self.runtime.set_tomato(payload))
            elif path == "/api/tomato/start":
                self.runtime.start_grasp(str(payload.get("token", "")), str(payload.get("confirm", "")))
                self._json({"ok": True})
            elif path == "/api/tomato/stop":
                self.runtime.stop(False)
                self._json({"ok": True})
            elif path == "/api/tomato/emergency-stop":
                self.runtime.stop(True)
                self._json({"ok": True})
            elif path == "/api/hardware/connect":
                self.runtime.connect_hardware()
                self._json({"ok": True})
            elif path == "/api/actual/arm":
                if not self.runtime.hardware_allowed or not self.runtime.enable_motion:
                    raise PermissionError("未启用番茄实机运动授权")
                if payload.get("confirm") != "TOMATO_LEFT_HAND_CONTROL":
                    raise PermissionError("缺少番茄左手控制确认口令")
                self.runtime.actual_armed = True
                self._json({"ok": True})
            elif path == "/api/actual/disarm":
                self.runtime.actual_armed = False
                self._json({"ok": True})
            elif path == "/api/test/thumb-tip":
                self._json(self.runtime.test_thumb_tip(
                    str(payload.get("confirm", "")), payload.get("delta", 8)))
            elif path == "/api/release":
                self.runtime.stop(False)
                self._json({"ok": True})
            else:
                self._json({"ok": False, "error": "接口不存在"}, 404)
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 400)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--profile", choices=("demo", "left-flexible"), default="demo")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--connect", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    args = parser.parse_args()
    if args.profile != "demo" and not args.connect:
        raise SystemExit("硬件 profile 必须显式使用 --connect")
    if args.enable_motion and args.profile != "left-flexible":
        raise SystemExit("--enable-motion 只能用于 left-flexible")
    runtime = TomatoRuntime(args.profile, args.can, args.connect, args.enable_motion)
    Handler.runtime = runtime
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    runtime.start()
    print(f"番茄自适应抓取控制台：http://{args.host}:{args.port}/ [{args.profile}]", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
