"""P5.5 Agent tools backed by real, read-only arm/camera adapters.

The task intentionally lives beside the P0--P3 validation task rather than in a
production manipulation task.  Its only arm-side writes are bounded CAN
telemetry/state-query frames needed to establish fresh feedback; no enable,
mode, target, trajectory, or emergency-stop operation is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from module.agent import AgentTool, RecoverableToolError
from module.runtime import RuntimeContext
from proto import Observation
from task._base_task import BaseTask


ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = ROOT / "botSDK" / "lbot_arm_api-master"
if TYPE_CHECKING:
    from tools.local_arm_can.advanced_controller import AdvancedArmController
    from tools.local_arm_can.service import ArmCanRuntime


SIDES = ("left", "right")
TARGETS = (
    "measured_pose",
    "x_plus_5mm",
    "x_minus_5mm",
    "y_plus_5mm",
    "y_minus_5mm",
    "z_plus_5mm",
    "z_minus_5mm",
)
TARGET_OFFSETS_M: dict[str, tuple[float, float, float]] = {
    "measured_pose": (0.0, 0.0, 0.0),
    "x_plus_5mm": (0.005, 0.0, 0.0),
    "x_minus_5mm": (-0.005, 0.0, 0.0),
    "y_plus_5mm": (0.0, 0.005, 0.0),
    "y_minus_5mm": (0.0, -0.005, 0.0),
    "z_plus_5mm": (0.0, 0.0, 0.005),
    "z_minus_5mm": (0.0, 0.0, -0.005),
}
POSITION_TOLERANCE_M = 2.0e-5
ROTATION_TOLERANCE_RAD = 2.0e-4
DETERMINISM_TOLERANCE_RAD = 1.0e-5


def _finite_vector(values: object, length: int, label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ValueError(f"{label} must contain {length} numeric values")
    if len(values) != length:
        raise ValueError(f"{label} must contain {length} numeric values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def _loopback_url(value: str, *, path: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("viewer URL must use the local HTTP loopback address")
    if parsed.path != path:
        raise ValueError(f"viewer URL path must be {path}")
    return value


class _LiveArmSession:
    """Own two vendor telemetry runtimes and never expose command methods."""

    def __init__(self, deployment: Mapping[str, Any]) -> None:
        arms = deployment.get("arms")
        if not isinstance(arms, Mapping):
            raise ValueError("deployment arms must be a mapping")
        self._deployment = arms
        self._runtimes: dict[str, Any] = {}
        self._controllers: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._closed = False

    def _start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("live read-only arm session is closed")
            if self._runtimes:
                return
            try:
                if str(VENDOR_ROOT) not in sys.path:
                    sys.path.insert(0, str(VENDOR_ROOT))
                from tools.local_arm_can.advanced_controller import AdvancedArmController
                from tools.local_arm_can.service import ArmCanRuntime

                for side, interface, allow_can0 in (
                    ("left", "can0", True),
                    ("right", "can1", False),
                ):
                    spec = self._deployment.get(side)
                    if not isinstance(spec, Mapping):
                        raise ValueError(f"missing deployment for {side} arm")
                    profile = (ROOT / str(spec["profile_path"])).resolve()
                    profile.relative_to(ROOT)
                    runtime = ArmCanRuntime(
                        interface,
                        arm=side,
                        allow_tx=True,
                        allow_can0=allow_can0,
                        poll_telemetry=True,
                    )
                    runtime.start()
                    self._runtimes[side] = runtime
                    self._controllers[side] = AdvancedArmController(
                        runtime, config_path=profile
                    )
                # Give both buses a bounded period to populate all seven joints.
                time.sleep(1.0)
            except BaseException:
                self._close_unlocked()
                raise

    def snapshot(self, side: str) -> tuple[dict[str, Any], Any]:
        if side not in SIDES:
            raise ValueError("side must be left or right")
        self._start()
        snapshot = self._runtimes[side].snapshot()
        return snapshot, self._controllers[side]

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        for runtime in self._runtimes.values():
            try:
                runtime.close()
            except BaseException:
                pass
        self._runtimes.clear()
        self._controllers.clear()
        self._closed = True


@dataclass
class LiveValidationState:
    state_calls: int = 0
    ik_calls: int = 0
    camera_calls: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False


class HardwareReadOnlyValidationTask(BaseTask):
    """P5.5 live, read-only Agent-to-hardware validation Task."""

    _TOOL_NAMES = frozenset(
        {"get_live_arm_state", "compute_live_shadow_ik", "get_live_camera_preview"}
    )

    def __init__(
        self,
        *,
        deployment_path: str = "config/robot_bimanual_offline.yaml",
        viewer_status_url: str = "http://127.0.0.1:8080/api/status",
        viewer_url: str = "http://127.0.0.1:8080/",
        rgb_stream_url: str = "http://127.0.0.1:8080/stream/rgb.mjpg",
        depth_stream_url: str = "http://127.0.0.1:8080/stream/depth.mjpg",
        required_tools: Sequence[str] = (
            "get_live_arm_state",
            "compute_live_shadow_ik",
            "get_live_camera_preview",
        ),
    ) -> None:
        if isinstance(required_tools, (str, bytes, bytearray)):
            raise TypeError("required_tools must be a sequence of tool names")
        required = tuple(str(name).strip() for name in required_tools)
        if not required or any(not name for name in required):
            raise ValueError("required_tools must contain at least one tool")
        unknown = set(required) - self._TOOL_NAMES
        if unknown:
            raise ValueError("unsupported live validation tools: " + ", ".join(sorted(unknown)))
        if len(set(required)) != len(required):
            raise ValueError("required_tools must not contain duplicates")
        self._deployment_path = (ROOT / deployment_path).resolve()
        self._deployment_path.relative_to(ROOT)
        self._viewer_status_url = _loopback_url(viewer_status_url, path="/api/status")
        self._viewer_url = _loopback_url(viewer_url, path="/")
        self._rgb_stream_url = _loopback_url(rgb_stream_url, path="/stream/rgb.mjpg")
        self._depth_stream_url = _loopback_url(depth_stream_url, path="/stream/depth.mjpg")
        self._required_tools = required
        self._state = LiveValidationState()
        self._session: _LiveArmSession | None = None
        super().__init__()

    @classmethod
    def validate_config_options(cls, options: Mapping[str, Any]) -> None:
        super().validate_config_options(options)
        allowed = {
            "deployment_path",
            "viewer_status_url",
            "viewer_url",
            "rgb_stream_url",
            "depth_stream_url",
            "required_tools",
        }
        unknown = set(options) - allowed
        if unknown:
            raise ValueError("unsupported live validation options: " + ", ".join(sorted(map(str, unknown))))

    @property
    def state(self) -> LiveValidationState:
        return self._state

    def build_prompt(self) -> str:
        return (
            "这是 P5.5 真机只读验证。只能调用当前白名单工具；允许读取真实机械臂和相机状态、"
            "计算 Shadow IK，但严禁使能、复位、模式切换、目标写入、轨迹、急停或任何运动。"
            "必须明确区分 hardware_accessed=true 与 motion_executed=false。"
        )

    def _require_context(self, context: RuntimeContext) -> Observation:
        return context.require_observation()

    def _execution_id(self, tool: str, ordinal: int) -> str:
        return f"p5_5-{tool}-{ordinal:04d}"

    def _load_deployment(self) -> Mapping[str, Any]:
        payload = json.loads(self._deployment_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("deployment config must be a mapping")
        return payload

    def _arm_session(self) -> _LiveArmSession:
        if self._session is None:
            self._session = _LiveArmSession(self._load_deployment())
        return self._session

    @staticmethod
    def _state_output(
        snapshot: Mapping[str, Any], controller: Any, execution_id: str
    ) -> Mapping[str, Any]:
        rows = snapshot.get("joints", [])
        joints = tuple(float(row["position"]) for row in rows)
        pose = controller.forward(joints)
        return {
            "execution_id": execution_id,
            "ok": True,
            "stage": "p5.5",
            "simulated": False,
            "hardware_accessed": True,
            "write_control_authorized": False,
            "motion_executed": False,
            "interface": snapshot["interface"],
            "arm": snapshot["arm"],
            "healthy_joint_count": snapshot["healthy_joint_count"],
            "all_joints_healthy": snapshot["all_joints_healthy"],
            "software_enabled": snapshot["software_enabled"],
            "last_error": snapshot["last_error"],
            "joint_positions_rad": list(joints),
            "joints": rows,
            "fk_pose": {
                "position_m": list(pose.position),
                "rpy_rad": list(pose.rpy),
                "frame_id": "world",
            },
            "seed_source": "current_feedback",
            "telemetry_only": True,
            "collision_checked": False,
        }

    def build_tools(self, context: RuntimeContext) -> list[AgentTool]:
        def get_live_arm_state(side: str) -> Mapping[str, Any]:
            self._require_context(context)
            if side not in SIDES:
                raise RecoverableToolError("side must be left or right")
            self._state.state_calls += 1
            execution_id = self._execution_id("get_live_arm_state", self._state.state_calls)
            snapshot, controller = self._arm_session().snapshot(side)
            output = dict(self._state_output(snapshot, controller, execution_id))
            self._state.events.append({"tool": "get_live_arm_state", "side": side})
            return output

        def compute_live_shadow_ik(side: str, target: str) -> Mapping[str, Any]:
            self._require_context(context)
            if side not in SIDES:
                raise RecoverableToolError("side must be left or right")
            if target not in TARGETS:
                raise RecoverableToolError("target is not an approved P5.5 shadow target")
            self._state.ik_calls += 1
            execution_id = self._execution_id("compute_live_shadow_ik", self._state.ik_calls)
            snapshot, controller = self._arm_session().snapshot(side)
            if not snapshot["all_joints_healthy"]:
                raise RecoverableToolError(
                    f"{side} feedback is stale: {snapshot['healthy_joint_count']}/7 joints"
                )
            seed = tuple(float(row["position"]) for row in snapshot["joints"])
            measured = controller.forward(seed)
            offset = TARGET_OFFSETS_M[target]
            target_position = tuple(
                float(value + delta)
                for value, delta in zip(measured.position, offset, strict=True)
            )
            result = controller.inverse(target_position, measured.rpy, seed)
            candidate = tuple(result.joints)
            returned = controller.forward(candidate)
            repeated = controller.inverse(target_position, measured.rpy, seed)
            lower, upper = controller.kinematics.joint_limits
            deltas = [float(a - b) for a, b in zip(candidate, seed, strict=True)]
            repeat_delta = [
                float(a - b) for a, b in zip(candidate, tuple(repeated.joints), strict=True)
            ]
            margins = [
                min(float(value) - float(lo), float(hi) - float(value))
                for value, lo, hi in zip(candidate, lower, upper, strict=True)
            ]
            self._state.events.append(
                {"tool": "compute_live_shadow_ik", "side": side, "target": target}
            )
            return {
                "execution_id": execution_id,
                "ok": True,
                "stage": "p5.5",
                "simulated": False,
                "hardware_accessed": True,
                "write_control_authorized": False,
                "motion_executed": False,
                "telemetry_only": True,
                "side": side,
                "target": target,
                "offset_m": list(offset),
                "seed_source": "current_feedback",
                "seed_joint_positions_rad": list(seed),
                "measured_fk_pose": {
                    "position_m": list(measured.position),
                    "rpy_rad": list(measured.rpy),
                    "frame_id": "world",
                },
                "solution_found": bool(result.success),
                "joint_positions_rad": list(candidate),
                "returned_fk_pose": {
                    "position_m": list(returned.position),
                    "rpy_rad": list(returned.rpy),
                    "frame_id": "world",
                },
                "position_error_m": float(result.position_error),
                "rotation_error_rad": float(result.rotation_error),
                "iterations": int(result.iterations),
                "delta_from_seed_rad": deltas,
                "max_abs_delta_rad": max(abs(value) for value in deltas),
                "j6_delta_rad": deltas[5],
                "j7_delta_rad": deltas[6],
                "within_limits": all(value >= 0.0 for value in margins),
                "minimum_limit_margin_rad": min(margins),
                "repeat_solution_found": bool(repeated.success),
                "repeat_max_abs_delta_rad": max(abs(value) for value in repeat_delta),
                "collision_checked": False,
            }

        def get_live_camera_preview() -> Mapping[str, Any]:
            self._require_context(context)
            self._state.camera_calls += 1
            execution_id = self._execution_id("get_live_camera_preview", self._state.camera_calls)
            request = Request(
                self._viewer_status_url,
                method="GET",
                headers={"Accept": "application/json"},
            )
            try:
                with urlopen(request, timeout=2.0) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                raise RecoverableToolError(
                    f"local Orbbec Viewer status unavailable: {type(exc).__name__}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise RecoverableToolError("camera status payload is not an object")
            streams = payload.get("streams", {})
            device = payload.get("device", {})
            if not isinstance(streams, Mapping) or not isinstance(device, Mapping):
                raise RecoverableToolError("camera status payload is malformed")
            rgb = streams.get("rgb", {})
            depth = streams.get("depth", {})
            if not isinstance(rgb, Mapping) or not isinstance(depth, Mapping):
                raise RecoverableToolError("camera RGB/Depth status is malformed")
            result = {
                "execution_id": execution_id,
                "ok": True,
                "stage": "p5.5",
                "simulated": False,
                "hardware_accessed": True,
                "motion_executed": False,
                "write_control_authorized": False,
                "viewer_url": self._viewer_url,
                "rgb_stream_url": self._rgb_stream_url,
                "depth_stream_url": self._depth_stream_url,
                "preview_opened": False,
                "stream_opened": False,
                "state": payload.get("state"),
                "device": dict(device),
                "rgb": dict(rgb),
                "depth": dict(depth),
                "rgb_depth_live": bool(rgb.get("live") is True and depth.get("live") is True),
            }
            self._state.events.append({"tool": "get_live_camera_preview"})
            return result

        tools = {
            "get_live_arm_state": AgentTool(
                name="get_live_arm_state",
                description="读取真实左/右机械臂当前七关节和 FK 状态；只读，不发送运动。",
                handler=get_live_arm_state,
                parameters={
                    "type": "object",
                    "properties": {"side": {"type": "string", "enum": list(SIDES)}},
                    "required": ["side"],
                    "additionalProperties": False,
                },
            ),
            "compute_live_shadow_ik": AgentTool(
                name="compute_live_shadow_ik",
                description="以真实当前反馈为 seed，对批准的 P5.5 邻近位姿做 IK/FK 回代；绝不运动。",
                handler=compute_live_shadow_ik,
                parameters={
                    "type": "object",
                    "properties": {
                        "side": {"type": "string", "enum": list(SIDES)},
                        "target": {"type": "string", "enum": list(TARGETS)},
                    },
                    "required": ["side", "target"],
                    "additionalProperties": False,
                },
            ),
            "get_live_camera_preview": AgentTool(
                name="get_live_camera_preview",
                description="读取本地 Orbbec Viewer 的真实 RGB/Depth 状态并返回受控预览 URL，不打开任意地址。",
                handler=get_live_camera_preview,
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
        }
        return [tools[name] for name in self._required_tools]

    def build_input(self, observation: Observation) -> Mapping[str, Any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "执行 P5.5 真机只读验证。按白名单调用所需工具；不得调用任何未列出的工具，"
                        f"当前 observation sequence={observation.sequence}。"
                    ),
                }
            ]
        }

    def is_complete(
        self, context: RuntimeContext, agent_result: Mapping[str, Any]
    ) -> bool:
        del context
        messages = agent_result.get("messages", [])
        if not isinstance(messages, list):
            return False
        successful = {
            message.get("name")
            for message in messages
            if isinstance(message, Mapping)
            and message.get("role") == "tool"
            and message.get("status") == "success"
        }
        return set(self._required_tools).issubset(successful)

    def close(self) -> None:
        if self._state.closed:
            return
        self._state.closed = True
        if self._session is not None:
            self._session.close()
        super().close()


__all__ = ["HardwareReadOnlyValidationTask", "LiveValidationState"]
