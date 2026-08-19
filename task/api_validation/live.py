"""Telemetry-only Agent tools for explicitly admitted hardware validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from module.agent import AgentTool, RecoverableToolError
from module.runtime import RuntimeContext
from proto import Observation
from task._base_task import BaseTask


ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = ROOT / "botSDK" / "lbot_arm_api-master"
HARDWARE_ADMISSION_ENV = "LINKAGENT_ALLOW_READ_ONLY_HARDWARE"
SIDES = ("left", "right")
TARGET_OFFSETS_M: dict[str, tuple[float, float, float]] = {
    "measured_pose": (0.0, 0.0, 0.0),
    "x_plus_5mm": (0.005, 0.0, 0.0),
    "x_minus_5mm": (-0.005, 0.0, 0.0),
    "y_plus_5mm": (0.0, 0.005, 0.0),
    "y_minus_5mm": (0.0, -0.005, 0.0),
    "z_plus_5mm": (0.0, 0.0, 0.005),
    "z_minus_5mm": (0.0, 0.0, -0.005),
}
_TOOL_NAMES = frozenset(
    {"get_live_arm_state", "compute_live_shadow_ik", "get_live_camera_preview"}
)


def _loopback_url(value: str, *, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError("viewer URL must be a string")
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("viewer URL must use local HTTP loopback")
    if parsed.path != path or parsed.params or parsed.query or parsed.fragment:
        raise ValueError(f"viewer URL must use the exact path {path}")
    return value


def _fetch_viewer_status(url: str) -> Mapping[str, Any]:
    request = Request(url, method="GET", headers={"Accept": "application/json"})
    with urlopen(request, timeout=2.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("camera status payload must be an object")
    return payload


class _LiveArmSession:
    """Lazily open only the requested arm's bounded telemetry runtime."""

    def __init__(self, deployment: Mapping[str, Any]) -> None:
        arms = deployment.get("arms")
        if not isinstance(arms, Mapping):
            raise ValueError("deployment arms must be a mapping")
        self._deployment = arms
        self._runtimes: dict[str, Any] = {}
        self._controllers: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._closed = False

    def _start_side(self, side: str) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("live telemetry session is closed")
            if side in self._runtimes:
                return
            spec = self._deployment.get(side)
            if not isinstance(spec, Mapping):
                raise ValueError(f"missing deployment for {side} arm")
            interface = spec.get("can_interface")
            profile_path = spec.get("profile_path")
            if not isinstance(interface, str) or not interface:
                raise ValueError(f"missing CAN interface for {side} arm")
            if not isinstance(profile_path, str) or not profile_path:
                raise ValueError(f"missing profile path for {side} arm")
            profile = (ROOT / profile_path).resolve()
            profile.relative_to(ROOT)
            if str(VENDOR_ROOT) not in sys.path:
                sys.path.insert(0, str(VENDOR_ROOT))
            from tools.local_arm_can.advanced_controller import AdvancedArmController
            from tools.local_arm_can.service import ArmCanRuntime

            runtime = ArmCanRuntime(
                interface,
                arm=side,
                allow_tx=True,
                allow_can0=interface == "can0",
                poll_telemetry=True,
            )
            try:
                runtime.start()
                time.sleep(1.0)
                controller = AdvancedArmController(runtime, config_path=profile)
            except BaseException:
                runtime.close()
                raise
            self._runtimes[side] = runtime
            self._controllers[side] = controller

    def snapshot(self, side: str) -> tuple[Mapping[str, Any], Any]:
        if side not in SIDES:
            raise ValueError("side must be left or right")
        self._start_side(side)
        return self._runtimes[side].snapshot(), self._controllers[side]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            runtimes = tuple(self._runtimes.values())
            self._runtimes.clear()
            self._controllers.clear()
        for runtime in runtimes:
            runtime.close()


@dataclass
class LiveValidationState:
    state_calls: int = 0
    ik_calls: int = 0
    camera_calls: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False


class HardwareReadOnlyValidationTask(BaseTask):
    """Expose arm telemetry, Shadow IK and loopback camera status only.

    Real adapters remain inaccessible unless both the Task configuration and
    ``LINKAGENT_ALLOW_READ_ONLY_HARDWARE=1`` explicitly admit access. The arm
    adapter may transmit bounded telemetry/state-query frames, but no enable,
    mode, target, trajectory, hand, or emergency-stop method is exposed.
    """

    def __init__(
        self,
        *,
        hardware_access_enabled: bool = False,
        deployment_path: str = "config/robot_bimanual_offline.yaml",
        viewer_status_url: str = "http://127.0.0.1:8080/api/status",
        viewer_url: str = "http://127.0.0.1:8080/",
        rgb_stream_url: str = "http://127.0.0.1:8080/stream/rgb.mjpg",
        depth_stream_url: str = "http://127.0.0.1:8080/stream/depth.mjpg",
        required_tools: Sequence[str] = tuple(sorted(_TOOL_NAMES)),
        session_factory: Callable[[Mapping[str, Any]], Any] | None = None,
        viewer_fetcher: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        if not isinstance(hardware_access_enabled, bool):
            raise TypeError("hardware_access_enabled must be a boolean")
        required = _normalize_tools(required_tools)
        self._hardware_access_enabled = hardware_access_enabled
        self._deployment_path = (ROOT / deployment_path).resolve()
        self._deployment_path.relative_to(ROOT)
        self._viewer_status_url = _loopback_url(viewer_status_url, path="/api/status")
        self._viewer_url = _loopback_url(viewer_url, path="/")
        self._rgb_stream_url = _loopback_url(
            rgb_stream_url, path="/stream/rgb.mjpg"
        )
        self._depth_stream_url = _loopback_url(
            depth_stream_url, path="/stream/depth.mjpg"
        )
        self._required_tools = required
        self._session_factory = session_factory or _LiveArmSession
        self._viewer_fetcher = viewer_fetcher or _fetch_viewer_status
        self._session: Any | None = None
        self._state = LiveValidationState()
        super().__init__()

    @classmethod
    def validate_config_options(cls, options: Mapping[str, Any]) -> None:
        super().validate_config_options(options)
        allowed = {
            "hardware_access_enabled",
            "deployment_path",
            "viewer_status_url",
            "viewer_url",
            "rgb_stream_url",
            "depth_stream_url",
            "required_tools",
        }
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(
                "unsupported live validation options: "
                + ", ".join(sorted(map(str, unknown)))
            )

    @property
    def state(self) -> LiveValidationState:
        return self._state

    def build_prompt(self) -> str:
        return (
            "This is telemetry-only hardware validation. Use only the exact "
            "allowlist. Reading fresh arm state, computing Shadow IK, and reading "
            "loopback camera status are allowed; enable, mode, reset, target, "
            "trajectory, hand control, emergency stop, and motion are unavailable."
        )

    def _admit(self) -> None:
        if not self._hardware_access_enabled:
            raise RecoverableToolError(
                "live hardware access is disabled by Task configuration"
            )
        if os.environ.get(HARDWARE_ADMISSION_ENV) != "1":
            raise RecoverableToolError(
                f"live hardware access requires {HARDWARE_ADMISSION_ENV}=1"
            )

    def _load_deployment(self) -> Mapping[str, Any]:
        payload = json.loads(self._deployment_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("deployment config must be a mapping")
        return payload

    def _arm_session(self) -> Any:
        if self._session is None:
            self._session = self._session_factory(self._load_deployment())
        return self._session

    def build_tools(self, context: RuntimeContext) -> list[AgentTool]:
        def require_context() -> Observation:
            return context.require_observation()

        def execution_id(name: str, ordinal: int) -> str:
            return f"hardware-read-only-{name}-{ordinal:04d}"

        def get_live_arm_state(side: str) -> Mapping[str, Any]:
            require_context()
            self._admit()
            _validate_side(side)
            snapshot, controller = self._arm_session().snapshot(side)
            self._require_fresh_feedback(snapshot, side)
            self._state.state_calls += 1
            self._state.events.append({"tool": "get_live_arm_state", "side": side})
            return self._state_output(
                snapshot,
                controller,
                execution_id("get_live_arm_state", self._state.state_calls),
            )

        def compute_live_shadow_ik(side: str, target: str) -> Mapping[str, Any]:
            require_context()
            self._admit()
            _validate_side(side)
            if target not in TARGET_OFFSETS_M:
                raise RecoverableToolError("target is not an approved Shadow IK target")
            snapshot, controller = self._arm_session().snapshot(side)
            self._require_fresh_feedback(snapshot, side)
            seed = tuple(float(row["position"]) for row in snapshot["joints"])
            measured = controller.forward(seed)
            offset = TARGET_OFFSETS_M[target]
            target_position = tuple(
                float(value + delta)
                for value, delta in zip(measured.position, offset, strict=True)
            )
            result = controller.inverse(target_position, measured.rpy, seed)
            if not result.success:
                raise RecoverableToolError("Shadow IK did not converge")
            candidate = tuple(float(value) for value in result.joints)
            returned = controller.forward(candidate)
            repeated = controller.inverse(target_position, measured.rpy, seed)
            if not repeated.success:
                raise RecoverableToolError("Shadow IK repeatability check failed")
            lower, upper = controller.kinematics.joint_limits
            margins = [
                min(value - float(lo), float(hi) - value)
                for value, lo, hi in zip(candidate, lower, upper, strict=True)
            ]
            deltas = [value - start for value, start in zip(candidate, seed, strict=True)]
            repeats = [
                value - float(other)
                for value, other in zip(candidate, repeated.joints, strict=True)
            ]
            self._state.ik_calls += 1
            self._state.events.append(
                {"tool": "compute_live_shadow_ik", "side": side, "target": target}
            )
            return {
                "execution_id": execution_id(
                    "compute_live_shadow_ik", self._state.ik_calls
                ),
                **_safety_flags(),
                "side": side,
                "target": target,
                "offset_m": list(offset),
                "seed_source": "current_feedback",
                "seed_joint_positions_rad": list(seed),
                "measured_fk_pose": _pose_dict(measured),
                "solution_found": True,
                "joint_positions_rad": list(candidate),
                "returned_fk_pose": _pose_dict(returned),
                "position_error_m": float(result.position_error),
                "rotation_error_rad": float(result.rotation_error),
                "iterations": int(result.iterations),
                "delta_from_seed_rad": deltas,
                "max_abs_delta_rad": max(abs(value) for value in deltas),
                "within_limits": all(value >= 0.0 for value in margins),
                "minimum_limit_margin_rad": min(margins),
                "repeat_max_abs_delta_rad": max(abs(value) for value in repeats),
                "collision_checked": False,
            }

        def get_live_camera_preview() -> Mapping[str, Any]:
            require_context()
            self._admit()
            try:
                payload = self._viewer_fetcher(self._viewer_status_url)
            except Exception as exc:
                raise RecoverableToolError(
                    f"local Orbbec Viewer status unavailable: {type(exc).__name__}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise RecoverableToolError("camera status payload must be an object")
            streams = payload.get("streams")
            device = payload.get("device")
            if not isinstance(streams, Mapping) or not isinstance(device, Mapping):
                raise RecoverableToolError("camera status payload is malformed")
            rgb = streams.get("rgb")
            depth = streams.get("depth")
            if not isinstance(rgb, Mapping) or not isinstance(depth, Mapping):
                raise RecoverableToolError("camera stream status is malformed")
            self._state.camera_calls += 1
            self._state.events.append({"tool": "get_live_camera_preview"})
            return {
                "execution_id": execution_id(
                    "get_live_camera_preview", self._state.camera_calls
                ),
                **_safety_flags(),
                "viewer_url": self._viewer_url,
                "rgb_stream_url": self._rgb_stream_url,
                "depth_stream_url": self._depth_stream_url,
                "preview_opened": False,
                "stream_opened": False,
                "state": payload.get("state"),
                "device": dict(device),
                "rgb": dict(rgb),
                "depth": dict(depth),
                "rgb_depth_live": bool(
                    rgb.get("live") is True and depth.get("live") is True
                ),
            }

        tools = {
            "get_live_arm_state": AgentTool(
                name="get_live_arm_state",
                description="Read one arm's fresh seven-joint telemetry and FK; no motion.",
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
                description="Compute bounded Shadow IK from current telemetry; no target write.",
                handler=compute_live_shadow_ik,
                parameters={
                    "type": "object",
                    "properties": {
                        "side": {"type": "string", "enum": list(SIDES)},
                        "target": {
                            "type": "string",
                            "enum": list(TARGET_OFFSETS_M),
                        },
                    },
                    "required": ["side", "target"],
                    "additionalProperties": False,
                },
            ),
            "get_live_camera_preview": AgentTool(
                name="get_live_camera_preview",
                description="Read Orbbec RGB-D status from an exact loopback URL.",
                handler=get_live_camera_preview,
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
        }
        return [tools[name] for name in self._required_tools]

    @staticmethod
    def _require_fresh_feedback(snapshot: Mapping[str, Any], side: str) -> None:
        if snapshot.get("all_joints_healthy") is not True:
            count = snapshot.get("healthy_joint_count", 0)
            raise RecoverableToolError(f"{side} feedback is stale: {count}/7 joints")
        rows = snapshot.get("joints")
        if not isinstance(rows, Sequence) or len(rows) != 7:
            raise RecoverableToolError(f"{side} telemetry must contain seven joints")

    @staticmethod
    def _state_output(
        snapshot: Mapping[str, Any], controller: Any, execution_id: str
    ) -> Mapping[str, Any]:
        rows = snapshot["joints"]
        joints = tuple(float(row["position"]) for row in rows)
        pose = controller.forward(joints)
        return {
            "execution_id": execution_id,
            **_safety_flags(),
            "interface": snapshot.get("interface"),
            "arm": snapshot.get("arm"),
            "healthy_joint_count": snapshot.get("healthy_joint_count"),
            "all_joints_healthy": True,
            "software_enabled": snapshot.get("software_enabled"),
            "last_error": snapshot.get("last_error"),
            "joint_positions_rad": list(joints),
            "joints": list(rows),
            "fk_pose": _pose_dict(pose),
            "seed_source": "current_feedback",
            "collision_checked": False,
        }

    def build_input(self, observation: Observation) -> Mapping[str, Any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Run telemetry-only hardware validation with the exact "
                        f"allowlist at observation sequence={observation.sequence}."
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
        succeeded = {
            message.get("name")
            for message in messages
            if isinstance(message, Mapping)
            and message.get("role") == "tool"
            and message.get("status") == "success"
        }
        return set(self._required_tools).issubset(succeeded)

    def close(self) -> None:
        if self._state.closed:
            return
        self._state.closed = True
        try:
            if self._session is not None:
                self._session.close()
        finally:
            super().close()


def _normalize_tools(required_tools: Sequence[str]) -> tuple[str, ...]:
    if isinstance(required_tools, (str, bytes, bytearray)):
        raise TypeError("required_tools must be a sequence of tool names")
    required = tuple(str(name).strip() for name in required_tools)
    if not required or any(not name for name in required):
        raise ValueError("required_tools must contain at least one tool")
    unknown = set(required) - _TOOL_NAMES
    if unknown:
        raise ValueError("unsupported live validation tools: " + ", ".join(sorted(unknown)))
    if len(set(required)) != len(required):
        raise ValueError("required_tools must not contain duplicates")
    return required


def _validate_side(side: str) -> None:
    if side not in SIDES:
        raise RecoverableToolError("side must be left or right")


def _pose_dict(pose: Any) -> dict[str, Any]:
    return {
        "position_m": [float(value) for value in pose.position],
        "rpy_rad": [float(value) for value in pose.rpy],
        "frame_id": "world",
    }


def _safety_flags() -> dict[str, Any]:
    return {
        "ok": True,
        "stage": "hardware-read-only",
        "simulated": False,
        "hardware_accessed": True,
        "telemetry_query_frames_authorized": True,
        "write_control_authorized": False,
        "motion_executed": False,
    }


__all__ = [
    "HARDWARE_ADMISSION_ENV",
    "HardwareReadOnlyValidationTask",
    "LiveValidationState",
]
