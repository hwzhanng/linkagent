"""Read-only, deterministic API-to-skill validation for P0 through P3.

This Task is intentionally independent from the robot gateways and from the
harvest/grasp Tasks.  It exercises the same Agent -> named skill boundary while
making it impossible for a probe to acquire a hardware lease, open a CAN bus,
or fetch a camera URL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from module.agent import AgentTool, RecoverableToolError
from module.runtime import RuntimeContext
from proto import Observation
from task._base_task import BaseTask


_DEFAULT_REQUIRED_TOOLS = (
    "get_validation_capabilities",
    "probe_inverse_kinematics",
    "probe_camera_preview",
)

_ALL_TOOLS = frozenset(
    {
        "get_validation_capabilities",
        "probe_inverse_kinematics",
        "probe_camera_preview",
        "probe_nonce",
    }
)


@dataclass
class ValidationProbeState:
    """Small audit trail proving which offline handlers were dispatched."""

    capability_calls: int = 0
    ik_calls: int = 0
    camera_calls: int = 0
    nonce_calls: int = 0
    last_ik_target: tuple[float, float, float] | None = None
    last_camera: tuple[str, int] | None = None
    last_nonce: str | None = None
    closed: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)


class HardwareCapabilityValidationTask(BaseTask):
    """Offline validation of capability discovery, IK and camera skill routing.

    All handlers are deterministic and read-only.  The result schema deliberately
    includes explicit safety flags so a report cannot be mistaken for hardware
    evidence.
    """

    _TOOL_NAMES = _ALL_TOOLS

    def __init__(
        self,
        *,
        stage: str = "p0",
        required_tools: Sequence[str] = _DEFAULT_REQUIRED_TOOLS,
        expected_nonce: str | None = None,
    ) -> None:
        if not isinstance(stage, str) or stage.strip().lower() not in {"p0", "p1", "p2", "p3"}:
            raise ValueError("stage must be one of 'p0', 'p1', 'p2', or 'p3'")
        if isinstance(required_tools, (str, bytes, bytearray)):
            raise TypeError("required_tools must be a sequence of tool names")
        try:
            normalized = tuple(str(name).strip() for name in required_tools)
        except TypeError as exc:
            raise TypeError("required_tools must be a sequence of tool names") from exc
        if not normalized or any(not name for name in normalized):
            raise ValueError("required_tools must contain at least one non-empty name")
        unknown = set(normalized) - self._TOOL_NAMES
        if unknown:
            raise ValueError(
                "required_tools contains unsupported probes: "
                + ", ".join(sorted(unknown))
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("required_tools must not contain duplicates")
        if expected_nonce is not None and (
            not isinstance(expected_nonce, str) or not expected_nonce.strip()
        ):
            raise ValueError("expected_nonce must be a non-empty string or None")
        self._stage = stage.strip().lower()
        self._required_tools = normalized
        self._expected_nonce = expected_nonce.strip() if expected_nonce is not None else None
        self._state = ValidationProbeState()
        self._fake_bundle = None
        self._camera_health = None
        if self._stage == "p3":
            self._build_p3_replay_backends()
        super().__init__()

    @classmethod
    def validate_config_options(cls, options: Mapping[str, Any]) -> None:
        super().validate_config_options(options)
        allowed = {"stage", "required_tools", "expected_nonce"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(
                "unsupported hardware validation task options: "
                + ", ".join(sorted(str(key) for key in unknown))
            )

    @property
    def state(self) -> ValidationProbeState:
        """Return the mutable audit state for offline tests and reports."""

        return self._state

    def build_prompt(self) -> str:
        return (
            "Run the requested offline capability probes only. This validation "
            "Task never controls a robot, opens CAN, or reads a live camera. "
            "Use each allowed probe at most once and finish only after every "
            f"required probe succeeds: {', '.join(self._required_tools)}."
        )

    def _build_p3_replay_backends(self) -> None:
        """Construct deterministic fake/read-only backends for P3 only."""

        from module.robotics import (
            FakeArmSdkGateway,
            FakeG20SdkGateway,
            FakeGatewayFaults,
            LinkBotG20ShadowBundle,
            ManualClock,
            MockHandSynergyMapper,
            MockPickScenario,
            MockPickWorld,
        )
        from proto import ArmSide

        clock = ManualClock()
        mapper = MockHandSynergyMapper()
        world = MockPickWorld(
            side=ArmSide.RIGHT,
            scenario=MockPickScenario(),
            clock=clock,
            mapper=mapper,
        )
        self._fake_bundle = LinkBotG20ShadowBundle(
            side=ArmSide.RIGHT,
            arm_gateway=FakeArmSdkGateway(world, FakeGatewayFaults()),
            hand_gateway=FakeG20SdkGateway(world, mapper, FakeGatewayFaults()),
            clock=clock,
            evidence_source="p3-fake-gateway",
        )
        self._fake_bundle.open()
        self._camera_health = {
            "device_id": "orbbec-replay-001",
            "calibration_id": "orbbec_gemini_pro_mock_calibration_v1",
            "rgb_live": True,
            "depth_live": True,
            "simulated": True,
            "evidence_sources": ("replay:rgb", "replay:aligned-depth"),
        }

    def build_tools(self, context: RuntimeContext) -> list[AgentTool]:
        # Requiring an observation keeps the probe on the same explicit runtime
        # boundary as production Tasks, while the observation contents are never
        # forwarded to a device or network.
        def _require_offline_context() -> Observation:
            return context.require_observation()

        def _execution_id(tool_name: str, ordinal: int) -> str:
            return f"{self._stage}-{tool_name}-{ordinal:04d}"

        def get_validation_capabilities() -> Mapping[str, Any]:
            """Report the read-only capabilities available in this offline probe."""

            _require_offline_context()
            self._state.capability_calls += 1
            execution_id = _execution_id(
                "get_validation_capabilities", self._state.capability_calls
            )
            self._state.events.append({"tool": "get_validation_capabilities"})
            return {
                "execution_id": execution_id,
                "ok": True,
                "stage": self._stage,
                "simulated": True,
                "hardware_accessed": False,
                "motion_executed": False,
                "network_accessed": False,
                "read_only_capabilities": [
                    "health_snapshot",
                    "arm_state_snapshot",
                    "hand_state_snapshot",
                    "tactile_state_snapshot",
                    "compute_fk",
                    "compute_ik",
                    "camera_preview_replay",
                ],
                "blocked_capabilities": [
                    "enable_or_disable_actuators",
                    "send_joint_trajectory",
                    "send_ik_target",
                    "live_camera_stream",
                ],
            }

        def probe_inverse_kinematics(
            target_x_m: float,
            target_y_m: float,
            target_z_m: float,
        ) -> Mapping[str, Any]:
            """Compute deterministic fake IK without commanding an arm."""

            _require_offline_context()
            target = (target_x_m, target_y_m, target_z_m)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in target
            ):
                raise RecoverableToolError("IK target coordinates must be finite numbers")
            if any(abs(float(value)) > 1.0 for value in target):
                raise RecoverableToolError("offline IK target is outside the 1 m probe workspace")
            x, y, z = (float(value) for value in target)
            if self._stage == "p3":
                return self._probe_p3_inverse_kinematics((x, y, z))
            joints = (
                round(y, 6),
                round(-z, 6),
                round(x, 6),
                round((x - y) / 2.0, 6),
                round((y + z) / 2.0, 6),
                round((z - x) / 2.0, 6),
                0.0,
            )
            self._state.ik_calls += 1
            execution_id = _execution_id("probe_inverse_kinematics", self._state.ik_calls)
            self._state.last_ik_target = (x, y, z)
            self._state.events.append(
                {"tool": "probe_inverse_kinematics", "target_m": [x, y, z]}
            )
            return {
                "execution_id": execution_id,
                "ok": True,
                "stage": self._stage,
                "simulated": True,
                "hardware_accessed": False,
                "motion_executed": False,
                "target_xyz_m": [x, y, z],
                "joint_positions_rad": list(joints),
                "position_residual_m": 0.0,
                "solver": "deterministic_offline_probe",
            }

        def probe_nonce(nonce: str) -> Mapping[str, Any]:
            """Echo a caller nonce to prove one real model tool dispatch."""

            _require_offline_context()
            if not isinstance(nonce, str) or not nonce.strip():
                raise RecoverableToolError("nonce must be a non-empty string")
            normalized = nonce.strip()
            if self._expected_nonce is not None and normalized != self._expected_nonce:
                raise RecoverableToolError("nonce did not match the requested probe value")
            self._state.nonce_calls += 1
            self._state.last_nonce = normalized
            execution_id = _execution_id("probe_nonce", self._state.nonce_calls)
            self._state.events.append({"tool": "probe_nonce", "nonce": normalized})
            return {
                "execution_id": execution_id,
                "ok": True,
                "stage": self._stage,
                "simulated": True,
                "hardware_accessed": False,
                "motion_executed": False,
                "nonce": normalized,
            }

        def probe_camera_preview(camera_name: str = "wrist", frame_id: int = 0) -> Mapping[str, Any]:
            """Return replay metadata for a camera frame without opening a stream."""

            _require_offline_context()
            if not isinstance(camera_name, str) or not camera_name.strip():
                raise RecoverableToolError("camera_name must be a non-empty string")
            if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
                raise RecoverableToolError("frame_id must be an integer >= 0")
            name = camera_name.strip()
            self._state.camera_calls += 1
            execution_id = _execution_id("probe_camera_preview", self._state.camera_calls)
            self._state.last_camera = (name, frame_id)
            self._state.events.append(
                {"tool": "probe_camera_preview", "camera": name, "frame_id": frame_id}
            )
            if self._stage == "p3":
                return self._probe_p3_camera_preview(
                    name=name, frame_id=frame_id, execution_id=execution_id
                )
            return {
                "execution_id": execution_id,
                "ok": True,
                "stage": self._stage,
                "simulated": True,
                "hardware_accessed": False,
                "network_accessed": False,
                "stream_opened": False,
                "camera": name,
                "frame_id": frame_id,
                "frame_available": True,
                "encoding": "metadata-only",
                "width": 640,
                "height": 480,
                "payload_ref": f"offline://camera/{name}/{frame_id}",
            }

        available_tools = {
            "get_validation_capabilities": AgentTool(
                name="get_validation_capabilities",
                description="List safe read-only capabilities and explicitly blocked hardware actions.",
                handler=get_validation_capabilities,
            ),
            "probe_inverse_kinematics": AgentTool(
                name="probe_inverse_kinematics",
                description="Run deterministic offline IK and return a candidate joint vector without motion.",
                handler=probe_inverse_kinematics,
            ),
            "probe_camera_preview": AgentTool(
                name="probe_camera_preview",
                description="Return metadata for an offline camera replay frame without opening a live stream.",
                handler=probe_camera_preview,
            ),
            "probe_nonce": AgentTool(
                name="probe_nonce",
                description="Echo a nonce to prove one and only one local skill dispatch.",
                handler=probe_nonce,
            ),
        }
        return [available_tools[name] for name in self._required_tools]

    def _probe_p3_inverse_kinematics(
        self, target: tuple[float, float, float]
    ) -> Mapping[str, Any]:
        from proto import ArmSide, Pose

        if self._fake_bundle is None:
            raise RuntimeError("P3 fake gateway is not initialized")
        side = ArmSide.RIGHT
        state = self._fake_bundle.arm.get_state(side)
        target_pose = Pose(target, (1.0, 0.0, 0.0, 0.0), "world")
        ik_result = self._fake_bundle.arm.ik(
            side, target_pose, state.joint_positions_rad
        )
        if not ik_result.success:
            raise RecoverableToolError(f"fake gateway IK failed: {ik_result.error_code}")
        fk_result = self._fake_bundle.arm.fk(side, ik_result.joint_positions_rad)
        if not fk_result.success or fk_result.pose is None:
            raise RecoverableToolError("fake gateway FK回代失败")
        residual = tuple(
            fk_result.pose.position_m[index] - target[index] for index in range(3)
        )
        margins = tuple(math.pi - abs(value) for value in ik_result.joint_positions_rad)
        self._state.ik_calls += 1
        execution_id = _execution_id_for_state(
            self._stage, "probe_inverse_kinematics", self._state.ik_calls
        )
        self._state.last_ik_target = target
        self._state.events.append(
            {"tool": "probe_inverse_kinematics", "target_m": list(target)}
        )
        return {
            "execution_id": execution_id,
            "ok": True,
            "stage": self._stage,
            "simulated": True,
            "hardware_accessed": False,
            "motion_executed": False,
            "side": side.value,
            "target_pose": target_pose.as_dict(),
            "seed_source": "fake_gateway_current_feedback",
            "seed_joint_positions_rad": list(state.joint_positions_rad),
            "solution_found": True,
            "joint_positions_rad": list(ik_result.joint_positions_rad),
            "fk_pose": fk_result.pose.as_dict(),
            "position_residual_m": list(residual),
            "max_position_residual_m": max(abs(value) for value in residual),
            "joint_limit_margin_rad": list(margins),
            "collision_checked": False,
            "collision_check_reason": "fake_gateway_has_no_collision_model",
            "solver": "fake_gateway",
        }

    def _probe_p3_camera_preview(
        self, *, name: str, frame_id: int, execution_id: str
    ) -> Mapping[str, Any]:
        if self._camera_health is None:
            raise RuntimeError("P3 replay camera is not initialized")
        snapshot = self._camera_health
        return {
            "execution_id": execution_id,
            "ok": True,
            "stage": self._stage,
            "simulated": snapshot["simulated"],
            "hardware_accessed": False,
            "motion_executed": False,
            "network_accessed": False,
            "stream_opened": False,
            "preview_opened": False,
            "camera": name,
            "frame_id": frame_id,
            "device_id": snapshot["device_id"],
            "calibration_id": snapshot["calibration_id"],
            "rgb_live": snapshot["rgb_live"],
            "depth_live": snapshot["depth_live"],
            "evidence_sources": list(snapshot["evidence_sources"]),
            "rgb_payload_ref": f"replay://{snapshot['device_id']}/rgb/{frame_id}",
            "depth_payload_ref": f"replay://{snapshot['device_id']}/depth/{frame_id}",
        }

    def build_input(self, observation: Observation) -> Mapping[str, Any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Run the {_stage_label(self._stage)} validation probes. "
                        f"Observation sequence={observation.sequence}; no live device is available."
                    ),
                }
            ]
        }

    def is_complete(
        self,
        context: RuntimeContext,
        agent_result: Mapping[str, Any],
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
        self._state.closed = True
        errors: list[BaseException] = []
        for backend in (self._camera_health, self._fake_bundle):
            if backend is None:
                continue
            try:
                close = getattr(backend, "close", None)
                if close is not None:
                    close()
            except BaseException as exc:
                errors.append(exc)
        super().close()
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("P3 replay backend cleanup failed", errors)


def _stage_label(stage: str) -> str:
    return {
        "p0": "P0 offline baseline",
        "p1": "P1 scripted API-to-skill",
        "p2": "P2 real API read-only",
        "p3": "P3 API to fake gateway and replay",
    }[stage]


def _execution_id_for_state(stage: str, tool_name: str, ordinal: int) -> str:
    return f"{stage}-{tool_name}-{ordinal:04d}"


__all__ = ["HardwareCapabilityValidationTask", "ValidationProbeState"]
