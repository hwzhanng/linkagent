"""Deterministic offline Agent-to-skill validation for P0 through P3."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from module.agent import AgentTool, RecoverableToolError
from module.runtime import RuntimeContext
from proto import Observation
from task._base_task import BaseTask


_DEFAULT_TOOLS = (
    "get_validation_capabilities",
    "probe_inverse_kinematics",
    "probe_camera_preview",
)
_ALL_TOOLS = frozenset((*_DEFAULT_TOOLS, "probe_nonce"))


@dataclass
class ValidationProbeState:
    """Small in-memory audit trail for offline validation."""

    capability_calls: int = 0
    ik_calls: int = 0
    camera_calls: int = 0
    nonce_calls: int = 0
    closed: bool = False
    events: list[dict[str, Any]] = field(default_factory=list)


class HardwareCapabilityValidationTask(BaseTask):
    """Expose only simulated capability, IK and camera probes to an Agent."""

    def __init__(
        self,
        *,
        stage: str = "p0",
        required_tools: Sequence[str] = _DEFAULT_TOOLS,
        expected_nonce: str | None = None,
    ) -> None:
        normalized_stage = str(stage).strip().lower()
        if normalized_stage not in {"p0", "p1", "p2", "p3"}:
            raise ValueError("stage must be one of p0, p1, p2, or p3")
        required = _normalize_required_tools(required_tools, _ALL_TOOLS)
        if expected_nonce is not None and (
            not isinstance(expected_nonce, str) or not expected_nonce.strip()
        ):
            raise ValueError("expected_nonce must be a non-empty string or None")
        self._stage = normalized_stage
        self._required_tools = required
        self._expected_nonce = expected_nonce.strip() if expected_nonce else None
        self._state = ValidationProbeState()
        self._fake_bundle: Any | None = None
        if self._stage == "p3":
            self._fake_bundle = self._build_fake_bundle()
        super().__init__()

    @classmethod
    def validate_config_options(cls, options: Mapping[str, Any]) -> None:
        super().validate_config_options(options)
        unknown = set(options) - {"stage", "required_tools", "expected_nonce"}
        if unknown:
            raise ValueError(
                "unsupported hardware validation options: "
                + ", ".join(sorted(map(str, unknown)))
            )

    @property
    def state(self) -> ValidationProbeState:
        return self._state

    def build_prompt(self) -> str:
        return (
            "Run only the requested offline capability probes. No probe may open "
            "CAN, access a live camera, authorize control, or execute motion. "
            f"Required probes: {', '.join(self._required_tools)}."
        )

    def build_tools(self, context: RuntimeContext) -> list[AgentTool]:
        def require_context() -> Observation:
            return context.require_observation()

        def execution_id(name: str, ordinal: int) -> str:
            return f"{self._stage}-{name}-{ordinal:04d}"

        def get_validation_capabilities() -> Mapping[str, Any]:
            require_context()
            self._state.capability_calls += 1
            self._state.events.append({"tool": "get_validation_capabilities"})
            return {
                "execution_id": execution_id(
                    "get_validation_capabilities", self._state.capability_calls
                ),
                "ok": True,
                "stage": self._stage,
                "simulated": True,
                "hardware_accessed": False,
                "motion_executed": False,
                "write_control_authorized": False,
                "network_accessed": False,
                "read_only_capabilities": [
                    "health_snapshot",
                    "compute_ik",
                    "camera_preview_replay",
                ],
                "blocked_capabilities": [
                    "actuator_control",
                    "target_write",
                    "trajectory",
                    "live_camera",
                ],
            }

        def probe_inverse_kinematics(
            target_x_m: float,
            target_y_m: float,
            target_z_m: float,
        ) -> Mapping[str, Any]:
            require_context()
            target = _finite_target((target_x_m, target_y_m, target_z_m))
            if any(abs(value) > 1.0 for value in target):
                raise RecoverableToolError(
                    "offline IK target is outside the 1 m probe workspace"
                )
            if self._stage == "p3":
                result = self._probe_fake_gateway_ik(target)
            else:
                x, y, z = target
                joints = (
                    round(y, 6),
                    round(-z, 6),
                    round(x, 6),
                    round((x - y) / 2.0, 6),
                    round((y + z) / 2.0, 6),
                    round((z - x) / 2.0, 6),
                    0.0,
                )
                result = {
                    "solution_found": True,
                    "joint_positions_rad": list(joints),
                    "position_residual_m": 0.0,
                    "solver": "deterministic_offline_probe",
                }
            self._state.ik_calls += 1
            self._state.events.append(
                {"tool": "probe_inverse_kinematics", "target_m": list(target)}
            )
            return {
                "execution_id": execution_id(
                    "probe_inverse_kinematics", self._state.ik_calls
                ),
                "ok": True,
                "stage": self._stage,
                "simulated": True,
                "hardware_accessed": False,
                "motion_executed": False,
                "write_control_authorized": False,
                "target_xyz_m": list(target),
                **result,
            }

        def probe_camera_preview(
            camera_name: str = "wrist", frame_id: int = 0
        ) -> Mapping[str, Any]:
            require_context()
            if not isinstance(camera_name, str) or not camera_name.strip():
                raise RecoverableToolError("camera_name must be a non-empty string")
            if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
                raise RecoverableToolError("frame_id must be an integer >= 0")
            self._state.camera_calls += 1
            name = camera_name.strip()
            self._state.events.append(
                {"tool": "probe_camera_preview", "camera": name, "frame_id": frame_id}
            )
            scheme = "replay" if self._stage == "p3" else "offline"
            return {
                "execution_id": execution_id(
                    "probe_camera_preview", self._state.camera_calls
                ),
                "ok": True,
                "stage": self._stage,
                "simulated": True,
                "hardware_accessed": False,
                "motion_executed": False,
                "write_control_authorized": False,
                "network_accessed": False,
                "stream_opened": False,
                "preview_opened": False,
                "camera": name,
                "frame_id": frame_id,
                "frame_available": True,
                "encoding": "metadata-only",
                "width": 640,
                "height": 480,
                "payload_ref": f"{scheme}://camera/{name}/{frame_id}",
            }

        def probe_nonce(nonce: str) -> Mapping[str, Any]:
            require_context()
            if not isinstance(nonce, str) or not nonce.strip():
                raise RecoverableToolError("nonce must be a non-empty string")
            normalized = nonce.strip()
            if self._expected_nonce is not None and normalized != self._expected_nonce:
                raise RecoverableToolError("nonce did not match the requested value")
            self._state.nonce_calls += 1
            self._state.events.append({"tool": "probe_nonce", "nonce": normalized})
            return {
                "execution_id": execution_id("probe_nonce", self._state.nonce_calls),
                "ok": True,
                "stage": self._stage,
                "simulated": True,
                "hardware_accessed": False,
                "motion_executed": False,
                "write_control_authorized": False,
                "nonce": normalized,
            }

        tools = {
            "get_validation_capabilities": AgentTool(
                name="get_validation_capabilities",
                description="List simulated capabilities and blocked hardware actions.",
                handler=get_validation_capabilities,
            ),
            "probe_inverse_kinematics": AgentTool(
                name="probe_inverse_kinematics",
                description="Compute deterministic offline IK without commanding hardware.",
                handler=probe_inverse_kinematics,
            ),
            "probe_camera_preview": AgentTool(
                name="probe_camera_preview",
                description="Return metadata for an offline camera replay frame.",
                handler=probe_camera_preview,
            ),
            "probe_nonce": AgentTool(
                name="probe_nonce",
                description="Echo an expected nonce to prove one bounded tool dispatch.",
                handler=probe_nonce,
            ),
        }
        return [tools[name] for name in self._required_tools]

    def _build_fake_bundle(self) -> Any:
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
        bundle = LinkBotG20ShadowBundle(
            side=ArmSide.RIGHT,
            arm_gateway=FakeArmSdkGateway(world, FakeGatewayFaults()),
            hand_gateway=FakeG20SdkGateway(world, mapper, FakeGatewayFaults()),
            clock=clock,
            evidence_source="p3-fake-gateway",
        )
        bundle.open()
        return bundle

    def _probe_fake_gateway_ik(
        self, target: tuple[float, float, float]
    ) -> Mapping[str, Any]:
        from proto import ArmSide, Pose

        if self._fake_bundle is None:
            raise RuntimeError("P3 fake gateway is unavailable")
        side = ArmSide.RIGHT
        state = self._fake_bundle.arm.get_state(side)
        target_pose = Pose(target, (1.0, 0.0, 0.0, 0.0), "world")
        ik = self._fake_bundle.arm.ik(side, target_pose, state.joint_positions_rad)
        if not ik.success:
            raise RecoverableToolError(f"fake gateway IK failed: {ik.error_code}")
        fk = self._fake_bundle.arm.fk(side, ik.joint_positions_rad)
        if not fk.success or fk.pose is None:
            raise RecoverableToolError("fake gateway FK back-check failed")
        residual = [fk.pose.position_m[index] - target[index] for index in range(3)]
        return {
            "solution_found": True,
            "joint_positions_rad": list(ik.joint_positions_rad),
            "position_residual_m": residual,
            "max_position_residual_m": max(abs(value) for value in residual),
            "collision_checked": False,
            "solver": "fake_gateway",
        }

    def build_input(self, observation: Observation) -> Mapping[str, Any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Run {self._stage} offline validation at observation "
                        f"sequence={observation.sequence}. No live device is available."
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
            if self._fake_bundle is not None:
                self._fake_bundle.close()
        finally:
            super().close()


def _normalize_required_tools(
    required_tools: Sequence[str], supported: frozenset[str]
) -> tuple[str, ...]:
    if isinstance(required_tools, (str, bytes, bytearray)):
        raise TypeError("required_tools must be a sequence of tool names")
    required = tuple(str(name).strip() for name in required_tools)
    if not required or any(not name for name in required):
        raise ValueError("required_tools must contain at least one tool")
    unknown = set(required) - supported
    if unknown:
        raise ValueError("unsupported validation tools: " + ", ".join(sorted(unknown)))
    if len(set(required)) != len(required):
        raise ValueError("required_tools must not contain duplicates")
    return required


def _finite_target(values: Sequence[object]) -> tuple[float, float, float]:
    if len(values) != 3 or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise RecoverableToolError("IK target coordinates must be finite numbers")
    target = (float(values[0]), float(values[1]), float(values[2]))
    if not all(math.isfinite(value) for value in target):
        raise RecoverableToolError("IK target coordinates must be finite numbers")
    return target


__all__ = ["HardwareCapabilityValidationTask", "ValidationProbeState"]
