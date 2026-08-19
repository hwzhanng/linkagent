"""Offline acceptance tests for Agent capability and hardware-read-only Tasks."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from module.agent import RecoverableToolError
from module.runtime import CompletionReason, RuntimeContext
from module.tools.env import MockEnv
from script.run import build_runtime_from_file
from task.api_validation import (
    HardwareCapabilityValidationTask,
    HardwareReadOnlyValidationTask,
)
from task.api_validation.live import HARDWARE_ADMISSION_ENV
from task.factory import TASK_REGISTRY


def _context() -> tuple[MockEnv, RuntimeContext]:
    env = MockEnv(observations=[{"sequence": 0, "timestamp_s": 0.0}])
    context = RuntimeContext(env)
    context.update_observation(env.reset())
    return env, context


def test_offline_configuration_completes_without_hardware() -> None:
    runtime = build_runtime_from_file("config/hardware_capability_validation_offline.yaml")
    report = runtime.run(seed=7)

    assert report.completed
    assert report.control_steps == 1
    assert report.last_agent_result is not None
    tool_messages = [
        message
        for message in report.last_agent_result["messages"]
        if message.get("role") == "tool"
    ]
    assert [message["name"] for message in tool_messages] == [
        "get_validation_capabilities",
        "probe_inverse_kinematics",
        "probe_camera_preview",
    ]
    assert all(message["status"] == "success" for message in tool_messages)
    assert all(message["content"]["simulated"] is True for message in tool_messages)
    assert all(
        message["content"]["hardware_accessed"] is False
        for message in tool_messages
    )
    assert all(
        message["content"]["motion_executed"] is False
        for message in tool_messages
    )


def test_checked_in_live_configuration_keeps_hardware_admission_closed() -> None:
    runtime = build_runtime_from_file("config/hardware_read_only_validation.yaml")
    with patch.dict("os.environ", {}, clear=True):
        report = runtime.run(seed=7)

    assert report.completed is False
    assert report.reason is CompletionReason.STEP_LIMIT
    assert report.last_agent_result is not None
    tool_messages = [
        message
        for message in report.last_agent_result["messages"]
        if message.get("role") == "tool"
    ]
    assert len(tool_messages) == 3
    assert all(message["status"] == "error" for message in tool_messages)
    assert all(
        "disabled by Task configuration"
        in message["content"]["error"]["message"]
        for message in tool_messages
    )


def test_offline_probe_allowlist_and_p3_fake_gateway_are_read_only() -> None:
    task = HardwareCapabilityValidationTask(
        stage="p3",
        required_tools=("probe_inverse_kinematics", "probe_camera_preview"),
    )
    env, context = _context()
    tools = {tool.name: tool for tool in task.build_tools(context)}

    assert set(tools) == {"probe_inverse_kinematics", "probe_camera_preview"}
    ik = tools["probe_inverse_kinematics"].invoke(
        {"target_x_m": 0.0, "target_y_m": 0.0, "target_z_m": 0.70}
    )
    camera = tools["probe_camera_preview"].invoke(
        {"camera_name": "wrist", "frame_id": 0}
    )

    assert ik["solution_found"] is True
    assert ik["max_position_residual_m"] == 0.0
    assert ik["hardware_accessed"] is False
    assert ik["motion_executed"] is False
    assert camera["payload_ref"] == "replay://camera/wrist/0"
    assert camera["network_accessed"] is False
    task.close()
    task.close()
    env.close()
    assert task.state.closed


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_live_arm_state", {"side": "right"}),
        (
            "compute_live_shadow_ik",
            {"side": "right", "target": "z_plus_5mm"},
        ),
        ("get_live_camera_preview", {}),
    ],
)
def test_live_tools_fail_before_adapter_construction_when_config_gate_is_closed(
    name: str, arguments: dict[str, str]
) -> None:
    constructed = 0

    def forbidden_factory(deployment):
        del deployment
        nonlocal constructed
        constructed += 1
        raise AssertionError("hardware adapter must not be constructed")

    task = HardwareReadOnlyValidationTask(
        hardware_access_enabled=False,
        required_tools=(name,),
        session_factory=forbidden_factory,
        viewer_fetcher=lambda _url: (_ for _ in ()).throw(
            AssertionError("camera adapter must not be called")
        ),
    )
    env, context = _context()
    tool = task.build_tools(context)[0]

    with patch.dict("os.environ", {HARDWARE_ADMISSION_ENV: "1"}):
        with pytest.raises(RecoverableToolError, match="disabled by Task"):
            tool.invoke(arguments)

    assert constructed == 0
    task.close()
    env.close()


def test_live_tools_require_environment_admission_before_adapter_construction() -> None:
    task = HardwareReadOnlyValidationTask(
        hardware_access_enabled=True,
        required_tools=("get_live_arm_state",),
        session_factory=lambda _deployment: (_ for _ in ()).throw(
            AssertionError("hardware adapter must not be constructed")
        ),
    )
    env, context = _context()
    tool = task.build_tools(context)[0]

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RecoverableToolError, match=HARDWARE_ADMISSION_ENV):
            tool.invoke({"side": "right"})

    task.close()
    env.close()


def test_three_live_tools_use_one_fake_side_and_never_authorize_motion() -> None:
    session = _FakeSession()
    viewer_calls: list[str] = []

    def viewer_fetcher(url: str):
        viewer_calls.append(url)
        return {
            "state": "running",
            "device": {"name": "Orbbec Gemini Pro", "serial": "fake-001"},
            "streams": {
                "rgb": {"live": True, "width": 1280, "height": 720},
                "depth": {"live": True, "width": 1280, "height": 800},
            },
        }

    task = HardwareReadOnlyValidationTask(
        hardware_access_enabled=True,
        required_tools=(
            "get_live_arm_state",
            "compute_live_shadow_ik",
            "get_live_camera_preview",
        ),
        session_factory=lambda _deployment: session,
        viewer_fetcher=viewer_fetcher,
    )
    env, context = _context()
    tools = {tool.name: tool for tool in task.build_tools(context)}

    with patch.dict("os.environ", {HARDWARE_ADMISSION_ENV: "1"}):
        state = tools["get_live_arm_state"].invoke({"side": "right"})
        ik = tools["compute_live_shadow_ik"].invoke(
            {"side": "right", "target": "z_plus_5mm"}
        )
        camera = tools["get_live_camera_preview"].invoke({})

    assert session.requested_sides == ["right", "right"]
    assert state["all_joints_healthy"] is True
    assert state["fk_pose"]["frame_id"] == "world"
    assert ik["solution_found"] is True
    assert ik["within_limits"] is True
    assert ik["collision_checked"] is False
    assert camera["rgb_depth_live"] is True
    assert viewer_calls == ["http://127.0.0.1:8080/api/status"]
    for result in (state, ik, camera):
        assert result["hardware_accessed"] is True
        assert result["telemetry_query_frames_authorized"] is True
        assert result["write_control_authorized"] is False
        assert result["motion_executed"] is False

    task.close()
    task.close()
    env.close()
    assert session.closed


def test_live_task_rejects_non_loopback_camera_and_preserves_task_registry() -> None:
    with pytest.raises(ValueError, match="loopback"):
        HardwareReadOnlyValidationTask(
            viewer_status_url="http://192.168.1.10:8080/api/status"
        )

    assert {
        "locate_target",
        "agri_pick",
        "harvest_cycle",
        "hardware_capability_validation",
        "hardware_read_only_validation",
    }.issubset(TASK_REGISTRY)


class _FakeController:
    def __init__(self) -> None:
        self.kinematics = SimpleNamespace(
            joint_limits=(tuple([-3.0] * 7), tuple([3.0] * 7))
        )

    def forward(self, joints):
        joints = tuple(float(value) for value in joints)
        return SimpleNamespace(
            position=(0.40 + joints[0] * 0.01, -0.10, 0.70),
            rpy=(0.0, 0.0, 0.0),
        )

    def inverse(self, target_position, target_rpy, seed):
        del target_rpy
        seed = tuple(float(value) for value in seed)
        desired_first = (float(target_position[0]) - 0.40) / 0.01
        joints = (desired_first, *seed[1:])
        return SimpleNamespace(
            success=True,
            joints=joints,
            position_error=0.0,
            rotation_error=0.0,
            iterations=2,
        )


class _FakeSession:
    def __init__(self) -> None:
        self.requested_sides: list[str] = []
        self.closed = False
        self.controller = _FakeController()

    def snapshot(self, side: str):
        self.requested_sides.append(side)
        rows = [
            {
                "joint_number": index + 1,
                "position": 0.05 * index,
                "healthy": True,
            }
            for index in range(7)
        ]
        return (
            {
                "interface": "fake-can1",
                "arm": side,
                "healthy_joint_count": 7,
                "all_joints_healthy": True,
                "software_enabled": False,
                "last_error": None,
                "joints": rows,
            },
            self.controller,
        )

    def close(self) -> None:
        self.closed = True
