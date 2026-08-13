"""P0/P1 acceptance tests for the offline API-to-skill validation Task."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from module.agent import (
    AgentTool,
    ArkOpenAIResponsesModel,
    TopAgent,
)
from module.runtime import RuntimeContext
from module.tools.env import MockEnv
from script.run import build_runtime_from_file
from task.api_validation import HardwareCapabilityValidationTask


ROOT = Path(__file__).resolve().parents[1]
TEST_API_KEY = "p1-scripted-provider-key"


class RecordingTransport:
    """Minimal scripted provider transport; it never opens a network socket."""

    def __init__(self, responses: Sequence[Mapping[str, object]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def post_json(self, url: str, *, headers: Mapping[str, str], payload: Mapping[str, object], timeout_s: float) -> Mapping[str, object]:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": json.loads(json.dumps(payload)),
                "timeout_s": timeout_s,
            }
        )
        if not self._responses:
            raise AssertionError("scripted provider called more times than expected")
        return self._responses.pop(0)


def _observation_env() -> MockEnv:
    return MockEnv(observations=[{"sequence": 0, "timestamp_s": 0.0}])


class P0OfflineAcceptanceTests(unittest.TestCase):
    def test_offline_configuration_completes_and_reports_no_hardware_access(self) -> None:
        runtime = build_runtime_from_file(ROOT / "config/hardware_capability_validation_offline.yaml")
        report = runtime.run(seed=7)

        self.assertTrue(report.completed)
        self.assertEqual(report.control_steps, 1)
        self.assertIsNotNone(report.last_agent_result)
        assert report.last_agent_result is not None
        tool_messages = [
            message
            for message in report.last_agent_result["messages"]
            if isinstance(message, Mapping) and message.get("role") == "tool"
        ]
        self.assertEqual(
            [message["name"] for message in tool_messages],
            [
                "get_validation_capabilities",
                "probe_inverse_kinematics",
                "probe_camera_preview",
            ],
        )
        for message in tool_messages:
            self.assertEqual(message["status"], "success")
            content = message["content"]
            self.assertIsInstance(content, Mapping)
            self.assertTrue(content["simulated"])
            self.assertFalse(content["hardware_accessed"])

    def test_probe_allowlist_is_read_only_and_invalid_ik_is_recoverable(self) -> None:
        task = HardwareCapabilityValidationTask(stage="p0")
        env = _observation_env()
        context = RuntimeContext(env)
        observation = env.reset()
        context.update_observation(observation)
        tools = task.build_tools(context)
        self.assertEqual(
            {tool.name for tool in tools},
            {
                "get_validation_capabilities",
                "probe_inverse_kinematics",
                "probe_camera_preview",
            },
        )
        self.assertTrue(all("url" not in json.dumps(tool.spec.as_dict()).lower() for tool in tools))

        agent = TopAgent(
            model=_offline_model(
                [
                    {
                        "name": "probe_inverse_kinematics",
                        "arguments": {"target_x_m": 4.0, "target_y_m": 0.0, "target_z_m": 0.0},
                    }
                ]
            ),
            tools=tools,
            max_iterations=2,
            max_tool_calls=1,
            max_tool_calls_per_turn=1,
        )
        result = agent.invoke(task.build_input(observation))
        tool_message = next(message for message in result["messages"] if message.get("role") == "tool")
        self.assertEqual(tool_message["status"], "error")
        self.assertEqual(task.state.ik_calls, 0)
        task.close()
        env.close()
        self.assertTrue(task.state.closed)


class P1ScriptedApiRoundTripTests(unittest.TestCase):
    def test_openai_responses_round_trip_dispatches_ik_skill_without_motion(self) -> None:
        transport = RecordingTransport(
            [
                {
                    "id": "resp-p1-1",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-p1-ik",
                            "name": "probe_inverse_kinematics",
                            "arguments": '{"target_x_m":0.18,"target_y_m":-0.04,"target_z_m":0.32}',
                        }
                    ],
                },
                {
                    "id": "resp-p1-2",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "offline IK probe accepted"}],
                        }
                    ],
                },
            ]
        )
        task = HardwareCapabilityValidationTask(stage="p1", required_tools=("probe_inverse_kinematics",))
        env = _observation_env()
        context = RuntimeContext(env)
        observation = env.reset()
        context.update_observation(observation)
        with patch.dict(os.environ, {"ARK_API_KEY": TEST_API_KEY}):
            model = ArkOpenAIResponsesModel(
                model="scripted-p1-model",
                transport=transport,
                timeout_s=2.0,
            )
            result = TopAgent(
                model=model,
                tools=task.build_tools(context),
                system_prompt=task.build_prompt(),
                max_iterations=3,
                max_tool_calls=1,
                max_tool_calls_per_turn=1,
            ).invoke(task.build_input(observation))

        self.assertEqual(result["output"], "offline IK probe accepted")
        self.assertTrue(task.is_complete(context, result))
        self.assertEqual(task.state.ik_calls, 1)
        self.assertEqual(len(transport.calls), 2)
        function_output = transport.calls[1]["payload"]["input"][0]
        self.assertEqual(function_output["call_id"], "call-p1-ik")
        decoded = json.loads(function_output["output"])
        self.assertEqual(decoded["status"], "success")
        self.assertEqual(decoded["content"]["execution_id"], "p1-probe_inverse_kinematics-0001")
        self.assertFalse(decoded["content"]["motion_executed"])
        self.assertFalse(decoded["content"]["hardware_accessed"])
        task.close()
        env.close()


class P3FakeGatewayReplayTests(unittest.TestCase):
    def test_fake_gateway_ik_fk_and_replay_are_read_only(self) -> None:
        task = HardwareCapabilityValidationTask(
            stage="p3",
            required_tools=("probe_inverse_kinematics", "probe_camera_preview"),
        )
        env = _observation_env()
        context = RuntimeContext(env)
        observation = env.reset()
        context.update_observation(observation)
        tools = {tool.name: tool for tool in task.build_tools(context)}
        ik = tools["probe_inverse_kinematics"].invoke(
            {"target_x_m": 0.0, "target_y_m": 0.0, "target_z_m": 0.70}
        )
        camera = tools["probe_camera_preview"].invoke(
            {"camera_name": "wrist", "frame_id": 0}
        )
        self.assertTrue(ik["solution_found"])
        self.assertEqual(ik["max_position_residual_m"], 0.0)
        self.assertFalse(ik["motion_executed"])
        self.assertFalse(ik["hardware_accessed"])
        self.assertEqual(camera["device_id"], "orbbec-replay-001")
        self.assertTrue(camera["rgb_live"])
        self.assertTrue(camera["depth_live"])
        self.assertFalse(camera["preview_opened"])
        task.close()
        env.close()
        self.assertTrue(task.state.closed)


def _offline_model(tool_sequence: Sequence[Mapping[str, object]]):
    from module.agent.models.mock import MockChatModel

    return MockChatModel(tool_sequence=tool_sequence, final_output="offline probe")


if __name__ == "__main__":
    unittest.main()
