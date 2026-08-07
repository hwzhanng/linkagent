"""Offline contract tests for Ark-compatible provider model adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import io
import json
import os
from pathlib import Path
from typing import Any, Literal
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from module.agent import (
    AgentTool,
    AgentToolSpec,
    ArkAnthropicMessagesModel,
    ArkOpenAIResponsesModel,
    ProviderConfigurationError,
    ProviderProtocolError,
    ProviderTransportError,
    TopAgent,
    UrllibJsonTransport,
)
from module.tools.env import ENV_REGISTRY, MockEnv
from script.run import build_runtime_from_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_API_KEY = "provider-test-key"


class RecordingTransport:
    """Return scripted JSON responses and retain detached request snapshots."""

    def __init__(self, responses: Sequence[Mapping[str, Any] | BaseException]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": json.loads(json.dumps(payload, ensure_ascii=False)),
                "timeout_s": timeout_s,
            }
        )
        if not self._responses:
            raise AssertionError("fake provider was called too many times")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self._body[:limit]


class ToolSchemaTests(unittest.TestCase):
    def test_tool_schema_is_inferred_detached_and_handler_free(self) -> None:
        def lookup_weather(
            city: str,
            days: int = 1,
            units: Literal["c", "f"] = "c",
        ) -> dict[str, object]:
            return {"city": city, "days": days, "units": units}

        tool = AgentTool(
            "lookup_weather",
            "Look up a deterministic weather record.",
            lookup_weather,
        )
        exposed = tool.spec.as_dict()

        self.assertNotIn("handler", exposed)
        self.assertEqual(exposed["parameters"]["type"], "object")
        self.assertEqual(exposed["parameters"]["required"], ["city"])
        self.assertEqual(
            exposed["parameters"]["properties"],
            {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "units": {"enum": ["c", "f"]},
            },
        )
        self.assertFalse(exposed["parameters"]["additionalProperties"])

        exposed["parameters"]["properties"]["city"]["type"] = "number"
        self.assertEqual(
            tool.spec.as_dict()["parameters"]["properties"]["city"]["type"],
            "string",
        )
        with self.assertRaises(TypeError):
            tool.parameters["properties"] = {}  # type: ignore[index]

    def test_invalid_or_secret_bearing_tool_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "root type must be 'object'"):
            AgentToolSpec(
                "bad_schema",
                "Invalid schema.",
                parameters={"type": "array", "properties": {}},
            )
        with self.assertRaisesRegex(ValueError, "not defined"):
            AgentToolSpec(
                "bad_required",
                "Invalid required field.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": ["missing"],
                },
            )
        with self.assertRaises(TypeError):
            AgentToolSpec(
                "secret_input",
                "A model must never be asked for provider credentials.",
                parameters={
                    "type": "object",
                    "properties": {"api_key": {"type": "string"}},
                },
            )


class OpenAIResponsesAdapterTests(unittest.TestCase):
    def test_top_agent_performs_complete_responses_tool_round_trip(self) -> None:
        transport = RecordingTransport(
            (
                {
                    "id": "resp-1",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-weather-1",
                            "name": "lookup_weather",
                            "arguments": '{"city":"北京"}',
                        }
                    ],
                },
                {
                    "id": "resp-2",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "工具结果已确认。"}
                            ],
                        }
                    ],
                },
            )
        )
        handler_calls: list[str] = []

        def lookup_weather(city: str) -> dict[str, str]:
            handler_calls.append(city)
            return {"city": city, "condition": "sunny"}

        with patch.dict(os.environ, {"ARK_API_KEY": TEST_API_KEY}):
            model = ArkOpenAIResponsesModel(
                model="doubao-seed-2.0-pro",
                transport=transport,
                timeout_s=12.5,
                max_output_tokens=512,
            )
            self.assertNotIn(TEST_API_KEY, repr(vars(model)))
            result = TopAgent(
                model,
                (
                    AgentTool(
                        "lookup_weather",
                        "Return a weather record for one city.",
                        lookup_weather,
                    ),
                ),
                system_prompt="Use the allowed tool.",
            ).invoke(
                {"messages": [{"role": "user", "content": "查询北京天气"}]}
            )

        self.assertEqual(handler_calls, ["北京"])
        self.assertEqual(result["output"], "工具结果已确认。")
        self.assertEqual(len(transport.calls), 2)
        first, second = transport.calls
        self.assertEqual(
            first["url"],
            "https://ark.cn-beijing.volces.com/api/coding/v3/responses",
        )
        self.assertEqual(first["headers"]["Authorization"], f"Bearer {TEST_API_KEY}")
        self.assertEqual(first["payload"]["model"], "doubao-seed-2.0-pro")
        self.assertTrue(first["payload"]["store"])
        self.assertEqual(first["payload"]["instructions"], "Use the allowed tool.")
        self.assertEqual(
            first["payload"]["tools"][0]["parameters"]["required"],
            ["city"],
        )
        self.assertNotIn("handler", json.dumps(first["payload"]))
        self.assertEqual(second["payload"]["previous_response_id"], "resp-1")
        self.assertNotIn("tools", second["payload"])
        output_item = second["payload"]["input"][0]
        self.assertEqual(output_item["type"], "function_call_output")
        self.assertEqual(output_item["call_id"], "call-weather-1")
        decoded_output = json.loads(output_item["output"])
        self.assertEqual(decoded_output["status"], "success")
        self.assertEqual(decoded_output["content"]["condition"], "sunny")

    def test_responses_protocol_rejects_malformed_critical_fields(self) -> None:
        malformed = (
            (
                {"status": "completed", "output": []},
                "non-empty id",
            ),
            (
                {"id": "resp", "status": "failed", "output": []},
                "failed response",
            ),
            (
                {
                    "id": "resp",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "safe_tool",
                            "arguments": "not-json",
                        }
                    ],
                },
                "not valid JSON",
            ),
            (
                {
                    "id": "resp",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "Invalid Tool Name",
                            "arguments": "{}",
                        }
                    ],
                },
                "invalid tool call",
            ),
            (
                {
                    "id": "resp",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "safe_tool",
                            "arguments": "{}",
                        },
                        {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "safe_tool",
                            "arguments": "{}",
                        },
                    ],
                },
                "duplicate tool call ids",
            ),
        )
        for response, expected in malformed:
            with self.subTest(expected=expected):
                transport = RecordingTransport((response,))
                with patch.dict(os.environ, {"ARK_API_KEY": TEST_API_KEY}):
                    model = ArkOpenAIResponsesModel(
                        model="doubao-seed-2.0-pro",
                        transport=transport,
                    )
                    with self.assertRaisesRegex(ProviderProtocolError, expected):
                        model.invoke(
                            ({"role": "user", "content": "hello"},),
                            (),
                        )


class AnthropicAdapterTests(unittest.TestCase):
    def test_top_agent_performs_complete_anthropic_tool_round_trip(self) -> None:
        transport = RecordingTransport(
            (
                {
                    "id": "msg-1",
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu-weather-1",
                            "name": "lookup_weather",
                            "input": {"city": "北京"},
                        }
                    ],
                },
                {
                    "id": "msg-2",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "工具结果已确认。"}],
                },
            )
        )

        def lookup_weather(city: str) -> dict[str, str]:
            return {"city": city, "condition": "sunny"}

        with patch.dict(os.environ, {"ARK_API_KEY": TEST_API_KEY}):
            model = ArkAnthropicMessagesModel(
                model="doubao-seed-2.0-pro",
                transport=transport,
            )
            result = TopAgent(
                model,
                (
                    AgentTool(
                        "lookup_weather",
                        "Return a weather record for one city.",
                        lookup_weather,
                    ),
                ),
                system_prompt="Use the allowed tool.",
            ).invoke(
                {"messages": [{"role": "user", "content": "查询北京天气"}]}
            )

        self.assertEqual(result["output"], "工具结果已确认。")
        self.assertEqual(len(transport.calls), 2)
        first, second = transport.calls
        self.assertEqual(
            first["url"],
            "https://ark.cn-beijing.volces.com/api/coding/v1/messages",
        )
        self.assertEqual(first["headers"]["Authorization"], f"Bearer {TEST_API_KEY}")
        self.assertEqual(first["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(first["payload"]["system"], "Use the allowed tool.")
        self.assertEqual(
            first["payload"]["tools"][0]["input_schema"]["required"],
            ["city"],
        )
        self.assertEqual(
            [message["role"] for message in second["payload"]["messages"]],
            ["user", "assistant", "user"],
        )
        assistant_blocks = second["payload"]["messages"][1]["content"]
        self.assertEqual(assistant_blocks[0]["type"], "tool_use")
        tool_result = second["payload"]["messages"][2]["content"][0]
        self.assertEqual(tool_result["type"], "tool_result")
        self.assertEqual(tool_result["tool_use_id"], "toolu-weather-1")
        self.assertFalse(tool_result["is_error"])

    def test_parallel_anthropic_tool_results_share_one_user_turn(self) -> None:
        transport = RecordingTransport(
            (
                {
                    "id": "msg-1",
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu-1",
                            "name": "first_tool",
                            "input": {},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu-2",
                            "name": "second_tool",
                            "input": {},
                        },
                    ],
                },
                {
                    "id": "msg-2",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "done"}],
                },
            )
        )
        with patch.dict(os.environ, {"ARK_API_KEY": TEST_API_KEY}):
            model = ArkAnthropicMessagesModel(
                model="doubao-seed-2.0-pro",
                transport=transport,
            )
            TopAgent(
                model,
                (
                    AgentTool("first_tool", "Return first.", lambda: "first"),
                    AgentTool("second_tool", "Return second.", lambda: "second"),
                ),
            ).invoke({"messages": [{"role": "user", "content": "run both"}]})

        second_messages = transport.calls[1]["payload"]["messages"]
        self.assertEqual([item["role"] for item in second_messages], ["user", "assistant", "user"])
        result_blocks = second_messages[-1]["content"]
        self.assertEqual(len(result_blocks), 2)
        self.assertEqual(
            {block["tool_use_id"] for block in result_blocks},
            {"toolu-1", "toolu-2"},
        )

    def test_anthropic_protocol_rejects_empty_or_malformed_content(self) -> None:
        for response, expected in (
            ({"content": []}, "neither text nor tool_use"),
            ({"content": [{"type": "tool_use", "id": "x", "name": "safe_tool", "input": []}]}, "input must be an object"),
            ({"content": [{"type": "tool_use", "id": "x", "name": "Invalid Tool Name", "input": {}}]}, "invalid tool call"),
            (
                {
                    "content": [
                        {"type": "tool_use", "id": "x", "name": "safe_tool", "input": {}},
                        {"type": "tool_use", "id": "x", "name": "safe_tool", "input": {}},
                    ]
                },
                "duplicate tool call ids",
            ),
        ):
            with self.subTest(expected=expected):
                transport = RecordingTransport((response,))
                with patch.dict(os.environ, {"ARK_API_KEY": TEST_API_KEY}):
                    model = ArkAnthropicMessagesModel(
                        model="doubao-seed-2.0-pro",
                        transport=transport,
                    )
                    with self.assertRaisesRegex(ProviderProtocolError, expected):
                        model.invoke(
                            ({"role": "user", "content": "hello"},),
                            (),
                        )


class ProviderBoundaryTests(unittest.TestCase):
    def test_missing_key_and_unsafe_base_url_fail_before_network(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                ProviderConfigurationError,
                "ARK_API_KEY.*not set",
            ):
                ArkOpenAIResponsesModel(model="doubao-seed-2.0-pro")
        with patch.dict(os.environ, {"ARK_API_KEY": TEST_API_KEY}):
            with self.assertRaisesRegex(ValueError, "must use https"):
                ArkOpenAIResponsesModel(
                    model="doubao-seed-2.0-pro",
                    base_url="http://example.com/v3",
                )
            with self.assertRaisesRegex(ValueError, "cannot contain credentials"):
                ArkOpenAIResponsesModel(
                    model="doubao-seed-2.0-pro",
                    base_url="https://user:password@example.com/v3",
                )

    def test_missing_key_is_detected_before_environment_creation(self) -> None:
        created_envs: list[MockEnv] = []

        def build_tracking_env(**options: Any) -> MockEnv:
            env = MockEnv(**options)
            created_envs.append(env)
            return env

        with (
            patch.dict(ENV_REGISTRY, {"mock": build_tracking_env}),
            patch.dict(os.environ, {}, clear=True),
        ):
            with self.assertRaises(ProviderConfigurationError):
                build_runtime_from_file(
                    PROJECT_ROOT / "config" / "ark_openai.yaml"
                )
        self.assertEqual(created_envs, [])

    def test_real_profiles_build_without_network_and_default_remains_offline(self) -> None:
        with patch.dict(os.environ, {"ARK_API_KEY": TEST_API_KEY}):
            openai_runtime = build_runtime_from_file(
                PROJECT_ROOT / "config" / "ark_openai.yaml"
            )
            anthropic_runtime = build_runtime_from_file(
                PROJECT_ROOT / "config" / "ark_anthropic.yaml"
            )
        openai_runtime.close()
        anthropic_runtime.close()

        with patch.dict(os.environ, {}, clear=True):
            default_runtime = build_runtime_from_file(
                PROJECT_ROOT / "config" / "default.yaml"
            )
            report = default_runtime.run()
        self.assertTrue(report.completed)

    def test_transport_redacts_http_errors_and_wraps_network_failures(self) -> None:
        transport = UrllibJsonTransport()
        http_error = HTTPError(
            "https://example.com/responses",
            401,
            "unauthorized",
            hdrs=None,
            fp=io.BytesIO(
                json.dumps(
                    {"error": {"message": "api_key='LEAKED TEST SECRET'"}}
                ).encode("utf-8")
            ),
        )
        with patch("module.agent.models._http.urlopen", side_effect=http_error):
            with self.assertRaises(ProviderTransportError) as caught:
                transport.post_json(
                    "https://example.com/responses",
                    headers={"Authorization": f"Bearer {TEST_API_KEY}"},
                    payload={"model": "test"},
                    timeout_s=1.0,
                )
        self.assertNotIn("LEAKED TEST SECRET", str(caught.exception))
        self.assertIn("<redacted>", str(caught.exception))

        with patch(
            "module.agent.models._http.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with self.assertRaisesRegex(
                ProviderTransportError,
                "connection refused",
            ):
                transport.post_json(
                    "https://example.com/responses",
                    headers={"Authorization": f"Bearer {TEST_API_KEY}"},
                    payload={"model": "test"},
                    timeout_s=1.0,
                )

    def test_transport_enforces_response_body_limit(self) -> None:
        transport = UrllibJsonTransport(max_response_bytes=1_024)
        oversized = b"{" + (b"x" * 1_024) + b"}"
        with patch(
            "module.agent.models._http.urlopen",
            return_value=FakeHTTPResponse(oversized),
        ):
            with self.assertRaisesRegex(
                ProviderProtocolError,
                "exceeded the configured byte limit",
            ):
                transport.post_json(
                    "https://example.com/responses",
                    headers={"Authorization": f"Bearer {TEST_API_KEY}"},
                    payload={"model": "test"},
                    timeout_s=1.0,
                )


if __name__ == "__main__":
    unittest.main()
