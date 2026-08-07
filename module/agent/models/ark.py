"""Ark Coding Plan adapters for OpenAI Responses and Anthropic Messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
import re
from typing import Any

from module.agent._contracts import AgentToolSpec, ModelTurn, ToolCall
from module.agent.models._http import (
    JsonTransport,
    ProviderConfigurationError,
    ProviderProtocolError,
    UrllibJsonTransport,
    validate_provider_base_url,
)
from module.agent.models.base import BaseChatModel


_ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_MAX_ARGUMENT_BYTES = 65_536


class _ArkModelBase(BaseChatModel):
    """Shared validated configuration for credential-safe Ark adapters."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str = "ARK_API_KEY",
        timeout_s: float = 60.0,
        max_output_tokens: int = 1_024,
        transport: JsonTransport | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("provider model must be a non-empty string")
        normalized_model = model.strip()
        if len(normalized_model) > 256 or any(
            ord(character) < 32 for character in normalized_model
        ):
            raise ValueError(
                "provider model must contain at most 256 printable characters"
            )
        if not isinstance(api_key_env, str) or not _ENV_NAME_PATTERN.fullmatch(
            api_key_env
        ):
            raise ValueError(
                "api_key_env must be an upper-case environment variable name"
            )
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TypeError("timeout_s must be a number")
        if not 0.1 <= float(timeout_s) <= 300.0:
            raise ValueError("timeout_s must be between 0.1 and 300")
        if isinstance(max_output_tokens, bool) or not isinstance(
            max_output_tokens, int
        ):
            raise TypeError("max_output_tokens must be an integer")
        if not 1 <= max_output_tokens <= 65_536:
            raise ValueError("max_output_tokens must be between 1 and 65536")
        selected_transport = transport or UrllibJsonTransport()
        if not isinstance(selected_transport, JsonTransport):
            raise TypeError("transport must implement JsonTransport")

        self._model_name = normalized_model
        self._base_url = validate_provider_base_url(base_url)
        self._api_key_env = api_key_env
        self._timeout_s = float(timeout_s)
        self._max_output_tokens = max_output_tokens
        self._transport = selected_transport
        # Fail before the composition root acquires Env/Task resources. The
        # actual credential is deliberately not retained on the model object.
        self._require_api_key()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def api_key_env(self) -> str:
        return self._api_key_env

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    @property
    def max_output_tokens(self) -> int:
        return self._max_output_tokens

    def _require_api_key(self) -> str:
        api_key = os.environ.get(self._api_key_env)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ProviderConfigurationError(
                f"required provider credential environment variable "
                f"{self._api_key_env!r} is not set"
            )
        normalized = api_key.strip()
        if len(normalized) > 4_096 or any(
            not 33 <= ord(character) <= 126 for character in normalized
        ):
            raise ProviderConfigurationError(
                f"provider credential in {self._api_key_env!r} is invalid"
            )
        return normalized

    def _post(
        self,
        endpoint: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._transport.post_json(
            f"{self._base_url}{endpoint}",
            headers=headers,
            payload=payload,
            timeout_s=self._timeout_s,
        )


class ArkOpenAIResponsesModel(_ArkModelBase):
    """OpenAI-compatible Responses API adapter with stateful tool continuation."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/coding/v3",
        api_key_env: str = "ARK_API_KEY",
        timeout_s: float = 60.0,
        max_output_tokens: int = 1_024,
        transport: JsonTransport | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            timeout_s=timeout_s,
            max_output_tokens=max_output_tokens,
            transport=transport,
        )
        self._previous_response_id: str | None = None
        self._submitted_tool_call_ids: set[str] = set()

    def reset(self) -> None:
        self._previous_response_id = None
        self._submitted_tool_call_ids.clear()

    def invoke(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[AgentToolSpec],
    ) -> ModelTurn:
        _validate_model_inputs(messages, tools)
        submitted_ids: tuple[str, ...] = ()
        if self._previous_response_id is None:
            payload = self._initial_payload(messages, tools)
        else:
            tool_outputs, submitted_ids = self._pending_tool_outputs(messages)
            if not tool_outputs:
                raise ProviderProtocolError(
                    "Responses continuation requires at least one new tool output"
                )
            payload = {
                "model": self._model_name,
                "previous_response_id": self._previous_response_id,
                "input": tool_outputs,
                "max_output_tokens": self._max_output_tokens,
            }

        api_key = self._require_api_key()
        response = self._post(
            "/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "realagent/0.2",
            },
            payload=payload,
        )
        turn, response_id = _parse_openai_response(response)
        self._previous_response_id = response_id
        self._submitted_tool_call_ids.update(submitted_ids)
        return turn

    def _initial_payload(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[AgentToolSpec],
    ) -> dict[str, Any]:
        instructions: list[str] = []
        input_items: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role in {"system", "developer"}:
                instructions.append(_content_text(message.get("content", "")))
                continue
            if role not in {"user", "assistant"}:
                raise ProviderProtocolError(
                    f"unsupported initial Responses message role {role!r}"
                )
            input_items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": _content_text(message.get("content", "")),
                }
            )
        if not input_items:
            raise ProviderProtocolError(
                "Responses request requires at least one user/assistant message"
            )
        payload: dict[str, Any] = {
            "model": self._model_name,
            "store": True,
            "input": input_items,
            "max_output_tokens": self._max_output_tokens,
        }
        if instructions:
            payload["instructions"] = "\n\n".join(instructions)
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.as_dict()["parameters"],
                }
                for tool in tools
            ]
        return payload

    def _pending_tool_outputs(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
        outputs: list[dict[str, Any]] = []
        submitted_ids: list[str] = []
        for message in messages:
            if message.get("role") != "tool":
                continue
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ProviderProtocolError(
                    "tool result message requires a non-empty tool_call_id"
                )
            if call_id in self._submitted_tool_call_ids:
                continue
            output = json.dumps(
                {
                    "status": message.get("status", "success"),
                    "content": message.get("content"),
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
            )
            submitted_ids.append(call_id)
        return outputs, tuple(submitted_ids)


class ArkAnthropicMessagesModel(_ArkModelBase):
    """Anthropic-compatible Messages API adapter using full neutral history."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/coding",
        api_key_env: str = "ARK_API_KEY",
        timeout_s: float = 60.0,
        max_output_tokens: int = 1_024,
        transport: JsonTransport | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            timeout_s=timeout_s,
            max_output_tokens=max_output_tokens,
            transport=transport,
        )

    def invoke(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[AgentToolSpec],
    ) -> ModelTurn:
        _validate_model_inputs(messages, tools)
        system, provider_messages = _anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": self._model_name,
            "max_tokens": self._max_output_tokens,
            "messages": provider_messages,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.as_dict()["parameters"],
                }
                for tool in tools
            ]

        api_key = self._require_api_key()
        response = self._post(
            "/v1/messages",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "User-Agent": "realagent/0.2",
            },
            payload=payload,
        )
        return _parse_anthropic_response(response)


def _validate_model_inputs(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[AgentToolSpec],
) -> None:
    if isinstance(messages, (str, bytes, bytearray)) or not isinstance(
        messages, Sequence
    ):
        raise TypeError("provider messages must be a sequence")
    if isinstance(tools, (str, bytes, bytearray)) or not isinstance(tools, Sequence):
        raise TypeError("provider tools must be a sequence")
    if any(not isinstance(message, Mapping) for message in messages):
        raise TypeError("provider messages must contain mappings")
    if any(not isinstance(tool, AgentToolSpec) for tool in tools):
        raise TypeError("provider tools must contain AgentToolSpec values")


def _parse_openai_response(
    response: Mapping[str, Any],
) -> tuple[ModelTurn, str]:
    response_id = response.get("id")
    if not isinstance(response_id, str) or not response_id:
        raise ProviderProtocolError("Responses response requires a non-empty id")
    if len(response_id) > 1_024 or any(ord(character) < 32 for character in response_id):
        raise ProviderProtocolError("Responses response id is invalid")
    status = response.get("status")
    if not isinstance(status, str) or not status:
        raise ProviderProtocolError("Responses response requires a status string")
    if len(status) > 128 or any(ord(character) < 32 for character in status):
        raise ProviderProtocolError("Responses response status is invalid")
    if status == "failed" or response.get("error") is not None:
        raise ProviderProtocolError("Responses provider reported a failed response")
    output = response.get("output")
    if isinstance(output, (str, bytes, bytearray)) or not isinstance(output, Sequence):
        raise ProviderProtocolError("Responses output must be a sequence")

    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for item in output:
        if not isinstance(item, Mapping):
            raise ProviderProtocolError("Responses output items must be objects")
        item_type = item.get("type")
        if item_type == "function_call":
            name = item.get("name")
            call_id = item.get("call_id")
            if not isinstance(name, str) or not name:
                raise ProviderProtocolError("function_call requires a name")
            if not isinstance(call_id, str) or not call_id:
                raise ProviderProtocolError("function_call requires a call_id")
            arguments = _decode_tool_arguments(item.get("arguments"))
            calls.append(
                _provider_tool_call(
                    name=name,
                    arguments=arguments,
                    call_id=call_id,
                    protocol="Responses",
                )
            )
        elif item_type == "message":
            content = item.get("content", [])
            if isinstance(content, str):
                text_parts.append(content)
                continue
            if isinstance(content, (bytes, bytearray)) or not isinstance(
                content, Sequence
            ):
                raise ProviderProtocolError("Responses message content must be a sequence")
            for block in content:
                if not isinstance(block, Mapping):
                    raise ProviderProtocolError(
                        "Responses message content blocks must be objects"
                    )
                if block.get("type") in {"output_text", "text", "refusal"}:
                    text = block.get("text", block.get("refusal"))
                    if isinstance(text, str):
                        text_parts.append(text)
        elif item_type == "text":
            text = item.get("text", item.get("content"))
            if isinstance(text, str):
                text_parts.append(text)
        else:
            # Reasoning and provider-owned built-in tool items do not belong in
            # the local custom-function contract and are intentionally ignored.
            continue
    if not text_parts and not calls:
        raise ProviderProtocolError(
            "Responses output contained neither text nor function calls"
        )
    return (
        _provider_model_turn(
            content="\n".join(text_parts),
            calls=calls,
            stop_reason=status,
            protocol="Responses",
        ),
        response_id,
    )


def _anthropic_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    result: list[dict[str, Any]] = []

    def append_blocks(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if result and result[-1]["role"] == role:
            result[-1]["content"].extend(blocks)
        else:
            result.append({"role": role, "content": blocks})

    for message in messages:
        role = message.get("role")
        if role in {"system", "developer"}:
            system_parts.append(_content_text(message.get("content", "")))
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ProviderProtocolError(
                    "tool result message requires a non-empty tool_call_id"
                )
            append_blocks(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": _content_text(message.get("content", "")),
                        "is_error": message.get("status") == "error",
                    }
                ],
            )
            continue
        if role not in {"user", "assistant"}:
            raise ProviderProtocolError(
                f"unsupported Anthropic message role {role!r}"
            )
        blocks: list[dict[str, Any]] = []
        content_text = _content_text(message.get("content", ""))
        if content_text:
            blocks.append({"type": "text", "text": content_text})
        tool_calls = message.get("tool_calls", [])
        if isinstance(tool_calls, (str, bytes, bytearray)) or not isinstance(
            tool_calls, Sequence
        ):
            raise ProviderProtocolError("assistant tool_calls must be a sequence")
        for raw_call in tool_calls:
            if not isinstance(raw_call, Mapping):
                raise ProviderProtocolError("assistant tool_calls must contain objects")
            call_id = raw_call.get("id")
            name = raw_call.get("name")
            arguments = raw_call.get("arguments", {})
            if not isinstance(call_id, str) or not call_id:
                raise ProviderProtocolError("assistant tool call requires an id")
            if not isinstance(name, str) or not name:
                raise ProviderProtocolError("assistant tool call requires a name")
            if not isinstance(arguments, Mapping):
                raise ProviderProtocolError(
                    "assistant tool call arguments must be a mapping"
                )
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": dict(arguments),
                }
            )
        append_blocks(role, blocks or [{"type": "text", "text": ""}])
    if not result:
        raise ProviderProtocolError(
            "Anthropic request requires at least one user/assistant message"
        )
    return "\n\n".join(system_parts), result


def _parse_anthropic_response(response: Mapping[str, Any]) -> ModelTurn:
    content = response.get("content")
    if isinstance(content, (str, bytes, bytearray)) or not isinstance(
        content, Sequence
    ):
        raise ProviderProtocolError("Anthropic response content must be a sequence")
    stop_reason = response.get("stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, str):
        raise ProviderProtocolError("Anthropic stop_reason must be a string or null")
    if isinstance(stop_reason, str) and (
        len(stop_reason) > 128
        or any(ord(character) < 32 for character in stop_reason)
    ):
        raise ProviderProtocolError("Anthropic stop_reason is invalid")

    text_parts: list[str] = []
    calls: list[ToolCall] = []
    for block in content:
        if not isinstance(block, Mapping):
            raise ProviderProtocolError("Anthropic content blocks must be objects")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ProviderProtocolError("Anthropic text block requires text")
            text_parts.append(text)
        elif block_type == "tool_use":
            call_id = block.get("id")
            name = block.get("name")
            arguments = block.get("input", {})
            if not isinstance(call_id, str) or not call_id:
                raise ProviderProtocolError("Anthropic tool_use requires an id")
            if not isinstance(name, str) or not name:
                raise ProviderProtocolError("Anthropic tool_use requires a name")
            if not isinstance(arguments, Mapping):
                raise ProviderProtocolError("Anthropic tool_use input must be an object")
            calls.append(
                _provider_tool_call(
                    name=name,
                    arguments=arguments,
                    call_id=call_id,
                    protocol="Anthropic",
                )
            )
        else:
            # Thinking/signature blocks are provider internals, not local tools.
            continue
    if not text_parts and not calls:
        raise ProviderProtocolError(
            "Anthropic response contained neither text nor tool_use blocks"
        )
    return _provider_model_turn(
        content="\n".join(text_parts),
        calls=calls,
        stop_reason=stop_reason or "completed",
        protocol="Anthropic",
    )


def _provider_model_turn(
    *,
    content: str,
    calls: Sequence[ToolCall],
    stop_reason: str,
    protocol: str,
) -> ModelTurn:
    """Build a neutral turn while preserving provider-error classification."""

    call_ids = [call.id for call in calls]
    if len(call_ids) != len(set(call_ids)):
        raise ProviderProtocolError(
            f"{protocol} response contained duplicate tool call ids"
        )
    try:
        return ModelTurn(
            content=content,
            tool_calls=tuple(calls),
            stop_reason=stop_reason,
        )
    except (TypeError, ValueError) as exc:
        raise ProviderProtocolError(
            f"{protocol} response contained an invalid tool call"
        ) from exc


def _provider_tool_call(
    *,
    name: str,
    arguments: Mapping[str, Any],
    call_id: str,
    protocol: str,
) -> ToolCall:
    """Validate provider-owned identifiers as a protocol error."""

    try:
        return ToolCall(name=name, arguments=arguments, id=call_id)
    except (TypeError, ValueError) as exc:
        raise ProviderProtocolError(
            f"{protocol} response contained an invalid tool call"
        ) from exc


def _decode_tool_arguments(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        raise ProviderProtocolError("function_call arguments must be JSON text")
    if len(value.encode("utf-8")) > _MAX_ARGUMENT_BYTES:
        raise ProviderProtocolError("function_call arguments exceeded the byte limit")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ProviderProtocolError(
            "function_call arguments were not valid JSON"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise ProviderProtocolError("function_call arguments must decode to an object")
    return decoded


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ProviderProtocolError(
            "provider message content was not JSON-compatible"
        ) from exc


__all__ = ["ArkAnthropicMessagesModel", "ArkOpenAIResponsesModel"]
