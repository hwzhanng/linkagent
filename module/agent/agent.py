"""Bounded, dependency-neutral model/tool execution loop."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import logging
from typing import Any

from module.agent._contracts import (
    AgentInvoker,
    AgentTool,
    AgentToolSpec,
    ModelTurn,
    RecoverableToolError,
    ToolCall,
    ToolCallLimitError,
    _make_json_safe,
)
from module.agent.models.base import BaseChatModel


logger = logging.getLogger(__name__)


class TopAgent(AgentInvoker):
    """Run a chat model and an explicit tool whitelist in a bounded loop.

    This is the *inner* agent loop.  The outer runtime remains responsible for
    observations, environment control steps, completion checks, and resource
    lifecycle.  Every :meth:`invoke` call builds a new local transcript and
    resets the model's per-invocation state.

    Args:
        model: Neutral chat-model adapter.
        tools: Complete task-scoped capability whitelist.
        system_prompt: Optional system instruction prepended to each invocation.
        max_iterations: Maximum number of model turns per invocation.
        max_tool_calls: Maximum number of tool calls executed per invocation.
        max_tool_calls_per_turn: Maximum calls accepted in one model turn.
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: Iterable[AgentTool] = (),
        *,
        system_prompt: str = "",
        max_iterations: int = 8,
        max_tool_calls: int = 8,
        max_tool_calls_per_turn: int = 4,
    ) -> None:
        if not isinstance(model, BaseChatModel):
            raise TypeError("model must be a BaseChatModel instance")
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise TypeError("max_iterations must be an integer")
        if not 1 <= max_iterations <= 128:
            raise ValueError("max_iterations must be between 1 and 128")
        for field_name, value, hard_limit in (
            ("max_tool_calls", max_tool_calls, 256),
            ("max_tool_calls_per_turn", max_tool_calls_per_turn, 64),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if not 1 <= value <= hard_limit:
                raise ValueError(
                    f"{field_name} must be between 1 and {hard_limit}"
                )
        if max_tool_calls_per_turn > max_tool_calls:
            raise ValueError(
                "max_tool_calls_per_turn must not exceed max_tool_calls"
            )

        tool_list = tuple(tools)
        tool_by_name: dict[str, AgentTool] = {}
        for index, tool in enumerate(tool_list):
            if not isinstance(tool, AgentTool):
                raise TypeError(f"tools[{index}] must be an AgentTool")
            if tool.name in tool_by_name:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            tool_by_name[tool.name] = tool

        self._model = model
        self._tools = tool_list
        self._tool_specs = tuple(tool.spec for tool in tool_list)
        self._tool_by_name = tool_by_name
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._max_tool_calls = max_tool_calls
        self._max_tool_calls_per_turn = max_tool_calls_per_turn

    @property
    def model(self) -> BaseChatModel:
        """Return the configured model adapter."""

        return self._model

    @property
    def tools(self) -> tuple[AgentTool, ...]:
        """Return the immutable task-scoped tool whitelist."""

        return self._tools

    @property
    def max_iterations(self) -> int:
        """Return the configured upper bound on model turns."""

        return self._max_iterations

    @property
    def tool_specs(self) -> tuple[AgentToolSpec, ...]:
        """Return the non-executable metadata passed to model adapters."""

        return self._tool_specs

    @property
    def max_tool_calls(self) -> int:
        """Return the total tool-call budget for one invocation."""

        return self._max_tool_calls

    @property
    def max_tool_calls_per_turn(self) -> int:
        """Return the per-model-turn tool-call budget."""

        return self._max_tool_calls_per_turn

    def invoke(self, agent_input: Mapping[str, Any]) -> dict[str, Any]:
        """Run one independent inner loop.

        ``agent_input`` must contain a ``messages`` sequence.  Tool failures,
        including unknown tool names, become ``role="tool"`` messages with
        ``status="error"``.  They are not raised out of the loop, allowing the
        next model turn to recover or choose another action.

        Returns:
            A dictionary with exactly ``messages``, ``output``,
            ``stop_reason``, and ``iterations``.
        """

        messages = self._build_initial_messages(agent_input)
        self._model.reset()
        last_output: Any = ""
        executed_tool_calls = 0

        for iteration in range(1, self._max_iterations + 1):
            # A fresh JSON-safe copy prevents a model adapter from mutating the
            # transcript owned by this invocation.
            model_messages = tuple(
                _make_json_safe(message) for message in messages
            )
            turn = self._model.invoke(model_messages, self._tool_specs)
            if not isinstance(turn, ModelTurn):
                raise TypeError(
                    f"model.invoke() returned {type(turn).__name__}, expected ModelTurn"
                )

            assistant_message, normalized_calls = self._assistant_message(
                turn,
                iteration=iteration,
            )
            messages.append(assistant_message)
            last_output = assistant_message["content"]

            if not normalized_calls:
                result = {
                    "messages": messages,
                    "output": last_output,
                    "stop_reason": turn.stop_reason or "completed",
                    "iterations": iteration,
                }
                logger.debug("agent invocation result=%s", result)
                return result

            call_count = len(normalized_calls)
            if call_count > self._max_tool_calls_per_turn:
                raise ToolCallLimitError(
                    "model requested too many tools in one turn: "
                    f"{call_count} > {self._max_tool_calls_per_turn}"
                )
            if executed_tool_calls + call_count > self._max_tool_calls:
                raise ToolCallLimitError(
                    "model exceeded the invocation tool-call budget: "
                    f"{executed_tool_calls + call_count} > {self._max_tool_calls}"
                )
            executed_tool_calls += call_count

            for call, call_id in normalized_calls:
                messages.append(self._execute_tool(call, call_id=call_id))

        logger.warning(
            "agent reached max_iterations=%d with pending tool workflow",
            self._max_iterations,
        )
        result = {
            "messages": messages,
            "output": last_output,
            "stop_reason": "max_iterations",
            "iterations": self._max_iterations,
        }
        logger.debug("agent invocation result=%s", result)
        return result

    def _build_initial_messages(
        self,
        agent_input: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(agent_input, Mapping):
            raise TypeError("agent_input must be a mapping")
        if "messages" not in agent_input:
            raise ValueError("agent_input requires a 'messages' field")

        raw_messages = agent_input["messages"]
        if isinstance(raw_messages, (str, bytes, bytearray)) or not isinstance(
            raw_messages, Sequence
        ):
            raise TypeError("agent_input['messages'] must be a sequence of mappings")

        messages: list[dict[str, Any]] = []
        if self._system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": _make_json_safe(self._system_prompt),
                }
            )

        for index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, Mapping):
                raise TypeError(f"agent_input['messages'][{index}] must be a mapping")
            role = raw_message.get("role")
            if not isinstance(role, str) or not role.strip():
                raise ValueError(
                    f"agent_input['messages'][{index}] requires a non-empty string role"
                )
            if "content" not in raw_message:
                raise ValueError(
                    f"agent_input['messages'][{index}] requires a 'content' field"
                )
            safe_message = _make_json_safe(raw_message)
            if not isinstance(safe_message, dict):  # Defensive; mappings normalize to dicts.
                raise TypeError("normalized input message must be a dictionary")
            messages.append(safe_message)
        return messages

    @staticmethod
    def _assistant_message(
        turn: ModelTurn,
        *,
        iteration: int,
    ) -> tuple[dict[str, Any], list[tuple[ToolCall, str]]]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": _make_json_safe(turn.content),
        }
        normalized_calls: list[tuple[ToolCall, str]] = []
        if turn.tool_calls:
            serialized_calls: list[dict[str, Any]] = []
            for index, call in enumerate(turn.tool_calls, start=1):
                call_id = call.id or f"tool-call-{iteration}-{index}"
                normalized_calls.append((call, call_id))
                serialized_calls.append(call.as_dict(fallback_id=call_id))
            message["tool_calls"] = serialized_calls
        return message, normalized_calls

    def _execute_tool(self, call: ToolCall, *, call_id: str) -> dict[str, Any]:
        logger.info("agent selected tool %s", call.name)
        tool = self._tool_by_name.get(call.name)
        if tool is None:
            logger.warning("model requested unknown tool %s", call.name)
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "name": call.name,
                "status": "error",
                "content": {
                    "error": {
                        "type": "unknown_tool",
                        "message": f"unknown tool {call.name!r}",
                    }
                },
            }

        try:
            result = tool.invoke(call.arguments)
        except RecoverableToolError as exc:
            safe_error_message = _make_json_safe(str(exc))
            logger.warning(
                "tool %s reported recoverable failure type=%s message=%s",
                call.name,
                type(exc).__name__,
                safe_error_message,
            )
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "name": call.name,
                "status": "error",
                "content": {
                    "error": {
                        "type": type(exc).__name__,
                        "message": safe_error_message,
                    }
                },
            }

        logger.info("tool %s completed successfully", call.name)
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": call.name,
            "status": "success",
            "content": _make_json_safe(result),
        }


__all__ = ["TopAgent"]
