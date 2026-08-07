"""Deterministic chat model for local development and unit tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
import logging
from typing import Any

from module.agent._contracts import AgentToolSpec, ModelTurn, ToolCall
from module.agent.models.base import BaseChatModel


logger = logging.getLogger(__name__)


class MockChatModel(BaseChatModel):
    """Emit a configured sequence of tool calls followed by a final answer.

    One configured call is emitted per model turn.  This deliberately exercises
    the same model -> tool -> model continuation used by a real adapter.  The
    sequence restarts whenever :meth:`reset` is called; ``TopAgent`` calls it at
    the beginning of every independent invocation.

    Args:
        tool_sequence: Ordered tool-call mappings, names, or ``ToolCall`` DTOs.
        final_output: Content returned after all scripted calls have been made.
        final_stop_reason: Terminal reason attached to the final turn.
    """

    def __init__(
        self,
        *,
        tool_sequence: Iterable[ToolCall | Mapping[str, Any] | str] = (),
        final_output: Any = "Mock model completed.",
        final_stop_reason: str = "completed",
    ) -> None:
        if isinstance(tool_sequence, (str, bytes, bytearray)):
            raise TypeError("tool_sequence must be an iterable of tool calls")
        if not isinstance(final_stop_reason, str) or not final_stop_reason:
            raise ValueError("final_stop_reason must be a non-empty string")

        calls: list[ToolCall] = []
        for index, value in enumerate(tool_sequence, start=1):
            call = ToolCall.from_value(value)
            if not call.id:
                call = replace(call, id=f"mock-tool-call-{index}")
            calls.append(call)

        self._tool_sequence = tuple(calls)
        self._final_output = final_output
        self._final_stop_reason = final_stop_reason
        self._cursor = 0

    @property
    def tool_sequence(self) -> tuple[ToolCall, ...]:
        """Return the immutable configured call sequence."""

        return self._tool_sequence

    def reset(self) -> None:
        """Restart the deterministic script for a new outer invocation."""

        self._cursor = 0

    def invoke(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[AgentToolSpec],
    ) -> ModelTurn:
        """Return the next scripted call, or the configured final answer."""

        # Accessing the lengths catches obviously invalid direct calls while
        # keeping the mock independent of the transcript's concrete container.
        message_count = len(messages)
        tool_count = len(tools)
        logger.debug(
            "mock model invocation: messages=%d exposed_tools=%d cursor=%d",
            message_count,
            tool_count,
            self._cursor,
        )

        if self._cursor < len(self._tool_sequence):
            call = self._tool_sequence[self._cursor]
            self._cursor += 1
            return ModelTurn(tool_calls=(call,))

        return ModelTurn(
            content=self._final_output,
            stop_reason=self._final_stop_reason,
        )


__all__ = ["MockChatModel"]
