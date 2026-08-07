"""Base contract for dependency-neutral chat-model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from module.agent._contracts import AgentToolSpec, ModelTurn


class BaseChatModel(ABC):
    """Minimal interface required by :class:`module.agent.TopAgent`.

    Concrete adapters translate the neutral messages and tool descriptions to
    their provider's protocol and translate one provider response back into a
    :class:`ModelTurn`.  Provider credentials, transport, and retry behavior
    belong in those adapters rather than in the agent loop.
    """

    def reset(self) -> None:
        """Reset per-invocation adapter state.

        Network-backed models are normally stateless and can keep this no-op.
        Scripted models override it so reusing a ``TopAgent`` still starts each
        outer-loop invocation with a fresh model script.
        """

    @abstractmethod
    def invoke(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[AgentToolSpec],
    ) -> ModelTurn:
        """Return the next assistant turn for the supplied transcript."""


__all__ = ["BaseChatModel"]
