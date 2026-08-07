"""Public, lazily loaded dependency-neutral agent API."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "AgentInvoker",
    "AgentTool",
    "AgentToolSpec",
    "ArkAnthropicMessagesModel",
    "ArkOpenAIResponsesModel",
    "BaseChatModel",
    "MODEL_REGISTRY",
    "MockChatModel",
    "ModelBuilder",
    "ModelTurn",
    "RecoverableToolError",
    "ToolCall",
    "TopAgent",
    "ToolCallLimitError",
    "build_model",
    "register_model",
    "JsonTransport",
    "ProviderConfigurationError",
    "ProviderModelError",
    "ProviderProtocolError",
    "ProviderTransportError",
    "UrllibJsonTransport",
]


_EXPORTS = {
    "AgentInvoker": ("module.agent._contracts", "AgentInvoker"),
    "AgentTool": ("module.agent._contracts", "AgentTool"),
    "AgentToolSpec": ("module.agent._contracts", "AgentToolSpec"),
    "ModelTurn": ("module.agent._contracts", "ModelTurn"),
    "RecoverableToolError": ("module.agent._contracts", "RecoverableToolError"),
    "ToolCall": ("module.agent._contracts", "ToolCall"),
    "TopAgent": ("module.agent.agent", "TopAgent"),
    "ToolCallLimitError": ("module.agent._contracts", "ToolCallLimitError"),
    "BaseChatModel": ("module.agent.models.base", "BaseChatModel"),
    "MODEL_REGISTRY": ("module.agent.models.factory", "MODEL_REGISTRY"),
    "ModelBuilder": ("module.agent.models.factory", "ModelBuilder"),
    "build_model": ("module.agent.models.factory", "build_model"),
    "register_model": ("module.agent.models.factory", "register_model"),
    "MockChatModel": ("module.agent.models.mock", "MockChatModel"),
    "ArkAnthropicMessagesModel": (
        "module.agent.models.ark",
        "ArkAnthropicMessagesModel",
    ),
    "ArkOpenAIResponsesModel": (
        "module.agent.models.ark",
        "ArkOpenAIResponsesModel",
    ),
    "JsonTransport": ("module.agent.models._http", "JsonTransport"),
    "ProviderConfigurationError": (
        "module.agent.models._http",
        "ProviderConfigurationError",
    ),
    "ProviderModelError": (
        "module.agent.models._http",
        "ProviderModelError",
    ),
    "ProviderProtocolError": (
        "module.agent.models._http",
        "ProviderProtocolError",
    ),
    "ProviderTransportError": (
        "module.agent.models._http",
        "ProviderTransportError",
    ),
    "UrllibJsonTransport": (
        "module.agent.models._http",
        "UrllibJsonTransport",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
