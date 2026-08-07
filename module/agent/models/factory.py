"""Explicit chat-model registry and construction helper."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

from module.agent.models.ark import (
    ArkAnthropicMessagesModel,
    ArkOpenAIResponsesModel,
)
from module.agent.models.base import BaseChatModel
from module.agent.models.mock import MockChatModel


ModelBuilder: TypeAlias = Callable[..., BaseChatModel]


# Registration is intentionally explicit.  Importing arbitrary modules or
# scanning directories would make the available model surface unpredictable.
MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "ark_anthropic_messages": ArkAnthropicMessagesModel,
    "ark_openai_responses": ArkOpenAIResponsesModel,
    "mock": MockChatModel,
}


def register_model(
    name: str,
    builder: ModelBuilder,
    *,
    replace: bool = False,
) -> None:
    """Register a model constructor under a stable configuration name.

    Args:
        name: Value accepted by the model configuration's ``type`` field.
        builder: Callable accepting configuration keys as keyword arguments.
        replace: Permit replacing an existing registration when true.

    Raises:
        ValueError: If the name is empty or already registered.
        TypeError: If ``builder`` is not callable.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("model registration name must be a non-empty string")
    if not callable(builder):
        raise TypeError("model builder must be callable")
    normalized_name = name.strip()
    if normalized_name in MODEL_REGISTRY and not replace:
        raise ValueError(f"model type {normalized_name!r} is already registered")
    MODEL_REGISTRY[normalized_name] = builder


def build_model(
    config: str | Mapping[str, Any],
    **overrides: Any,
) -> BaseChatModel:
    """Build a registered model from a type name or configuration mapping.

    Mapping configuration must contain a ``type`` field.  Remaining fields are
    passed unchanged to the registered builder, with explicit keyword
    ``overrides`` taking precedence.  The input mapping is never mutated.

    Examples:
        ``build_model("mock")``

        ``build_model({"type": "mock", "tool_sequence": ["locate"]})``
    """

    if isinstance(config, str):
        model_type = config
        parameters: dict[str, Any] = {}
    elif isinstance(config, Mapping):
        parameters = dict(config)
        try:
            model_type = parameters.pop("type")
        except KeyError as exc:
            raise ValueError("model configuration requires a 'type' field") from exc
    else:
        raise TypeError("model config must be a type string or mapping")

    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("model configuration 'type' must be a non-empty string")
    if any(not isinstance(key, str) for key in parameters):
        raise TypeError("model configuration keys must be strings")

    parameters.update(overrides)
    try:
        builder = MODEL_REGISTRY[model_type]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown model type {model_type!r}; available types: {available}"
        ) from exc

    model = builder(**parameters)
    if not isinstance(model, BaseChatModel):
        raise TypeError(
            f"model builder {model_type!r} returned {type(model).__name__}, "
            "expected BaseChatModel"
        )
    return model


__all__ = ["MODEL_REGISTRY", "ModelBuilder", "build_model", "register_model"]
