"""Dependency-neutral contracts used by the agent layer.

The objects in this module intentionally depend only on the Python standard
library.  They form the boundary between tasks, chat-model adapters, and the
bounded agent loop without exposing any vendor SDK types.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import inspect
from itertools import islice
import json
import math
from pathlib import Path
import re
import types
from types import MappingProxyType
from typing import (
    Any,
    Literal,
    Protocol,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)

from module.redaction import redact_text


_MAX_CONTENT_DEPTH = 12
_MAX_COLLECTION_ITEMS = 100
_MAX_STRING_LENGTH = 16_384
_MAX_TOOL_SCHEMA_BYTES = 65_536
_MAX_TOOL_CALLS_IN_MODEL_TURN = 64
_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_SECRET_FIELD_NAMES = {
    "apikey",
    "authorization",
    "accesstoken",
    "clientsecret",
    "credential",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
    "signingkey",
    "token",
}


@runtime_checkable
class AgentInvoker(Protocol):
    """Structural contract consumed by the outer runtime.

    Implementations must treat each call as a fresh outer-loop step.  In
    particular, an implementation must not implicitly append messages from a
    previous call.
    """

    def invoke(self, agent_input: Mapping[str, Any]) -> dict[str, Any]:
        """Run one independent agent invocation and return a stable result."""


class RecoverableToolError(RuntimeError):
    """Expected semantic or argument failure that a model may safely recover from."""


class ToolCallLimitError(RuntimeError):
    """A model response exceeded the configured tool-execution budget."""


@dataclass(frozen=True, slots=True)
class AgentToolSpec:
    """Non-executable tool metadata visible to a chat-model adapter."""

    name: str
    description: str
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    )

    def __post_init__(self) -> None:
        normalized_name, normalized_description = _validate_tool_identity(
            self.name,
            self.description,
        )
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "description", normalized_description)
        object.__setattr__(
            self,
            "parameters",
            _normalize_tool_parameters(self.parameters),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the model-facing representation."""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": _thaw_json(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class AgentTool:
    """A named callable that a task explicitly allows an agent to execute.

    ``TopAgent`` only executes instances supplied in its ``tools`` iterable;
    there is no process-wide tool registry.  Arguments are always passed to
    ``handler`` as keyword arguments, which keeps ordinary typed Python
    functions convenient to expose.

    Args:
        name: Model-visible unique tool name.
        description: Concise model-visible description of the operation.
        handler: Callable invoked as ``handler(**arguments)``.
        parameters: Optional model-visible object JSON schema. When omitted,
            a conservative schema is inferred from the handler signature.
    """

    name: str
    description: str
    handler: Callable[..., Any] = field(repr=False)
    parameters: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        normalized_name, normalized_description = _validate_tool_identity(
            self.name,
            self.description,
        )
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")
        raw_parameters = self.parameters
        if raw_parameters is None:
            raw_parameters = _infer_tool_parameters(self.handler)
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "description", normalized_description)
        object.__setattr__(
            self,
            "parameters",
            _normalize_tool_parameters(raw_parameters),
        )

    @classmethod
    def from_callable(
        cls,
        handler: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> AgentTool:
        """Create a tool using a callable's name and docstring as defaults."""

        inferred_name = name or getattr(handler, "__name__", "")
        inferred_description = description
        if inferred_description is None:
            inferred_description = (getattr(handler, "__doc__", "") or "").strip()
        return cls(
            name=inferred_name,
            description=inferred_description,
            handler=handler,
            parameters=parameters,
        )

    def invoke(self, arguments: Mapping[str, Any] | None = None) -> Any:
        """Execute the tool with a copied mapping of keyword arguments."""

        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise TypeError("tool arguments must be a mapping")
        prepared_arguments = dict(arguments)
        try:
            inspect.signature(self.handler).bind(**prepared_arguments)
        except (TypeError, ValueError) as exc:
            raise RecoverableToolError(
                f"invalid arguments for tool {self.name!r}: {exc}"
            ) from exc
        return self.handler(**prepared_arguments)

    def as_dict(self) -> dict[str, Any]:
        """Return the small model-facing tool description."""

        return self.spec.as_dict()

    @property
    def spec(self) -> AgentToolSpec:
        """Return non-executable metadata safe to expose to a model adapter."""

        if self.parameters is None:  # pragma: no cover - normalized in __post_init__.
            raise AssertionError("tool parameters were not normalized")
        return AgentToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A dependency-neutral request from a model to invoke one tool."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        normalized_name = _validate_tool_name(self.name)
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool call arguments must be a mapping")
        if not isinstance(self.id, str):
            raise TypeError("tool call id must be a string")
        normalized_id = self.id.strip()
        if len(normalized_id) > 128 or any(ord(character) < 32 for character in normalized_id):
            raise ValueError(
                "tool call id must contain at most 128 printable characters"
            )
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "id", normalized_id)
        # Detach the DTO from a mutable caller-owned configuration mapping.
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))

    @classmethod
    def from_value(cls, value: ToolCall | Mapping[str, Any] | str) -> ToolCall:
        """Normalize a convenient string or mapping representation."""

        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        if not isinstance(value, Mapping):
            raise TypeError("tool calls must be ToolCall, mapping, or string values")

        unknown_keys = set(value) - {"id", "name", "arguments"}
        if unknown_keys:
            names = ", ".join(sorted(str(key) for key in unknown_keys))
            raise ValueError(f"unsupported tool call fields: {names}")
        if "name" not in value:
            raise ValueError("tool call mapping requires a 'name' field")
        return cls(
            name=value["name"],
            arguments=value.get("arguments", {}),
            id=value.get("id", ""),
        )

    def as_dict(self, *, fallback_id: str | None = None) -> dict[str, Any]:
        """Return a JSON-safe representation for an assistant message."""

        call_id = self.id or fallback_id or ""
        return {
            "id": call_id,
            "name": self.name,
            "arguments": _make_json_safe(self.arguments),
        }


@dataclass(frozen=True, slots=True)
class ModelTurn:
    """One assistant turn returned by a :class:`BaseChatModel`.

    A turn may contain text, tool calls, or both.  ``stop_reason`` is optional;
    ``TopAgent`` supplies the stable ``"completed"`` default when a terminal
    turn omits it.
    """

    content: Any = ""
    tool_calls: Sequence[ToolCall] = field(default_factory=tuple)
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.tool_calls, (str, bytes, bytearray)) or not isinstance(
            self.tool_calls, Sequence
        ):
            raise TypeError("model turn tool_calls must be a sequence")
        if len(self.tool_calls) > _MAX_TOOL_CALLS_IN_MODEL_TURN:
            raise ValueError(
                "model turn contains too many tool calls; "
                f"hard limit is {_MAX_TOOL_CALLS_IN_MODEL_TURN}"
            )
        normalized = tuple(ToolCall.from_value(call) for call in self.tool_calls)
        object.__setattr__(self, "tool_calls", normalized)
        if self.stop_reason is not None and not isinstance(self.stop_reason, str):
            raise TypeError("model turn stop_reason must be a string or None")


def _make_json_safe(
    value: Any,
    *,
    _depth: int = 0,
    _seen: set[int] | None = None,
) -> Any:
    """Convert message content into a bounded, JSON-compatible value.

    Binary payloads are represented only by their byte length.  Collection
    sizes and nesting depth are capped so observations such as images, masks,
    or point clouds cannot accidentally make the agent transcript enormous.
    Opaque objects are represented by their qualified type, never by ``repr``.
    """

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        redacted = _redact_text(value)
        if len(redacted) <= _MAX_STRING_LENGTH:
            return redacted
        omitted = len(redacted) - _MAX_STRING_LENGTH
        return f"{redacted[:_MAX_STRING_LENGTH]}<... {omitted} characters omitted>"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": "binary", "size_bytes": len(value)}
    if isinstance(value, Enum):
        return _make_json_safe(value.value, _depth=_depth, _seen=_seen)
    if isinstance(value, Path):
        return _make_json_safe(str(value), _depth=_depth, _seen=_seen)
    if _depth >= _MAX_CONTENT_DEPTH:
        return {"type": "truncated", "reason": "maximum depth reached"}

    seen = _seen if _seen is not None else set()
    track_identity = isinstance(value, (Mapping, Sequence, set, frozenset)) or is_dataclass(
        value
    )
    value_id = id(value)
    if track_identity:
        if value_id in seen:
            return {"type": "truncated", "reason": "cyclic reference"}
        seen.add(value_id)

    try:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for index, (key, item) in enumerate(
                islice(value.items(), _MAX_COLLECTION_ITEMS),
                start=1,
            ):
                if isinstance(key, str):
                    normalized_key = _make_json_safe(
                        key,
                        _depth=_depth + 1,
                        _seen=seen,
                    )
                    # The string branch always produces a string.
                    assert isinstance(normalized_key, str)
                    secret_field = _is_secret_field(key)
                else:
                    # Never call str/repr on caller-owned key objects: those
                    # methods may be effectful or contain credential material.
                    normalized_key = f"<non-string-key:{index}>"
                    secret_field = False
                safe_key = _unique_mapping_key(result, normalized_key)
                if secret_field:
                    result[safe_key] = "<redacted>"
                else:
                    result[safe_key] = _make_json_safe(
                        item,
                        _depth=_depth + 1,
                        _seen=seen,
                    )
            if len(value) > _MAX_COLLECTION_ITEMS:
                result["__truncated_items__"] = len(value) - _MAX_COLLECTION_ITEMS
            return result

        if is_dataclass(value) and not isinstance(value, type):
            return {
                item.name: (
                    "<redacted>"
                    if _is_secret_field(item.name)
                    else _make_json_safe(
                        getattr(value, item.name),
                        _depth=_depth + 1,
                        _seen=seen,
                    )
                )
                for item in fields(value)
            }

        if isinstance(value, Sequence):
            result = [
                _make_json_safe(item, _depth=_depth + 1, _seen=seen)
                for item in islice(value, _MAX_COLLECTION_ITEMS)
            ]
            if len(value) > _MAX_COLLECTION_ITEMS:
                result.append(
                    {"type": "truncated", "items_omitted": len(value) - _MAX_COLLECTION_ITEMS}
                )
            return result

        if isinstance(value, (set, frozenset)):
            # Sets have no stable semantic order, so sort their safe string forms.
            safe_items = [
                _make_json_safe(item, _depth=_depth + 1, _seen=seen)
                for item in islice(value, _MAX_COLLECTION_ITEMS)
            ]
            result = sorted(safe_items, key=lambda item: str(item))
            if len(value) > _MAX_COLLECTION_ITEMS:
                result.append(
                    {"type": "truncated", "items_omitted": len(value) - _MAX_COLLECTION_ITEMS}
                )
            return result
    finally:
        if track_identity:
            seen.discard(value_id)

    value_type = type(value)
    return {
        "type": "opaque",
        "class": f"{value_type.__module__}.{value_type.__qualname__}",
    }


def _redact_text(value: str) -> str:
    """Compatibility wrapper around the shared message/log redactor."""

    return redact_text(value)


def _unique_mapping_key(result: Mapping[str, Any], candidate: str) -> str:
    """Return a deterministic collision-free key after safe normalization."""

    if candidate not in result:
        return candidate
    suffix = 2
    while True:
        marker = f"#{suffix}"
        normalized = f"{candidate[: _MAX_STRING_LENGTH - len(marker)]}{marker}"
        if normalized not in result:
            return normalized
        suffix += 1


def _is_secret_field(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return normalized in _SECRET_FIELD_NAMES or any(
        normalized.endswith(secret_name)
        for secret_name in _SECRET_FIELD_NAMES
    )


def _infer_tool_parameters(handler: Callable[..., Any]) -> dict[str, Any]:
    """Infer a conservative object schema from one trusted local callable."""

    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "tool parameters must be provided when the handler signature "
            "cannot be inspected"
        ) from exc
    try:
        type_hints = get_type_hints(handler)
    except (AttributeError, NameError, TypeError):
        type_hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ValueError(
                "tool handlers cannot use positional-only parameters"
            )
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise ValueError(
                "tool handlers with *args/**kwargs require an explicit "
                "parameters schema"
            )
        annotation = type_hints.get(parameter.name, parameter.annotation)
        properties[parameter.name] = _annotation_schema(annotation)
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _annotation_schema(annotation: Any) -> dict[str, Any]:
    """Translate common Python annotations into a small JSON Schema subset."""

    if annotation is inspect.Signature.empty or annotation is Any:
        return {}
    if isinstance(annotation, str):
        return {
            "str": {"type": "string"},
            "int": {"type": "integer"},
            "float": {"type": "number"},
            "bool": {"type": "boolean"},
        }.get(annotation, {})
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is type(None):
        return {"type": "null"}

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in {types.UnionType, Union}:
        return {"anyOf": [_annotation_schema(item) for item in arguments]}
    if origin is Literal:
        safe_values = _make_json_safe(list(arguments))
        if not isinstance(safe_values, list):  # pragma: no cover - list branch.
            raise TypeError("literal values did not normalize to a list")
        return {"enum": safe_values}
    if origin in {list, tuple, set, frozenset, Sequence}:
        item_annotation = arguments[0] if arguments else Any
        return {"type": "array", "items": _annotation_schema(item_annotation)}
    if origin in {dict, Mapping} or annotation in {dict, Mapping}:
        return {"type": "object"}
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {"enum": [_make_json_safe(item.value) for item in annotation]}
    return {}


def _normalize_tool_parameters(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate, detach, bound, and freeze a provider-visible tool schema."""

    if not isinstance(parameters, Mapping):
        raise TypeError("tool parameters must be a mapping")
    safe_parameters = _make_json_safe(parameters)
    if not isinstance(safe_parameters, dict):
        raise TypeError("normalized tool parameters must be a dictionary")
    try:
        encoded = json.dumps(
            safe_parameters,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - safe converter guard.
        raise TypeError("tool parameters must be JSON-compatible") from exc
    if len(encoded.encode("utf-8")) > _MAX_TOOL_SCHEMA_BYTES:
        raise ValueError(
            f"tool parameters exceed {_MAX_TOOL_SCHEMA_BYTES} encoded bytes"
        )
    normalized = json.loads(encoded)
    if normalized.get("type") != "object":
        raise ValueError("tool parameters root type must be 'object'")
    properties = normalized.get("properties")
    if not isinstance(properties, dict):
        raise TypeError("tool parameters properties must be a mapping")
    if any(not isinstance(name, str) or not name for name in properties):
        raise ValueError("tool parameter names must be non-empty strings")
    if any(not isinstance(schema, dict) for schema in properties.values()):
        raise TypeError("each tool parameter property must be a schema mapping")
    required = normalized.get("required", [])
    if not isinstance(required, list) or any(
        not isinstance(name, str) for name in required
    ):
        raise TypeError("tool parameters required must be a list of strings")
    unknown_required = set(required) - set(properties)
    if unknown_required:
        names = ", ".join(sorted(unknown_required))
        raise ValueError(f"required tool parameters are not defined: {names}")
    if len(required) != len(set(required)):
        raise ValueError("required tool parameters must be unique")
    additional = normalized.get("additionalProperties", False)
    if not isinstance(additional, bool):
        raise TypeError("tool parameters additionalProperties must be a bool")
    normalized["additionalProperties"] = additional
    return _freeze_json(normalized)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validate_tool_identity(name: object, description: object) -> tuple[str, str]:
    normalized_name = _validate_tool_name(name)
    if not isinstance(description, str):
        raise TypeError("tool description must be a string")
    normalized_description = _redact_text(description.strip())
    if not normalized_description:
        raise ValueError("tool description must be a non-empty string")
    return normalized_name, normalized_description


def _validate_tool_name(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool name must be a non-empty string")
    normalized_name = name.strip()
    if not _TOOL_NAME_PATTERN.fullmatch(normalized_name):
        raise ValueError(
            "tool name must be lower snake_case beginning with a letter and "
            "contain at most 128 characters"
        )
    return normalized_name


__all__ = [
    "AgentInvoker",
    "AgentTool",
    "AgentToolSpec",
    "ModelTurn",
    "RecoverableToolError",
    "ToolCall",
    "ToolCallLimitError",
]
