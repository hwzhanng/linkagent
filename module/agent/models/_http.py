"""Bounded JSON-over-HTTPS transport for provider model adapters."""

from __future__ import annotations

from collections.abc import Mapping
import json
import socket
from typing import Any, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from module.redaction import redact_text


_DEFAULT_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_REQUEST_BYTES = 4 * 1024 * 1024


class ProviderModelError(RuntimeError):
    """Base class for safe provider adapter failures."""


class ProviderConfigurationError(ProviderModelError):
    """Provider configuration or its referenced environment is invalid."""


class ProviderTransportError(ProviderModelError):
    """The HTTPS request failed or returned a non-success status."""


class ProviderProtocolError(ProviderModelError):
    """The provider request or response violated the expected JSON protocol."""


@runtime_checkable
class JsonTransport(Protocol):
    """Injectable provider transport used by production and fake adapters."""

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """POST a JSON mapping and return one decoded JSON mapping."""


class UrllibJsonTransport:
    """Standard-library HTTPS transport with bounded response decoding."""

    def __init__(self, *, max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES) -> None:
        if isinstance(max_response_bytes, bool) or not isinstance(
            max_response_bytes, int
        ):
            raise TypeError("max_response_bytes must be an integer")
        if not 1_024 <= max_response_bytes <= 64 * 1024 * 1024:
            raise ValueError(
                "max_response_bytes must be between 1024 and 67108864"
            )
        self._max_response_bytes = max_response_bytes

    @property
    def max_response_bytes(self) -> int:
        """Return the largest accepted provider response body."""

        return self._max_response_bytes

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """POST one bounded HTTPS request without logging headers or payload."""

        _validate_https_url(url)
        if not isinstance(headers, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in headers.items()
        ):
            raise TypeError("provider headers must be a string mapping")
        if not isinstance(payload, Mapping):
            raise TypeError("provider payload must be a mapping")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TypeError("provider timeout_s must be a number")
        if not 0.1 <= float(timeout_s) <= 300.0:
            raise ValueError("provider timeout_s must be between 0.1 and 300")
        try:
            request_body = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ProviderProtocolError(
                "provider request payload is not JSON-compatible"
            ) from exc
        if len(request_body) > _MAX_REQUEST_BYTES:
            raise ProviderProtocolError(
                "provider request exceeded the encoded byte limit"
            )

        request = Request(
            url,
            data=request_body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with urlopen(request, timeout=float(timeout_s)) as response:
                body = response.read(self._max_response_bytes + 1)
        except HTTPError as exc:
            try:
                body = exc.read(self._max_response_bytes + 1)
            finally:
                exc.close()
            detail = _safe_provider_error_detail(body)
            raise ProviderTransportError(
                f"provider returned HTTP {exc.code}: {detail}"
            ) from None
        except (URLError, TimeoutError, socket.timeout) as exc:
            reason = getattr(exc, "reason", exc)
            raise ProviderTransportError(
                "provider request failed: " + redact_text(str(reason))
            ) from None

        if len(body) > self._max_response_bytes:
            raise ProviderProtocolError(
                "provider response exceeded the configured byte limit"
            )
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderProtocolError(
                "provider response was not valid UTF-8 JSON"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ProviderProtocolError("provider response root must be an object")
        return decoded


def validate_provider_base_url(base_url: str) -> str:
    """Validate and normalize a credential-free HTTPS provider Base URL."""

    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("provider base_url must be a non-empty string")
    normalized = base_url.strip().rstrip("/")
    _validate_https_url(normalized)
    parsed = urlsplit(normalized)
    if parsed.query or parsed.fragment:
        raise ValueError("provider base_url cannot contain query or fragment")
    return normalized


def _validate_https_url(url: str) -> None:
    if not isinstance(url, str) or not url:
        raise ValueError("provider URL must be a non-empty string")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("provider URL must use https with a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("provider URL cannot contain credentials")


def _safe_provider_error_detail(body: bytes) -> str:
    """Extract a bounded, redacted error summary without exposing raw bodies."""

    if len(body) > _DEFAULT_MAX_RESPONSE_BYTES:
        return "response body exceeded the safe error limit"
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "provider returned a non-JSON error body"
    if not isinstance(decoded, Mapping):
        return "provider returned an unstructured error"
    error = decoded.get("error", decoded)
    if not isinstance(error, Mapping):
        return "provider returned an unstructured error"
    safe_parts: list[str] = []
    for key in ("type", "code", "message"):
        value = error.get(key)
        if isinstance(value, (str, int, float, bool)):
            rendered = redact_text(str(value))
            if len(rendered) > 1_024:
                rendered = rendered[:1_024] + "<truncated>"
            safe_parts.append(f"{key}={rendered}")
    return ", ".join(safe_parts) or "provider error details unavailable"


__all__ = [
    "JsonTransport",
    "ProviderConfigurationError",
    "ProviderModelError",
    "ProviderProtocolError",
    "ProviderTransportError",
    "UrllibJsonTransport",
    "validate_provider_base_url",
]
