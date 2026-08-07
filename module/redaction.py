"""Small, dependency-free text redaction shared by messages and logs."""

from __future__ import annotations

import re


_SECRET_NAME = (
    r"(?:[a-z0-9]+[_-])*(?:api[_-]?key|private[_-]?key|signing[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|authorization|credential|password|"
    r"secret|token)"
)
_QUOTED_SECRET_PATTERN = re.compile(
    rf"(?i)(?P<prefix>[\"']?{_SECRET_NAME}[\"']?\s*[=:]\s*)"
    r'''(?P<value>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')'''
)
_UNQUOTED_AUTHORIZATION_PATTERN = re.compile(
    r"(?i)(?P<prefix>[\"']?(?:[a-z0-9]+[_-])*authorization[\"']?"
    r"\s*[=:]\s*)(?P<value>[^\"'\s][^,;}\r\n]*)"
)
_UNQUOTED_SECRET_PATTERN = re.compile(
    rf"(?i)(?P<prefix>[\"']?{_SECRET_NAME}[\"']?\s*[=:]\s*)"
    r"(?P<value>[^\s,;}\"']+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")


def redact_text(value: str) -> str:
    """Redact common credential assignments without logging their values.

    A quoted assignment is removed through its matching closing quote, including
    embedded spaces. Unquoted credential assignments intentionally stop at the
    first token; unquoted Authorization values consume the rest of their field.
    """

    if not isinstance(value, str):
        raise TypeError("redaction input must be a string")
    redacted = _QUOTED_SECRET_PATTERN.sub(_replace_quoted_secret, value)
    redacted = _UNQUOTED_AUTHORIZATION_PATTERN.sub(
        r"\g<prefix><redacted>", redacted
    )
    redacted = _UNQUOTED_SECRET_PATTERN.sub(r"\g<prefix><redacted>", redacted)
    return _BEARER_PATTERN.sub("Bearer <redacted>", redacted)


def _replace_quoted_secret(match: re.Match[str]) -> str:
    quoted_value = match.group("value")
    quote = quoted_value[0]
    return f"{match.group('prefix')}{quote}<redacted>{quote}"
