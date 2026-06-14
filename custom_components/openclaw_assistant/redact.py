"""Helpers for keeping secrets and user content out of logs and diagnostics.

The hard rule for this integration is that API keys, bearer tokens,
``Authorization`` headers, full prompts, assistant responses and raw SSE
payloads must never reach the logs or the diagnostics download. These helpers
centralise that policy so the rest of the code can stay terse and safe.

Like :mod:`.sse`, this module has no Home Assistant dependency so it can be
unit-tested on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

#: Placeholder substituted for any redacted value.
REDACTED: Final = "**REDACTED**"

#: Header names whose values must never be logged (compared case-insensitively).
SENSITIVE_HEADERS: Final = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "api-key",
        "x-api-key",
        "openai-api-key",
    }
)

#: Mapping keys whose values must never be exposed in diagnostics/logs.
SENSITIVE_KEYS: Final = frozenset(
    {"api_key", "apikey", "token", "bearer", "authorization", "password", "secret"}
)


def redact_url(url: str) -> str:
    """Return a URL safe to log: drop user-info and the query string.

    A base URL or endpoint can legitimately smuggle a secret in its user-info
    (``https://user:pass@host``) or query string (``?api_key=...``). Both are
    removed; the scheme, host, port and path are preserved for debugging. If a
    query string was present it is replaced with a ``redacted`` marker so the
    log still shows that one existed without revealing it.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:
        return REDACTED

    if not parts.scheme and not parts.netloc:
        # Not a parseable absolute URL; redact wholesale rather than risk a leak.
        return REDACTED

    host = parts.hostname or ""
    if ":" in host:
        # IPv6 literal — re-add the brackets the parser stripped.
        host = f"[{host}]"
    netloc = host
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"

    # Strip matrix parameters (``;key=value``) from each path segment; they can
    # smuggle a secret (e.g. ``;jsessionid=...``) past query-string redaction.
    path = "/".join(segment.split(";", 1)[0] for segment in parts.path.split("/"))

    query = "redacted" if parts.query else ""
    return urlunsplit((parts.scheme, netloc, path, query, ""))


def redact_headers(headers: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``headers`` with sensitive values masked."""
    return {
        key: (REDACTED if key.lower() in SENSITIVE_HEADERS else value)
        for key, value in headers.items()
    }


def _redact_value(value: Any) -> Any:
    """Recursively redact nested mappings and sequences of mappings."""
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of a config/diagnostics mapping with secrets masked.

    Keys whose lowercased name is in :data:`SENSITIVE_KEYS` are masked; nested
    mappings — including those inside lists/tuples — are redacted recursively.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(key, str) and key.lower() in SENSITIVE_KEYS:
            result[key] = REDACTED
        else:
            result[key] = _redact_value(value)
    return result


def redact_text(text: Any) -> str:
    """Replace free text (prompts, responses) with a length-only placeholder.

    Never returns the original characters, so the result is always safe to
    interpolate into a log message - including in exception handlers.
    """
    if text is None:
        return "<none>"
    try:
        length = len(text)
    except TypeError:
        length = 0
    return f"<redacted {length} chars>"


def byte_len(data: Any) -> int:
    """Return a best-effort byte length for logging, never the content itself."""
    if data is None:
        return 0
    if isinstance(data, (bytes, bytearray)):
        return len(data)
    if isinstance(data, str):
        return len(data.encode("utf-8"))
    try:
        return len(str(data).encode("utf-8"))
    except Exception:
        return 0
