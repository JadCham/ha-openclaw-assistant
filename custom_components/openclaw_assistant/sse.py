"""Incremental parser for OpenAI-compatible Server-Sent Events (SSE).

This module is intentionally free of any Home Assistant or ``aiohttp``
dependency so the streaming wire protocol can be unit-tested in isolation.
It turns the raw network chunks of an OpenAI-style ``/v1/chat/completions``
streaming response into the assistant text fragments they carry.

Security note: nothing in this module logs. Callers that want to report a
malformed payload receive only its *byte length* via ``on_parse_error`` -
never the payload content itself - so raw model/user text can never leak into
logs through this code path.
"""

from __future__ import annotations

import codecs
from collections.abc import Callable, Iterable, Iterator
import json
from typing import Any, Final

#: The literal sentinel line that terminates an OpenAI-compatible stream.
DONE: Final = "[DONE]"

#: Returned by :func:`parse_payload` to signal the stream is finished. It is a
#: unique object (truthy) so callers must test it with ``is`` *before* the
#: ordinary truthiness check used for text fragments.
DONE_SENTINEL: Final = object()


class SSEDecoder:
    """Turn raw SSE byte/str chunks into the values of their ``data:`` fields.

    The decoder is stateful: feed it the chunks of a stream in order via
    :meth:`decode` and it returns the complete ``data:`` payloads produced so
    far, buffering any trailing partial line until more bytes arrive. Call
    :meth:`flush` once the stream ends to recover a final line that was not
    newline-terminated.

    It handles every framing quirk a real OpenAI-compatible server produces:

    * multiple events packed into a single network chunk,
    * a single event split across two or more network chunks,
    * ``\\n`` and ``\\r\\n`` line endings,
    * comment / keepalive lines that begin with ``:`` (e.g. ``: ping``),
    * non-``data`` SSE fields (``event:``, ``id:``, ``retry:``) which are
      ignored,
    * blank lines used as event separators.
    """

    def __init__(self) -> None:
        """Initialise an empty decoder."""
        self._buffer = ""
        # Incremental UTF-8 decoder so a multibyte character split across two
        # network chunks is reassembled instead of mangled into U+FFFD. The
        # client feeds raw bytes from ``aiohttp`` at arbitrary byte boundaries,
        # so per-chunk decoding would corrupt non-ASCII text (accents, CJK,
        # emoji). The decoder holds any trailing partial sequence until the
        # next ``decode`` call.
        self._utf8 = codecs.getincrementaldecoder("utf-8")("replace")

    def decode(self, chunk: bytes | bytearray | str) -> list[str]:
        """Feed one chunk and return the ``data:`` payloads it completed."""
        if isinstance(chunk, (bytes, bytearray)):
            self._buffer += self._utf8.decode(bytes(chunk))
        else:
            self._buffer += chunk
        return self._extract_lines()

    def _extract_lines(self) -> list[str]:
        payloads: list[str] = []
        while True:
            newline = self._buffer.find("\n")
            if newline == -1:
                break
            line = self._buffer[:newline]
            self._buffer = self._buffer[newline + 1 :]
            if (payload := _parse_line(line)) is not None:
                payloads.append(payload)
        return payloads

    def flush(self) -> list[str]:
        """Return a payload buffered in a final line with no trailing newline."""
        # Flush any dangling partial multibyte sequence first.
        self._buffer += self._utf8.decode(b"", final=True)
        if not self._buffer:
            return []
        line = self._buffer
        self._buffer = ""
        payload = _parse_line(line)
        return [payload] if payload is not None else []


def _parse_line(line: str) -> str | None:
    """Return the value of a ``data:`` field, or ``None`` for any other line."""
    if line.endswith("\r"):
        line = line[:-1]
    if not line or line.startswith(":"):
        # Blank line (event boundary) or a comment / keepalive line.
        return None
    if not line.startswith("data:"):
        # Some other SSE field (event:/id:/retry:); irrelevant to us.
        return None
    value = line[len("data:") :]
    # Per the SSE spec strip exactly one optional leading space after the colon.
    if value.startswith(" "):
        value = value[1:]
    return value


def extract_content(payload: Any) -> str | None:
    """Safely pull the assistant text out of one parsed chat-completion object.

    Looks at ``choices[0].delta.content`` (the streaming shape) and falls back
    to ``choices[0].message.content`` (the shape a server returns when it
    ignores ``stream: true`` and replies with a single JSON object). Returns
    ``None`` whenever there is no text to emit: role-only deltas, empty
    deltas, finish-reason-only deltas, tool-call deltas, and the trailing
    usage chunk whose ``choices`` array is empty.
    """
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        # Empty list = usage-only chunk; guard before indexing choices[0].
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None

    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str) and content:
            return content

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content:
            return content

    return None


def extract_error(payload: Any) -> str | None:
    """Return a safe error code/type if a chat-completion object is an error.

    OpenAI-compatible servers sometimes return an error as a top-level
    ``{"error": {...}}`` object — occasionally even with HTTP 200. We surface a
    short, non-sensitive identifier (``code``/``type``) so the caller can fail
    loudly instead of treating it as an empty response. The free-form
    ``message`` is intentionally *not* returned, to avoid leaking content.
    """
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if error is None:
        return None
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
        return str(code) if code else "unknown"
    return "unknown"


def parse_payload(
    payload: str,
    *,
    on_parse_error: Callable[[int], None] | None = None,
) -> Any:
    """Interpret a single decoded ``data:`` payload string.

    Returns :data:`DONE_SENTINEL` for the ``[DONE]`` terminator, a non-empty
    ``str`` of assistant text to emit, or ``None`` when there is nothing to
    emit (blank payload, JSON with no text, or malformed JSON). On malformed
    JSON ``on_parse_error`` is invoked with the payload's UTF-8 byte length
    only - never its content.
    """
    stripped = payload.strip()
    if not stripped:
        return None
    if stripped == DONE:
        return DONE_SENTINEL
    try:
        obj = json.loads(stripped)
    except (ValueError, TypeError):
        if on_parse_error is not None:
            on_parse_error(len(payload.encode("utf-8")))
        return None
    return extract_content(obj)


def iter_text_deltas(
    payloads: Iterable[str],
    *,
    on_parse_error: Callable[[int], None] | None = None,
) -> Iterator[str]:
    """Yield assistant text fragments from decoded ``data:`` payload strings.

    Stops permanently at the ``[DONE]`` sentinel. This is the synchronous,
    fully testable core of the streaming pipeline; the async client feeds
    :class:`SSEDecoder` output through :func:`parse_payload` directly.
    """
    for payload in payloads:
        result = parse_payload(payload, on_parse_error=on_parse_error)
        if result is DONE_SENTINEL:
            return
        if result:
            yield result
