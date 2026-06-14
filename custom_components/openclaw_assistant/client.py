"""Thin async HTTP client for an OpenClaw OpenAI-compatible endpoint.

The client owns the wire concerns - building the ``/v1/chat/completions``
request, streaming the SSE response, falling back to a plain JSON body when
the server ignores ``stream: true``, mapping HTTP failures onto typed errors,
and honouring cancellation - so the conversation entity can stay focused on
the Home Assistant chat-log contract.

It uses Home Assistant's shared aiohttp ``ClientSession`` and never performs
blocking I/O. Nothing here logs prompts, responses, raw payloads, tokens or
``Authorization`` headers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any, Final

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import LOGGER
from .sse import (
    DONE_SENTINEL,
    SSEDecoder,
    extract_content,
    extract_error,
    parse_payload,
)

#: Connect timeout (seconds) used for the streaming request regardless of the
#: configured per-read timeout, so a dead host fails fast.
_CONNECT_TIMEOUT: Final = 30


class OpenClawError(HomeAssistantError):
    """Base error for all OpenClaw client failures."""


class OpenClawAuthError(OpenClawError):
    """Raised on HTTP 401/403 (bad or missing token)."""


class OpenClawConnectionError(OpenClawError):
    """Raised when the OpenClaw host cannot be reached."""


class OpenClawTimeoutError(OpenClawError):
    """Raised when the OpenClaw request times out."""


class OpenClawResponseError(OpenClawError):
    """Raised on other non-success HTTP responses (404, 5xx, ...)."""

    def __init__(self, message: str, status: int) -> None:
        """Store the HTTP status alongside the message."""
        super().__init__(message)
        self.status = status


def _build_endpoint(base_url: str) -> str:
    """Build the chat-completions URL from a user-supplied base URL.

    Accepts a bare host (``http://openclaw.local:18789``), a ``/v1`` root, or
    the full endpoint, and always resolves to a single clean
    ``.../v1/chat/completions`` URL.
    """
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


class OpenClawClient:
    """Streaming client for an OpenClaw chat-completions endpoint."""

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout: int,
        verify_tls: bool,
    ) -> None:
        """Initialise the client against Home Assistant's shared session."""
        # Shared, HA-managed session - never closed by us.
        self._session = async_get_clientsession(hass, verify_ssl=verify_tls)
        self._endpoint = _build_endpoint(base_url)
        self._api_key = api_key or None
        self._model = model
        self._timeout = timeout

    @property
    def endpoint(self) -> str:
        """Return the resolved chat-completions endpoint (no secrets)."""
        return self._endpoint

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @staticmethod
    def _raise_for_status(status: int) -> None:
        """Map an HTTP status onto a typed error, logging only the status."""
        if status in (401, 403):
            raise OpenClawAuthError(f"OpenClaw request failed: status={status}")
        if status >= 400:
            raise OpenClawResponseError(
                f"OpenClaw request failed: status={status}", status
            )

    @staticmethod
    def _log_parse_error(byte_count: int) -> None:
        """Log a malformed SSE payload without leaking its content."""
        LOGGER.debug(
            "OpenClaw stream chunk parse failed: invalid JSON, bytes=%d", byte_count
        )

    async def async_validate(self) -> None:
        """Send a tiny non-streaming request to verify connectivity and auth.

        Deliberately avoids ``/v1/models`` (not every OpenClaw build exposes
        it) and uses a minimal chat-completions call instead. Raises an
        :class:`OpenClawError` subclass on failure; returns ``None`` on success.
        """
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "max_tokens": 1,
        }
        timeout = aiohttp.ClientTimeout(total=min(self._timeout, _CONNECT_TIMEOUT))
        try:
            async with self._session.post(
                self._endpoint,
                json=body,
                headers=self._headers(),
                timeout=timeout,
            ) as resp:
                self._raise_for_status(resp.status)
                # Drain and discard the body; we only care that it succeeded.
                await resp.read()
        except OpenClawError:
            raise
        except TimeoutError as err:
            raise OpenClawTimeoutError("OpenClaw request timed out") from err
        except aiohttp.ClientError as err:
            raise OpenClawConnectionError("Could not reach OpenClaw") from err

    async def stream_chat_completion(
        self, messages: Iterable[Mapping[str, Any]]
    ) -> AsyncIterator[str]:
        """Stream assistant text fragments for ``messages``.

        Requests with ``stream: true`` and yields text as it arrives. If the
        server replies with a plain JSON object instead of an SSE stream, the
        full message content is yielded as a single fragment so the caller
        still gets an answer. Raises an :class:`OpenClawError` subclass on
        transport/HTTP failure and propagates :class:`asyncio.CancelledError`
        untouched when Assist is interrupted.
        """
        body: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "stream": True,
        }
        # No total cap: a long answer is fine. sock_read is an inactivity
        # timeout so a stalled upstream still fails. sock_connect fails fast
        # on a dead host.
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=min(self._timeout, _CONNECT_TIMEOUT),
            sock_read=self._timeout,
        )
        try:
            async with self._session.post(
                self._endpoint,
                json=body,
                headers=self._headers(),
                timeout=timeout,
            ) as resp:
                self._raise_for_status(resp.status)
                content_type = (resp.headers.get("Content-Type") or "").lower()
                if "text/event-stream" in content_type:
                    async for text in self._iter_sse(resp):
                        yield text
                else:
                    # Server ignored stream: true and returned a JSON object.
                    async for text in self._iter_json(resp):
                        yield text
        except OpenClawError:
            raise
        except TimeoutError as err:
            raise OpenClawTimeoutError("OpenClaw request timed out") from err
        except aiohttp.ClientError as err:
            raise OpenClawConnectionError("Error talking to OpenClaw") from err

    async def _iter_sse(self, resp: aiohttp.ClientResponse) -> AsyncIterator[str]:
        """Decode an SSE response body into assistant text fragments."""
        decoder = SSEDecoder()
        async for chunk in resp.content.iter_any():
            for payload in decoder.decode(chunk):
                result = parse_payload(payload, on_parse_error=self._log_parse_error)
                if result is DONE_SENTINEL:
                    return
                if result:
                    yield result
        for payload in decoder.flush():
            result = parse_payload(payload, on_parse_error=self._log_parse_error)
            if result is DONE_SENTINEL:
                return
            if result:
                yield result

    async def _iter_json(self, resp: aiohttp.ClientResponse) -> AsyncIterator[str]:
        """Parse a non-streaming JSON chat-completion body into one fragment."""
        try:
            # content_type=None: accept whatever Content-Type the server set.
            data = await resp.json(content_type=None)
        except (aiohttp.ClientError, ValueError) as err:
            raise OpenClawResponseError(
                "OpenClaw returned an unreadable response", resp.status
            ) from err
        # Some servers return an error object with HTTP 200; fail loudly instead
        # of reporting an empty response. Only a safe code/type is surfaced.
        if (error_code := extract_error(data)) is not None:
            raise OpenClawResponseError(
                f"OpenClaw returned an error: {error_code}", resp.status
            )
        if (text := extract_content(data)) is not None:
            yield text
