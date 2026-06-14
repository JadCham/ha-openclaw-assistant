"""End-to-end streaming tests against the real Home Assistant ChatLog.

These exercise the actual ``ChatLog.async_add_delta_content_stream`` contract
and the OpenClaw client's SSE/JSON handling with a mocked aiohttp session, so
they verify the streaming-TTS mechanism: that text deltas reach the chat log's
``delta_listener`` *incrementally*, not in one buffered blob.

The module is skipped entirely when Home Assistant is not installed (e.g. the
lightweight lint/test CI job), so the pure-module tests can run on their own.
"""

from __future__ import annotations

import pytest

pytest.importorskip("homeassistant")

from homeassistant.components.conversation import ConversationInput
from homeassistant.components.conversation.chat_log import (
    AssistantContent,
    ChatLog,
    SystemContent,
    ToolResultContent,
    UserContent,
)
from homeassistant.core import Context
from homeassistant.helpers import intent

from custom_components.openclaw_assistant import client as client_mod
from custom_components.openclaw_assistant.client import (
    OpenClawAuthError,
    OpenClawClient,
    OpenClawConnectionError,
    OpenClawResponseError,
    OpenClawTimeoutError,
)
from custom_components.openclaw_assistant.conversation import (
    OpenClawConversationEntity,
    _user_facing_error,
)

_USER_MSG = [{"role": "user", "content": "hi"}]


class _FakeEntry:
    """Minimal config-entry stand-in for the conversation entity."""

    def __init__(self, *, data=None, options=None, runtime_data=None) -> None:
        self.entry_id = "test_entry"
        self.title = "OpenClaw (ha-voice)"
        self.data = data or {}
        self.options = options or {}
        self.runtime_data = runtime_data


def _make_entity(client=None, *, data=None, options=None) -> OpenClawConversationEntity:
    entity = OpenClawConversationEntity(
        _FakeEntry(data=data, options=options, runtime_data=client)
    )
    entity.entity_id = "conversation.openclaw_test"
    return entity


def _user_input(text: str = "hi") -> ConversationInput:
    return ConversationInput(
        text=text,
        context=Context(),
        conversation_id="conv-1",
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="conversation.openclaw_test",
    )


class _FakeHass:
    """Minimal stand-in: ChatLog only touches ``hass.data`` without tools."""

    def __init__(self) -> None:
        self.data: dict = {}


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    def __init__(self, *, status=200, headers=None, chunks=None, json_data=None):
        self.status = status
        self.headers = headers or {}
        self.content = _FakeContent(chunks or [])
        self._json = json_data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def read(self):
        return b""

    async def json(self, content_type=None):
        return self._json


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.posted: list[tuple] = []

    def post(self, url, **kwargs):
        self.posted.append((url, kwargs))
        return self._response


def _make_client(
    monkeypatch, response: _FakeResponse
) -> tuple[OpenClawClient, _FakeSession]:
    session = _FakeSession(response)
    monkeypatch.setattr(client_mod, "async_get_clientsession", lambda *a, **k: session)
    client = OpenClawClient(
        _FakeHass(),
        base_url="http://openclaw.local:18789",
        api_key="secret-token",
        model="ha-voice",
        timeout=30,
        verify_tls=True,
    )
    return client, session


# --- ChatLog streaming mechanism --------------------------------------------


async def test_deltas_reach_listener_incrementally():
    """The core promise: each fragment is delivered to the listener as it lands."""
    chat_log = ChatLog(_FakeHass(), "conv-1")
    chat_log.content.append(UserContent(content="hello"))

    seen: list[dict] = []
    chat_log.delta_listener = lambda _cl, delta: seen.append(delta)

    async def deltas():
        yield {"role": "assistant"}
        yield {"content": "Hello "}
        yield {"content": "world"}

    added = [
        content
        async for content in chat_log.async_add_delta_content_stream(
            "conversation.openclaw", deltas()
        )
    ]

    # The listener saw the role opener and then each content fragment, in order
    # and incrementally (more than one content delta) - this is what feeds TTS.
    content_deltas = [d.get("content") for d in seen if "content" in d]
    assert content_deltas == ["Hello ", "world"]

    # A single assistant message was recorded, with the concatenated text.
    assert len(added) == 1
    assert isinstance(added[0], AssistantContent)
    assert added[0].content == "Hello world"
    assert chat_log.content[-1].content == "Hello world"


# --- Client SSE streaming ---------------------------------------------------


async def test_client_streams_sse_fragments(monkeypatch):
    wire = (
        b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"Two"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" steps"}}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    # Deliberately split the wire across awkward byte boundaries.
    chunks = [wire[:40], wire[40:95], wire[95:]]
    response = _FakeResponse(
        status=200,
        headers={"Content-Type": "text/event-stream"},
        chunks=chunks,
    )
    client, session = _make_client(monkeypatch, response)

    fragments = [frag async for frag in client.stream_chat_completion(_USER_MSG)]
    assert fragments == ["Two", " steps"]

    # The request was made with stream:true and the bearer header (never logged).
    url, kwargs = session.posted[0]
    assert url.endswith("/v1/chat/completions")
    assert kwargs["json"]["stream"] is True
    assert kwargs["headers"]["Authorization"] == "Bearer secret-token"


async def test_client_falls_back_to_non_streaming_json(monkeypatch):
    response = _FakeResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        json_data={
            "choices": [{"message": {"role": "assistant", "content": "full answer"}}]
        },
    )
    client, _ = _make_client(monkeypatch, response)

    fragments = [frag async for frag in client.stream_chat_completion(_USER_MSG)]
    assert fragments == ["full answer"]


async def test_client_maps_auth_error(monkeypatch):
    response = _FakeResponse(status=401, headers={"Content-Type": "application/json"})
    client, _ = _make_client(monkeypatch, response)

    with pytest.raises(OpenClawAuthError):
        async for _ in client.stream_chat_completion(_USER_MSG):
            pass


async def test_client_maps_server_error(monkeypatch):
    response = _FakeResponse(status=500, headers={"Content-Type": "application/json"})
    client, _ = _make_client(monkeypatch, response)

    with pytest.raises(OpenClawResponseError) as excinfo:
        async for _ in client.stream_chat_completion(_USER_MSG):
            pass
    assert excinfo.value.status == 500


async def test_client_sse_routing_with_charset_suffix(monkeypatch):
    wire = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
    response = _FakeResponse(
        status=200,
        headers={"Content-Type": "text/event-stream; charset=utf-8"},
        chunks=[wire],
    )
    client, _ = _make_client(monkeypatch, response)
    assert [f async for f in client.stream_chat_completion(_USER_MSG)] == ["hi"]


async def test_client_json_routing_with_charset_suffix(monkeypatch):
    response = _FakeResponse(
        status=200,
        headers={"Content-Type": "application/json; charset=utf-8"},
        json_data={"choices": [{"message": {"content": "full"}}]},
    )
    client, _ = _make_client(monkeypatch, response)
    assert [f async for f in client.stream_chat_completion(_USER_MSG)] == ["full"]


async def test_client_json_error_body_raises_even_on_200(monkeypatch):
    response = _FakeResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        json_data={"error": {"type": "invalid_request_error"}},
    )
    client, _ = _make_client(monkeypatch, response)
    with pytest.raises(OpenClawResponseError):
        async for _ in client.stream_chat_completion(_USER_MSG):
            pass


async def test_client_json_empty_content_yields_nothing(monkeypatch):
    response = _FakeResponse(
        status=200,
        headers={"Content-Type": "application/json"},
        json_data={"choices": [{"message": {"content": None}}]},
    )
    client, _ = _make_client(monkeypatch, response)
    assert [f async for f in client.stream_chat_completion(_USER_MSG)] == []


# --- Conversation entity: message mapping -----------------------------------


def test_build_messages_configured_prompt_wins_over_framework():
    entity = _make_entity(data={"system_prompt": "be brief"})
    chat_log = ChatLog(_FakeHass(), "c")
    chat_log.content = [
        SystemContent(content="framework prompt"),
        UserContent(content="hi"),
        AssistantContent(agent_id="x", content="hello"),
        AssistantContent(agent_id="x", content=None),  # empty -> skipped
        UserContent(content="more"),
    ]
    msgs = entity._build_messages(chat_log)
    assert msgs[0] == {"role": "system", "content": "be brief"}
    assert {"role": "system", "content": "framework prompt"} not in msgs
    assert {"role": "user", "content": "hi"} in msgs
    assert {"role": "assistant", "content": "hello"} in msgs
    # No empty assistant content leaked through.
    assert all(m["content"] for m in msgs)


def test_build_messages_uses_framework_prompt_when_unconfigured():
    entity = _make_entity(data={})
    chat_log = ChatLog(_FakeHass(), "c")
    chat_log.content = [
        SystemContent(content="framework prompt"),
        UserContent(content="hi"),
    ]
    msgs = entity._build_messages(chat_log)
    assert msgs[0] == {"role": "system", "content": "framework prompt"}


def test_build_messages_ignores_tool_results():
    entity = _make_entity(data={})
    chat_log = ChatLog(_FakeHass(), "c")
    chat_log.content = [
        SystemContent(content=""),
        UserContent(content="hi"),
        ToolResultContent(
            agent_id="x", tool_call_id="1", tool_name="t", tool_result={}
        ),
    ]
    msgs = entity._build_messages(chat_log)
    assert all(m["role"] in ("system", "user", "assistant") for m in msgs)


# --- Conversation entity: handle-message paths ------------------------------


class _FakeStreamClient:
    def __init__(self, fragments) -> None:
        self._fragments = fragments

    async def stream_chat_completion(self, messages):
        for fragment in self._fragments:
            yield fragment


class _FakeErrorClient:
    def __init__(self, exc) -> None:
        self._exc = exc

    async def stream_chat_completion(self, messages):
        raise self._exc
        yield  # pragma: no cover - makes this an async generator


async def test_entity_streams_and_returns_speech():
    entity = _make_entity(_FakeStreamClient(["Hello ", "world"]), data={})
    chat_log = ChatLog(_FakeHass(), "c")
    chat_log.content.append(UserContent(content="hi"))

    result = await entity._async_handle_message(_user_input(), chat_log)
    assert result.response.speech["plain"]["speech"] == "Hello world"
    assert result.response.error_code is None


async def test_entity_empty_response_returns_error():
    entity = _make_entity(_FakeStreamClient([]), data={})
    chat_log = ChatLog(_FakeHass(), "c")
    chat_log.content.append(UserContent(content="hi"))

    result = await entity._async_handle_message(_user_input(), chat_log)
    assert result.response.error_code == intent.IntentResponseErrorCode.UNKNOWN


async def test_entity_client_error_returns_spoken_error():
    entity = _make_entity(_FakeErrorClient(OpenClawAuthError("nope")), data={})
    chat_log = ChatLog(_FakeHass(), "c")
    chat_log.content.append(UserContent(content="hi"))

    result = await entity._async_handle_message(_user_input(), chat_log)
    assert result.response.error_code == intent.IntentResponseErrorCode.UNKNOWN
    assert "credential" in result.response.speech["plain"]["speech"].lower()


def test_user_facing_error_messages_are_distinct_and_safe():
    auth = _user_facing_error(OpenClawAuthError("x"))
    timeout = _user_facing_error(OpenClawTimeoutError("x"))
    conn = _user_facing_error(OpenClawConnectionError("x"))
    assert "credential" in auth.lower()
    assert "too long" in timeout.lower()
    assert "reach" in conn.lower()
    # The raw exception text must not be echoed into the spoken message.
    assert "x" not in {auth, timeout, conn}
