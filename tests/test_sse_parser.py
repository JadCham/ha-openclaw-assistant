"""Unit tests for the OpenAI-compatible SSE parser (sse.py)."""

from __future__ import annotations

import sse


def _chunk(*objs: object) -> str:
    """Render objects as SSE ``data:`` events (JSON), one per event."""
    import json

    return "".join(f"data: {json.dumps(o)}\n\n" for o in objs)


def _delta(content: str) -> dict:
    return {"choices": [{"index": 0, "delta": {"content": content}}]}


def collect(chunks, *, on_parse_error=None):
    """Feed byte/str chunks through the decoder, return (texts, saw_done)."""
    decoder = sse.SSEDecoder()
    out: list[str] = []
    for chunk in chunks:
        for payload in decoder.decode(chunk):
            result = sse.parse_payload(payload, on_parse_error=on_parse_error)
            if result is sse.DONE_SENTINEL:
                return out, True
            if result:
                out.append(result)
    for payload in decoder.flush():
        result = sse.parse_payload(payload, on_parse_error=on_parse_error)
        if result is sse.DONE_SENTINEL:
            return out, True
        if result:
            out.append(result)
    return out, False


# --- Normal streaming -------------------------------------------------------


def test_normal_stream_concatenates_content():
    body = _chunk(_delta("Hello"), _delta(", "), _delta("world"))
    texts, _ = collect([body.encode()])
    assert texts == ["Hello", ", ", "world"]
    assert "".join(texts) == "Hello, world"


def test_role_only_first_chunk_yields_nothing():
    role_chunk = {
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]
    }
    body = _chunk(role_chunk) + _chunk(_delta("Hi"))
    texts, _ = collect([body.encode()])
    assert texts == ["Hi"]


def test_done_sentinel_stops_stream():
    body = _chunk(_delta("first")) + "data: [DONE]\n\n" + _chunk(_delta("never"))
    texts, saw_done = collect([body.encode()])
    assert saw_done is True
    assert texts == ["first"]


def test_empty_delta_yields_nothing():
    body = _chunk({"choices": [{"index": 0, "delta": {}}]})
    texts, _ = collect([body.encode()])
    assert texts == []


def test_finish_reason_chunk_yields_nothing():
    body = _chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    texts, _ = collect([body.encode()])
    assert texts == []


def test_usage_chunk_with_empty_choices_does_not_crash():
    body = _chunk({"choices": [], "usage": {"total_tokens": 20}})
    texts, _ = collect([body.encode()])
    assert texts == []


def test_null_content_delta_yields_nothing():
    body = _chunk({"choices": [{"index": 0, "delta": {"content": None}}]})
    texts, _ = collect([body.encode()])
    assert texts == []


# --- Framing edge cases -----------------------------------------------------


def test_multiple_events_in_one_chunk():
    body = _chunk(_delta("a"), _delta("b"), _delta("c"))
    texts, _ = collect([body.encode()])  # whole body in a single chunk
    assert texts == ["a", "b", "c"]


def test_event_split_across_chunks():
    body = _chunk(_delta("split-test"))
    raw = body.encode()
    # Split at every single byte to maximally stress the line buffer.
    chunks = [raw[i : i + 1] for i in range(len(raw))]
    texts, _ = collect(chunks)
    assert texts == ["split-test"]


def test_split_in_middle_of_data_prefix():
    # "dat" + "a: {json}\n\n" — the 'data:' prefix itself is split.
    full = "data: " + '{"choices":[{"delta":{"content":"X"}}]}' + "\n\n"
    texts, _ = collect([full[:3], full[3:]])
    assert texts == ["X"]


def test_crlf_line_endings():
    payload = '{"choices":[{"delta":{"content":"crlf"}}]}'
    body = f"data: {payload}\r\n\r\n"
    texts, _ = collect([body.encode()])
    assert texts == ["crlf"]


def test_keepalive_and_comment_lines_skipped():
    body = ": keep-alive\n\n: OPENROUTER PROCESSING\n\n" + _chunk(
        _delta("after-keepalive")
    )
    texts, _ = collect([body.encode()])
    assert texts == ["after-keepalive"]


def test_other_sse_fields_ignored():
    body = "event: message\nid: 42\nretry: 1000\n" + _chunk(_delta("payload"))
    texts, _ = collect([body.encode()])
    assert texts == ["payload"]


def test_data_without_leading_space_accepted():
    # Some servers emit "data:{json}" with no space after the colon.
    body = 'data:{"choices":[{"delta":{"content":"nospace"}}]}\n\n'
    texts, _ = collect([body.encode()])
    assert texts == ["nospace"]


def test_trailing_event_without_newline_flushed():
    body = 'data: {"choices":[{"delta":{"content":"tail"}}]}'  # no trailing newline
    texts, _ = collect([body.encode()])
    assert texts == ["tail"]


# --- Malformed JSON: never leak content -------------------------------------


def test_malformed_json_reports_byte_length_only():
    seen: list[int] = []
    bad = 'data: {"choices":[{"delta":{"content":"oops"  BROKEN\n\n'
    good = _chunk(_delta("recovered"))
    texts, _ = collect([(bad + good).encode()], on_parse_error=seen.append)
    # Parsing continues past the bad event.
    assert texts == ["recovered"]
    # The callback received exactly one int (a byte count), never the content.
    assert len(seen) == 1
    assert isinstance(seen[0], int)
    assert seen[0] > 0


def test_malformed_json_without_callback_is_silently_skipped():
    bad = "data: not-json-at-all\n\n"
    texts, _ = collect([(bad + _chunk(_delta("ok"))).encode()])
    assert texts == ["ok"]


# --- Non-streaming fallback shape -------------------------------------------


def test_extract_content_handles_non_streaming_message():
    obj = {"choices": [{"message": {"role": "assistant", "content": "full answer"}}]}
    assert sse.extract_content(obj) == "full answer"


def test_extract_content_prefers_delta_then_message():
    obj = {"choices": [{"delta": {"content": "d"}, "message": {"content": "m"}}]}
    assert sse.extract_content(obj) == "d"


def test_extract_content_guards_non_dict_and_empty():
    assert sse.extract_content(None) is None
    assert sse.extract_content({"choices": []}) is None
    assert sse.extract_content({"choices": "nope"}) is None
    assert sse.extract_content({}) is None


# --- iter_text_deltas convenience over a payload list -----------------------


def test_iter_text_deltas_stops_at_done():
    import json

    payloads = [
        json.dumps(_delta("one")),
        json.dumps(_delta("two")),
        "[DONE]",
        json.dumps(_delta("three")),
    ]
    assert list(sse.iter_text_deltas(payloads)) == ["one", "two"]


def test_full_representative_wire_stream():
    """The exact byte pattern a real server produces, fed in one blob."""
    wire = (
        'data: {"id":"c","object":"chat.completion.chunk","choices":'
        '[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
        ": keep-alive\n\n"
        'data: {"id":"c","object":"chat.completion.chunk","choices":'
        '[{"index":0,"delta":{"content":"Two"},"finish_reason":null}]}\n\n'
        'data: {"id":"c","object":"chat.completion.chunk","choices":'
        '[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: {"id":"c","object":"chat.completion.chunk","choices":[],'
        '"usage":{"prompt_tokens":18,"completion_tokens":2,"total_tokens":20}}\n\n'
        "data: [DONE]\n\n"
    )
    texts, saw_done = collect([wire.encode()])
    assert texts == ["Two"]
    assert saw_done is True


# --- Multibyte UTF-8 split across chunks (regression) ------------------------


def test_multibyte_utf8_split_at_every_byte_boundary():
    """A multibyte char split across network chunks must not corrupt text."""
    import json

    content = "café 😀 中文 — ünïcödé"
    # ensure_ascii=False so the wire bytes actually contain multibyte sequences
    # (a real server may send raw UTF-8 rather than \\uXXXX escapes).
    payload = json.dumps(_delta(content), ensure_ascii=False)
    raw = f"data: {payload}\n\n".encode()

    for split in range(1, len(raw)):
        texts, _ = collect([raw[:split], raw[split:]])
        assert texts == [content], f"corruption when split at byte {split}: {texts!r}"

    # Maximal fragmentation: one byte per chunk.
    texts, _ = collect([raw[i : i + 1] for i in range(len(raw))])
    assert texts == [content]


# --- Error-object detection -------------------------------------------------


def test_extract_error_detects_error_object():
    assert (
        sse.extract_error({"error": {"code": "context_length_exceeded"}})
        == "context_length_exceeded"
    )
    assert (
        sse.extract_error({"error": {"type": "invalid_request_error"}})
        == "invalid_request_error"
    )
    assert sse.extract_error({"error": "boom"}) == "unknown"
    assert sse.extract_error({"error": {}}) == "unknown"


def test_extract_error_none_for_normal_payloads():
    assert sse.extract_error({"choices": [{"delta": {"content": "hi"}}]}) is None
    assert sse.extract_error({}) is None
    assert sse.extract_error(None) is None
