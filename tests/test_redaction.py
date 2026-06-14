"""Unit tests for the redaction helpers (redact.py)."""

from __future__ import annotations

import redact

# --- URL redaction ----------------------------------------------------------


def test_redact_url_strips_query_token():
    url = "http://openclaw.local:18789/v1/chat/completions?api_key=supersecret&x=1"
    out = redact.redact_url(url)
    assert "supersecret" not in out
    assert "openclaw.local" in out
    assert "18789" in out
    assert "/v1/chat/completions" in out


def test_redact_url_strips_userinfo():
    url = "https://user:p4ssw0rd@host.example/v1"
    out = redact.redact_url(url)
    assert "p4ssw0rd" not in out
    assert "user" not in out
    assert "host.example" in out


def test_redact_url_preserves_scheme_host_path():
    out = redact.redact_url("http://openclaw.local:18789/v1/chat/completions")
    assert out == "http://openclaw.local:18789/v1/chat/completions"


def test_redact_url_marks_query_presence_without_leaking():
    out = redact.redact_url("https://h/x?token=abc")
    assert "abc" not in out
    assert out.endswith("?redacted")


def test_redact_url_empty_and_invalid():
    assert redact.redact_url("") == ""
    # A bare host with no scheme is treated as not-a-URL and fully redacted.
    assert redact.redact_url("not a url at all") == redact.REDACTED


def test_redact_url_strips_fragment():
    out = redact.redact_url("http://h:18789/v1#secrettoken")
    assert "secrettoken" not in out
    assert out == "http://h:18789/v1"


def test_redact_url_preserves_ipv6_brackets():
    out = redact.redact_url("http://[::1]:8080/v1/chat/completions")
    assert out == "http://[::1]:8080/v1/chat/completions"


def test_redact_url_strips_path_matrix_params():
    out = redact.redact_url("https://host/v1;api_key=SECRET/chat/completions")
    assert "SECRET" not in out
    assert out == "https://host/v1/chat/completions"
    out2 = redact.redact_url("https://host/path;jsessionid=SECRET")
    assert "SECRET" not in out2
    assert out2 == "https://host/path"


# --- Header redaction -------------------------------------------------------


def test_redact_headers_masks_authorization_case_insensitive():
    headers = {"Authorization": "Bearer abc123", "Content-Type": "application/json"}
    out = redact.redact_headers(headers)
    assert out["Authorization"] == redact.REDACTED
    assert "abc123" not in str(out)
    assert out["Content-Type"] == "application/json"


def test_redact_headers_masks_api_key_variants():
    headers = {"x-api-key": "k", "api-key": "k2", "OpenAI-API-Key": "k3"}
    out = redact.redact_headers(headers)
    assert out["x-api-key"] == redact.REDACTED
    assert out["api-key"] == redact.REDACTED
    assert out["OpenAI-API-Key"] == redact.REDACTED


# --- Mapping redaction ------------------------------------------------------


def test_redact_mapping_masks_secret_keys():
    data = {"base_url": "http://h", "api_key": "secret", "model": "ha-voice"}
    out = redact.redact_mapping(data)
    assert out["api_key"] == redact.REDACTED
    assert out["base_url"] == "http://h"
    assert out["model"] == "ha-voice"
    assert "secret" not in str(out)


def test_redact_mapping_recurses_into_nested():
    data = {"outer": {"token": "t", "keep": "v"}}
    out = redact.redact_mapping(data)
    assert out["outer"]["token"] == redact.REDACTED
    assert out["outer"]["keep"] == "v"


def test_redact_mapping_recurses_into_lists():
    data = {"items": [{"api_key": "LEAK", "keep": "v"}, {"token": "t2"}]}
    out = redact.redact_mapping(data)
    assert out["items"][0]["api_key"] == redact.REDACTED
    assert out["items"][0]["keep"] == "v"
    assert out["items"][1]["token"] == redact.REDACTED
    assert "LEAK" not in str(out)
    assert "t2" not in str(out)


# --- Free-text redaction ----------------------------------------------------


def test_redact_text_never_returns_content():
    secret = "the user said something private"
    out = redact.redact_text(secret)
    assert secret not in out
    assert "private" not in out
    assert str(len(secret)) in out


def test_redact_text_none():
    assert redact.redact_text(None) == "<none>"


def test_byte_len_counts_utf8_not_content():
    out = redact.byte_len("héllo")  # é is 2 bytes in UTF-8
    assert out == 6
    assert redact.byte_len(b"abc") == 3
    assert redact.byte_len(None) == 0
