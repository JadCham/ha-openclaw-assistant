# OpenClaw Assistant for Home Assistant

[![Validate](https://github.com/JadCham/ha-openclaw-assistant/actions/workflows/validate.yml/badge.svg)](https://github.com/JadCham/ha-openclaw-assistant/actions/workflows/validate.yml)
[![Tests](https://github.com/JadCham/ha-openclaw-assistant/actions/workflows/tests.yml/badge.svg)](https://github.com/JadCham/ha-openclaw-assistant/actions/workflows/tests.yml)
[![hacs](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://hacs.xyz)

A Home Assistant custom integration that lets **Home Assistant Assist** use
**OpenClaw** as its conversation agent, with **true streamed response
deltas**.

```
HA Assist  →  OpenClaw (OpenAI-compatible, streaming)  →  ChatLog deltas  →  streaming TTS
```

The whole point is latency: assistant text is pushed into the Home Assistant
chat log **as it streams from OpenClaw**, so a streaming-capable TTS engine
(e.g. a patched Deepgram TTS entity) can begin speaking *before* OpenClaw has
finished generating the full answer.

> This is a **conversation agent that calls OpenClaw**. It is *not* a bridge
> that lets OpenClaw control Home Assistant. By design it does not expose your
> entities or tools to the model.

## How it works

1. You speak to Assist; Home Assistant hands the utterance to this integration.
2. The integration replays the conversation history as OpenAI-style `messages`
   and calls `POST {base_url}/v1/chat/completions` with `stream: true`.
3. A small, well-tested SSE parser turns OpenClaw's streaming chunks into text
   fragments.
4. Each fragment is fed into Home Assistant's `ChatLog` via
   `async_add_delta_content_stream(...)` — **no full-response buffering**.
5. Home Assistant forwards those deltas to your selected streaming TTS engine.

If OpenClaw ignores `stream: true` and returns a single JSON object, the
integration transparently falls back to reading the full message.

## Requirements

- Home Assistant **2026.6.0** or newer.
- An OpenClaw instance exposing an **OpenAI-compatible**
  `/v1/chat/completions` endpoint.
- A streaming-capable TTS entity if you want to hear the streaming benefit
  (Assist will work with any TTS, but non-streaming TTS waits for the full
  answer).

## Installation (HACS)

This integration is installed as a **HACS custom repository**.

1. In Home Assistant, open **HACS**.
2. Click the **⋮** menu (top-right) → **Custom repositories**.
3. Enter the repository URL:
   `https://github.com/JadCham/ha-openclaw-assistant`
4. Select category **Integration** and click **Add**.
5. Find **OpenClaw Assistant** in the list and **Download** it.
6. **Restart Home Assistant**.

> The integration icon may appear generic until it is added to
> [home-assistant/brands](https://github.com/home-assistant/brands); this does
> not affect functionality.

### Manual installation (alternative)

Copy `custom_components/openclaw_assistant` into your Home Assistant
`config/custom_components/` directory and restart.

## Configuration

1. Go to **Settings → Devices & services → Add integration**.
2. Search for **OpenClaw Assistant**.
3. Fill in:
   - **Base URL** — e.g. `http://openclaw.local:18789`
   - **API key or token** — optional; sent as a bearer token. Leave blank if
     OpenClaw is unauthenticated.
   - **Model or agent identifier** — e.g. `ha-voice`
   - **System prompt** — optional instructions prepended to every conversation.
   - **Request timeout** — seconds (default `120`; this is an *inactivity*
     timeout while streaming, so long answers are fine).
   - **Verify TLS certificate** — leave on unless you are using a self-signed
     certificate on a trusted network.
   - **Test the connection now** — sends a tiny request to confirm the URL and
     token before saving. Untick to save without validating.

Settings other than the base URL/token can be changed later via
**Configure** (the options flow); changing the base URL or token uses
**Reconfigure**. Both reload the integration automatically.

### Selecting it as your voice agent

1. Go to **Settings → Voice assistants** and edit (or create) an Assist
   pipeline.
2. Set **Conversation agent** to **OpenClaw Assistant**.
3. Set **Text-to-speech** to a **streaming-capable** engine (e.g. a patched
   Deepgram TTS entity).

## Verifying streaming TTS

Use a prompt long enough to make streaming obvious:

```
Explain the plan for debugging a slow Home Assistant voice pipeline in ten concise steps.
```

Expected behaviour:

- Home Assistant enters conversation processing.
- OpenClaw streaming starts.
- This integration emits assistant text deltas into the chat log.
- **TTS begins speaking before the full answer is complete.**

The benchmark: in the HA state/log history, TTS should start while the
conversation is *still processing*, not only after the full response is done.

If TTS still waits until the end, check whether:

- HA Assist is actually consuming chat-log deltas (pipeline debug).
- The TTS provider buffers `message_gen` (not all engines stream).
- The selected pipeline is really using this conversation entity.

## Security

- API keys, bearer tokens, `Authorization` headers and full URLs containing
  secrets are **never logged**.
- Full prompts, full assistant responses and raw SSE chunks are **never
  logged**. Malformed chunks are reported by **byte length only**.
- TLS verification is **on by default**.
- No shell/subprocess execution, no `eval`/`exec`, no dynamic imports from user
  input, no arbitrary file access.
- The integration uses Home Assistant's shared aiohttp client, explicit
  timeouts, and propagates cancellation so interrupting Assist aborts the
  upstream request.
- Diagnostics downloads redact all secrets.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install pytest ruff

ruff check .
python -m compileall custom_components/openclaw_assistant tests
pytest -q
```

The SSE parser (`sse.py`) and redaction helpers (`redact.py`) are pure modules
with no Home Assistant dependency, so their unit tests run without a Home
Assistant test harness.

## License

[MIT](LICENSE)
