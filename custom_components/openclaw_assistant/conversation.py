"""Conversation platform: an Assist agent backed by OpenClaw.

This entity receives a Home Assistant Assist utterance, replays the
conversation history into OpenAI-compatible ``messages``, asks OpenClaw for a
streamed completion, and feeds the text fragments into the Home Assistant
chat log as delta content. Because the deltas are added incrementally,
Home Assistant forwards them to a streaming-capable TTS engine while the
model is still generating - TTS starts before the answer is finished.

The integration is intentionally minimal: it does not expose Home Assistant
entities/tools to the model. It is a conversation agent that calls OpenClaw,
not a bridge that lets OpenClaw control Home Assistant.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal

from homeassistant.components import conversation
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .client import (
    OpenClawAuthError,
    OpenClawClient,
    OpenClawConnectionError,
    OpenClawError,
    OpenClawTimeoutError,
)
from .const import (
    CONF_MODEL,
    CONF_SYSTEM_PROMPT,
    DEFAULT_MODEL,
    DOMAIN,
    LOGGER,
    MANUFACTURER,
)
from .redact import redact_text

if TYPE_CHECKING:
    from . import OpenClawConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: OpenClawConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the OpenClaw conversation entity from a config entry."""
    async_add_entities([OpenClawConversationEntity(config_entry)])


class OpenClawConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    """A Home Assistant Assist conversation agent powered by OpenClaw."""

    _attr_has_entity_name = True
    _attr_name = None
    # Advertise streaming so Assist wires our deltas to streaming TTS.
    _attr_supports_streaming = True

    def __init__(self, entry: OpenClawConfigEntry) -> None:
        """Initialise the entity for a config entry."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        model = {**entry.data, **entry.options}.get(CONF_MODEL, DEFAULT_MODEL)
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer=MANUFACTURER,
            model=model,
            entry_type=DeviceEntryType.SERVICE,
        )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """OpenClaw handles language selection itself; accept everything."""
        return MATCH_ALL

    @property
    def _client(self) -> OpenClawClient:
        return self.entry.runtime_data

    @property
    def _system_prompt(self) -> str:
        return {**self.entry.data, **self.entry.options}.get(CONF_SYSTEM_PROMPT, "")

    async def async_added_to_hass(self) -> None:
        """Register as the conversation agent for this config entry."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Deregister the conversation agent."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    def _build_messages(self, chat_log: conversation.ChatLog) -> list[dict[str, str]]:
        """Map the HA chat log onto OpenAI-compatible ``messages``.

        We manage the system prompt ourselves rather than via the HA LLM API,
        because the MVP deliberately does not expose HA entities/tools to the
        model. Tool-result content is skipped (no tools in this pass).
        """
        messages: list[dict[str, str]] = []
        system_prompt = self._system_prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for content in chat_log.content:
            if isinstance(content, conversation.SystemContent):
                # Only honour a non-empty framework system message when the
                # user has not configured their own prompt.
                if content.content and not system_prompt:
                    messages.append({"role": "system", "content": content.content})
            elif isinstance(content, conversation.UserContent):
                messages.append({"role": "user", "content": content.content})
            elif isinstance(content, conversation.AssistantContent) and content.content:
                messages.append({"role": "assistant", "content": content.content})
            # conversation.ToolResultContent is intentionally ignored.
        return messages

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process one utterance: stream OpenClaw's reply into the chat log."""
        messages = self._build_messages(chat_log)
        model = {**self.entry.data, **self.entry.options}.get(CONF_MODEL, DEFAULT_MODEL)
        LOGGER.debug(
            "OpenClaw request: model=%s messages=%d last_user=%s",
            model,
            len(messages),
            redact_text(user_input.text),
        )

        async def deltas() -> AsyncIterator[conversation.AssistantContentDeltaDict]:
            # The first delta must carry the role to open a new assistant
            # message; subsequent deltas carry text fragments.
            yield {"role": "assistant"}
            async for fragment in self._client.stream_chat_completion(messages):
                if fragment:
                    yield {"content": fragment}

        try:
            added = [
                content
                async for content in chat_log.async_add_delta_content_stream(
                    self.entity_id, deltas()
                )
            ]
        except OpenClawError as err:
            LOGGER.error("OpenClaw conversation failed: %s", err)
            return self._error_result(user_input, _user_facing_error(err))
        except HomeAssistantError as err:
            LOGGER.error("OpenClaw conversation failed: %s", err)
            return self._error_result(
                user_input, "Sorry, there was a problem talking to OpenClaw."
            )

        if not any(
            isinstance(content, conversation.AssistantContent) for content in added
        ):
            LOGGER.warning("OpenClaw returned an empty response")
            return self._error_result(
                user_input, "Sorry, OpenClaw did not return a response."
            )

        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    def _error_result(
        self, user_input: conversation.ConversationInput, message: str
    ) -> conversation.ConversationResult:
        """Build a spoken error response without crashing the pipeline."""
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_error(intent.IntentResponseErrorCode.UNKNOWN, message)
        return conversation.ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
        )


def _user_facing_error(err: OpenClawError) -> str:
    """Return a friendly, secret-free spoken message for a client error."""
    if isinstance(err, OpenClawAuthError):
        return "Sorry, OpenClaw rejected the credentials."
    if isinstance(err, OpenClawTimeoutError):
        return "Sorry, OpenClaw took too long to respond."
    if isinstance(err, OpenClawConnectionError):
        return "Sorry, I could not reach OpenClaw."
    return "Sorry, there was a problem talking to OpenClaw."
