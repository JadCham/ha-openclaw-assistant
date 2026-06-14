"""Diagnostics for OpenClaw Assistant (secrets and content always redacted)."""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant

from . import OpenClawConfigEntry
from .const import CONF_BASE_URL
from .redact import redact_mapping, redact_url


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: OpenClawConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry with all secrets redacted."""
    return {
        # redact_mapping masks api_key/token/etc; redact_url drops user-info
        # and the query string from the base URL.
        "entry": {
            "title": entry.title,
            "data": redact_mapping(entry.data),
            "options": redact_mapping(entry.options),
            "base_url": redact_url(entry.data.get(CONF_BASE_URL, "")),
        },
        "client": {
            "endpoint": redact_url(entry.runtime_data.endpoint),
        },
    }
