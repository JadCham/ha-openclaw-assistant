"""The OpenClaw Assistant integration.

Wires a config entry up to a single OpenClaw conversation entity. The
per-entry :class:`~.client.OpenClawClient` is stored on ``entry.runtime_data``
so the conversation platform can reach it. Setup never performs a network
call, so Home Assistant starts cleanly even when OpenClaw is offline; any
connectivity problem surfaces (gracefully) at conversation time instead.
"""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .client import OpenClawClient
from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_TIMEOUT,
    CONF_VERIFY_TLS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    DEFAULT_VERIFY_TLS,
)

PLATFORMS: list[Platform] = [Platform.CONVERSATION]

type OpenClawConfigEntry = ConfigEntry[OpenClawClient]


def _build_client(hass: HomeAssistant, entry: OpenClawConfigEntry) -> OpenClawClient:
    """Construct a client from the entry's merged data + options."""
    settings = {**entry.data, **entry.options}
    return OpenClawClient(
        hass,
        base_url=settings[CONF_BASE_URL],
        api_key=settings.get(CONF_API_KEY),
        model=settings.get(CONF_MODEL, DEFAULT_MODEL),
        timeout=settings.get(CONF_TIMEOUT, DEFAULT_TIMEOUT),
        verify_tls=settings.get(CONF_VERIFY_TLS, DEFAULT_VERIFY_TLS),
    )


async def async_setup_entry(hass: HomeAssistant, entry: OpenClawConfigEntry) -> bool:
    """Set up OpenClaw Assistant from a config entry."""
    entry.runtime_data = _build_client(hass, entry)
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: OpenClawConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
