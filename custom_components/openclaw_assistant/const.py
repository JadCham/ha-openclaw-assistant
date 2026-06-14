"""Constants for the OpenClaw Assistant integration."""

from __future__ import annotations

import logging
from typing import Final

DOMAIN: Final = "openclaw_assistant"
LOGGER: logging.Logger = logging.getLogger(__package__)

# Config / options keys.
CONF_BASE_URL: Final = "base_url"
CONF_API_KEY: Final = "api_key"
CONF_MODEL: Final = "model"
CONF_SYSTEM_PROMPT: Final = "system_prompt"
CONF_TIMEOUT: Final = "timeout"
CONF_VERIFY_TLS: Final = "verify_tls"
CONF_VERIFY_CONNECTION: Final = "verify_connection"

# Defaults.
DEFAULT_MODEL: Final = "ha-voice"
DEFAULT_TIMEOUT: Final = 120
DEFAULT_VERIFY_TLS: Final = True
DEFAULT_VERIFY_CONNECTION: Final = True
DEFAULT_SYSTEM_PROMPT: Final = ""

# Bounds for the timeout selector (seconds).
MIN_TIMEOUT: Final = 5
MAX_TIMEOUT: Final = 600

# Manufacturer string shown on the created device.
MANUFACTURER: Final = "OpenClaw"
