"""Config and options flows for OpenClaw Assistant."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from homeassistant.config_entries import (
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithReload,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
import voluptuous as vol

from .client import (
    OpenClawAuthError,
    OpenClawClient,
    OpenClawError,
    OpenClawTimeoutError,
)
from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_SYSTEM_PROMPT,
    CONF_TIMEOUT,
    CONF_VERIFY_CONNECTION,
    CONF_VERIFY_TLS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    DEFAULT_VERIFY_CONNECTION,
    DEFAULT_VERIFY_TLS,
    DOMAIN,
    LOGGER,
    MAX_TIMEOUT,
    MIN_TIMEOUT,
)

_API_KEY_FIELD = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
_PROMPT_FIELD = TextSelector(TextSelectorConfig(multiline=True))
_BOOL = BooleanSelector()
_TIMEOUT_FIELD = NumberSelector(
    NumberSelectorConfig(
        min=MIN_TIMEOUT,
        max=MAX_TIMEOUT,
        step=1,
        mode=NumberSelectorMode.BOX,
        unit_of_measurement="seconds",
    )
)


def _user_schema() -> vol.Schema:
    """Schema for the initial setup step (includes the base URL)."""
    return vol.Schema(
        {
            vol.Required(CONF_BASE_URL): TextSelector(
                TextSelectorConfig(type=TextSelectorType.URL)
            ),
            vol.Optional(CONF_API_KEY): _API_KEY_FIELD,
            vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): TextSelector(),
            vol.Optional(CONF_SYSTEM_PROMPT): _PROMPT_FIELD,
            vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): _TIMEOUT_FIELD,
            vol.Optional(CONF_VERIFY_TLS, default=DEFAULT_VERIFY_TLS): _BOOL,
            vol.Optional(
                CONF_VERIFY_CONNECTION, default=DEFAULT_VERIFY_CONNECTION
            ): _BOOL,
        }
    )


def _options_schema() -> vol.Schema:
    """Schema for the options step (everything except the base URL/token)."""
    return vol.Schema(
        {
            vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): TextSelector(),
            vol.Optional(CONF_SYSTEM_PROMPT): _PROMPT_FIELD,
            vol.Optional(CONF_TIMEOUT, default=DEFAULT_TIMEOUT): _TIMEOUT_FIELD,
            vol.Optional(CONF_VERIFY_TLS, default=DEFAULT_VERIFY_TLS): _BOOL,
            vol.Optional(CONF_VERIFY_CONNECTION, default=False): _BOOL,
        }
    )


async def _async_validate(
    hass: HomeAssistant, settings: Mapping[str, Any]
) -> dict[str, str]:
    """Validate connectivity/auth. Return an errors dict (empty on success)."""
    if not settings.get(CONF_VERIFY_CONNECTION):
        return {}
    client = OpenClawClient(
        hass,
        base_url=settings[CONF_BASE_URL],
        api_key=settings.get(CONF_API_KEY),
        model=settings.get(CONF_MODEL, DEFAULT_MODEL),
        timeout=int(settings.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)),
        verify_tls=settings.get(CONF_VERIFY_TLS, DEFAULT_VERIFY_TLS),
    )
    try:
        await client.async_validate()
    except OpenClawAuthError:
        return {"base": "invalid_auth"}
    except OpenClawTimeoutError:
        return {"base": "timeout_connect"}
    except OpenClawError:
        return {"base": "cannot_connect"}
    except Exception as err:
        # Log only the exception *type* so no URL/token/content can leak.
        LOGGER.error(
            "Unexpected error validating OpenClaw connection: %s", type(err).__name__
        )
        return {"base": "unknown"}
    return {}


def _clean_setup_data(user_input: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise initial-setup input into the entry data dict."""
    data = dict(user_input)
    data.pop(CONF_VERIFY_CONNECTION, None)
    if CONF_TIMEOUT in data:
        data[CONF_TIMEOUT] = int(data[CONF_TIMEOUT])
    if CONF_BASE_URL in data:
        data[CONF_BASE_URL] = str(data[CONF_BASE_URL]).strip()
    # Drop empty optional strings so they don't shadow defaults.
    if not data.get(CONF_API_KEY):
        data.pop(CONF_API_KEY, None)
    if not data.get(CONF_SYSTEM_PROMPT):
        data.pop(CONF_SYSTEM_PROMPT, None)
    return data


def _clean_options(user_input: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise options input. Empty system prompt is kept so it can be cleared."""
    data = dict(user_input)
    data.pop(CONF_VERIFY_CONNECTION, None)
    if CONF_TIMEOUT in data:
        data[CONF_TIMEOUT] = int(data[CONF_TIMEOUT])
    data.setdefault(CONF_SYSTEM_PROMPT, "")
    return data


class OpenClawConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the OpenClaw Assistant config flow."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial setup step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = await _async_validate(self.hass, user_input)
            if not errors:
                data = _clean_setup_data(user_input)
                title = f"OpenClaw ({data.get(CONF_MODEL, DEFAULT_MODEL)})"
                return self.async_create_entry(title=title, data=data)

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                _user_schema(), user_input or {}
            ),
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle reconfiguration of an existing entry (e.g. URL/token change)."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = await _async_validate(self.hass, user_input)
            if not errors:
                return self.async_update_reload_and_abort(
                    entry, data=_clean_setup_data(user_input)
                )
            suggested: Mapping[str, Any] = user_input
        else:
            suggested = {
                **entry.data,
                CONF_VERIFY_CONNECTION: DEFAULT_VERIFY_CONNECTION,
            }

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(_user_schema(), suggested),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(config_entry) -> OpenClawOptionsFlow:
        """Return the options flow handler."""
        return OpenClawOptionsFlow()


class OpenClawOptionsFlow(OptionsFlowWithReload):
    """Edit the tunable OpenClaw settings; auto-reloads the entry on save."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        errors: dict[str, str] = {}
        if user_input is not None:
            # Validate against the entry's stored base URL / token.
            merged = {**self.config_entry.data, **user_input}
            errors = await _async_validate(self.hass, merged)
            if not errors:
                return self.async_create_entry(data=_clean_options(user_input))
            suggested: Mapping[str, Any] = user_input
        else:
            suggested = {
                **self.config_entry.data,
                **self.config_entry.options,
                CONF_VERIFY_CONNECTION: False,
            }

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                _options_schema(), suggested
            ),
            errors=errors,
        )
