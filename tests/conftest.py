"""Pytest configuration for the pure-module unit tests.

The SSE parser and redaction helpers are deliberately free of any Home
Assistant dependency, so we put the integration directory on ``sys.path`` and
import them as standalone modules. This avoids importing the package
``__init__`` (which pulls in Home Assistant) and lets the tests run with
nothing more than pytest installed.
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PKG_DIR = _REPO_ROOT / "custom_components" / "openclaw_assistant"

# The package dir lets the pure-module tests do ``import sse`` / ``import redact``
# without pulling in Home Assistant. The repo root lets the (optional) Home
# Assistant integration test import ``custom_components.openclaw_assistant.*``.
sys.path.insert(0, str(_PKG_DIR))
sys.path.insert(0, str(_REPO_ROOT))
