"""Test-time defaults for the webhook tests.

The application's ``Settings`` defaults to ``data_dir=/data`` (the
in-container path). On a developer machine outside Docker that path is
read-only / doesn't exist. Set DATA_DIR + WHITELIST envvars BEFORE the
config module is imported so settings construction succeeds.

This file is autoloaded by pytest from the tests/ directory.
"""

from __future__ import annotations

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="ongiini-test-")
os.environ.setdefault("DATA_DIR", _TMP)
os.environ.setdefault("WHITELIST", "")
# Defensive: prevent any test from accidentally talking to the real
# WhatsApp number. Tests should patch these out, but if a test slips
# through with an unmocked send, the empty token causes send_text to
# log a warning rather than make a real call.
os.environ.setdefault("WHATSAPP_TOKEN", "")
os.environ.setdefault("WHATSAPP_PHONE_ID", "")
os.environ.setdefault("WHATSAPP_APP_SECRET", "")
os.environ.setdefault("TAVILY_API_KEY", "")
