"""Which settings names are secrets, and how a secret is shown (SEC20-01).

``backend/src/core/secret_redaction.py`` is what keeps the settings routes from handing out the
Cloudflare tunnel token, the ngrok authtoken and the maps key. These tests pin the classifier
(both directions: a name that merely sits next to a secret word must stay visible), the shape of
the operator view, and the write rules.
"""

from __future__ import annotations

import pytest

from backend.src.core.secret_redaction import (
    MIN_LENGTH_FOR_TAIL,
    TAIL_LENGTH,
    is_hint_name,
    is_secret_name,
    merge_secret_update,
    redact_secrets,
    secret_is_set,
    secret_tail,
)

LONG = "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"


@pytest.mark.parametrize(
    "name",
    [
        "cloudflare_token",
        "ngrok_token",
        "ngrok_authtoken",
        "google_api_key",
        "api_key",
        "apiKey",
        "APIKey",
        "googleApiKey",
        "cloudflareTunnelToken",
        "password",
        "password_hash",
        "totp_secret",
        "client_secret",
        "backup_codes",
        "recovery_codes",
        "private_key",
        "signing_key",
        "ssh-key",
        "webhook_url",
        "Authorization",
        "session_id",
        "wifi_psk",
        "access_token",
        "refresh_token",
    ],
)
def test_names_that_hold_secrets_are_recognised(name):
    assert is_secret_name(name), name


@pytest.mark.parametrize(
    "name",
    [
        "provider",
        "style",
        "bypass_external",
        "custom_domain",
        "auto_tls",
        "method",
        "enabled",
        "level",
        "auth_level",
        "session_timeout_minutes",
        "require_https",
        "auto_lock_manual_control",
        "google_billing_warnings",
        "branding_checksum",
        "profile_version",
        "hotkey",
        "monkey",
        "keyboard",
        "key_count_hint",
        "latency_targets",
        "pi5_ms",
        "",
    ],
)
def test_names_that_only_look_similar_stay_visible(name):
    assert not is_secret_name(name), name


def test_only_strings_can_be_names():
    assert not is_secret_name(None)
    assert not is_secret_name(7)
    assert not is_secret_name(("token",))


def test_hint_names_are_the_derived_keys_of_a_secret_only():
    assert is_hint_name("google_api_key_set")
    assert is_hint_name("cloudflare_token_last4")
    assert not is_hint_name("auto_tls_set")  # not derived from a secret
    assert not is_hint_name("google_api_key")
    assert not is_hint_name("_set")
    assert not is_hint_name(None)


def test_set_and_tail_of_a_value():
    assert secret_is_set("x") is True
    assert secret_is_set("") is False
    assert secret_is_set("   ") is False
    assert secret_is_set(None) is False
    assert secret_is_set(0) is False
    assert secret_is_set(["a"]) is True

    assert secret_tail(LONG) == LONG[-TAIL_LENGTH:]
    assert (
        secret_tail("x" * (MIN_LENGTH_FOR_TAIL - 1)) is None
    )  # four characters would be a quarter
    assert secret_tail("x" * MIN_LENGTH_FOR_TAIL) == "xxxx"
    assert secret_tail(f"  {LONG}  ") == LONG[-TAIL_LENGTH:]  # padding never shifts the tail
    assert secret_tail(12345678901234567890) is None
    assert secret_tail(None) is None


def test_the_open_view_drops_secrets_and_says_nothing_about_them():
    stored = {
        "provider": "google",
        "google_api_key": LONG,
        "api_key": None,
        "nested": {"ngrok_token": LONG, "keep": 1, "deeper": [{"password": "pw", "ok": True}]},
        "list": [{"cloudflare_token": LONG}, "plain"],
    }
    assert redact_secrets(stored) == {
        "provider": "google",
        "nested": {"keep": 1, "deeper": [{"ok": True}]},
        "list": [{}, "plain"],
    }


def test_the_operator_view_reports_presence_and_a_safe_tail_only():
    stored = {
        "provider": "google",
        "google_api_key": LONG,
        "ngrok_token": "short-secret",
        "cloudflare_token": "",
        "api_key": None,
    }
    view = redact_secrets(stored, hints=True)
    assert view == {
        "provider": "google",
        "google_api_key_set": True,
        "google_api_key_last4": LONG[-4:],
        "ngrok_token_set": True,  # too short for a tail
        "cloudflare_token_set": False,
        "api_key_set": False,
    }
    assert LONG not in repr(view) and "short-secret" not in repr(view)


def test_redaction_never_mutates_what_it_is_given():
    stored = {"google_api_key": LONG, "nested": {"token": LONG}}
    redact_secrets(stored, hints=True)
    assert stored == {"google_api_key": LONG, "nested": {"token": LONG}}


class TestMergeSecretUpdate:
    CURRENT = {"provider": "osm", "google_api_key": LONG, "ngrok_token": "old-ngrok-token-value"}

    def test_plain_fields_merge_like_a_dict_update(self):
        merged = merge_secret_update(self.CURRENT, {"provider": "google", "style": "hybrid"})
        assert merged == {**self.CURRENT, "provider": "google", "style": "hybrid"}

    def test_an_untouched_password_input_keeps_the_stored_secret(self):
        for empty in ("", "   ", "\n"):
            merged = merge_secret_update(self.CURRENT, {"google_api_key": empty})
            assert merged["google_api_key"] == LONG, repr(empty)

    def test_a_new_value_replaces_it_verbatim(self):
        merged = merge_secret_update(self.CURRENT, {"google_api_key": "  new key  "})
        assert merged["google_api_key"] == "  new key  "

    def test_an_explicit_null_clears_it_and_stores_none(self):
        merged = merge_secret_update(self.CURRENT, {"google_api_key": None})
        assert merged["google_api_key"] is None
        assert merged["ngrok_token"] == "old-ngrok-token-value"
        assert redact_secrets(merged, hints=True)["google_api_key_set"] is False

    def test_null_for_a_never_set_secret_stores_none(self):
        assert merge_secret_update({}, {"google_api_key": None}) == {"google_api_key": None}

    def test_a_null_for_a_plain_field_is_stored_as_given(self):
        assert merge_secret_update(self.CURRENT, {"style": None})["style"] is None

    def test_the_derived_keys_of_a_read_view_are_never_stored(self):
        echoed = redact_secrets(self.CURRENT, hints=True)
        assert "google_api_key_set" in echoed
        assert merge_secret_update(self.CURRENT, echoed) == self.CURRENT

    def test_nothing_is_modified_in_place(self):
        before = dict(self.CURRENT)
        merge_secret_update(self.CURRENT, {"google_api_key": None, "provider": "google"})
        assert self.CURRENT == before
