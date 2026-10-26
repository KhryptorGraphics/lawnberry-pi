"""SEC20-10: a link key pasted into ``thor_link.yaml`` must never reach an error, a log or a repr.

The wiring built its error from ``str(ValidationError)``, which prints ``input_value`` (a prefix and
a suffix of a long string: 188 of the 256 bits of a 64-hex key), and from PyYAML's message, which
quotes the offending source line. ``main.py`` logs that text at CRITICAL and keeps it in
``app.state.thor_link_error``. Now nothing the file holds is repeated, a setting named like secret
material is refused by name, and the other places a key travels on the link-start path are checked
the same way: no substring of 6 or more characters of the key may appear anywhere.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.src.core.secrets_manager import SecretsManager
from backend.src.models.thor_link import LinkKeyError, parse_key, validate_key
from backend.src.services.thor_link_service import ThorLinkService, resolve_link_key
from backend.src.services.thor_link_wiring import (
    CONFIG_FILE,
    FenceSettings,
    ThorLinkConfigError,
    ThorLinkSettings,
    ThorLinkStartError,
    _looks_like_secret_name,
    load_thor_link_settings,
    start_thor_link,
)
from backend.src.services.thor_plan_follower import GnssSnapshot
from backend.src.services.tractor_service import TractorControlService
from tests.unit._thor_link_rig import THOR
from tests.unit.test_thor_link_wiring import _stub_the_rest_of_the_backend

HEX64 = "9f2c4b7a1d8e3f605a7b9c1d2e4f60718293a4b5c6d7e8f90a1b2c3d4e5f6071"
PASSPHRASE = "violet-anvil-quokka-7731-zephyr-mango"
SECRETS = {"hex64": HEX64, "passphrase": PASSPHRASE}
REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"
TRIANGLE = "[[1, 2], [3, 4], [5, 6]]"


def leaked(secret: str, text: str, size: int = 6) -> list[str]:
    """Every ``size``-character window of ``secret`` that appears in ``text`` (none is allowed)."""
    return sorted(
        {secret[i : i + size] for i in range(len(secret) - size + 1)} & _present(secret, text, size)
    )


def _present(secret: str, text: str, size: int) -> set[str]:
    return {
        secret[i : i + size] for i in range(len(secret) - size + 1) if secret[i : i + size] in text
    }


def everything_said_by(exc: BaseException) -> str:
    """``str``, ``repr``, ``args`` and the traceback of ``exc`` and of everything chained to it,
    suppressed context included: what a log line, a debugger or an error reporter could print."""
    out, seen, stack = [], set(), [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        out += [
            str(e),
            repr(e),
            repr(e.args),
            *traceback.format_exception(type(e), e, e.__traceback__),
        ]
        stack += [e.__cause__, e.__context__]
    return "\n".join(out)


def logged(caplog) -> str:
    parts = [caplog.text]
    for record in caplog.records:
        parts += [record.getMessage(), record.exc_text or ""]
        if record.exc_info:
            parts += traceback.format_exception(*record.exc_info)
    return "\n".join(parts)


# Where a key can be pasted by mistake: as a setting of its own, as the value of a real setting
# (right type or wrong), nested, as the whole file, or next to a YAML syntax error.
CASES = {
    "key:": "key: {K}\n",
    "enabled and key:": "enabled: false\nkey: {K}\n",
    "thor_link_key:": "thor_link_key: {K}\n",
    "THOR_LINK_KEY:": "THOR_LINK_KEY: {K}\n",
    "psk:": "psk: {K}\n",
    "password:": "password: {K}\n",
    "thor_host:": "enabled: true\nthor_host: {K}\n",
    "bind_host:": "bind_host: {K}\n",
    "allowed_sources list": "allowed_sources: [{K}]\n",
    "allowed_sources scalar": "allowed_sources: {K}\n",
    "timeout_s (wrong type)": "timeout_s: {K}\n",
    "enabled (wrong type)": "enabled: {K}\n",
    "thor_port (wrong type)": "thor_port: {K}\n",
    "require_fix list": "require_fix: [{K}]\n",
    "geofence.key": "geofence: {{key: {K}, boundary: " + TRIANGLE + "}}\n",
    "geofence.margin_m (wrong type)": "geofence: {{margin_m: {K}, boundary: " + TRIANGLE + "}}\n",
    "geofence.boundary (wrong type)": "geofence: {{boundary: {K}}}\n",
    "geofence (wrong type)": "geofence: {K}\n",
    "the whole file": "{K}\n",
    "a list document": "- {K}\n",
    "an unknown setting named by the key": "{K}: true\n",
    "yaml error after the key": "key: {K}\n  bad: [\n",
    "yaml error inside the value": "key: a: {K}\n",
    "yaml error unclosed": "enabled: [unclosed {K}\n",
}


@pytest.mark.parametrize("which", sorted(SECRETS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_nothing_the_file_holds_reaches_the_error(tmp_path, caplog, case, which):
    secret = SECRETS[which]
    (tmp_path / CONFIG_FILE).write_text(CASES[case].format(K=secret))
    with caplog.at_level(logging.NOTSET):
        with pytest.raises(ThorLinkConfigError) as raised:
            load_thor_link_settings(tmp_path)
    assert leaked(secret, everything_said_by(raised.value) + logged(caplog)) == []
    assert raised.value.__cause__ is None and raised.value.__context__ is None, "nothing chained"


@pytest.mark.parametrize("which", sorted(SECRETS))
async def test_the_critical_log_line_and_the_recorded_error_never_hold_the_key(
    monkeypatch, tmp_path, caplog, which
):
    """The review's repro through the real lifespan: what ``main.py`` logs and stores."""
    import backend.src.main as main

    secret = SECRETS[which]
    monkeypatch.setenv("SIM_MODE", "1")
    (tmp_path / CONFIG_FILE).write_text(f"enabled: true\nthor_host: 127.0.0.1\nkey: {secret}\n")
    monkeypatch.setenv("LAWNBERRY_CONFIG_DIR", str(tmp_path))
    tractor = TractorControlService(config={"enabled": True})
    _stub_the_rest_of_the_backend(monkeypatch, main, tractor, [])
    with caplog.at_level(logging.NOTSET):
        async with main.lifespan(main.app):
            error = main.app.state.thor_link_error
    records = [r for r in caplog.records if "Thor link NOT started" in r.getMessage()]
    assert len(records) == 1 and records[0].levelno == logging.CRITICAL
    assert error and "THOR_LINK_KEY" in error and "'key'" in error
    assert leaked(secret, error + logged(caplog)) == []
    await tractor.shutdown()


@pytest.mark.parametrize(
    "name",
    ["key", "thor_link_key", "THOR_LINK_KEY", "link-key", "hmacKey", "hmac_key", "secret", "psk"]
    + ["password", "passphrase", "token", "LinkSecret", "api.key"],
)
def test_a_setting_named_like_secret_material_is_refused_by_name_only(tmp_path, name):
    for body, where in (
        (f"{name}: {HEX64}\n", ""),
        (f"geofence: {{{name}: {HEX64}, boundary: {TRIANGLE}}}\n", "geofence: "),
    ):
        (tmp_path / CONFIG_FILE).write_text(body)
        with pytest.raises(ThorLinkConfigError) as raised:
            load_thor_link_settings(tmp_path)
        text = str(raised.value)
        assert f"{where}setting '{name}' looks like secret material" in text, text
        assert "THOR_LINK_KEY (environment) or config/secrets.json, never in this file" in text
        assert leaked(HEX64, everything_said_by(raised.value)) == []


def test_several_secret_names_are_all_named_and_nothing_else_is(tmp_path):
    (tmp_path / CONFIG_FILE).write_text(f"key: {PASSPHRASE}\npsk: {HEX64}\nenabled: false\n")
    with pytest.raises(ThorLinkConfigError, match="settings 'key', 'psk' look like") as raised:
        load_thor_link_settings(tmp_path)
    assert leaked(HEX64, str(raised.value)) == [] and leaked(PASSPHRASE, str(raised.value)) == []


def test_a_bare_settings_model_does_not_print_what_it_rejected_either():
    for build in (
        lambda: ThorLinkSettings(timeout_s=HEX64),
        lambda: ThorLinkSettings(thor_host=PASSPHRASE),
        lambda: ThorLinkSettings(key=HEX64),
        lambda: FenceSettings(boundary=HEX64),
        lambda: FenceSettings(boundary=[(1, 2), (3, 4), (5, 6)], key=HEX64),
    ):
        with pytest.raises(ValidationError) as raised:
            build()
        said = everything_said_by(raised.value)
        assert leaked(HEX64, said) == [] and leaked(PASSPHRASE, said) == []


def test_a_typo_is_still_named_but_a_name_that_is_not_one_is_not_repeated(tmp_path):
    (tmp_path / CONFIG_FILE).write_text("enabled: true\nthor_hots: 1.2.3.4\n")
    with pytest.raises(ThorLinkConfigError, match="thor_hots: Extra inputs are not permitted"):
        load_thor_link_settings(tmp_path)
    (tmp_path / CONFIG_FILE).write_text(f"{HEX64}: true\n")
    with pytest.raises(ThorLinkConfigError, match="<name not shown>") as raised:
        load_thor_link_settings(tmp_path)
    assert leaked(HEX64, str(raised.value)) == []


def test_no_real_setting_looks_like_secret_material():
    """The refusal by name can then only ever refuse a name the file has no use for."""
    names = set(ThorLinkSettings.model_fields) | set(FenceSettings.model_fields)
    assert names and not [n for n in names if _looks_like_secret_name(n)]


def test_the_shipped_config_still_loads_and_leaves_the_link_off():
    cfg = load_thor_link_settings(REPO_CONFIG)
    assert (REPO_CONFIG / CONFIG_FILE).is_file() and cfg.enabled is False


# --------------- the rest of the link-start path: no key, prefix or suffix of it ---------------

# (what is written in the environment, the same as bytes if it is valid hex else None)
BAD_KEYS = {
    "not hex": ("zz" + HEX64, None),
    "too short": (HEX64[:40], bytes.fromhex(HEX64[:40])),
    "one byte repeated": ("ab" * 40, bytes.fromhex("ab" * 40)),
    "short pattern": ((PASSPHRASE.encode()[:11] * 4).hex(), PASSPHRASE.encode()[:11] * 4),
    "few distinct bytes": (bytes(range(8)).hex() * 6, bytes(range(8)) * 6),
}


@pytest.mark.parametrize("name", sorted(BAD_KEYS))
async def test_a_refused_key_is_never_repeated_by_any_step_of_the_start(
    monkeypatch, tmp_path, caplog, name
):
    text, raw = BAD_KEYS[name]
    monkeypatch.setenv("SIM_MODE", "0")
    monkeypatch.setenv("LAWN_SECRETS_PATH", str(tmp_path / "secrets.json"))
    monkeypatch.setenv("THOR_LINK_KEY", text)
    tractor = TractorControlService(config={"enabled": True})
    steps = [
        lambda: parse_key(text),
        lambda: resolve_link_key(SecretsManager(str(tmp_path / "s.json"))),
    ]
    if raw is not None:  # a refused *key*, as opposed to text that is not even hex
        steps += [lambda: validate_key(raw), lambda: ThorLinkService(tractor, THOR, key=raw)]
    errors: list[BaseException] = []
    with caplog.at_level(logging.NOTSET):
        for step in steps:
            with pytest.raises(LinkKeyError) as raised:
                step()
            errors.append(raised.value)
        with pytest.raises(ThorLinkStartError) as started:
            await start_thor_link(
                tractor,
                safety_monitor_running=True,
                gnss=lambda: GnssSnapshot(None, None, "none", None),
                settings=ThorLinkSettings(enabled=True, thor_host="127.0.0.1"),
            )
        errors.append(started.value)
    said = "\n".join(everything_said_by(e) for e in errors) + logged(caplog)
    assert leaked(text, said) == [] and (raw is None or leaked(raw.hex(), said) == [])
