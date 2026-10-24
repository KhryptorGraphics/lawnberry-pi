"""THOR-02: the link key is the only real control, so a placeholder or typed key is refused.

Key validation used to be length only: ``00`` x 32, ``aa`` x 32 and the hex of a repeated word were
all accepted, so a typo'd or placeholder key meant an authenticated-forgery risk. Now every way in
(``validate_key``, ``parse_key``, ``encode``/``decode_*``, the service, the Thor client) refuses
them with an explicit reason that never echoes the key, and the published sim key is refused
anywhere but in simulation.
"""

import random

import pytest

from backend.src.core.secrets_manager import SecretsManager
from backend.src.models.thor_link import (
    KEY_MIN_BYTES,
    KEY_MIN_DISTINCT_BYTES,
    KEY_MIN_PERIOD_BYTES,
    PUBLISHED_KEYS,
    SIM_LINK_KEY,
    Downlink,
    LinkKeyError,
    Uplink,
    decode_downlink,
    decode_uplink,
    encode,
    parse_key,
    refuse_published_key,
    validate_key,
)
from backend.src.services.thor_link_service import ThorLinkService, resolve_link_key
from backend.src.services.tractor_service import TractorControlService
from tests.unit._thor_link_rig import THOR

_RNG = random.Random(4217)
STRONG = _RNG.randbytes(32)
STRONG_64 = _RNG.randbytes(64)


def _pattern(unit: bytes, length: int = 40) -> bytes:
    return (unit * (length // len(unit) + 1))[:length]


def _over_values(values, length: int, *, seed: int) -> bytes:
    """Random bytes drawn from only ``values`` (all of them present), in no repeating pattern."""
    rng = random.Random(seed)
    values = list(values)
    return bytes(values) + bytes(rng.choice(values) for _ in range(length - len(values)))


WEAK = {
    "all zero (32)": (b"\x00" * 32, "all zero"),
    "all zero (64)": (b"\x00" * 64, "all zero"),
    "aa x 32": (b"\xaa" * 32, "one byte repeated"),
    "ff x 64": (b"\xff" * 64, "one byte repeated"),
    "ascii x 40": (b"x" * 40, "one byte repeated"),
    "two byte pattern": (_pattern(b"\x01\x02"), "2-byte pattern"),
    "repeated word": (b"password" * 4, "8-byte pattern"),
    "word, not a whole number of times": (_pattern(b"hunter2!", 37), "8-byte pattern"),
    "longest refused unit (15 bytes)": (_pattern(bytes(range(1, 16)), 45), "15-byte pattern"),
    "two byte values, random order": (_over_values([7, 200], 48, seed=9), "only 2 distinct"),
    "eight byte values": (_over_values(range(8), 64, seed=10), "only 8 distinct"),
}


@pytest.mark.parametrize("name", sorted(WEAK))
def test_a_degenerate_key_is_refused_with_an_explicit_reason(name):
    key, reason = WEAK[name]
    with pytest.raises(LinkKeyError, match=reason) as why:
        validate_key(key)
    assert key.hex() not in str(why.value) and repr(key) not in str(why.value)
    assert len(key) >= KEY_MIN_BYTES, "refused for what it is, not because it is short"


@pytest.mark.parametrize("name", sorted(WEAK))
def test_every_entry_point_refuses_the_same_keys(name, monkeypatch):
    key, reason = WEAK[name]
    monkeypatch.setenv("SIM_MODE", "1")  # even in simulation a degenerate key is refused
    for refuse in (
        lambda: parse_key(key.hex()),
        lambda: parse_key(key.hex(), simulation=True),
        lambda: encode(Downlink(sid=1, ack=2, plan_id="p"), key),
        lambda: encode(Uplink(seq=0, t_ms=0), key),
        lambda: decode_downlink(b"y" * 64, key),
        lambda: decode_uplink(b"y" * 64, key),
        lambda: ThorLinkService(TractorControlService(config={}), THOR, key=key),
    ):
        with pytest.raises(LinkKeyError, match=reason):
            refuse()


def test_the_environment_and_the_secrets_store_cannot_provide_a_degenerate_key(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("SIM_MODE", "0")
    store = tmp_path / "secrets.json"
    monkeypatch.setenv("LAWN_SECRETS_PATH", str(store))
    for key, reason in WEAK.values():
        monkeypatch.setenv("THOR_LINK_KEY", key.hex())
        with pytest.raises(LinkKeyError, match=reason):
            resolve_link_key(SecretsManager(store_path=str(store)))
        with pytest.raises(LinkKeyError, match=reason):
            ThorLinkService(TractorControlService(config={}), THOR)
    monkeypatch.delenv("THOR_LINK_KEY")
    sm = SecretsManager(store_path=str(store))
    sm.set("THOR_LINK_KEY", (b"\x00" * 32).hex())
    with pytest.raises(LinkKeyError, match="all zero"):
        resolve_link_key(sm)


def test_a_random_key_is_never_refused():
    rng = random.Random(31337)
    for length in (32, 33, 40, 64, 128):
        for _ in range(1500):
            key = rng.randbytes(length)
            assert validate_key(key) == key
    assert validate_key(bytearray(STRONG)) == STRONG, (
        "a bytearray is accepted and returned as bytes"
    )
    assert parse_key(STRONG.hex().upper()) == STRONG and parse_key(f"  {STRONG.hex()}\n") == STRONG
    assert validate_key(STRONG_64) == STRONG_64


def test_the_limits_sit_where_the_documentation_says():
    """A 15-byte unit repeated is refused, a 16-byte one (128 bits of secret) is not; the same
    for the number of distinct byte values."""
    unit16 = bytes(random.Random(7).sample(range(256), KEY_MIN_PERIOD_BYTES))  # 16 distinct bytes
    assert validate_key(unit16 * 2) == unit16 * 2
    unit15 = _RNG.randbytes(KEY_MIN_PERIOD_BYTES - 1)
    with pytest.raises(LinkKeyError, match="15-byte pattern"):
        validate_key(unit15 * 3)
    with pytest.raises(LinkKeyError, match="15-byte pattern"):
        validate_key(unit15 * 2 + unit15[:5])
    few = [bytes([i]) for i in range(KEY_MIN_DISTINCT_BYTES - 1)]
    rng = random.Random(5)
    low = bytes(range(KEY_MIN_DISTINCT_BYTES - 1)) + b"".join(rng.choice(few) for _ in range(50))
    assert len(set(low)) == KEY_MIN_DISTINCT_BYTES - 1
    with pytest.raises(LinkKeyError, match=f"only {KEY_MIN_DISTINCT_BYTES - 1} distinct"):
        validate_key(low)
    enough = [bytes([i]) for i in range(KEY_MIN_DISTINCT_BYTES)]
    ok = bytes(range(KEY_MIN_DISTINCT_BYTES)) + b"".join(rng.choice(enough) for _ in range(40))
    assert len(set(ok)) == KEY_MIN_DISTINCT_BYTES and validate_key(ok) == ok


def test_short_and_malformed_keys_keep_their_own_messages():
    for bad in (b"", b"short", STRONG[: KEY_MIN_BYTES - 1], None, "a-string-not-bytes" * 4, 7):
        with pytest.raises(LinkKeyError, match=f"at least {KEY_MIN_BYTES} bytes"):
            validate_key(bad)
    with pytest.raises(LinkKeyError, match="hex"):
        parse_key("zz" * 40)


def test_the_published_sim_key_passes_validation_but_not_outside_simulation():
    assert SIM_LINK_KEY in PUBLISHED_KEYS
    assert validate_key(SIM_LINK_KEY) == SIM_LINK_KEY, "encode/decode must work under simulation"
    assert refuse_published_key(SIM_LINK_KEY, simulation=True) == SIM_LINK_KEY
    assert refuse_published_key(STRONG, simulation=False) == STRONG
    with pytest.raises(LinkKeyError, match="SIM_MODE=1"):
        refuse_published_key(SIM_LINK_KEY, simulation=False)
    # a configured key that IS the published key: the hex of it is refused outside simulation
    with pytest.raises(LinkKeyError, match="SIM_MODE=1"):
        parse_key(SIM_LINK_KEY.hex())
    with pytest.raises(LinkKeyError, match="SIM_MODE=1"):
        parse_key(SIM_LINK_KEY.hex(), simulation=False)
    assert parse_key(SIM_LINK_KEY.hex(), simulation=True) == SIM_LINK_KEY


def test_the_environment_cannot_smuggle_the_published_key_in_outside_simulation(
    monkeypatch, tmp_path
):
    store = tmp_path / "secrets.json"
    monkeypatch.setenv("LAWN_SECRETS_PATH", str(store))
    monkeypatch.setenv("THOR_LINK_KEY", SIM_LINK_KEY.hex())
    monkeypatch.setenv("SIM_MODE", "0")
    with pytest.raises(LinkKeyError, match="SIM_MODE=1"):
        resolve_link_key(SecretsManager(store_path=str(store)))
    monkeypatch.setenv("SIM_MODE", "1")
    assert resolve_link_key(SecretsManager(store_path=str(store))) == SIM_LINK_KEY
