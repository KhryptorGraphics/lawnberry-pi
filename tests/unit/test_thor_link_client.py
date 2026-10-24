"""scripts/thor_link_client.py: key policy, what the Thor answers, and what it refuses to.

The Thor side of the link has obligations the Pi cannot enforce for it (see
``docs/tractor-platform.md``, "Thor side"); these tests pin them: no key means no start, only
authentic and fresh uplinks are answered, every answer echoes ``sid``/``ack``, and anything
the planner cannot vouch for becomes a pause instead of silence or a guess.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from backend.src.models import thor_link as wire
from backend.src.models.thor_link import (
    MAX_DATAGRAM_BYTES,
    SIM_LINK_KEY,
    Downlink,
    LinkKeyError,
    Uplink,
    Waypoint,
    decode_downlink,
    decode_uplink,
    encode,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "thor_link_client.py"
_spec = importlib.util.spec_from_file_location("scripts.thor_link_client", SCRIPT)
client_mod = importlib.util.module_from_spec(_spec)
sys.modules["scripts.thor_link_client"] = client_mod
_spec.loader.exec_module(client_mod)
ThorLinkClient = client_mod.ThorLinkClient
StaticPlanner = client_mod.StaticPlanner
load_key = client_mod.load_key

KEY_HEX = "3e" * 40
KEY = bytes.fromhex(KEY_HEX)
OTHER_KEY = bytes.fromhex("c1" * 40)
PI = ("10.0.0.7", 47101)


def wps(n: int, blade: bool = False) -> list[Waypoint]:
    return [Waypoint(lat=45.0 + i * 1e-5, lon=-93.0, blade=blade) for i in range(n)]


def up(seq: int = 0, sid: int = 77, **kw) -> Uplink:
    base = {"lat": 45.0, "lon": -93.0, "fix": "rtk_fixed", "sid": sid, "seq": seq, "t_ms": seq}
    return Uplink(**{**base, **kw})


def datagram(seq: int = 0, sid: int = 77, key: bytes = KEY, **kw) -> bytes:
    return encode(up(seq, sid, **kw), key)


# ------------------------------------- key -------------------------------------


def _key_file(tmp_path, text=KEY_HEX, mode=0o600):
    path = tmp_path / "thor.key"
    path.write_text(text)
    path.chmod(mode)
    return path


def test_the_key_comes_from_a_private_file_or_the_environment(tmp_path):
    assert load_key(key_file=_key_file(tmp_path), env={}) == KEY
    assert load_key(env={"THOR_LINK_KEY": KEY_HEX}) == KEY
    # an explicit file wins over whatever the environment holds
    assert load_key(key_file=_key_file(tmp_path), env={"THOR_LINK_KEY": "ab" * 40}) == KEY


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o666])
def test_a_key_file_other_users_can_read_is_refused(tmp_path, mode):
    with pytest.raises(LinkKeyError, match="chmod 600") as err:
        load_key(key_file=_key_file(tmp_path, mode=mode), env={})
    assert KEY_HEX not in str(err.value)


@pytest.mark.parametrize("text", ["not hex at all", "ab" * 8, "", "zz" * 40])
def test_a_malformed_or_short_key_is_refused_without_echoing_it(tmp_path, text):
    with pytest.raises(LinkKeyError) as err:
        load_key(key_file=_key_file(tmp_path, text), env={})
    assert text.strip() == "" or text not in str(err.value)
    with pytest.raises(LinkKeyError):
        load_key(env={"THOR_LINK_KEY": text or "  "})


def test_a_missing_key_file_is_a_key_error_not_a_crash(tmp_path):
    with pytest.raises(LinkKeyError, match="cannot read key file"):
        load_key(key_file=tmp_path / "absent.key", env={})


def test_without_a_key_there_is_no_fallback_to_anything_weaker():
    with pytest.raises(LinkKeyError, match="THOR_LINK_KEY"):
        load_key(env={})
    with pytest.raises(LinkKeyError, match="THOR_LINK_KEY"):
        load_key(env={"SIM_MODE": "1"})  # SIM_MODE alone does not select the public key


def test_the_public_sim_key_needs_both_the_flag_and_sim_mode():
    assert load_key(sim=True, env={"SIM_MODE": "1"}) == SIM_LINK_KEY
    with pytest.raises(LinkKeyError, match="SIM_MODE=1"):
        load_key(sim=True, env={})
    with pytest.raises(LinkKeyError, match="SIM_MODE=1"):
        load_key(sim=True, env={"SIM_MODE": "0"})
    # ... and a configured key is never replaced by it
    assert load_key(sim=True, env={"SIM_MODE": "1", "THOR_LINK_KEY": KEY_HEX}) == KEY


# ------------------------------ the static planner ------------------------------


def test_a_static_plan_is_served_in_windows_that_follow_the_pis_progress():
    plan = wps(40)
    p = StaticPlanner(plan, plan_id="bench")
    first = p.on_uplink(up(0))
    assert (first.sid, first.ack, first.plan_id, first.start_index) == (77, 0, "bench", 0)
    assert first.waypoints == plan[:16] and not first.pause
    later = p.on_uplink(up(1, waypoint_index=20))
    assert later.start_index == 20 and later.waypoints == plan[20:36]
    tail = p.on_uplink(up(2, waypoint_index=30))
    assert tail.start_index == 30 and tail.waypoints == plan[30:40]


def test_progress_never_goes_backwards_and_cannot_pass_the_end():
    p = StaticPlanner(wps(10))
    p.on_uplink(up(0, waypoint_index=6))
    assert p.on_uplink(up(1, waypoint_index=2)).start_index == 6, "a stale report is ignored"
    done = p.on_uplink(up(2, waypoint_index=10_000))
    assert done.pause and done.start_index == 10 and done.waypoints == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"lat": None},
        {"lon": None},
        {"fix": "none"},
        {"fix": "gps"},
        {"fix": "rtk_float"},
        {"estop": True},
        {"hold_fault": True},
    ],
)
def test_the_planner_pauses_instead_of_planning_from_a_bad_report(overrides):
    reply = StaticPlanner(wps(5)).on_uplink(up(0, **overrides))
    assert reply.pause and reply.waypoints == []
    assert (reply.sid, reply.ack) == (77, 0), "a pause is echoed too, or the Pi ignores it"


def test_a_hold_that_is_only_waiting_for_a_plan_is_not_a_reason_to_pause():
    """The Pi starts held; the Thor's plan is what ends the hold. Pausing on it would deadlock."""
    assert not StaticPlanner(wps(5)).on_uplink(up(0, safe_hold=True)).pause


def test_an_empty_plan_is_refused_up_front():
    with pytest.raises(ValueError):
        StaticPlanner([])


# --------------------------------- what is answered ---------------------------------


def test_an_authentic_uplink_is_answered_with_an_authentic_echoing_downlink():
    c = ThorLinkClient(KEY, StaticPlanner(wps(3), "p").on_uplink)
    reply = c.handle(datagram(5, sid=9001), PI)
    down = decode_downlink(reply, KEY)
    assert (down.sid, down.ack, down.plan_id) == (9001, 5, "p") and len(down.waypoints) == 3
    assert c.counters["answered"] == 1 and c.last_uplink.seq == 5
    with pytest.raises(wire.LinkAuthError):
        decode_uplink(reply, KEY)  # it is a downlink: the direction label differs


def test_nothing_is_answered_unless_it_authenticates():
    c = ThorLinkClient(KEY, StaticPlanner(wps(3)).on_uplink)
    good = datagram(0)
    flipped = bytearray(good)
    flipped[len(flipped) // 2] ^= 0x01
    assert c.handle(datagram(0, key=OTHER_KEY), PI) is None
    assert c.handle(bytes(flipped), PI) is None
    assert c.handle(good[:-1], PI) is None
    assert c.handle(b"", PI) is None and c.handle(good[:30], PI) is None
    assert c.handle(encode(Downlink(sid=77, ack=0, plan_id="x"), KEY), PI) is None, (
        "wrong direction"
    )
    assert c.handle(b"\x00" * (MAX_DATAGRAM_BYTES + 1), PI) is None
    assert c.counters["auth"] == 6 and c.counters["oversize"] == 1
    assert c.counters["answered"] == 0 and c.last_uplink is None


def test_authentic_but_invalid_content_is_dropped_as_malformed():
    body = json.dumps({"v": 2, "sid": 77, "seq": 0, "t_ms": 0, "fix": 5, "detections": "x"})
    data = wire._tag(KEY, wire._UPLINK_DOMAIN, body.encode()) + body.encode()
    c = ThorLinkClient(KEY, StaticPlanner(wps(3)).on_uplink)
    assert c.handle(data, PI) is None and c.counters["malformed"] == 1


def test_replayed_and_reordered_uplinks_are_not_answered_twice():
    c = ThorLinkClient(KEY, StaticPlanner(wps(3)).on_uplink)
    assert c.handle(datagram(4), PI) is not None
    assert c.handle(datagram(4), PI) is None, "the same datagram again"
    assert c.handle(datagram(3), PI) is None, "an older one arriving late"
    assert c.handle(datagram(5), PI) is not None
    assert c.counters["replay"] == 2 and c.counters["answered"] == 2


def test_a_pi_restart_is_a_new_session_and_starts_counting_again():
    c = ThorLinkClient(KEY, StaticPlanner(wps(3)).on_uplink)
    assert c.handle(datagram(900, sid=1), PI) is not None
    assert c.handle(datagram(0, sid=2), PI) is not None, "new sid: its counter starts over"
    assert c.handle(datagram(901, sid=1), PI) is not None, "the old session is still tracked"
    assert c.handle(datagram(0, sid=2), PI) is None


def test_the_session_table_is_bounded():
    c = ThorLinkClient(KEY, StaticPlanner(wps(3)).on_uplink)
    for sid in range(1, 60):
        assert c.handle(datagram(0, sid=sid), PI) is not None
    assert len(c._last_seq) == client_mod.MAX_SESSIONS


def test_only_the_named_pi_is_answered_when_an_allowlist_is_given():
    c = ThorLinkClient(KEY, StaticPlanner(wps(3)).on_uplink, allowed_pi=["10.0.0.7"])
    assert c.handle(datagram(0), ("10.0.0.99", 1)) is None and c.counters["source"] == 1
    assert c.handle(datagram(0), PI) is not None


def test_an_operator_pause_overrides_the_plan_but_keeps_the_echo():
    called = []

    def planner(u):
        called.append(u)
        return StaticPlanner(wps(3)).on_uplink(u)

    c = ThorLinkClient(KEY, planner)
    c.paused = True
    down = decode_downlink(c.handle(datagram(3, sid=5), PI), KEY)
    assert down.pause and down.waypoints == [] and (down.sid, down.ack) == (5, 3)
    assert called == [], "a paused Thor does not even consult the planner"
    c.paused = False
    assert not decode_downlink(c.handle(datagram(4, sid=5), PI), KEY).pause


def test_a_planner_that_fails_answers_with_a_pause_not_with_silence(caplog):
    def broken(_up):
        raise RuntimeError("planner bug")

    c = ThorLinkClient(KEY, broken)
    with caplog.at_level("ERROR"):
        down = decode_downlink(c.handle(datagram(0), PI), KEY)
    assert down.pause and down.plan_id == "error" and c.counters["planner_error"] == 1


def test_a_planner_answer_that_cannot_be_sent_also_becomes_a_pause():
    def too_big(u):
        return Downlink.model_construct(
            v=2, sid=u.sid, ack=u.seq, plan_id="p" * 1500, start_index=0, waypoints=[], pause=False
        )

    down = decode_downlink(ThorLinkClient(KEY, too_big).handle(datagram(0), PI), KEY)
    assert down.pause and down.plan_id == "error"


def test_garbage_never_raises_and_is_never_answered():
    c = ThorLinkClient(KEY, StaticPlanner(wps(3)).on_uplink)
    rng = random.Random(7)
    for n in list(range(0, 70)) + [200, 1199, 1200, 1201, 4000]:
        assert c.handle(rng.randbytes(n), PI) is None
    assert c.counters["answered"] == 0


def test_the_summary_says_what_the_pi_last_reported():
    c = ThorLinkClient(KEY, StaticPlanner(wps(3)).on_uplink)
    assert "no uplink yet" in c.summary()
    c.handle(datagram(0, safe_hold=True, waypoint_index=2), PI)
    assert "index=2" in c.summary() and "hold=True" in c.summary()


# ---------------------------------- the strategist ----------------------------------


def test_the_strategist_plans_its_first_transit_from_where_the_mower_really_is():
    from backend.src.nav.thor_strategist import ThorStrategist

    m = 111_195.0

    def ll(east, north):
        return 45.0 + north / m, -93.0 + east / (m * math.cos(math.radians(45.0)))

    def build():
        # a no-go block across the bottom of the yard, between the mower and the first stripe
        return ThorStrategist(
            [ll(0, 0), ll(30, 0), ll(30, 30), ll(0, 30)],
            [[ll(12, -1), ll(18, -1), ll(18, 8), ll(12, 8)]],
            plan_id="yard",
        )

    here = ll(28, 1.4)
    report = up(0, lat=here[0], lon=here[1])

    plain = build()
    legs = [plain.frame.xy(here)] + [(leg.x, leg.y) for leg in plain.legs[:3]]
    assert not plain.leg_is_legal(legs[0], legs[1]), "the unadapted plan starts across the block"

    adapted = build()
    reply = client_mod.StrategistPlanner(adapted).on_uplink(report)
    assert not reply.pause and reply.plan_id == "yard"
    start = adapted.frame.xy(here)
    pts = [start] + [adapted.frame.xy((w.lat, w.lon)) for w in reply.waypoints[:6]]
    assert all(adapted.leg_is_legal(a, b) for a, b in zip(pts, pts[1:], strict=False))
    # replanning happens once per session, not on every uplink
    calls: list = []
    real_replan = adapted.replan
    adapted.replan = lambda start=None: (calls.append(start), real_replan(start=start))[1]
    planner = client_mod.StrategistPlanner(adapted)
    planner.on_uplink(up(1, lat=here[0], lon=here[1]))
    planner.on_uplink(up(2, lat=here[0], lon=here[1]))
    assert len(calls) == 1, "one replan for the session, not one per uplink"
    planner.on_uplink(up(0, sid=78, lat=here[0], lon=here[1]))
    assert len(calls) == 2, "a restarted Pi with nothing reached gets its own first transit"
    planner.on_uplink(up(0, sid=79, lat=here[0], lon=here[1], waypoint_index=3))
    assert len(calls) == 2, "but never once the Pi reports progress: that plan is under way"


def test_the_strategist_is_not_replanned_from_a_report_it_cannot_trust():
    from backend.src.nav.thor_strategist import ThorStrategist

    s = ThorStrategist([(45.0, -93.0), (45.0, -92.999), (45.0009, -92.999), (45.0009, -93.0)])
    before = [(leg.x, leg.y) for leg in s.legs]
    planner = client_mod.StrategistPlanner(s)
    assert planner.on_uplink(up(0, fix="rtk_float")).pause
    assert planner.on_uplink(up(1, lat=None, lon=None)).pause
    assert [(leg.x, leg.y) for leg in s.legs] == before


def test_plans_load_from_json_files(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {"plan_id": "bench-9", "waypoints": [{"lat": 45.0, "lon": -93.0, "blade": True}]}
        )
    )
    planner = client_mod.load_planner(plan=str(plan), boundary=None, require_fix="rtk_fixed")
    down = planner(up(0))
    assert down.plan_id == "bench-9" and down.waypoints[0].blade is True
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps([{"lat": 45.0, "lon": -93.0}]))
    assert (
        client_mod.load_planner(plan=str(bare), boundary=None, require_fix="rtk_fixed")(
            up(0)
        ).plan_id
        == "static"
    )
    boundary = tmp_path / "yard.json"
    boundary.write_text(
        json.dumps({"plan_id": "yard", "boundary": [[45, -93], [45, -92.999], [45.0009, -92.999]]})
    )
    assert client_mod.load_planner(plan=None, boundary=str(boundary), require_fix="rtk_fixed")
    with pytest.raises(ValueError, match="exactly one"):
        client_mod.load_planner(plan=None, boundary=None, require_fix="rtk_fixed")
    with pytest.raises(ValueError, match="exactly one"):
        client_mod.load_planner(plan=str(plan), boundary=str(boundary), require_fix="rtk_fixed")


# ------------------------------- real sockets, real CLI -------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ask(port: int, data: bytes, timeout: float = 0.4) -> bytes | None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(timeout)
        s.sendto(data, ("127.0.0.1", port))
        try:
            return s.recvfrom(2048)[0]
        except TimeoutError:
            return None


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("THOR_LINK_KEY", "SIM_MODE")}
    env.update(extra)
    env["PYTHONPATH"] = str(SCRIPT.parents[1])
    return env


def test_the_command_line_serves_a_plan_pauses_on_a_signal_and_stops_cleanly(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([{"lat": 45.0, "lon": -93.0}, {"lat": 45.0001, "lon": -93.0}]))
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--listen",
            f"127.0.0.1:{port}",
            "--plan",
            str(plan),
            "--key-file",
            str(_key_file(tmp_path)),
            "--pi",
            "127.0.0.1",
        ],
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        seq = 0
        reply = None
        deadline = time.monotonic() + 20
        while reply is None and time.monotonic() < deadline:  # interpreter and imports take a while
            reply = _ask(port, datagram(seq, sid=31))
            seq += 1
        assert reply is not None, "the client never answered"
        down = decode_downlink(reply, KEY)
        assert (down.sid, down.plan_id) == (31, "static") and len(down.waypoints) == 2
        assert _ask(port, datagram(seq + 1, sid=31, key=OTHER_KEY)) is None, "wrong key: silence"
        proc.send_signal(signal.SIGUSR1)
        time.sleep(0.3)
        assert decode_downlink(_ask(port, datagram(seq + 2, sid=31)), KEY).pause
        proc.send_signal(signal.SIGUSR2)
        time.sleep(0.3)
        assert not decode_downlink(_ask(port, datagram(seq + 3, sid=31)), KEY).pause
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()


@pytest.mark.parametrize(
    ("args", "env", "message"),
    [
        (["--plan", "plan.json"], {}, "no link key"),
        (["--plan", "plan.json", "--sim"], {}, "SIM_MODE=1"),
        (["--plan", "plan.json", "--key-file", "absent.key"], {}, "cannot read key file"),
        (["--plan", "plan.json"], {"THOR_LINK_KEY": "abcd"}, "at least 32 bytes"),
        (["--sim"], {"SIM_MODE": "1"}, "exactly one"),
    ],
)
def test_the_command_line_refuses_to_start_without_a_usable_key_or_plan(
    tmp_path, args, env, message
):
    (tmp_path / "plan.json").write_text(json.dumps([{"lat": 45.0, "lon": -93.0}]))
    args = [str(tmp_path / a) if a in ("plan.json", "absent.key") else a for a in args]
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--listen", f"127.0.0.1:{_free_port()}", *args],
        env=_env(**env),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 2 and message in done.stderr
    assert env.get("THOR_LINK_KEY", "no-key-set") not in done.stderr, "a key is never echoed"


async def test_the_endpoint_answers_over_a_real_udp_socket():
    c = await ThorLinkClient.serve(KEY, StaticPlanner(wps(2), "net").on_uplink, "127.0.0.1", 0)
    try:
        loop = asyncio.get_running_loop()
        got: asyncio.Future = loop.create_future()

        class Pi(asyncio.DatagramProtocol):
            def datagram_received(self, data, addr):
                got.set_result((data, addr))

        transport, _ = await loop.create_datagram_endpoint(Pi, remote_addr=("127.0.0.1", c.port))
        transport.sendto(datagram(1, sid=8))
        data, addr = await asyncio.wait_for(got, 2.0)
        assert addr[1] == c.port and decode_downlink(data, KEY).plan_id == "net"
        transport.close()
    finally:
        c.close()
