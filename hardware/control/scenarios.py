"""Deterministic failure scenarios and acceptance tables for the offline controller model.

Run with ``python -m hardware.control.scenarios --output-dir <dir>``. Output is JSON plus a
readable Markdown report. Nothing here touches hardware or the network, and no result is a
physical claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hardware.control.dual_controller_sim import (
    SEL_MASTER_US,
    SEL_SLAVE_US,
    ContactFault,
    Simulation,
    fresh_pair,
)

# Inventory status is a property of the real machine, not of the model.
INTERFACE_STATUS = {
    "pi_pto_permission": "owned_candidate (HiLetGo, contact/rating unverified)",
    "alta_pto_permission": "CONDITIONAL (second contact not in verified inventory)",
    "pi_engine_stop": "CONDITIONAL (contact form and engine interface BLOCKED)",
    "alta_engine_stop": "CONDITIONAL (contact form and engine interface BLOCKED)",
    "steering_power_relay": "owned allocation, host ownership unresolved",
    "starter": "not modelled (OEM PARK interlock blocks autonomous crank)",
    "pololu_2806": "owned, bench validation pending",
}


def _scenario(name: str) -> dict:
    builder = globals().get(f"_scenario_{name}")
    if builder is None:
        raise KeyError(f"unknown scenario {name!r}")
    return builder()


def _summary(sim: Simulation, *, scenario: str, steps: list[dict]) -> dict:
    snap = sim.snapshot()
    return {
        "scenario": scenario,
        "transport": sim.transport.value,
        "interface_status": INTERFACE_STATUS,
        "steps": steps,
        "final": snap,
        "limitations": [
            "Requests are commands; no bar position, clutch release, blade RPM or engine "
            "state is observed.",
            "Both PTO contacts welded closed with OEM ON remains feed-permitted and unsafe.",
            "Both engine-stop contacts stuck open defeats the modelled kill path.",
            "A wedged-but-heartbeating host is invisible to heartbeat supervision.",
            "Simultaneous host loss and mux/servo power loss are unprotected by construction.",
        ],
    }


def _scenario_healthy_pair() -> dict:
    sim = fresh_pair()
    steps = [{"t_ms": sim.now_ms, "note": "reciprocal validation + human authorization"}]
    sim.step()
    steps.append({"t_ms": sim.now_ms, "note": "steady heartbeat", "state": sim.snapshot()})
    return _summary(sim, scenario="healthy_pair", steps=steps)


def _scenario_pi_loss() -> dict:
    sim = fresh_pair()
    sim.pto_chain.oem_pto_allowed = True
    before = sim.snapshot()
    sim.pi_alive = False
    steps = [{"t_ms": sim.now_ms, "note": "Pi power/process lost", "state": before}]
    # Pi PCA keeps its last pulses and K_PI keeps its state: a crash is not a de-energize.
    sim.pi_pwm.driven_by_live_host = False
    for _ in range(4):
        sim.step()
        steps.append({"t_ms": sim.now_ms, "note": "Alta alone", "state": sim.snapshot()})
    return _summary(sim, scenario="pi_loss", steps=steps)


def _scenario_alta_loss() -> dict:
    sim = fresh_pair()
    sim.pto_chain.oem_pto_allowed = True
    sim.alta_alive = False
    sim.alta_pwm.driven_by_live_host = False
    steps = [{"t_ms": sim.now_ms, "note": "Alta power/process lost"}]
    for _ in range(4):
        sim.step()
        steps.append({"t_ms": sim.now_ms, "note": "Pi alone", "state": sim.snapshot()})
    return _summary(sim, scenario="alta_loss", steps=steps)


def _scenario_link_split() -> dict:
    sim = fresh_pair()
    sim.pto_chain.oem_pto_allowed = True
    sim.link_up = False
    steps = [{"t_ms": sim.now_ms, "note": "peer link severed, both hosts alive"}]
    for _ in range(4):
        sim.step()
        steps.append(
            {"t_ms": sim.now_ms, "note": "no split-brain election", "state": sim.snapshot()}
        )
    return _summary(sim, scenario="link_split", steps=steps)


def _scenario_sel_frozen_slave() -> dict:
    sim = fresh_pair()
    sim.alta_pwm.write(3, SEL_SLAVE_US)
    for _ in range(3):
        sim.mux.drive_sel(sim.now_ms, SEL_SLAVE_US)
    sim.alta_alive = False  # PCA keeps the last valid slave-select pulses
    sim.alta_pwm.driven_by_live_host = False
    sim.pi_alive = True
    steps = [{"t_ms": sim.now_ms, "note": "Alta frozen with valid slave SEL"}]
    for _ in range(3):
        sim.step()
        steps.append({"t_ms": sim.now_ms, "note": "frozen SEL", "state": sim.snapshot()})
    return _summary(sim, scenario="sel_frozen_slave", steps=steps)


def _scenario_sel_missing() -> dict:
    sim = fresh_pair()
    sim.alta_alive = False
    sim.alta_pwm.powered = False
    steps = [{"t_ms": sim.now_ms, "note": "Alta gone, SEL pulses stop"}]
    for _ in range(10):
        sim.step()
        steps.append(
            {"t_ms": sim.now_ms, "note": "FAILMODE OFF -> master", "state": sim.snapshot()}
        )
    return _summary(sim, scenario="sel_missing", steps=steps)


def _scenario_pto_permission_matrix() -> dict:
    sim = Simulation()
    rows = []
    for oem in (False, True):
        for k_pi in (False, True):
            for k_alta in (False, True):
                sim.pto_chain.oem_pto_allowed = oem
                sim.pto_chain.pi.coil_energized = k_pi
                sim.pto_chain.alta.coil_energized = k_alta
                rows.append(
                    {
                        "oem_pto_allowed": oem,
                        "k_pi_closed": sim.pto_chain.pi.closed,
                        "k_alta_closed": sim.pto_chain.alta.closed,
                        "feed_permitted": sim.pto_chain.feed_permitted,
                    }
                )
    return _summary(sim, scenario="pto_permission_matrix", steps=rows)


def _scenario_welded_pto() -> dict:
    sim = fresh_pair()
    sim.pto_chain.oem_pto_allowed = True
    sim.pto_chain.pi.fault = ContactFault.STUCK_CLOSED
    sim.alta_alive = False
    steps = []
    for _ in range(4):
        sim.step()
        steps.append(
            {
                "t_ms": sim.now_ms,
                "note": "K_PI welded; Alta opens its own",
                "state": sim.snapshot(),
            }
        )
    sim2 = fresh_pair()
    sim2.pto_chain.oem_pto_allowed = True
    sim2.pto_chain.pi.fault = ContactFault.STUCK_CLOSED
    sim2.pto_chain.alta.fault = ContactFault.STUCK_CLOSED
    sim2.alta_alive = False
    for _ in range(4):
        sim2.step()
    steps.append(
        {
            "t_ms": sim2.now_ms,
            "note": "both contacts welded: feed stays permitted and unsafe",
            "state": sim2.snapshot(),
        }
    )
    return _summary(sim, scenario="welded_pto", steps=steps)


def _scenario_engine_kill_parallel() -> dict:
    sim = Simulation()
    rows = []
    for pi_fault in (ContactFault.HEALTHY, ContactFault.STUCK_OPEN):
        for alta_fault in (ContactFault.HEALTHY, ContactFault.STUCK_OPEN):
            pair = sim.engine_pair
            pair.pi.fault = pi_fault
            pair.alta.fault = alta_fault
            pair.pi.coil_energized = False  # both hosts requesting stop
            pair.alta.coil_energized = False
            rows.append(
                {
                    "pi_contact": pi_fault.value,
                    "alta_contact": alta_fault.value,
                    "kill_ground_requested": pair.kill_ground_requested,
                }
            )
    return _summary(sim, scenario="engine_kill_parallel", steps=rows)


def _scenario_no_auto_rearm() -> dict:
    sim = fresh_pair()
    sim.link_up = False
    for _ in range(4):
        sim.step()
    latched = sim.snapshot()
    sim.link_up = True
    for _ in range(6):
        sim.step()
    recovered = sim.snapshot()
    refused = sim.pi.rearm(operator_confirmed=False, now_ms=sim.now_ms)
    accepted = sim.pi.rearm(operator_confirmed=True, now_ms=sim.now_ms)
    after = sim.snapshot()
    steps = [
        {"t_ms": sim.now_ms, "note": "fault latched", "state": latched},
        {"t_ms": sim.now_ms, "note": "link restored, no auto-rearm", "state": recovered},
        {"operator_confirmed_false": refused, "operator_confirmed_true": accepted},
        {"t_ms": sim.now_ms, "note": "rearm returns to validated only", "state": after},
    ]
    return _summary(sim, scenario="no_auto_rearm", steps=steps)


def _scenario_reboot_with_latch() -> dict:
    sim = fresh_pair()
    sim.link_up = False
    for _ in range(4):
        sim.step()
    store = sim.latch_store
    rebuilt = Simulation(latch_store=store)
    steps = [
        {"note": "shared latch store retained", "rebuilt_pi_state": rebuilt.pi.state.value},
    ]
    for _ in range(4):
        rebuilt.step()
    steps.append({"note": "after healthy exchange", "state": rebuilt.snapshot()})
    cold = Simulation()
    steps.append({"note": "empty store cold boot", "cold_pi_state": cold.pi.state.value})
    return _summary(rebuilt, scenario="reboot_with_latch", steps=steps)


def _scenario_mux_power_loss() -> dict:
    sim = fresh_pair()
    sim.mux.powered = False
    sim.step()
    steps = [{"t_ms": sim.now_ms, "note": "mux supply lost (outputs_low)", "state": sim.snapshot()}]
    sim.servo_power = False
    sim.step()
    steps.append({"t_ms": sim.now_ms, "note": "servo power also lost", "state": sim.snapshot()})
    return _summary(sim, scenario="mux_power_loss", steps=steps)


def _scenario_neutral_not_commandable() -> dict:
    sim = fresh_pair()
    sim.alta.outputs = type(sim.alta.outputs)(steering_neutral=False)
    sim.pi_alive = False
    sim.pi_pwm.driven_by_live_host = False
    for _ in range(4):
        sim.step()
    return _summary(
        sim,
        scenario="neutral_not_commandable",
        steps=[{"t_ms": sim.now_ms, "note": "capability absent", "state": sim.snapshot()}],
    )


def _scenario_pi_pwm_frozen() -> dict:
    sim = fresh_pair()
    sim.pi_pwm.driven_by_live_host = False  # PCA wedged, host still heartbeating
    steps = []
    for _ in range(4):
        sim.step()
        steps.append(
            {"t_ms": sim.now_ms, "note": "Pi alive but PWM frozen", "state": sim.snapshot()}
        )
    return _summary(sim, scenario="pi_pwm_frozen", steps=steps)


SCENARIOS = (
    "healthy_pair",
    "pi_loss",
    "alta_loss",
    "link_split",
    "sel_frozen_slave",
    "sel_missing",
    "pto_permission_matrix",
    "welded_pto",
    "engine_kill_parallel",
    "no_auto_rearm",
    "reboot_with_latch",
    "mux_power_loss",
    "neutral_not_commandable",
    "pi_pwm_frozen",
)


def truth_table_rows() -> list[dict]:
    """SEL/mux behavior as a table independent of any scenario run."""
    probes = [
        ("valid below threshold", True, SEL_MASTER_US, "off"),
        ("valid above threshold", True, SEL_SLAVE_US, "off"),
        ("invalid, FAILMODE off", False, 0, "off"),
        ("invalid, FAILMODE on", False, 0, "on"),
    ]
    rows = []
    for label, valid, pulse, failmode in probes:
        mux = Simulation().mux
        mux.failmode_on = failmode == "on"
        if valid:
            for _ in range(3):
                mux.drive_sel(0, pulse)
        rows.append(
            {
                "condition": label,
                "sel_valid": valid,
                "sel_us": pulse,
                "failmode": failmode,
                "expected_source": mux.source(0).value,
                "is_neutral_claim": False,
            }
        )
    return rows


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# Dual-controller offline simulation report",
        "",
        "Generated by `python -m hardware.control.scenarios`. Logic and modelled electrical",
        "state only; **no physical outcome is observed or claimed**.",
        "",
        "## SEL / mux source table",
        "",
        "| Condition | SEL valid | SEL µs | FAILMODE | Modelled source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in truth_table_rows():
        lines.append(
            f"| {row['condition']} | {row['sel_valid']} | {row['sel_us']} | "
            f"{row['failmode']} | {row['expected_source']} |"
        )
    lines += ["", "## Scenarios", ""]
    for result in results:
        final = result["final"]
        lines += [
            f"### {result['scenario']}",
            "",
            f"- Pi: `{final['pi_state']}` (fault: {final['pi_fault']})",
            f"- Alta: `{final['alta_state']}` (fault: {final['alta_fault']})",
            f"- Neutral requested: {final['neutral_requested']}; "
            f"source `{final['selected_source']}`; pulses {final['selected_pwm_us']}",
            f"- PTO OFF requested: {final['pto_off_requested']}; "
            f"modelled clutch feed permitted: {final['modeled_clutch_feed_permitted']}",
            f"- Engine OFF requested: {final['engine_off_requested']}; "
            f"modelled kill ground requested: {final['modeled_kill_ground_requested']}",
            f"- Capability failures: {final['capability_failures'] or 'none'}",
            f"- Physical observations: {final['physical_observations']}",
            "",
        ]
    lines += ["## Interface inventory status", ""]
    for key, value in INTERFACE_STATUS.items():
        lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hardware/control/reports"),
        help="directory for the JSON and Markdown reports",
    )
    args = parser.parse_args(argv)
    results = [_scenario(name) for name in SCENARIOS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render_markdown(results), encoding="utf-8")
    print(f"wrote {len(results)} scenarios to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
