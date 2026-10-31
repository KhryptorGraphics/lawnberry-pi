"""Behavioral tests for the offline dual-controller shutdown model.

Deterministic and injected-time: no sleeps, no hardware, no network. Each test asserts
observable model behavior at a failure boundary, not implementation wiring. A model that
reports a physical outcome as known would fail these tests by design.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from hardware.control.dual_controller_sim import (
    NEUTRAL_NOT_COMMANDABLE,
    PEER_TIMEOUT_MS,
    SEL_SLAVE_US,
    ContactFault,
    ControllerState,
    HostRole,
    MuxSource,
    PwmCalibration,
    Simulation,
    Transport,
    fresh_pair,
)

HEARTBEAT_MS = 200


def _authorized() -> Simulation:
    return fresh_pair()


class TestHealthyPair:
    def test_both_hosts_reach_run_authorized(self):
        sim = _authorized()
        assert sim.pi.state is ControllerState.RUN_AUTHORIZED
        assert sim.alta.state is ControllerState.RUN_AUTHORIZED

    def test_healthy_pair_issues_no_stop_requests(self):
        sim = _authorized()
        sim.step()
        snap = sim.snapshot()
        assert snap["pto_off_requested"] is False
        assert snap["engine_off_requested"] is False
        assert snap["neutral_requested"] is False
        assert snap["capability_failures"] == []

    def test_pi_pulses_are_selected_while_healthy(self):
        sim = _authorized()
        sim.pi_pwm.write(0, 1600)
        sim.pi_pwm.write(2, 1400)
        sim.step()
        snap = sim.snapshot()
        assert snap["selected_source"] == MuxSource.PI.value
        assert snap["selected_pwm_us"] == [1600, 1400]

    def test_authorization_requires_human_confirmation(self):
        sim = Simulation()
        for _ in range(4):
            sim.step()
        assert sim.pi.authorize_run(operator_confirmed=False, now_ms=sim.now_ms) is False
        assert sim.pi.state is ControllerState.PEER_VALIDATED


class TestFreshnessBoundary:
    def test_heartbeat_accepted_just_before_timeout(self):
        sim = _authorized()
        assert PEER_TIMEOUT_MS == 600
        sim.now_ms += PEER_TIMEOUT_MS - 1
        assert sim.pi.peer_fresh(sim.now_ms) is True
        sim.pi.tick(sim.now_ms)
        assert sim.pi.state is ControllerState.RUN_AUTHORIZED

    def test_timeout_latches_at_exactly_six_hundred_ms(self):
        sim = _authorized()
        sim.now_ms += PEER_TIMEOUT_MS
        assert sim.pi.peer_fresh(sim.now_ms) is False
        sim.pi.tick(sim.now_ms)
        assert sim.pi.state is ControllerState.FAULT_LATCHED
        assert sim.pi.fault_reason == "peer_timeout"

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda hb: replace(hb, authenticated=False), id="auth"),
            pytest.param(lambda hb: replace(hb, version=99), id="version"),
            pytest.param(lambda hb: replace(hb, peer_boot_id="other-session"), id="wrong-session"),
            pytest.param(lambda hb: replace(hb, sequence=0), id="replayed-sequence"),
        ],
    )
    def test_invalid_heartbeats_do_not_refresh_freshness(self, mutate):
        sim = _authorized()
        stale_age = sim.now_ms
        message = mutate(sim.alta.heartbeat(stale_age))
        accepted = sim.pi.receive_heartbeat(message, stale_age + 599)
        assert accepted is False
        assert sim.pi.peer.last_rx_ms == stale_age
        sim.pi.tick(stale_age + 600)
        assert sim.pi.state is ControllerState.FAULT_LATCHED

    def test_retired_boot_id_never_regains_freshness(self):
        sim = _authorized()
        sim.pi.latch_store.retire(HostRole.ALTA, "alta-boot-1")
        message = sim.alta.heartbeat(sim.now_ms)
        assert sim.pi.receive_heartbeat(message, sim.now_ms) is False


class TestSingleHostFailure:
    def test_pi_loss_makes_alta_request_off_without_pi(self):
        sim = _authorized()
        sim.pto_chain.oem_pto_allowed = True
        assert sim.pto_chain.feed_permitted is True
        sim.pi_alive = False
        for _ in range(4):
            sim.step()
        snap = sim.snapshot()
        assert sim.alta.state is ControllerState.FAULT_LATCHED
        assert snap["pto_off_requested"] is True
        assert snap["engine_off_requested"] is True
        assert snap["modeled_clutch_feed_permitted"] is False
        assert snap["modeled_kill_ground_requested"] is True

    def test_pi_loss_selects_alta_neutral_after_sel_qualifies(self):
        sim = _authorized()
        sim.pi_alive = False
        for _ in range(4):
            sim.step()
        snap = sim.snapshot()
        assert snap["selected_source"] == MuxSource.ALTA.value
        assert snap["selected_pwm_us"] == [
            sim.calibration.left_neutral_us,
            sim.calibration.right_neutral_us,
        ]

    def test_alta_loss_leaves_pi_in_control_and_requests_off(self):
        sim = _authorized()
        sim.pto_chain.oem_pto_allowed = True
        sim.alta_alive = False
        for _ in range(4):
            sim.step()
        snap = sim.snapshot()
        assert sim.pi.state is ControllerState.FAULT_LATCHED
        assert snap["pto_off_requested"] is True
        assert snap["engine_off_requested"] is True
        assert snap["selected_source"] == MuxSource.PI.value
        assert snap["selected_pwm_us"] == [
            sim.calibration.left_neutral_us,
            sim.calibration.right_neutral_us,
        ]

    def test_alta_loss_never_leaves_slave_selection_active(self):
        sim = _authorized()
        sim.alta_pwm.write(3, SEL_SLAVE_US)
        sim.alta_alive = False
        sim.alta_pwm.powered = False
        for _ in range(10):
            sim.step()
        assert sim.snapshot()["selected_source"] == MuxSource.PI.value


class TestTransportModes:
    @pytest.mark.parametrize("transport", [Transport.HALOW, Transport.ETHERNET])
    def test_link_loss_latches_both_hosts_in_either_mode(self, transport):
        sim = fresh_pair(transport=transport)
        assert sim.transport is transport
        sim.link_up = False
        for _ in range(4):
            sim.step()
        assert sim.pi.state is ControllerState.FAULT_LATCHED
        assert sim.alta.state is ControllerState.FAULT_LATCHED

    def test_one_directional_loss_still_times_out_the_receiver(self):
        sim = _authorized()

        class OneWay(Simulation):
            def exchange(self):  # only Pi can be heard
                if self.link_up:
                    self.alta.receive_heartbeat(self.pi.heartbeat(self.now_ms), self.now_ms)

        sim.exchange = OneWay.exchange.__get__(sim)
        for _ in range(4):
            sim.step()
        # The receiver times out locally, and the still-listening peer then latches on the
        # reported fault. Both stop: a half-open link never keeps one host running.
        assert sim.pi.state is ControllerState.FAULT_LATCHED
        assert sim.pi.fault_reason == "peer_timeout"
        assert sim.alta.state is ControllerState.FAULT_LATCHED
        assert sim.alta.fault_reason == "peer_reported_fault"

    def test_no_implicit_transport_fallback(self):
        sim = fresh_pair(transport=Transport.ETHERNET)
        sim.link_up = False
        for _ in range(4):
            sim.step()
        assert sim.transport is Transport.ETHERNET


class TestPtoPermission:
    @pytest.mark.parametrize("oem", [False, True])
    @pytest.mark.parametrize("k_pi", [False, True])
    @pytest.mark.parametrize("k_alta", [False, True])
    def test_series_permission_requires_all_three(self, oem, k_pi, k_alta):
        sim = Simulation()
        sim.pto_chain.oem_pto_allowed = oem
        sim.pto_chain.pi.coil_energized = k_pi
        sim.pto_chain.alta.coil_energized = k_alta
        assert sim.pto_chain.feed_permitted is (oem and k_pi and k_alta)

    def test_oem_off_always_blocks_the_feed(self):
        sim = Simulation()
        sim.pto_chain.oem_pto_allowed = False
        sim.pto_chain.pi.coil_energized = True
        sim.pto_chain.alta.coil_energized = True
        assert sim.pto_chain.feed_permitted is False

    def test_either_host_opens_its_own_contact_independently(self):
        sim = _authorized()
        sim.pto_chain.oem_pto_allowed = True
        sim.alta.request_pto_off()
        assert sim.pto_chain.alta.closed is False
        assert sim.pto_chain.pi.closed is True
        assert sim.pto_chain.feed_permitted is False

    def test_welded_pi_contact_is_still_stopped_by_alta(self):
        sim = _authorized()
        sim.pto_chain.oem_pto_allowed = True
        sim.pto_chain.pi.fault = ContactFault.STUCK_CLOSED
        sim.pi_alive = False
        for _ in range(4):
            sim.step()
        assert sim.snapshot()["modeled_clutch_feed_permitted"] is False

    def test_welded_alta_contact_is_still_stopped_by_pi(self):
        sim = _authorized()
        sim.pto_chain.oem_pto_allowed = True
        sim.pto_chain.alta.fault = ContactFault.STUCK_CLOSED
        sim.alta_alive = False
        for _ in range(4):
            sim.step()
        assert sim.snapshot()["modeled_clutch_feed_permitted"] is False

    def test_both_welded_remains_feeding_and_is_reported_unsafe(self):
        sim = _authorized()
        sim.pto_chain.oem_pto_allowed = True
        sim.pto_chain.pi.fault = ContactFault.STUCK_CLOSED
        sim.pto_chain.alta.fault = ContactFault.STUCK_CLOSED
        sim.alta_alive = False
        for _ in range(4):
            sim.step()
        snap = sim.snapshot()
        # The model must expose this, not manufacture a shutdown it cannot achieve.
        assert snap["pto_off_requested"] is True
        assert snap["modeled_clutch_feed_permitted"] is True

    def test_coil_power_loss_opens_an_energize_to_allow_contact(self):
        sim = _authorized()
        sim.pto_chain.oem_pto_allowed = True
        sim.pto_chain.pi.coil_power = False
        assert sim.pto_chain.pi.closed is False
        assert sim.pto_chain.feed_permitted is False

    def test_missing_pto_capability_does_not_suppress_engine_request(self):
        sim = _authorized()
        sim.alta.outputs = type(sim.alta.outputs)(pto_contact=False, engine_stop=True)
        sim.enter = sim.alta.enter_latched_stop("manual_fault", sim.now_ms)
        snap = sim.snapshot()
        assert snap["engine_off_requested"] is True
        assert "requested_not_observed" in snap["capability_failures"]


class TestEngineStop:
    def test_either_parallel_contact_grounds_the_kill_node(self):
        sim = Simulation()
        sim.engine_pair.pi.coil_energized = False
        sim.engine_pair.alta.coil_energized = True
        assert sim.engine_pair.kill_ground_requested is True
        assert sim.engine_pair.run_permitted is False

    def test_one_stuck_open_contact_leaves_the_other_effective(self):
        sim = Simulation()
        sim.engine_pair.pi.fault = ContactFault.STUCK_OPEN
        sim.engine_pair.pi.coil_energized = False
        sim.engine_pair.alta.coil_energized = False
        assert sim.engine_pair.kill_ground_requested is True

    def test_both_stuck_open_defeats_the_modelled_kill_path(self):
        sim = Simulation()
        sim.engine_pair.pi.fault = ContactFault.STUCK_OPEN
        sim.engine_pair.alta.fault = ContactFault.STUCK_OPEN
        sim.engine_pair.pi.coil_energized = False
        sim.engine_pair.alta.coil_energized = False
        assert sim.engine_pair.kill_ground_requested is False

    def test_engine_request_is_independent_of_pto_contact_state(self):
        sim = _authorized()
        sim.alta.request_engine_off()
        assert sim.alta.requests.engine_off is True
        assert sim.pto_chain.alta.closed is True  # PTO untouched by the kill request


class TestStopTiming:
    def test_pto_and_engine_off_are_not_postponed_behind_neutral(self):
        sim = _authorized()
        sim.alta.enter_latched_stop("test", sim.now_ms)
        assert sim.alta.requests.pto_off is True
        assert sim.alta.requests.engine_off is True
        assert sim.alta.requests.steering_neutral is True

    def test_stop_is_idempotent(self):
        sim = _authorized()
        sim.alta.enter_latched_stop("test", sim.now_ms)
        first = sim.alta.requests
        sim.alta.enter_latched_stop("test", sim.now_ms)
        assert sim.alta.requests == first

    def test_neutral_request_survives_jammed_steering(self):
        sim = _authorized()
        sim.servo_power = False
        sim.alta.enter_latched_stop("jammed", sim.now_ms)
        snap = sim.snapshot()
        assert snap["neutral_requested"] is True
        assert snap["physical_observations"] is None


class TestRearmAndReboot:
    def test_recovery_does_not_auto_rearm(self):
        sim = _authorized()
        sim.link_up = False
        for _ in range(4):
            sim.step()
        sim.link_up = True
        for _ in range(6):
            sim.step()
        assert sim.pi.state is ControllerState.FAULT_LATCHED

    def test_rearm_returns_to_validated_not_run(self):
        sim = _authorized()
        sim.link_up = False
        for _ in range(4):
            sim.step()
        sim.link_up = True
        for _ in range(6):
            sim.step()
        assert sim.pi.rearm(operator_confirmed=True, now_ms=sim.now_ms) is True
        assert sim.pi.state is ControllerState.PEER_VALIDATED

    def test_rearm_refused_without_operator_confirmation(self):
        sim = _authorized()
        sim.link_up = False
        for _ in range(4):
            sim.step()
        sim.link_up = True
        for _ in range(6):
            sim.step()
        assert sim.pi.rearm(operator_confirmed=False, now_ms=sim.now_ms) is False

    def test_shared_latch_store_survives_host_reconstruction(self):
        sim = _authorized()
        sim.link_up = False
        for _ in range(4):
            sim.step()
        rebuilt = Simulation(latch_store=sim.latch_store)
        assert rebuilt.pi.state is ControllerState.FAULT_LATCHED
        for _ in range(4):
            rebuilt.step()
        assert rebuilt.pi.state is ControllerState.FAULT_LATCHED

    def test_empty_store_cold_boots_disarmed(self):
        assert Simulation().pi.state is ControllerState.BOOT_DISARMED


class TestMuxAndHonesty:
    def test_mux_power_loss_is_outputs_low_not_neutral(self):
        sim = _authorized()
        sim.mux.powered = False
        sim.step()
        snap = sim.snapshot()
        assert snap["selected_source"] == MuxSource.OUTPUTS_LOW.value
        assert snap["selected_pwm_us"] == [None, None]
        assert snap["neutral_pulse_selected"] is False

    def test_failmode_on_forces_low_and_is_never_neutral(self):
        sim = _authorized()
        sim.mux.failmode_on = True
        sim.mux.last_pulse_ms = None
        sim.step()
        assert sim.snapshot()["neutral_pulse_selected"] is False

    def test_missing_neutral_capability_is_reported(self):
        sim = _authorized()
        sim.alta.outputs = type(sim.alta.outputs)(steering_neutral=False)
        sim.pi_alive = False
        for _ in range(4):
            sim.step()
        assert NEUTRAL_NOT_COMMANDABLE in sim.snapshot()["capability_failures"]

    def test_frozen_pi_pwm_is_invisible_to_heartbeat_supervision(self):
        sim = _authorized()
        sim.pi_pwm.driven_by_live_host = False
        for _ in range(4):
            sim.step()
        snap = sim.snapshot()
        # Pi is still heartbeating, so no stop is triggered and the held angle persists.
        assert snap["pto_off_requested"] is False
        assert snap["selected_source"] == MuxSource.PI.value
        assert snap["physical_observations"] is None

    def test_snapshot_never_claims_a_physical_outcome(self):
        sim = _authorized()
        sim.pi_alive = False
        for _ in range(4):
            sim.step()
        snap = sim.snapshot()
        assert snap["physical_observations"] is None
        physical_keys = [k for k in snap if k.startswith("bar_") or k.endswith("_reached")]
        assert physical_keys == []

    def test_calibration_is_marked_unverified_by_default(self):
        assert PwmCalibration().physically_verified is False

    def test_module_source_imports_no_live_hardware_or_backend(self):
        """Read the module's own imports, not global sys.modules.

        Asserting on sys.modules fails when another test in the same session has already
        imported backend.* -- which the rest of this suite legitimately does. The property
        that matters is that THIS module imports nothing live, so inspect its source.
        """
        import ast
        import inspect

        from hardware.control import dual_controller_sim

        tree = ast.parse(inspect.getsource(dual_controller_sim))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        forbidden = {"RPi", "smbus2", "gpiozero", "periphery", "backend", "serial"}
        assert imported & forbidden == set(), f"live dependencies imported: {imported & forbidden}"
        # It may only use the standard library.
        assert imported <= {"dataclasses", "enum", "__future__"}, imported
