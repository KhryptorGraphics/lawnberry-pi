// CSG probes for the control_77603 electronics packaging (chassis, base, compute tower,
// strap bays, harness clamp, signal-board clips) inside the unchanged legacy enclosure.
//
// Each probe intersects REAL built solids. "empty" probes must render
// "Current top level object is empty" (no interference); "contact" probes must NOT (named
// faces meet: a seat, a stop, a capture). Contact proves only that the geometry meets where
// sampled; no retention force, stiffness, vibration or thermal claim follows from it.
//
// Run: openscad --export-format binstl -o /dev/null -D 'check="name"' [-D 'idx=N'] control_checks.scad
include <cleat_carrier.scad>
use <assemblies.scad>
use <board_carriers.scad>
use <compute_tower.scad>
use <chassis.scad>
use <board_refs.scad>
use <../enclosure_body.scad>
use <../enclosure_lid.scad>

check = "none";
idx = 0;

// Every printed part and payload inside the box (seated, assembled).
module all_printed() {
    chassis_base();
    chassis();
    ec_tower();
    for (i = [0 : len(ch_strips) - 1]) ec_carrier(i, cr_seat_shift);
}
module all_payloads() {
    ec_boards();
    for (i = [0 : len(ch_strips) - 1]) ec_payload(i, cr_seat_shift);
    ec_signal_boards();
}
module all_contents() { all_printed(); all_payloads(); }

// Carrier i (seated, with its payload), and everything else that could meet it.
module carrier_with_payload(i) { ec_carrier(i, cr_seat_shift); ec_payload(i, cr_seat_shift); }
module everything_but(i) {
    for (j = [0 : len(ch_strips) - 1]) if (j != i) carrier_with_payload(j);
    ec_tower();
    ec_boards();
    ec_signal_boards();
}

// Header pairs [lower level, lower pin height, upper level, upper socket depth].
hdr_pairs = [
    ["alta", male_pin_h, "hra", socket_h],
    ["pi", male_pin_h, "ai", s_pi_ai - hdr_base_h],
    ["ai", stack_pin_h, "lc", socket_h],
    ["lc", stack_pin_h, "hrp", socket_h],
];

// Strap-bay probe boxes: [w, d, h] at the bay's minimum and maximum capacity.
bay_probe_box = [bay_box_min, [bay_box_max[0], bay_box_max[1], 20]];
// The box rests 0.05 off the ledge top and the rib tips (its seat faces), so the probes
// never intersect coplanar faces.
module bay_box_local(b) {
    translate([bay_w / 2 - b[0] / 2, bay_ref_y - b[1] - 0.05, -bay_drop + bay_ledge_t + 0.05]) cube(b);
}
function bay_stop(d) = bay_d_k(bay_k_for_box(d));
hc_probe_d = [hc_bundle_d[0], hc_bundle_d[1]];
function hc_stop(d) = min([for (k = [0 : 15]) let(dk = hc_bundle_d[1] - rt_pitch * k) if (dk >= d) dk]);
module hc_bundle_local(d) { translate([hc_w / 2, bay_ref_y - d / 2 - 0.05, -bay_drop - 5]) cylinder(d = d, h = 60); }

// ---- Base and chassis on the box ------------------------------------------------------------------
if (check == "base_seats_on_bosses") {
    // CONTACT: lowered 0.05, the base's underside meets the four boss tops.
    intersection() { translate([0, 0, -0.05]) chassis_base(); enclosure_body(); }
} else if (check == "base_clear_body") {
    // EMPTY: raised 0.02 off its seat faces (the boss tops), the base meets nothing else.
    intersection() { translate([0, 0, 0.02]) chassis_base(); enclosure_body(); }
} else if (check == "chassis_clear_body") {
    intersection() { chassis(); enclosure_body(); }
} else if (check == "chassis_seats_on_base") {
    // CONTACT: the plate underside lands on the base top.
    intersection() { chassis(0, -0.05); chassis_base(false); }
} else if (check == "chassis_clear_base") {
    // EMPTY: raised 0.02 off its seat (the base top), the detent nub sits in its hole and
    // the hook lips stay off the chamfered window edges.
    intersection() { chassis(0, 0.02); chassis_base(); }
} else if (check == "chassis_hooks_hold") {
    // CONTACT: lifted 0.4, the chamfered window edges meet the hook lips (hold-down).
    intersection() { chassis(0, 0.4); chassis_base(false); }
} else if (check == "chassis_lowered_clear") {
    // EMPTY: the chassis lowered 6 mm +Y of its seat clears the box, the base (hooks rise
    // through its windows; the detent nub is pressed flat) and the bolt heads.
    intersection() {
        chassis(ch_slide, 0.05);
        union() { enclosure_body(); chassis_base(false); ec_bolt_heads(); }
    }
} else if (check == "chassis_detent_blocks") {
    // CONTACT: pushed 1 mm back toward the lowered pose, the plate meets the detent nub.
    intersection() { chassis(1, 0.02); chassis_base(true); }
} else if (check == "chassis_clear_bolt_heads") {
    intersection() { chassis(0, 0.02); union() { ec_bolt_heads(); chassis_base(); } }
// ---- Everything inside vs box features, keep-outs and the closed lid ------------------------------------
} else if (check == "contents_clear_keepouts") {
    // EMPTY: glands (+ cable bend), camera port mouth, camera-tower nuts, vent.
    intersection() { all_contents(); ec_keepouts(); }
} else if (check == "contents_clear_body") {
    intersection() {
        union() { ec_tower(); all_payloads(); for (i = [0 : len(ch_strips) - 1]) ec_carrier(i, cr_seat_shift); }
        enclosure_body();
    }
} else if (check == "contents_clear_lid") {
    intersection() { all_contents(); enclosure_lid_closed(); }
} else if (check == "patch_clear") {
    // EMPTY: the Ethernet patch (plug housings past the jack faces, cable stubs and the slack
    // channel) clears every printed part, payload, board, the body and the closed lid.
    intersection() {
        ec_patch();
        union() { all_contents(); enclosure_body(); enclosure_lid_closed(); }
    }
} else if (check == "patch_plug_seats") {
    // CONTACT: the plug length inside the jack (idx 0 Alta, 1 Pi 5) meets that board's jack.
    intersection() { ec_patch_plug(idx, "insert"); ec_board(idx == 0 ? "alta" : "pi"); }
} else if (check == "pigtails_clear") {
    // EMPTY: both pigtail envelopes clear every part, payload, board, the patch, the body, the
    // closed lid and the (assumed) bulkhead envelopes.
    intersection() {
        ec_pigtails();
        union() { all_contents(); ec_patch(); enclosure_body(); enclosure_lid_closed(); ec_lid_bulkheads(); }
    }
} else if (check == "lid_bulkheads_clear") {
    // EMPTY: the bulkhead envelopes under the lid clear every part, payload, board and the patch.
    intersection() { ec_lid_bulkheads(); union() { all_contents(); ec_patch(); } }
} else if (check == "pigtail_reaches_hat") {
    // CONTACT: 1 mm past the descent envelope, pigtail idx (0 Alta, 1 Pi) meets its HAT's RF end.
    intersection() { ec_pigtail_end(idx); ec_board(idx == 0 ? "hra" : "hrp"); }
} else if (check == "port_room_clear") {
    // EMPTY: no printed part or payload box sits in the USB-C / RJ45 / USB plug + bend room.
    intersection() {
        union() { all_printed(); for (i = [0 : len(ch_strips) - 1]) ec_payload(i, cr_seat_shift); ec_signal_boards(); }
        ec_port_room();
    }
// ---- Compute tower ----------------------------------------------------------------------------------------
} else if (check == "tower_seats_on_chassis") {
    intersection() { ctw_to_body(0, -0.05) tower_rack(); chassis(); }
} else if (check == "tower_clear_chassis") {
    // EMPTY: raised 0.02 off its seat (pedestal and pad tops), the rack meets nothing else.
    intersection() { ctw_to_body(0, 0.02) tower_rack(); chassis(); }
} else if (check == "tower_lowered_clear") {
    // EMPTY: lowered 6 mm -X of its seat, the shelf's keyhole entries pass the mushroom
    // heads (the lock finger, pressed down by the toe, is left out).
    intersection() { ctw_to_body(-ctw_slide, 0.05) tower_rack(); chassis(0, 0, false); }
} else if (check == "tower_studs_hold") {
    // CONTACT: lifted 0.9, the keyhole floors meet the mushroom heads.
    intersection() { ctw_to_body(0, 0.9) tower_rack(); chassis(); }
} else if (check == "tower_lock_blocks") {
    // CONTACT: pushed 1 mm back toward its lowered pose, the toe meets the lock nub.
    intersection() { ctw_to_body(-1, 0.02) tower_rack(); chassis(); }
} else if (check == "boards_clear_tower") {
    intersection() { ec_boards(); union() { ctw_to_body() tower_rack(); ctw_to_body() tower_clamp(0); } }
} else if (check == "stack_on_bosses") {
    // CONTACT: the Alta pushed down by the nominal stack gap meets the rack bosses.
    intersection() { translate([0, 0, -(ctw_stack_gap_mm() + 0.05)]) ec_board("alta"); ctw_to_body() tower_rack(); }
} else if (check == "clamp_pads_bear") {
    // CONTACT: the clamp pushed down by the nominal pad gap meets the top HaLow HAT.
    intersection() { ctw_to_body(0, -(ctw_pad_gap_mm() + 0.05)) tower_clamp(0); ec_board("hrp"); }
} else if (check == "header_engages") {
    // CONTACT: the lower board's pins still overlap the upper board's socket with the socket
    // raised by hdr_engage_min, i.e. engagement >= hdr_engage_min at the modelled heights.
    p = hdr_pairs[idx];
    intersection() {
        ctw_to_body() ctw_at_board(ctw_level_w(p[0])) rb_hdr_pins(p[1]);
        ctw_to_body() ctw_at_board(ctw_level_w(p[2]) + hdr_engage_min) rb_hdr_socket(p[3]);
    }
// ---- Carriers on their strips (idx = strip index 0..6) ------------------------------------------------------
} else if (check == "carrier_hook_seats") {
    // CONTACT: gravity-seated (+0.03 overdrive) the hook, plate and feet meet strip and panel.
    intersection() { ec_carrier(idx, cr_seat_shift + [0, 0.03, -0.03]); chassis(); }
} else if (check == "carrier_latch_under_land") {
    // CONTACT: lifted from the nominal pose past the latch gap, the tooth meets the land.
    intersection() { ec_carrier(idx, [0, 0, cr_latch_gap + 0.05]); chassis(); }
} else if (check == "carrier_nominal_clear") {
    // EMPTY: at the nominal pose every mating face is 0.4 off the strip and the panel.
    intersection() { ec_carrier(idx); chassis(); }
} else if (check == "carrier_clear_others") {
    // EMPTY: carrier idx + its payload vs every other carrier, payload, the tower and boards.
    intersection() { carrier_with_payload(idx); everything_but(idx); }
// ---- Strap bay and harness clamp grip (idx 0 = minimum, 1 = maximum) ----------------------------------------
} else if (check == "bay_holds_box") {
    // CONTACT: at the ratchet stop chosen for the box, the spring leaf meets the box.
    b = bay_probe_box[idx];
    intersection() { strap_bay("bar", bay_stop(b[1])); bay_box_local(b); }
} else if (check == "bay_strap_clear") {
    // EMPTY: 0.05 short of that stop (pawls in their grooves, catch faces just apart) the
    // strap bar clears the frame everywhere.
    b = bay_probe_box[idx];
    intersection() { strap_bay("frame"); strap_bay("bar", bay_stop(b[1]) - 0.05); }
} else if (check == "bay_pawls_bear") {
    // CONTACT: pushed 0.05 past that stop by the box, both pawls' catch faces meet a rack tooth.
    b = bay_probe_box[idx];
    intersection() { strap_bay("frame"); strap_bay("bar", bay_stop(b[1]) + 0.05); }
} else if (check == "bay_box_clear_frame") {
    // EMPTY: the box stands clear of the posts, ribs' sides and hanger (it rests on the
    // ledge top and the rib tips only: coplanar contact).
    b = bay_probe_box[idx];
    intersection() { strap_bay("frame"); bay_box_local(b); }
} else if (check == "harness_holds_bundle") {
    d = hc_probe_d[idx];
    intersection() { harness_clamp("bar", hc_stop(d)); hc_bundle_local(d); }
} else if (check == "harness_strap_clear") {
    d = hc_probe_d[idx];
    intersection() { harness_clamp("frame"); harness_clamp("bar", hc_stop(d) - 0.05); }
} else if (check == "harness_pawls_bear") {
    d = hc_probe_d[idx];
    intersection() { harness_clamp("frame"); harness_clamp("bar", hc_stop(d) + 0.05); }
// ---- Signal boards on the chassis clips (idx = sig_boards index 0..2) -----------------------------------------
} else if (check == "signal_boards_clear") {
    intersection() { ec_signal_boards(); chassis(); }
} else if (check == "signal_board_captured") {
    // CONTACT: pushed toward the panel by the pocket clearance, the board meets its pocket.
    intersection() { ec_signal_board(idx, [0, sig_cl + 0.05, 0]); chassis(); }
} else {
    assert(false, str("unknown control check: ", check));
}
