// CSG probes for the control_77603 electronics packaging (chassis, base, compute tower,
// strap bays, harness clamp, signal-board clips, INA3221 cradle, SMA service clip and the RF
// leads) inside the unchanged legacy enclosure.
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
use <ina_holder.scad>
use <sma_clip.scad>
use <../antenna_mount.scad>
use <../enclosure_body.scad>
use <../enclosure_lid.scad>

check = "none";
idx = 0;

// Every printed part and payload inside the box (seated, assembled; clamp at nominal travel).
// Payloads include the SMA clip's jacks and the lid cables' mated plugs.
module all_printed() {
    chassis_base();
    chassis();
    ec_tower();
    ec_ina();
    ec_sc_clip();
    for (i = [0 : len(ch_strips) - 1]) ec_carrier(i, cr_seat_shift);
}
module all_payloads() {
    ec_boards();
    for (i = [0 : len(ch_strips) - 1]) ec_payload(i, cr_seat_shift);
    ec_signal_boards();
    ec_ina_board();
    ec_ina_adafruit();
    ec_sc_hardware();
}
module all_contents() { all_printed(); all_payloads(); }

// Carrier i (seated, with its payload), and everything else that could meet it.
module carrier_with_payload(i) { ec_carrier(i, cr_seat_shift); ec_payload(i, cr_seat_shift); }
module everything_but(i) {
    for (j = [0 : len(ch_strips) - 1]) if (j != i) carrier_with_payload(j);
    ec_tower();
    ec_boards();
    ec_signal_boards();
    ec_ina();
    ec_ina_board();
    ec_ina_adafruit();
    ec_sc_clip();
    ec_sc_hardware();
    ec_tower_leads();
}

// Fixed to the chassis, the body or the closed lid: what clamp-borne parts (INA cradle, SMA clip)
// must clear at every clamp travel.
module ec_fixed() {
    chassis_base();
    chassis();
    for (i = [0 : len(ch_strips) - 1]) carrier_with_payload(i);
    ec_signal_boards();
    ec_keepouts();
    ec_port_room();
    enclosure_body();
    enclosure_lid_closed();
    ec_lid_bulkheads();
}
// Lid lifted lid_lift_slack: the lid, its under-lid envelopes.
module ec_lid_lifted() {
    translate([0, 0, lid_lift_slack]) { enclosure_lid_closed(); antenna_lid_keepout(); }
}
// Clip-frame annulus r0..r1 round joint i's axis, dz thick from z0.
module sc_ring(i, r0, r1, z0, dz) {
    translate([sc_joints[i][0], sc_joints[i][1], z0]) difference() {
        cylinder(r = r1, h = dz, $fn = 72);
        translate([0, 0, -1]) cylinder(r = r0, h = dz + 2, $fn = 72);
    }
}
function sc_travels() = [-ctw_adjust, 0, ctw_adjust_up];

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
} else if (check == "lid_bulkheads_clear") {
    // EMPTY: the lid's under-lid envelopes (HaLow stations, socket bosses, GNSS jack column) clear
    // every part, payload, board, the patch and the tower-side leads.
    intersection() { ec_lid_bulkheads(); union() { all_contents(); ec_patch(); ec_tower_leads(); } }
// ---- RF service joints: lid cables, tower-side leads, the lid's GNSS jack --------------------------
} else if (check == "pigtails_clear") {
    // EMPTY: the lid-side cable corridors (HaLow pigtails + slack loops, the GNSS jumper's U) clear
    // every part, payload and board (incl. the SMA clip, its jacks and the mated plugs: each corridor
    // ends 0.05 off its plug barrel), the patch, the tower-side leads, the body, the closed lid and
    // its under-lid envelopes, and the INA3221 cradle, largest board and I2C lead at clamp travel
    // -12 / 0 / +3.
    intersection() {
        ec_pigtails();
        union() {
            all_contents(); ec_patch(); ec_tower_leads();
            enclosure_body(); enclosure_lid_closed(); ec_lid_bulkheads();
            for (t = sc_travels()) {
                ina_at_clamp(t) ina_cradle("assembly", ina_d_min);
                ec_ina_board(0, ina_board_probe[1], t);
                ec_ina_adafruit(0, t);
                ec_ina_i2c_run(t);
            }
        }
    }
} else if (check == "lid_cable_reaches") {
    // CONTACT: idx = 2 cable + end. End 0: a 1 mm stub from the corridor meets the lid end's
    // envelope (HaLow station / lid GNSS jack column); end 1: a stub meets the plug barrel's end.
    i = floor(idx / 2);
    intersection() { ec_lid_cable_end(i, idx % 2); if (idx % 2 == 0) ec_lid_bulkheads(); else ec_sc_plug(i); }
} else if (check == "tower_leads_clear") {
    // EMPTY: the tower-side lead corridors (and the GNSS kit cable's spare coil) clear every part,
    // payload and board (each corridor ends 0.05 short of its board's U.FL block or 0.1 under its
    // jack's heat-shrink), the patch, the lid cables, the body, the closed lid and its envelopes, and
    // the INA3221 I2C lead. Tower-borne, nominal stack (like the boards).
    intersection() {
        ec_tower_leads();
        union() {
            all_contents(); ec_patch(); ec_pigtails(); ec_ina_i2c_run();
            enclosure_body(); enclosure_lid_closed(); ec_lid_bulkheads();
        }
    }
} else if (check == "tower_lead_reaches") {
    // CONTACT: idx = 2 lead + end. End 0: a 1 mm stub meets the board's U.FL (HC01P card, LC29H,
    // WM6108 card); end 1: a stub meets the jack's heat-shrink in the clip.
    i = floor(idx / 2);
    intersection() {
        ec_tower_lead_end(i, idx % 2);
        if (idx % 2 == 0) ec_board(["hrp", "lc", "hra"][i]); else ec_sc_jack(i);
    }
} else if (check == "gnss_jack_clear") {
    // EMPTY: the lid GNSS jack's under-lid envelope (AntennaMount: hex, heat-shrink) clears every
    // part, payload and board, the patch, the tower-side leads, the tower + boards at its lowered
    // pose (6 mm -X), and the INA cradle + largest board and the SMA clip + jacks + plugs at clamp
    // travel -12 / +3.
    intersection() {
        ec_lid_gnss_keepout();
        union() {
            all_contents(); ec_patch(); ec_tower_leads();
            ec_tower(-ctw_slide, 0.05);
            translate([-ctw_slide, 0, 0.05]) ec_boards();
            for (t = [-ctw_adjust, ctw_adjust_up]) {
                ina_at_clamp(t) ina_cradle("assembly", ina_d_min);
                ec_ina_board(0, ina_board_probe[1], t);
                ec_sc_clip(t);
                ec_sc_hardware(t);
            }
        }
    }
// ---- SMA service clip on the tower clamp ------------------------------------------------------------
} else if (check == "sc_clip_seats") {
    // CONTACT: lowered 0.05, the foot and the south bridge meet the crossbar and finger-band tops.
    intersection() { ec_sc_clip(0, 0, -0.05); ctw_to_body() tower_clamp(0); }
} else if (check == "sc_clip_clear_clamp") {
    // EMPTY: raised 0.02 off its seat, the clip meets nothing else of the clamp: jaws 0.2 off the
    // band sides and the frame's +-Y faces, leaf 0.4 off the crossbar's +X face, lip 0.4 under it.
    intersection() { ec_sc_clip(0, 0, 0.02); ctw_to_body() tower_clamp(0); }
} else if (check == "sc_clip_retained") {
    // CONTACT: lifted 0.45, the snap lip meets the crossbar's underside (hold-down).
    intersection() { ec_sc_clip(0, 0, 0.45); ctw_to_body() tower_clamp(0); }
} else if (check == "sc_clip_keyed") {
    // CONTACT: raised 0.02 off its seat and moved (idx 0..3: +x / -x / +y / -y by 0.25; idx 4..5:
    // turned +1 / -1 degree about the GNSS jack's axis), the clip meets the clamp: the jaws (and the
    // leaf) bound its play, so the plugs' mating torque turns it at most 1 degree.
    m = [[0.25, 0, 0], [-0.25, 0, 0], [0, 0.25, 0], [0, -0.25, 0]];
    c = sc_joints[1];
    intersection() {
        if (idx < 4) translate(m[idx]) ec_sc_clip(0, 0, 0.02);
        else translate([c[0], c[1], 0]) rotate([0, 0, idx == 4 ? 1 : -1]) translate([-c[0], -c[1], 0])
            ec_sc_clip(0, 0, 0.02);
        ctw_to_body() tower_clamp(0);
    }
} else if (check == "sc_jack_fit") {
    // EMPTY (clip frame, idx = joint): the jack's D body, hex, heat-shrink, washer and nut (top part
    // raised 0.05, under part lowered 0.05 off their seat faces) fit the land, D hole and counterbore.
    intersection() {
        sma_clip();
        union() {
            translate([0, 0, 0.05]) sma_jack_top_ref(sc_joints[idx], sc_plate_z[1]);
            translate([0, 0, -0.05]) sma_jack_under_ref(sc_joints[idx], sc_plate_z[1]);
        }
    }
} else if (check == "sc_hex_bears") {
    // CONTACT: lifted 0.05, the jack's body hex bears on the land's underside in the counterbore.
    intersection() { sma_clip(); translate([0, 0, 0.05]) sma_jack_under_ref(sc_joints[idx], sc_plate_z[1]); }
} else if (check == "sc_dflat_keys") {
    // CONTACT: turned 10 degrees in the land, the jack's D body meets the hole's flat (torque key).
    intersection() { sma_clip(); sma_dbody(sc_joints[idx], sc_plate_z[1] - gn_land_t_mm() + 0.2, 2.6, 10); }
} else if (check == "sc_seat_contact") {
    // CONTACT: lowered 0.05, the lock washer lands on the plate top.
    intersection() { sma_clip(); translate([0, 0, -0.05]) sma_nut_washer_ref(sc_joints[idx], sc_plate_z[1]); }
} else if (check == "sc_seat_backed") {
    // EMPTY: the whole washer annulus outside the hole (0.1 slab under it) and the hex's top face
    // outside the hole (0.1 slab over it) are backed by plate material.
    difference() {
        union() {
            sc_ring(idx, 3.3, gn_washer_d_mm() / 2, sc_plate_z[1] - 0.1, 0.099);
            sc_ring(idx, 3.3, 3.9, sc_plate_z[1] - gn_land_t_mm() + 0.001, 0.1);
        }
        sma_clip();
    }
} else if (check == "sc_clip_clear") {
    // EMPTY: at clamp travel -12 / 0 / +3 (idx 0..2) the clip, its jacks and the mated plugs clear
    // everything fixed to the chassis, the body or the closed lid: base, chassis, every carrier and
    // payload, signal boards, keep-outs, port rooms, body, closed lid, under-lid envelopes.
    t = sc_travels()[idx];
    intersection() { union() { ec_sc_clip(t); ec_sc_hardware(t); } ec_fixed(); }
} else if (check == "sc_clip_clear_tower") {
    // EMPTY: the clip, its jacks and the mated plugs clear the rack, every board, the patch, the
    // INA3221 cradle (bar at both extremes), its largest board, the Adafruit + plugged housing and
    // its I2C lead, the tower-side leads and the lid cables (clamp-borne parts move together, so
    // nominal travel covers them).
    intersection() {
        union() { ec_sc_clip(); ec_sc_hardware(); }
        union() {
            ctw_to_body() tower_rack(); ec_boards(); ec_patch();
            ina_at_clamp() ina_cradle("assembly", ina_d_min);
            ina_at_clamp() ina_cradle("bar", ina_d_print);
            ec_ina_board(0, ina_board_probe[1]);
            ec_ina_adafruit();
            ec_ina_i2c_run();
            ec_tower_leads();
            ec_pigtails();
        }
    }
} else if (check == "sc_sweep_clear") {
    // EMPTY: the 8 mm spanner's room at joint j = idx % 3 (slab sector and head ring on the plug nut,
    // the descent column beyond the head ring up to the hand height) is free. idx 0..2 lid off: of
    // every part, payload, board, the clip and all three jacks, the INA3221 I2C lead, the patch and
    // the body; idx 3..5 lid lifted 100 mm: also of the other two plugs, the lifted lid and its
    // envelopes and the other cables rising to it.
    j = idx % 3;
    intersection() {
        ec_sc_sweep(j);
        union() {
            for (i = [0 : len(ch_strips) - 1]) carrier_with_payload(i);
            chassis(); ec_tower(); ec_boards(); ec_signal_boards(); ec_patch(); enclosure_body();
            ec_ina(); ec_ina_board(); ec_ina_adafruit(); ec_ina_i2c_run();
            ec_sc_clip(); ec_sc_hardware(0, 0, 0, false);
            if (idx >= 3) {
                for (i = [0 : 2]) if (i != j) ec_sc_plug(i);
                ec_lid_lifted();
                ec_lid_cables_lifted(j);
            }
        }
    }
} else if (check == "sc_lift_clear") {
    // EMPTY: lid off (plugs gone with it), the clip and its jacks lift out with the tower: the column
    // over their footprint from the jack faces to the rim is free. idx 0: seated tower; idx 1: tower
    // slid 6 mm -X for its removal (bays B4 and B6 already unhooked), the clip checked at that pose.
    dx = idx == 0 ? 0 : -ctw_slide;
    z0 = ec_zc() + sc_face_z;
    intersection() {
        union() {
            translate([0, 0, z0]) linear_extrude(height = box_top - z0) projection()
                union() { ec_sc_clip(0, dx); ec_sc_hardware(0, dx, 0, false); }
            if (idx == 1) { ec_sc_clip(0, dx, 0.05); ec_sc_hardware(0, dx, 0.05, false); }
        }
        union() {
            chassis_base();
            chassis();
            for (i = [0 : len(ch_strips) - 1]) if (idx == 0 || (i != 3 && i != 5)) carrier_with_payload(i);
            ec_signal_boards();
            if (idx == 0) ec_patch();
            ec_keepouts();
            enclosure_body();
        }
    }
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
// ---- INA3221 cradle on the tower clamp (idx = ina_board_probe: 0 smallest, 1 largest board) ------
} else if (check == "ina_cradle_seats") {
    // CONTACT: lowered 0.05, the rails and root blocks meet the clamp's long bars.
    intersection() { ina_at_clamp(0, 0, -0.05) ina_cradle("frame"); ctw_to_body() tower_clamp(0); }
} else if (check == "ina_cradle_clear_clamp") {
    // EMPTY: raised 0.02 off its seat (the long bars' tops), frame and open bar meet nothing
    // else of the clamp: saddle tabs 0.4 off the crossbar, leaves 0.4 off the bars' outer
    // faces, snap lips 0.4 under the bars.
    intersection() { ina_at_clamp(0, 0, 0.02) ina_cradle("assembly", ina_d_print); ctw_to_body() tower_clamp(0); }
} else if (check == "ina_hooks_hold") {
    // CONTACT: lifted 0.45, both snap lips meet the long bars' undersides (hold-down).
    intersection() { ina_at_clamp(0, 0, 0.45) ina_cradle("frame"); ctw_to_body() tower_clamp(0); }
} else if (check == "ina_saddle_locates") {
    // CONTACT: slid 0.45 along x (idx 0: +x, the inner tab; idx 1: -x, the outer tab) the
    // saddle meets the front crossbar.
    intersection() {
        ina_at_clamp(0, idx == 0 ? 0.45 : -0.45, 0.02) ina_cradle("frame");
        ctw_to_body() tower_clamp(0);
    }
} else if (check == "ina_board_clear_frame") {
    // EMPTY: board idx, resting 0.05 off the jaw face and ledge, clears the frame (jaw lip
    // 0.3 over the PCB, parts ina_edge_keep[0] inside the jaw edge, ends clear of the rails).
    b = ina_board_probe[idx];
    intersection() { ina_cradle("frame"); translate(ina_board_origin(b)) rb_ina3221(b[0], b[1]); }
} else if (check == "ina_board_clear_bar") {
    // EMPTY: with the shoe face 0.05 off the board's far edge, the bar (shoe, fingers) clears it.
    b = ina_board_probe[idx];
    intersection() { ina_cradle("bar", ina_d_touch(b)); translate(ina_board_origin(b)) rb_ina3221(b[0], b[1]); }
} else if (check == "ina_jaw_holds") {
    // CONTACT: lifted by the lip clearance + 0.05, the board's near edge meets the jaw lip.
    b = ina_board_probe[idx];
    intersection() {
        ina_cradle("frame");
        translate(ina_board_origin(b) + [0, 0, ina_cl + 0.05]) rb_ina3221(b[0], b[1]);
    }
} else if (check == "ina_shoe_holds") {
    // CONTACT: with the shoe at the far edge, the lifted board meets the shoe lip.
    b = ina_board_probe[idx];
    intersection() {
        ina_cradle("bar", ina_d_touch(b));
        translate(ina_board_origin(b) + [0, 0, ina_cl + 0.05]) rb_ina3221(b[0], b[1]);
    }
} else if (check == "ina_bar_preloads") {
    // CONTACT: at the ratchet stop chosen for board idx the undeflected shoe overlaps the
    // board's far edge (the leaf's 1.0..2.5 mm preload).
    b = ina_board_probe[idx];
    intersection() { ina_cradle("bar", ina_stop(b[0])); translate(ina_board_origin(b)) rb_ina3221(b[0], b[1]); }
} else if (check == "ina_bar_clear") {
    // EMPTY: 0.05 short of that stop (pawls in their grooves) the bar clears the frame.
    b = ina_board_probe[idx];
    intersection() { ina_cradle("frame"); ina_cradle("bar", ina_stop(b[0]) - 0.05); }
} else if (check == "ina_pawls_bear") {
    // CONTACT: pushed 0.05 past that stop by the board, the pawls' catch faces meet a rack tooth.
    b = ina_board_probe[idx];
    intersection() { ina_cradle("frame"); ina_cradle("bar", ina_stop(b[0]) + 0.05); }
} else if (check == "ina_clear_others") {
    // EMPTY: the cradle (bar at its closest stop and fully open), the largest board and the
    // Adafruit with its plugged Dupont housing clear every other part, payload and board, the
    // rack, the patch, the SMA clip with its jacks and plugs, the tower-side leads, keep-outs, port
    // rooms, body.
    intersection() {
        union() {
            ina_at_clamp() ina_cradle("assembly", ina_d_min);
            ina_at_clamp() ina_cradle("bar", ina_d_print);
            ec_ina_board(0, ina_board_probe[1]);
            ec_ina_adafruit();
        }
        union() {
            chassis_base();
            chassis();
            ctw_to_body() tower_rack();
            for (i = [0 : len(ch_strips) - 1]) carrier_with_payload(i);
            ec_boards();
            ec_signal_boards();
            ec_patch();
            ec_sc_clip();
            ec_sc_hardware();
            ec_tower_leads();
            ec_keepouts();
            ec_port_room();
            enclosure_body();
        }
    }
} else if (check == "ina_travel_clear") {
    // EMPTY: at clamp travel -12 / 0 / +3 (idx 0..2: shorter or taller stacks) the cradle, the
    // largest board and the Adafruit's plugged Dupont housing clear everything fixed to the
    // chassis or the lid: base, chassis, every carrier and payload, signal boards, keep-outs,
    // body, closed lid, bulkheads, lid cables; and the SMA clip + jacks + plugs at the same travel.
    t = [-ctw_adjust, 0, ctw_adjust_up][idx];
    intersection() {
        union() {
            ina_at_clamp(t) ina_cradle("assembly", ina_d_min);
            ina_at_clamp(t) ina_cradle("bar", ina_d_print);
            ec_ina_board(0, ina_board_probe[1], t);
            ec_ina_adafruit(0, t);
        }
        union() {
            chassis_base();
            chassis();
            for (i = [0 : len(ch_strips) - 1]) carrier_with_payload(i);
            ec_signal_boards();
            ec_keepouts();
            enclosure_body();
            enclosure_lid_closed();
            ec_lid_bulkheads();
            ec_pigtails();
            ec_sc_clip(t);
            ec_sc_hardware(t);
        }
    }
} else if (check == "ina_access_clear") {
    // EMPTY: lid off (lid cables, plugs and bulkheads leave with it), the column over the largest
    // board's parts field, from the PCB top to the box rim, is free (the SMA clip and its jacks
    // included): drivers and plugs reach the terminals and headers from above. The bar is drawn
    // with its shoe at the board edge.
    b = ina_board_probe[1];
    o = ina_board_origin(b);
    z0 = o[2] + ina_t;
    intersection() {
        ina_at_clamp() translate([o[0] + ina_edge_keep[0], o[1] + ina_end_keep, z0])
            cube([b[0] - ina_edge_keep[0] - ina_edge_keep[1], b[1] - 2 * ina_end_keep,
                  box_top - (ctw_shelf_z() + ctw_clamp_top_w(0) + z0)]);
        union() {
            chassis_base();
            chassis();
            ec_tower();
            ina_at_clamp() ina_cradle("frame");
            ina_at_clamp() ina_cradle("bar", ina_d_touch(b));
            for (i = [0 : len(ch_strips) - 1]) carrier_with_payload(i);
            ec_boards();
            ec_signal_boards();
            ec_patch();
            ec_sc_clip();
            ec_sc_hardware(0, 0, 0, false);
            enclosure_body();
        }
    }
} else if (check == "ina_lift_clear") {
    // EMPTY: lid off, the cradle lifts straight off: the column over its footprint (bar at its
    // extremes, largest board), from its top plane to the rim, is free (the SMA clip and its jacks
    // stay on the clamp beside it). idx 0: seated tower; idx 1: tower slid 6 mm -X for its own
    // removal (bays B4 and B6 already unhooked), the cradle and the clip riding on it at that pose.
    dx = idx == 0 ? 0 : -ctw_slide;
    z0 = ctw_shelf_z() + ctw_clamp_top_w(0) + ina_z_top;
    intersection() {
        union() {
            translate([0, 0, z0]) linear_extrude(height = box_top - z0) projection()
                ina_at_clamp(0, dx) {
                    ina_cradle("assembly", ina_d_min);
                    ina_cradle("bar", ina_d_print);
                    translate(ina_board_origin(ina_board_probe[1])) rb_ina3221(ina_board_probe[1][0], ina_board_probe[1][1]);
                }
            if (idx == 1) {
                ina_at_clamp(0, dx, 0.05) ina_cradle("assembly", ina_d_min);
                ina_at_clamp(0, dx, 0.05) ina_cradle("bar", ina_d_print);
            }
        }
        union() {
            chassis_base();
            chassis();
            for (i = [0 : len(ch_strips) - 1]) if (idx == 0 || (i != 3 && i != 5)) carrier_with_payload(i);
            ec_signal_boards();
            if (idx == 0) ec_patch();
            ec_sc_clip(0, dx);
            ec_sc_hardware(0, dx, 0, false);
            ec_keepouts();
            enclosure_body();
        }
    }
// ---- INA3221 cradle with the Adafruit #6062 (header soldered, Dupont housing plugged) -------------
} else if (check == "ina_adafruit_clear") {
    // EMPTY: the Adafruit in its required orientation, every part inflated by ina_part_cl +
    // 0.05 in x and y (the board sits 0.05 off its seat faces here), clears idx 0: the frame
    // (jaw lip and ledge on the terminal edge, rails); idx 1: the bar with its shoe 0.05 off the
    // header edge (shoe lip and ledge, leaf, fingers). So each gripping feature keeps at least
    // ina_part_cl from the header body, the plugged Dupont housing, the terminal blocks, the
    // STEMMA QT sockets and the pads under them.
    g = ina_part_cl + 0.05;
    intersection() {
        if (idx == 0) ina_cradle("frame");
        else ina_cradle("bar", ina_d_touch(ina_board_adafruit));
        translate(ina_board_origin(ina_board_adafruit)) rb_ina3221_adafruit(g);
    }
} else if (check == "ina_adafruit_held") {
    // CONTACT: lifted by the lip clearance + 0.05, the Adafruit's PCB meets idx 0: the jaw lip
    // (terminal edge); idx 1: the shoe lip (header edge).
    intersection() {
        if (idx == 0) ina_cradle("frame");
        else ina_cradle("bar", ina_d_touch(ina_board_adafruit));
        translate(ina_board_origin(ina_board_adafruit) + [0, 0, ina_cl + 0.05]) rb_ina3221_adafruit();
    }
} else {
    assert(false, str("unknown control check: ", check));
}
