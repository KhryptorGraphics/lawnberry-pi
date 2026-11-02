// Electronics packaging scenes: the unchanged legacy enclosure body with the removable
// chassis, the compute tower and every interface carrier in place, plus reference solids of
// the boards and payloads. Scenes only; nothing here is printed.
//
// Payload boxes in the strap bays are DESIGN CAPACITY envelopes (the HiLetgo relays use
// the board-dims §9 ESTIMATE 42 x 26 x 75); they are not purchased-part outlines.
// The modules below (ec_*) are shared with control_checks.scad.
include <cleat_carrier.scad>
use <board_carriers.scad>
use <compute_tower.scad>
use <chassis.scad>
use <board_refs.scad>
use <../enclosure_body.scad>
use <../enclosure_lid.scad>
use <../antenna_mount.scad>

// ---- Carriers on their strips -------------------------------------------------------------
// Local offset of a carrier from its nominal pose: "seated" = gravity seat on the bevel.
function ec_seat() = cr_seat_shift;
function ec_is_bay(i) = i < len(lay_bays);
// Payload [w, d, h] of carrier i (bays: relay ESTIMATE or the slot's capacity envelope).
function ec_payload(i) =
    !ec_is_bay(i) ? [hc_bundle_d[1], hc_bundle_d[1], 45] :
    i < 2 ? relay_box :
    [bay_box_max[0], bay_box_max[1], lay_bays[i][3]];
// Bar position (back face distance from the rib plane) that clamps payload depth d.
function ec_bar_d(i, d) =
    ec_is_bay(i) ? bay_d_k(bay_k_for_box(d))
                 : min([for (k = [0 : 15]) let(dk = hc_bundle_d[1] - rt_pitch * k) if (dk >= d) dk]);

module ec_carrier_local(i, d_payload = undef) {
    pl = ec_payload(i);
    d = is_undef(d_payload) ? pl[1] : d_payload;
    if (ec_is_bay(i)) strap_bay("assembly", ec_bar_d(i, d));
    else harness_clamp("assembly", ec_bar_d(i, d));
}
module ec_payload_local(i, d_payload = undef) {
    pl = ec_payload(i);
    d = is_undef(d_payload) ? pl[1] : d_payload;
    W = ec_is_bay(i) ? bay_w : hc_w;
    // Payloads rest 0.05 off the ledge and the rib tips (no coplanar faces in the probes).
    if (ec_is_bay(i))
        translate([W / 2 - pl[0] / 2, bay_ref_y - d - 0.05, -bay_drop + bay_ledge_t + 0.05])
            rb_bay_box([pl[0], d, pl[2]]);
    else  // harness bundle, vertical between the posts
        translate([W / 2, bay_ref_y - d / 2 - 0.05, -bay_drop + 0.5]) color([0.15, 0.15, 0.15])
            cylinder(d = d, h = pl[2]);
}
// Carrier i at its strip, offset `off` (local) from the nominal pose; explode pulls it off
// its panel (local -y).
module ec_carrier(i, off = [0, 0, 0], explode = 0) {
    at_strip(i) translate(off + [0, -explode, 0]) ec_carrier_local(i);
}
module ec_payload(i, off = [0, 0, 0], explode = 0) {
    at_strip(i) translate(off + [0, -explode, 0]) ec_payload_local(i);
}
module ec_carriers(explode = 0, except = -1) {
    for (i = [0 : len(ch_strips) - 1]) if (i != except) {
        color(ec_is_bay(i) ? [0.20, 0.45, 0.70] : [0.30, 0.30, 0.32]) ec_carrier(i, ec_seat(), explode);
        ec_payload(i, ec_seat(), explode);
    }
}

// ---- Compute tower ------------------------------------------------------------------------------
module ec_tower(dx = 0, dz = 0) {
    color([0.85, 0.55, 0.15]) ctw_to_body(dx, dz) tower_rack();
    color([0.95, 0.70, 0.30]) ctw_to_body(dx, dz) tower_clamp(0);
}
module ec_board(level) {
    ctw_to_body() ctw_at_board(ctw_level_w(level)) {
        if (level == "alta") rb_alta();
        else if (level == "hra" || level == "hrp") rb_halow();
        else if (level == "pi") rb_pi5();
        else if (level == "ai") rb_aihat();
        else if (level == "lc") rb_lc29h();
    }
}
ec_levels = ["alta", "hra", "pi", "ai", "lc", "hrp"];
module ec_boards(dz = 0) { translate([0, 0, dz]) for (l = ec_levels) ec_board(l); }

// ---- Signal boards on the chassis clips ---------------------------------------------------------
module ec_signal_board(i, off = [0, 0, 0]) {
    b = sig_boards[i];
    at_face(b[1], b[2], 0) translate([0, -cr_plate_front, 0] + off)
        rb_plate_board(b[3], b[4], b[5][0], b[5][1], b[6], pca_plug_h, sig_board_back());
}
module ec_signal_boards(explode = 0) {
    for (i = [0 : len(sig_boards) - 1]) ec_signal_board(i, [0, -explode, 0]);
}

// ---- Keep-outs ------------------------------------------------------------------------------------
module ec_box(b) { translate(b[0]) cube(b[1] - b[0]); }
module ec_bolt_heads() {
    for (p = tray_fixings) translate([p[0], p[1], tray_bottom_z + cb_t - 0.01])
        cylinder(d = bolt_head_d, h = bolt_head_h + 0.01);
}
module ec_keepouts() {
    for (p = gland_centres) translate([p[0], p[1], box_floor - 0.01])
        cylinder(d = ch_gland_hole_d, h = 26);          // = enclosure_common gland_keepout()
    ec_box(ko_cam_port);
    for (n = ko_nuts) ec_box(n);
    ec_box(ko_vent);
}
// Connector and cable-bend room the printed parts must leave free.
module ec_port_room() { ec_box(ko_usbc); ec_box(ko_plugs); }

// ---- Pi 5 <-> Alta Ethernet patch (side-angle plugs, slack coiled in a channel) ------------------
// Jack i: 0 = Alta, 1 = Pi 5. [jack face x, plug centre y, plug centre z].
function ec_jack(i) =
    let(level = i == 0 ? "alta" : "pi",
        u = (i == 0 ? alta_pcb[0] : pi5_pcb[0]) + pi5_port_overhang)
    [ctw_body(u, 0)[0], ctw_ty + pi5_rj45_v,
     ctw_shelf_z() + ctw_level_w(level) + board_t + patch_jack_h / 2];
// part "insert": the plug length inside the jack; "out": housing past the jack face plus the
// cable stub to the slack channel.
module ec_patch_plug(i, part = "out") {
    j = ec_jack(i);
    y0 = j[1] - patch_plug_w / 2;
    color([0.85, 0.85, 0.30]) if (part == "insert")
        translate([j[0] - patch_insert, y0, j[2] - patch_plug_h / 2])
            cube([patch_insert, patch_plug_w, patch_plug_h]);
    else {
        translate([j[0] + 0.05, y0, j[2] - patch_plug_h / 2])
            cube([patch_body_out - 0.05, patch_plug_w, patch_plug_h]);
        translate([j[0] + patch_exit_x[0], patch_channel[0][1], j[2] - patch_cable_t / 2])
            cube([patch_exit_x[1] - patch_exit_x[0], y0 - patch_channel[0][1] + 0.01, patch_cable_t]);
    }
}
module ec_patch() {
    ec_patch_plug(0);
    ec_patch_plug(1);
    color([0.85, 0.85, 0.30, 0.5]) ec_box(patch_channel);
}

// ---- HaLow pigtails from the lid bulkheads ------------------------------------------------------
// AntennaMount's authoritative under-lid envelope (bulkhead stations + lid post-socket bosses).
module ec_lid_bulkheads() { antenna_lid_keepout(); }
module ec_pigtails() { color([0.10, 0.10, 0.10, 0.6]) for (r = pig_route) ec_box(r); }
// A 1 mm stub from the descent envelope toward pigtail i's HAT RF end.
module ec_pigtail_end(i) {
    e = pig_route[len(pig_route) - 1];
    translate([e[0][0] - 1, e[0][1], pig_end_z[i][0]]) cube([1.01, e[1][1] - e[0][1], pig_end_z[i][1] - pig_end_z[i][0]]);
}

// ---- Whole contents -----------------------------------------------------------------------------
// exploded_mm > 0: the chassis rises by e, its carriers and boards ride with it pulled
// 0.4 e off their panels, and the tower (with its boards) rises 2.6 e, clear of all of it.
module electronics_contents(exploded_mm = 0) {
    e = exploded_mm;
    color([0.40, 0.40, 0.42]) chassis_base();
    color([0.16, 0.55, 0.60]) chassis(0, e);
    ec_tower(0, 2.6 * e);
    ec_boards(2.6 * e);
    translate([0, 0, 2.6 * e]) ec_patch();
    if (e == 0) ec_pigtails();
    translate([0, 0, e]) {
        ec_carriers(0.4 * e);
        ec_signal_boards(0.4 * e);
    }
}

module electronics_packaging(exploded_mm = 0, lid_lift_mm = 0, body = true) {
    if (body) color([0.75, 0.75, 0.78, 0.30]) enclosure_body();
    electronics_contents(exploded_mm);
    if (lid_lift_mm > 0) color([0.75, 0.75, 0.78, 0.30]) translate([0, 0, lid_lift_mm]) enclosure_lid_closed();
}

// electronics_packaging | electronics_contents | electronics_exploded | electronics_lid_lifted
assembly = "electronics_packaging";

if (assembly == "electronics_packaging") electronics_packaging(0, 0);
else if (assembly == "electronics_contents") electronics_contents(0);
else if (assembly == "electronics_exploded") electronics_packaging(60, 0, false);
else if (assembly == "electronics_lid_lifted") electronics_packaging(0, 120);
else assert(false, str("unknown control assembly: ", assembly));
