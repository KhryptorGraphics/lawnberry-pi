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
use <ina_holder.scad>
use <sma_clip.scad>
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
        else if (level == "hra") rb_halow_wm1302();
        else if (level == "hrp") rb_halow_hc01p();
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

// ---- RF service joints: SMA clip, tower-side leads, lid-side cables -----------------------------
// AntennaMount's authoritative under-lid envelope (HaLow bulkhead stations, lid post-socket bosses
// and the GNSS SMA jack).
module ec_lid_bulkheads() { antenna_lid_keepout(); }
module ec_lid_halow_keepout() { halow_lid_keepout(); }
module ec_lid_gnss_keepout() { gnss_lid_keepout(); }
// Body z of the clamp frame's top at travel t (the clip frame's z = 0).
function ec_zc(t = 0) = ctw_shelf_z() + ctw_clamp_top_w(t);
// The SMA clip (clamp-borne) at clamp travel t, with the tower moved by (dx, dz).
module ec_sc_clip(t = 0, dx = 0, dz = 0) { color([0.95, 0.45, 0.10]) sc_at_clamp(t, dx, dz) sma_clip(); }
// Joint i's mated right-angle SMA plug (clip frame): coupling nut from its open end (past the jack
// face by the MIL-STD-348A overhang) over the thread, the body above it, the crimp barrel along
// sc_dir(i) at the cable axis. The body square is aligned with the barrel.
module ec_sc_plug_local(i) {
    j = sc_joints[i];
    n0 = sc_nut0_z();
    translate([j[0], j[1], 0]) rotate([0, 0, sc_barrel_dir[i]]) {
        translate([0, 0, n0]) cylinder(d = ra_w / cos(30), h = ra_nut_len, $fn = 6);
        translate([-ra_w / 2, -ra_w / 2, n0 + ra_nut_len - 0.01]) cube([ra_w, ra_w, ra_len - ra_nut_len + 0.01]);
        translate([0, 0, sc_barrel_z()]) rotate([0, 90, 0]) cylinder(d = ra_barrel_d, h = ra_barrel, $fn = 24);
    }
}
// Joint i's tower-side jack (clip frame): D body, hex, heat-shrink, lock washer and nut.
module ec_sc_jack_local(i) { sma_jack_ref(sc_joints[i], sc_plate_z[1]); }
// The clip's three jacks and (plugs = true) the lid cables' mated plugs, at clamp travel t.
module ec_sc_hardware(t = 0, dx = 0, dz = 0, plugs = true) {
    sc_at_clamp(t, dx, dz) {
        color([0.75, 0.76, 0.78]) for (i = [0 : 2]) ec_sc_jack_local(i);
        if (plugs) color([0.85, 0.75, 0.2]) for (i = [0 : 2]) ec_sc_plug_local(i);
    }
}
module ec_sc_jack(i, t = 0) { sc_at_clamp(t) ec_sc_jack_local(i); }
module ec_sc_plug(i, t = 0) { sc_at_clamp(t) ec_sc_plug_local(i); }
// Tower-side U.FL -> SMA-F leads (tower-borne, drawn at the nominal stack like the boards) and
// their 1 mm reach stubs: e = 0 onto the board's U.FL, e = 1 into the jack's heat-shrink.
module ec_tower_leads() { color([0.10, 0.10, 0.10, 0.6]) for (r = tl_route) ec_box(r[1]); }
module ec_tower_lead_end(i, e) { ec_box(tl_stub[i][e]); }
// Lid-side cable corridors (lid-borne, lid closed, clip at nominal travel): the HaLow pigtails
// with their slack loops and the GNSS jumper's U; reach stubs e = 0 into the lid end's envelope,
// e = 1 onto the plug barrel's end.
// Lid-side cable corridors, less the HaLow station envelopes their pigtails start in (that room is
// the bulkheads'; the corridors' first boxes and exit boxes run into it so the cable path is continuous).
module ec_pigtails() {
    color([0.10, 0.10, 0.10, 0.6]) difference() {
        for (r = ll_route) ec_box(r[1]);
        ec_lid_halow_keepout();
    }
}
module ec_lid_cable_end(i, e) { ec_box(ll_stub[i][e]); }
// Lid lifted lid_lift_slack, the lid cables left as dressed: each in its corridors and slack zone
// (the slack feeds the lift; the bulkheads' room under the lid is vacated, so nothing is cut), plus
// the space the lift adds (ll_lift_ext: the HaLow drops and the GNSS jack leg up to the lifted lid,
// the GNSS U risen). `except` leaves one cable out.
module ec_lid_cables_lifted(except = -1) {
    color([0.10, 0.10, 0.10, 0.6]) for (r = concat(ll_route, ll_lift_ext)) if (r[0] != except) ec_box(r[1]);
}
// The INA3221's I2C lead envelope (agreed 2026-11-02), riding the clamp at travel t.
module ec_ina_i2c_run(t = 0) { color([0.2, 0.6, 0.9, 0.4]) ina_at_clamp(t) for (b = ina_i2c_run) ec_box(b); }
// 8 mm spanner room for joint j at clamp travel t: the sector and the head's ring round the plug
// nut in the slab, then the column over the sector beyond the head ring up to sc_hand_top above
// the rim (the spanner comes down there and slides onto the nut).
module ec_sector2(r, a, h) {
    polygon(concat([[0, 0]], [for (k = [0 : 8]) r * [cos(a - h + 2 * h * k / 8), sin(a - h + 2 * h * k / 8)]]));
}
module ec_sc_sweep(j, t = 0) {
    c = sc_joints[j];
    a = sc_sweep_dir[j];
    z0 = ec_zc(t) + sc_sw_slab[0];
    z1 = ec_zc(t) + sc_sw_slab[1];
    translate([c[0], c[1], 0]) {
        translate([0, 0, z0]) linear_extrude(height = z1 - z0) difference() {
            union() { circle(r = sc_head_r, $fn = 48); ec_sector2(sc_sw_r, a, sc_sw_half); }
            circle(r = ra_w / cos(30) / 2 + 0.4, $fn = 24);
        }
        translate([0, 0, z1 - 0.01]) linear_extrude(height = box_top + sc_hand_top - z1) difference() {
            ec_sector2(sc_sw_r, a, sc_sw_half);
            circle(r = sc_head_r, $fn = 48);
        }
    }
}
// The lid's purchased radio hardware (N bulkheads, GNSS jack + the outside antenna cable's plug),
// drawn at the closed-lid position.
module ec_lid_radio_refs() {
    color([0.75, 0.76, 0.78]) { for (s = [-1, 1]) antenna_bulkhead_ref(s); gnss_jack_ref(); }
    color([0.10, 0.10, 0.10]) gnss_plug_ref();
}

// ---- INA3221 cradle on the tower clamp (ina_holder.scad) --------------------------------------
// Scene board: the largest common outline (CJMCU-3221, 31.7 across x 38.3 along). The bar sits
// at its ratchet stop for that board; its leaf is drawn undeflected, so the shoe overlaps the
// board edge by the preload (as the strap bays' leaves overlap their boxes).
module ec_ina(dz = 0, b = ina_board_cjmcu, travel = 0, dx = 0) {
    color([0.82, 0.82, 0.84]) ina_at_clamp(travel, dx, dz) ina_cradle("frame");
    color([0.62, 0.64, 0.68]) ina_at_clamp(travel, dx, dz) ina_cradle("bar", ina_stop(b[0]));
}
module ec_ina_board(dz = 0, b = ina_board_cjmcu, travel = 0, dx = 0) {
    ina_at_clamp(travel, dx, dz) translate(ina_board_origin(b)) rb_ina3221(b[0], b[1]);
}
// The Adafruit #6062 (the board whose header carries the I2C) in its required orientation, header
// soldered and a Dupont housing plugged (ESTIMATE envelope; the housing top is the highest point).
module ec_ina_adafruit(dz = 0, travel = 0, dx = 0, grow = 0) {
    ina_at_clamp(travel, dx, dz) translate(ina_board_origin(ina_board_adafruit)) rb_ina3221_adafruit(grow);
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
    ec_ina(2.6 * e);
    ec_ina_board(2.6 * e);
    ec_sc_clip(0, 0, 2.6 * e);
    ec_sc_hardware(0, 0, 2.6 * e, e == 0);
    if (e == 0) ec_tower_leads();
    translate([0, 0, e]) {
        ec_carriers(0.4 * e);
        ec_signal_boards(0.4 * e);
    }
}

// The lid-side cables are drawn closed (corridors) or, with the lid lifted, as dressed (the
// corridors plus the HaLow drops and GNSS jack leg up to the lifted lid, the GNSS U risen).
module electronics_packaging(exploded_mm = 0, lid_lift_mm = 0, body = true) {
    if (body) color([0.75, 0.75, 0.78, 0.30]) enclosure_body();
    electronics_contents(exploded_mm);
    if (exploded_mm == 0) {
        translate([0, 0, lid_lift_mm]) ec_lid_radio_refs();
        if (lid_lift_mm > 0) ec_lid_cables_lifted(); else ec_pigtails();
    }
    if (lid_lift_mm > 0) color([0.75, 0.75, 0.78, 0.30]) translate([0, 0, lid_lift_mm]) enclosure_lid_closed();
}

// electronics_packaging | electronics_contents | electronics_exploded | electronics_lid_lifted
assembly = "electronics_packaging";

if (assembly == "electronics_packaging") electronics_packaging(0, 0);
else if (assembly == "electronics_contents") electronics_contents(0);
else if (assembly == "electronics_exploded") electronics_packaging(60, 0, false);
else if (assembly == "electronics_lid_lifted") electronics_packaging(0, lid_lift_slack);
else assert(false, str("unknown control assembly: ", assembly));
