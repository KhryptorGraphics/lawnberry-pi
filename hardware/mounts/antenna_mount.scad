// HaLow antenna ports and mounts on the enclosure LID's back (-Y) edge: two purchased Rokland
// 10 dBi "Backcountry" 915 MHz omnis, one per HaLow radio (-X station: Alta, +X station: Pi 5),
// and the LC29H's GNSS port (an SMA bulkhead jack) near the compute tower.
// The body is unchanged; enclosure_lid.scad carries the ports and post sockets.
// PROTOTYPE GEOMETRY: no ingress (IP), wind, vibration, UV or load rating is claimed.
//
// Antenna: N-male base, 45 in = 1143 mm tall, fibreglass radome, aluminium base tube, two
// U-bolts + bracket supplied:
// https://store.rokland.com/products/10-dbi-backcountry-n-male-omni-outdoor-helium-915-mhz-antenna-48-for-rak-miner-2-nebra-indoor-bobcat-hotspots
// Base-tube diameter, length and mass are NOT published: the grips print to the MEASURED tube
// diameter (ant_tube_d_mm, design range 25-40).
//
// PORT: the lid plate (lid_thickness = 5) is the bulkhead panel, inside the cord-seal line and
// over the box interior. An N-female-to-U.FL bulkhead passes a D-flat hole; its flange/nut and
// O-ring seat on the lid's flat top (the printed bed face) and its nut/shoulder on the flat
// inner face, where the HEAD ribs are cut back within ant_rib_clear_r for spanner room. The
// jack and pigtail hang straight into the box. Nothing printed stands within 10 mm of the
// flange, so the lid top holds no ring of water round the joint; tape the N joint.
//
// MOUNT: the connector carries the antenna's weight only. A separately printed post stands
// inboard (+Y) of each antenna. Its spigot drops into a blind socket in a boss under the lid
// and a flexure snap clicks under a recess in the socket wall (bolt-free; pinch the tab to
// release). Two band grips per antenna (centres 66 mm apart) slide down the post's dovetail
// rails onto ledges and clamp the base tube concentric with the bulkhead axis.
// Load path: antenna -> grips -> post rails -> post column -> column foot bearing on the lid top
// + spigot bearing in the socket boss -> lid plate -> 12 x M4 -> body flange.
// U-bolt FALLBACK: horizontal slots through the post and grips take the supplied U-bolts.
//
// GNSS PORT (owner decisions 2026-11-02): one SMA-female bulkhead jack through the lid, at gn_xy
// over the box interior, inside the seal line. It is the bulkhead end of a purchased
// SMA-female-bulkhead-to-SMA-male-right-angle jumper: the jumper's plug screws onto the GNSS jack
// held by the tower's printed SMA clip (control_77603/sma_clip.scad), whose own U.FL (IPEX MHF I)
// lead stays mated to the LC29H(DA) HAT. The GNSS antenna stays OUTSIDE (legacy antenna_deck or
// anywhere with sky view) and its SMA-male cable screws onto the jack. The jack's body hex sits
// under the lid in a local counterbore that leaves a gn_land_t land (the 5 mm plate exceeds the
// bulkhead's 3.175 mm maximum panel); the D-flat hole keys the body (so no spanner reaches under
// the lid), and a lock washer + nut sit on the flat lid top. Nothing printed stands within 10 mm
// of the washer; tape the SMA joint as the N joints.
// The jumper leaves the bulkhead straight down (heat-shrink end z 161.1), below the clip's plug
// barrels (z ~176), so it needs a U-turn plus its 100 mm lid-lift slack under the jack: gn_xy sits
// over the free loft between the clip and the +X panel's PCA9685 plugs, not over the clip.
//
// Frames: BODY coordinates with the lid CLOSED (z = 0 body floor bottom, +Y forward). Station
// frame: origin on the antenna axis at the lid top, +Y toward the post; the -X station is the
// mirror image of the +X station.
include <enclosure_common.scad>

antenna_part = "grip";   // grip | post | keepout
post_side = 1;           // post print: +1 = +X (Pi) station, -1 = -X (Alta) station, mirrored
ant_tube_d_mm = 32;      // TO MEASURE: antenna base-tube OD; photo estimate 30-35 (brief)

// ---- Bulkhead (Amphenol Type N catalog, https://www.farnell.com/datasheets/1931349.pdf and
// bulkhead page https://www.farnell.com/datasheets/269482.pdf; purchased pigtail TO MEASURE) ----
ant_dhole_d = 16.3;           // catalog "U" D-flat hole Ø.641 (pairing hole vs body: EST)
ant_dhole_flat = 13.9;        // catalog flats .546 (EST pairing)
ant_panel_t = lid_thickness;  // the lid plate is the panel; 82-369 accepts <= 5.3
ant_nut_af = 19.1;            // catalog hex nut .750 A/F
ant_nut_t = 6;                // DESIGN CAPACITY nut + lock washer (thickness TO MEASURE)
ant_thread_d = 15.9;          // 5/8-24 UNEF coupling/body thread (catalog)
ant_top_d = 23;               // ESTIMATE: flange (82-369: 20.6) or nut + washer on the lid top
ant_top_t = 5;                // ESTIMATE
ant_jack_above = 18;          // ESTIMATE: lid top to the N mating face
ant_jack_below = 14.5;        // ESTIMATE: lid inner face to the cable exit
ant_cable_d = 3;              // envelope round the 1.13 mm cable (generic pigtail, BoardResearch)
ant_cable_bend_r = 6;         // ESTIMATE ~5 x OD; minimum bend radius not published

// ---- Placement ----
ant_ax = 64.5;                // stations at +-ant_ax: 129 mm apart, 0.39 wavelength at 915 MHz
ant_ay = -48;                 // nut/jack inside the -Y inner wall; hole inside the seal line
ant_rib_clear_r = 20;         // HEAD lid ribs cut back round each bulkhead (nut spanner)
ant_keepout_d = 30;           // under-lid station envelope agreed with control_77603
ant_keepout_z0 = 162;

// ---- GNSS sky-view port: SMA-female bulkhead (body coordinates) ----
// Published SMA-F D-flat bulkhead drawings (U.FL pigtails T and L share the bulkhead end with the
// SMA-F-bulkhead -> SMA-M jumper; the purchased jumper and leads are TO MEASURE against them).
// The same data size the three jacks on the tower's SMA clip (sma_bulkhead_* modules below):
//   T = Taoglas CAB.011, https://www.taoglas.com/datasheets/CAB.011.pdf p.7
//   L = Linx/TE CSH-SGFB, https://docs.rs-online.com/3709/A700000010108438.pdf and
//       https://www.farnell.com/datasheets/3970976.pdf p.2
//   M = MIL-STD-348A Fig. 310-1/310-2, https://www.takitek.co.jp/Datasheets/MIL-STD-348RevA.pdf
gn_xy = [51, 41.6];           // DESIGN: loft +X of the SMA clip; envelope 0.4 off the y 50 rib and the PCA9685 plugs
gn_thread_d = 6.35;           // 1/4-36 UNS-2A (T, L, M)
gn_body_flat = 5.8;           // T: across the body's D-flat (L 5.9)
gn_hole_d = 6.5;              // L recommended panel hole
gn_hole_flat = 6.0;           // L: flat to the opposite arc
gn_panel_max = 3.175;         // L note 12 (T: about 3.5, from M's plug-nut clearance)
gn_land_t = 3;                // DESIGN land left by the under-lid counterbore
gn_cbore_d = 14;              // DESIGN: body hex corners 9.24 + 2.4 a side
gn_thread_len = 11.4;         // T, L: body hex face to the jack (mating) face
gn_hex_af = 8;                // T, L body hex (under the land)
gn_hex_t = 3.4;               // T (L 2.6)
gn_nut_af = 8;                // T, L panel nut
gn_nut_t = 1.8;               // T (L 1.7)
gn_washer_d = 10.2;           // T, L toothed lock washer OD
gn_washer_t = 0.5;            // L
gn_shrink_len = 22;           // T heat-shrink 17 +-5 under the hex (maximum)
gn_shrink_d = 4.3;            // T: 3.8 +-0.5 at the hex (maximum)
gn_plug_nut_clear = 5.54;     // M Fig. 310-2: thread behind the jack face the mating plug nut needs
gn_plug_overhang = 3.43;      // M Fig. 310-1: plug nut past the mating plane (max)
gn_plug_d = 10;               // ESTIMATE straight SMA-male plug of the antenna cable (5/16 hex 9.2 A/C)
gn_plug_len = 25;             // ESTIMATE plug + boot above the jack face; then the cable leaves
gn_keepout_d = 16;            // under-lid envelope round the hex and heat-shrink
gn_keepout_z0 = 160;          // below the heat-shrink end (worst case z 161.1)
function gn_land_z0() = sma_land_z0(ant_lid_top());          // land underside (hex face)
function gn_jack_face_z() = sma_face_z(ant_lid_top());
function gn_shrink_z0() = sma_shrink_z0(ant_lid_top());
function ant_lid_top() = box_top + lid_thickness;

// ---- Antenna reference (ESTIMATES except the 1143 mm height) ----
ant_height = 1143;            // vendor: 45 in
ant_coupling_d = 22;          // ESTIMATE N-male coupling nut
ant_coupling_len = 20;        // ESTIMATE
ant_tube_len = 100;           // PHOTO ESTIMATE ~100 (brief); TO MEASURE
ant_radome_d = 30;            // ESTIMATE; clearance probes use ant_env_d
ant_env_d = 40;               // clearance envelope = largest tube in range
function ant_mate_z() = ant_jack_above;                  // station frame
function ant_base_z() = ant_mate_z() - 12;               // coupling nut bottom (ESTIMATE)
function ant_tube_z0() = ant_base_z() + ant_coupling_len;
function ant_top_z() = ant_base_z() + ant_height;

// ---- Post, rails, grips (station frame) ----
grip_h = 18;
grip_z = [36, 102];           // grip bottoms above the lid top; centres 66 mm apart
grip_rail_y = 30;             // post face, from the axis toward the post
grip_rail_tip_y = 24;         // dovetail tip
grip_rail_root_hw = 8;
grip_rail_tip_hw = 11;
grip_slide_clear = 0.25;
grip_half_w = 31;
grip_back_clear = 0.2;
grip_rib_crush = 0.3;         // crush-rib interference on the measured tube
grip_band_clear = 0.8;        // band loop radial clearance for axial insertion
grip_cradle_clear = 0.3;      // cradle bore over the tube; the ribs set the centre
grip_band_t = 2;
grip_tail_end_y = 14;
grip_tail_slot_top = 22;
grip_tab_h = 8;               // pull tab above the grip
grip_ledge_h = 6;
post_half_w = 31;
post_back_y = 46;             // column y 30..46; the back face is the print bed face
post_top = grip_z[1] + grip_h;
ubolt_leg_d = 9;              // DESIGN: takes M8 (and smaller) legs; TO MEASURE
ubolt_dx = [18, 25];          // leg offset range from the axis
// Spigot and socket: the spigot sits under the column's inboard part so its boss stays 2 mm
// inside the box's inner wall.
spig_x = [-28.8, 13.8];
spig_y = [34, post_back_y];
spig_d = 16;
sock_clear = 0.2;
sock_extra = 0.5;             // socket deeper than the spigot
sock_floor = 3;
boss_wall = 4;
// Snap: a leaf on the spigot's inboard face, rooted at the spigot tip, with a tab above the
// lid top. Its tooth (square top, 45-degree lead-in) clicks into a recess in the socket wall.
snap_leaf_t = 1.8;
snap_slit = 1.8;
snap_root_z = -13;
snap_tab_z = [2, 10];
snap_tooth = 1.0;
snap_tooth_top = -6.0;
snap_recess_d = 1.4;
snap_recess_z = [-10.2, -5.8];

function grip_r(d = ant_tube_d_mm) = d / 2;
function grip_rb(d = ant_tube_d_mm) = d / 2 + grip_band_clear;
function sock_x() = [spig_x[0] - sock_clear, spig_x[1] + sock_clear];
function sock_y() = [spig_y[0] - sock_clear, spig_y[1] + sock_clear];
function sock_z0() = -(spig_d + sock_extra);
function boss_x() = [sock_x()[0] - boss_wall, sock_x()[1] + boss_wall];
function boss_y() = [sock_y()[0] - 3.8, sock_y()[1] + 3.8];
function boss_z0() = sock_z0() - sock_floor;

assert(ant_tube_d_mm >= 25 && ant_tube_d_mm <= 40, "Grip range is 25-40 mm base tube");
assert(ant_panel_t <= 5.3, "Lid exceeds the 82-369 bulkhead's 5.3 mm maximum panel");
assert(ant_ax + ant_keepout_d / 2 <= box_inner_l / 2 - 2
       && ant_ay - ant_keepout_d / 2 >= -box_inner_w / 2 + 1,
       "Bulkhead station must hang inside the box's inner walls");
assert(ant_ax + boss_x()[1] <= box_inner_l / 2 - 2 && ant_ax + boss_x()[0] >= 0,
       "Socket boss must stay 2 mm inside the inner wall");
assert(boss_y()[0] > ant_keepout_d / 2, "Socket boss runs into the bulkhead station");
assert(ant_lid_top() + boss_z0() >= 170, "Socket boss hangs below z 170 (control_77603 reservation)");
assert(grip_rail_tip_y - 1 - grip_slide_clear > grip_r(40), "40 mm tube leaves no cradle wall");
assert(ubolt_dx[0] - ubolt_leg_d / 2 > grip_rail_tip_hw + 2, "U-bolt slot cuts the rail");
assert(ubolt_dx[1] + ubolt_leg_d / 2 <= min(grip_half_w, post_half_w) - 1.5,
       "U-bolt slot breaks out of the grip or post side");
assert(grip_z[0] > ant_tube_z0() && grip_z[1] + grip_h < ant_tube_z0() + ant_tube_len,
       "Grips must sit on the estimated base tube");
assert(grip_z[1] - grip_ledge_h - 9 - (grip_z[0] + grip_h) >= grip_h + 4,
       "Lower grip cannot be fitted through the gap under the upper ledge");
assert(snap_slit > snap_tooth + sock_clear, "Snap leaf cannot deflect clear of the socket wall");
assert(ant_lid_top() - ant_panel_t - ant_jack_below - ant_cable_bend_r - ant_cable_d / 2
       >= ant_keepout_z0, "Pigtail bend drops below the station floor");
assert(gn_land_t <= gn_panel_max, "GNSS land exceeds the SMA bulkhead's maximum panel");
assert(gn_thread_len - gn_land_t - gn_washer_t - gn_nut_t >= gn_plug_nut_clear,
       "Too little SMA thread above the nut for the mating plug's coupling nut");
assert(gn_cbore_d >= gn_hex_af / cos(30) + 2, "GNSS counterbore does not clear the body hex");
assert(abs(gn_xy[0]) + gn_keepout_d / 2 <= box_inner_l / 2 - 2
       && abs(gn_xy[1]) + gn_keepout_d / 2 <= box_inner_w / 2 - 2,
       "GNSS jack must hang inside the box's inner walls");
assert(gn_shrink_z0() >= gn_keepout_z0, "GNSS heat-shrink drops below its envelope");

// Accessors (`use` shares functions, never variables).
function ant_axis_xy(side) = [side * ant_ax, ant_ay];
function ant_lid_top_z() = ant_lid_top();
function ant_top_z_mm() = ant_lid_top() + ant_top_z();
function ant_insert_shift_mm() = (grip_rib_crush / sin(50) + grip_band_clear) / 2;
function ant_post_lift_mm() = snap_recess_z[1] - snap_tooth_top + 0.3;
function ant_snap_deflect_mm() = snap_tooth + sock_clear;
function ant_snap_free_z() = snap_root_z + 0.5;   // leaf free above its root block
function ant_top_d_mm() = ant_top_d;
function ant_top_t_mm() = ant_top_t;
function gn_xy_mm() = gn_xy;
function gn_washer_d_mm() = gn_washer_d;
function gn_top_h_mm() = gn_jack_face_z() + gn_plug_len - ant_lid_top();
function gn_nut_top_z() = sma_nut_top_z(ant_lid_top());
// SMA bulkhead data for control_77603 (the SMA clip's lands and jack heights).
function gn_land_t_mm() = gn_land_t;
function gn_thread_len_mm() = gn_thread_len;
function gn_plug_overhang_mm() = gn_plug_overhang;

module ant_mirror(side) { if (side < 0) mirror([1, 0, 0]) children(); else children(); }
// Station frame -> body coordinates (mirrored for the -X station).
module at_station(side) {
    translate([side * ant_ax, ant_ay, ant_lid_top()]) ant_mirror(side) children();
}
module box3(a, b) { translate(a) cube(b - a); }

// ---- Lid features (body coordinates, lid closed); enclosure_lid.scad maps them ----
module antenna_socket_boss(side) {
    at_station(side) box3([boss_x()[0], boss_y()[0], boss_z0()],
                          [boss_x()[1], boss_y()[1], -lid_thickness + 0.01]);
}
module antenna_lid_add() { for (s = [-1, 1]) antenna_socket_boss(s); }
module antenna_lid_rib_clear() {
    for (s = [-1, 1]) translate([s * ant_ax, ant_ay, box_top - 9])
        cylinder(r = ant_rib_clear_r, h = 9, $fn = 72);
}
module ant_snap_recess() {
    // In the socket's inboard wall; its bottom is a 45-degree chamfer (a print ceiling).
    x0 = sock_x()[0];
    translate([0, sock_y()[1], 0]) rotate([90, 0, 0])
        linear_extrude(height = sock_y()[1] - sock_y()[0])
            polygon([[x0 + 0.01, snap_recess_z[1]], [x0 - snap_recess_d, snap_recess_z[1]],
                     [x0 - snap_recess_d, snap_recess_z[0] + snap_recess_d],
                     [x0 + 0.01, snap_recess_z[0]]]);
}
module antenna_lid_cut() {
    for (s = [-1, 1]) at_station(s) {
        // D-flat bulkhead hole through the plate.
        translate([0, 0, -ant_panel_t - 1]) intersection() {
            cylinder(d = ant_dhole_d, h = ant_panel_t + 2, $fn = 64);
            translate([-ant_dhole_d, -ant_dhole_flat / 2, 0])
                cube([2 * ant_dhole_d, ant_dhole_flat, ant_panel_t + 2]);
        }
        // Blind post socket and its snap recess.
        box3([sock_x()[0], sock_y()[0], sock_z0()], [sock_x()[1], sock_y()[1], 1]);
        ant_snap_recess();
    }
    gnss_lid_cut();
}
// GNSS D profile (2D, axis at the origin): circle d cut by a flat flat_c from the centre, +X.
module gn_d2(d, flat_c) {
    intersection() { circle(d = d, $fn = 48); translate([-d, -d]) square([d + flat_c, 2 * d]); }
}
// SMA-F D-flat bulkhead on a printed land: the lid's GNSS port and the three jacks of the tower's
// SMA clip (control_77603/sma_clip.scad) share these. xy = jack axis (body); top = the land's top
// face (outside face of the panel). The land is gn_land_t thick; the body hex bears on its
// underside inside a gn_cbore_d counterbore that starts at z cb0.
function sma_land_z0(top) = top - gn_land_t;                  // land underside (hex face)
function sma_face_z(top) = sma_land_z0(top) + gn_thread_len;   // jack (mating) face
function sma_shrink_z0(top) = sma_land_z0(top) - gn_hex_t - gn_shrink_len;
function sma_nut_top_z(top) = top + gn_washer_t + gn_nut_t;
// D-flat hole through the land and the counterbore under it that makes the land.
module sma_land_cut(xy, top, cb0) {
    translate([xy[0], xy[1], 0]) {
        translate([0, 0, sma_land_z0(top) - 0.5]) linear_extrude(height = gn_land_t + 1.5)
            gn_d2(gn_hole_d, gn_hole_flat - gn_hole_d / 2);
        translate([0, 0, cb0]) cylinder(d = gn_cbore_d, h = sma_land_z0(top) - cb0, $fn = 72);
    }
}
module gnss_lid_cut() { sma_land_cut(gn_xy, ant_lid_top(), box_top - 1); }
// Under-lid envelopes agreed with control_77603 (the control probes call antenna_lid_keepout()).
// HaLow stations: jack, nut, pigtail bend and the socket boss.
module halow_lid_keepout() {
    for (s = [-1, 1]) {
        translate([s * ant_ax, ant_ay, ant_keepout_z0])
            cylinder(d = ant_keepout_d, h = box_top - ant_keepout_z0, $fn = 72);
        antenna_socket_boss(s);
    }
}
// GNSS jack: body hex and heat-shrink (gn_keepout_d). The D-flat keys the body, so no spanner
// room is kept round the hex.
module gnss_lid_keepout() {
    translate([gn_xy[0], gn_xy[1], gn_keepout_z0])
        cylinder(d = gn_keepout_d, h = box_top - gn_keepout_z0, $fn = 48);
}
module antenna_lid_keepout() { halow_lid_keepout(); gnss_lid_keepout(); }

// ---- Post (station frame; printed lying on its back face, see antenna_post_print) ----
module ant_rail_profile() {
    polygon([[-grip_rail_root_hw, grip_rail_y + 1], [grip_rail_root_hw, grip_rail_y + 1],
             [grip_rail_root_hw, grip_rail_y], [grip_rail_tip_hw, grip_rail_tip_y],
             [-grip_rail_tip_hw, grip_rail_tip_y], [-grip_rail_root_hw, grip_rail_y]]);
}
module ant_ledge(z_top) {
    // Shelf under a rail segment; its underside is a 45-degree corbel back to the post face.
    hull() {
        translate([-14, grip_rail_tip_y, z_top - grip_ledge_h])
            cube([28, grip_rail_y - grip_rail_tip_y + 1, grip_ledge_h]);
        translate([-14, grip_rail_y, z_top - grip_ledge_h - (grip_rail_y - grip_rail_tip_y) - 3])
            cube([28, 1, 1]);
    }
}
function snap_x() = spig_x[0];
module antenna_post_core() {
    xs = snap_x() + snap_leaf_t + snap_slit;          // core starts past the leaf and slit
    difference() {
        union() {
            box3([-post_half_w, grip_rail_y, 0], [post_half_w, post_back_y, post_top]);
            for (z = grip_z) {
                translate([0, 0, z]) linear_extrude(height = grip_h) ant_rail_profile();
                ant_ledge(z);
            }
        }
        // Notch round the snap tab at the column root.
        box3([-post_half_w - 1, spig_y[0] - 1, -1], [xs, post_back_y + 1, snap_tab_z[1] + 2]);
        // U-bolt FALLBACK leg slots through the column (diamond: prints without support).
        for (z = grip_z, s = [-1, 1]) hull() for (dx = ubolt_dx)
            translate([s * dx, grip_rail_tip_y - 1, z + grip_h / 2]) rotate([-90, 0, 0])
                rotate([0, 0, 45]) cylinder(d = ubolt_leg_d, h = post_back_y - grip_rail_tip_y + 2, $fn = 4);
    }
    // Spigot core (past the slit) and the root block that carries the leaf.
    box3([xs, spig_y[0], -spig_d], [spig_x[1], spig_y[1], 0.01]);
    box3([spig_x[0], spig_y[0], -spig_d], [spig_x[1], spig_y[1], snap_root_z + 0.01]);
}
module antenna_post_snap() {
    // Leaf, tooth and finger tab: one piece with the core through the root block.
    box3([snap_x(), spig_y[0], snap_root_z], [snap_x() + snap_leaf_t, spig_y[1], snap_tab_z[1]]);
    translate([0, spig_y[1], 0]) rotate([90, 0, 0]) linear_extrude(height = spig_y[1] - spig_y[0])
        polygon([[snap_x() + 0.01, snap_tooth_top - snap_tooth - 2.4],
                 [snap_x() - snap_tooth, snap_tooth_top - 2.4],
                 [snap_x() - snap_tooth, snap_tooth_top], [snap_x() + 0.01, snap_tooth_top]]);
    box3([-post_half_w - 0.5, spig_y[0], snap_tab_z[0]], [snap_x() + 0.01, spig_y[1], snap_tab_z[1]]);
}
module antenna_post_local() { antenna_post_core(); antenna_post_snap(); }
module antenna_post_placed(side) { at_station(side) antenna_post_local(); }
// Print pose: back face (y = post_back_y) on the bed, column axis along +Y.
module antenna_post_print(side = post_side) {
    translate([0, spig_d, post_back_y]) rotate([-90, 0, 0]) ant_mirror(side) antenna_post_local();
}

// ---- Grip (printed upright, bottom face on the bed; one body; same part at both stations) ----
// Local frame: origin on the antenna axis, +Y toward the post, z = 0 the grip bottom.
module grip_profile(d = ant_tube_d_mm) {
    r = grip_r(d);
    rb = grip_rb(d);
    t = grip_band_t;
    xo = rb + t;                               // tail outer face
    yb = grip_rail_y - grip_back_clear;
    // Cradle block with the dovetail slot, tail slot and pawl relief.
    difference() {
        translate([-grip_half_w, 0]) square([2 * grip_half_w, yb]);
        circle(r = r + grip_cradle_clear, $fn = 144);
        offset(delta = grip_slide_clear) ant_rail_profile();
        translate([rb - 0.5, -1]) square([t + 1, grip_tail_slot_top + 1]);
        translate([xo + 2.3, grip_tail_end_y - 2]) square([0.8, grip_tail_slot_top - grip_tail_end_y + 2]);
        translate([xo + 0.5, grip_tail_end_y - 2]) square([2.6, 2.6]);
    }
    // Crush ribs inside the cradle (interfere grip_rib_crush with the measured tube).
    for (a = [50, 90, 130]) rotate(a) translate([r + 1.2 - grip_rib_crush, 0]) circle(r = 1.2, $fn = 24);
    // Pawl tooth on the flexure: square top face blocks loosening, ramped underside lets the
    // tail pass when pulled tight.
    polygon([[xo + 0.55, grip_tail_end_y + 0.6], [xo - 0.6, grip_tail_end_y + 2.0],
             [xo + 0.55, grip_tail_end_y + 2.0]]);
    // Band loop (root fused on the -X side), and the ratchet tail.
    difference() {
        circle(r = rb + t, $fn = 144);
        circle(r = rb, $fn = 144);
        translate([-rb - t - 1, sin(5) * rb]) square([2 * (rb + t) + 2, rb + t + 1]);
        translate([0, -0.01]) square([rb + t + 1, rb + t + 1]);
    }
    difference() {
        polygon([[rb, -0.02], [xo, -0.02], [xo, grip_tail_end_y - 0.8],
                 [xo - 0.8, grip_tail_end_y], [rb, grip_tail_end_y]]);
        for (k = [1:5]) let(y0 = grip_tail_end_y - 1.6 * k - 0.4)
            polygon([[xo + 0.01, y0], [xo - 0.7, y0 + 1.4], [xo + 0.01, y0 + 1.4]]);
    }
}
module antenna_grip(d = ant_tube_d_mm) {
    rb = grip_rb(d);
    difference() {
        union() {
            linear_extrude(height = grip_h) grip_profile(d);
            // Pull tab on the tail end, standing above the grip.
            translate([rb, grip_tail_end_y - 4, grip_h - 0.01])
                cube([grip_band_t - 0.8, 3.2, grip_tab_h]);
        }
        // U-bolt FALLBACK leg slots, matching the post (diamond: prints without support).
        for (s = [-1, 1]) hull() for (dx = [max(ubolt_dx[0], grip_r(d) + ubolt_leg_d / 2 + 0.5), ubolt_dx[1]])
            translate([s * dx, -1, grip_h / 2]) rotate([-90, 0, 0]) rotate([0, 0, 45])
                cylinder(d = ubolt_leg_d, h = grip_rail_y + 2, $fn = 4);
    }
}
// Grip i (0 lower, 1 upper) of a station, installed (never mirrored: the rail is symmetric).
module antenna_grip_placed(i, side, d = ant_tube_d_mm) {
    translate([side * ant_ax, ant_ay, ant_lid_top() + grip_z[i]]) antenna_grip(d);
}

// ---- References (body coordinates) ----
module antenna_bulkhead_top_ref(side) {
    // Flange or nut + washer and O-ring on the lid top, and the N-female interface.
    at_station(side) {
        cylinder(d = ant_top_d, h = ant_top_t, $fn = 48);
        cylinder(d = ant_thread_d, h = ant_jack_above, $fn = 48);
    }
}
module antenna_bulkhead_under_ref(side) {
    // D-flat body through the panel and below, and the nut/shoulder on the inner face.
    at_station(side) {
        translate([0, 0, -ant_panel_t - ant_jack_below]) intersection() {
            cylinder(d = ant_thread_d - 0.2, h = ant_panel_t + ant_jack_below, $fn = 48);
            translate([-ant_thread_d, -(ant_dhole_flat - 0.2) / 2, 0])
                cube([2 * ant_thread_d, ant_dhole_flat - 0.2, ant_panel_t + ant_jack_below]);
        }
        translate([0, 0, -ant_panel_t - ant_nut_t])
            cylinder(d = ant_nut_af / cos(30), h = ant_nut_t, $fn = 6);
    }
}
module antenna_bulkhead_ref(side) { antenna_bulkhead_top_ref(side); antenna_bulkhead_under_ref(side); }
module antenna_pigtail_ref(side) {
    // Leaves the jack, turns +Y (inboard) on ant_cable_bend_r and runs to the station edge.
    z0 = -ant_panel_t - ant_jack_below;
    r = ant_cable_bend_r;
    at_station(side) {
        for (a = [0:15:75]) hull() for (b = [a, a + 15])
            translate([0, r - r * cos(b), z0 - r * sin(b)]) sphere(d = ant_cable_d, $fn = 16);
        hull() for (y = [r, ant_keepout_d / 2 - ant_cable_d / 2 - 0.1])
            translate([0, y, z0 - r]) sphere(d = ant_cable_d, $fn = 16);
    }
}
module antenna_ref(side, tube_d = ant_tube_d_mm, env = false) {
    at_station(side) {
        if (env) cylinder(d = ant_env_d, h = ant_top_z(), $fn = 64);
        else {
            translate([0, 0, ant_base_z()]) cylinder(d = ant_coupling_d, h = ant_coupling_len, $fn = 48);
            translate([0, 0, ant_tube_z0()]) cylinder(d = tube_d, h = ant_tube_len, $fn = 96);
            translate([0, 0, ant_tube_z0() + ant_tube_len])
                cylinder(d = min(ant_radome_d, tube_d),
                         h = ant_top_z() - ant_tube_z0() - ant_tube_len, $fn = 48);
        }
    }
}
// ---- SMA bulkhead jack references (body coordinates; xy, top as sma_land_cut) ----
// D-flat threaded body (the flat runs along the thread), keyed in the land's D hole.
module sma_dbody(xy, z0, h, turn = 0) {
    translate([xy[0], xy[1], z0]) rotate([0, 0, turn]) linear_extrude(height = h)
        gn_d2(gn_thread_d, gn_body_flat - gn_thread_d / 2);
}
// Under the land: D body through it, body hex under it, heat-shrink below.
module sma_jack_under_ref(xy, top) {
    sma_dbody(xy, sma_land_z0(top), gn_land_t);
    translate([xy[0], xy[1], 0]) {
        translate([0, 0, sma_land_z0(top) - gn_hex_t]) cylinder(d = gn_hex_af / cos(30), h = gn_hex_t, $fn = 6);
        translate([0, 0, sma_shrink_z0(top)]) cylinder(d = gn_shrink_d, h = gn_shrink_len + 0.01, $fn = 24);
    }
}
// On the land's top face: lock washer and nut ...
module sma_nut_washer_ref(xy, top) {
    translate([xy[0], xy[1], top]) {
        cylinder(d = gn_washer_d, h = gn_washer_t, $fn = 48);
        translate([0, 0, gn_washer_t - 0.01]) cylinder(d = gn_nut_af / cos(30), h = gn_nut_t + 0.01, $fn = 6);
    }
}
// ... and the threaded body up to the jack (mating) face.
module sma_jack_top_ref(xy, top) {
    sma_dbody(xy, top, sma_face_z(top) - top);
    sma_nut_washer_ref(xy, top);
}
module sma_jack_ref(xy, top) { sma_jack_top_ref(xy, top); sma_jack_under_ref(xy, top); }
// The lid's GNSS jack.
module gnss_dbody(z0, h, turn = 0) { sma_dbody(gn_xy, z0, h, turn); }
module gnss_jack_under_ref() { sma_jack_under_ref(gn_xy, ant_lid_top()); }
module gnss_nut_washer_ref() { sma_nut_washer_ref(gn_xy, ant_lid_top()); }
module gnss_jack_top_ref() { sma_jack_top_ref(gn_xy, ant_lid_top()); }
// The outside antenna cable's straight SMA-male plug, mated (its nut overhangs the jack face).
module gnss_plug_ref() {
    translate([gn_xy[0], gn_xy[1], gn_jack_face_z() - gn_plug_overhang])
        cylinder(d = gn_plug_d, h = gn_plug_overhang + gn_plug_len, $fn = 36);
}
module gnss_jack_ref() { gnss_jack_top_ref(); gnss_jack_under_ref(); }
// Both stations' purchased and printed hardware above and under the closed lid (views).
module antenna_lid_hardware(d = ant_tube_d_mm, antenna_len = 0) {
    for (s = [-1, 1]) {
        color([0.16, 0.55, 0.60]) {
            antenna_post_placed(s);
            for (i = [0, 1]) antenna_grip_placed(i, s, d);
        }
        color([0.75, 0.76, 0.78]) antenna_bulkhead_ref(s);
        color([0.1, 0.1, 0.1]) antenna_pigtail_ref(s);
        // antenna_len > 0 draws only the lower antenna_len of each antenna. Preview paints any
        // clip face that is coplanar with a visible surface over it (z-fighting stripes), so the
        // clip starts 1 mm inside the lid, and it is a coaxial cylinder whose facets (radius d,
        // the tube OD, around every segment) lie away from the posts' flat faces.
        color([0.92, 0.70, 0.25, 0.55]) {
            if (antenna_len > 0) intersection() {
                antenna_ref(s, d);
                translate([s * ant_ax, ant_ay, ant_lid_top() - 1])
                    cylinder(r = d, h = antenna_len + 1, $fn = 30);
            } else antenna_ref(s, d);
        }
    }
    color([0.75, 0.76, 0.78]) gnss_jack_ref();
    color([0.1, 0.1, 0.1]) gnss_plug_ref();
}

if (antenna_part == "grip") antenna_grip();
else if (antenna_part == "post") antenna_post_print(post_side);
else if (antenna_part == "keepout") antenna_lid_keepout();
else assert(false, "antenna_part: grip | post | keepout");
