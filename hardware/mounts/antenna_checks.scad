// CSG probes for the two lid-mounted HaLow antenna stations (antenna_mount.scad,
// enclosure_lid.scad). Every probe intersects REAL built solids: the reprinted lid, the HEAD
// body, the printed posts and grips, the published reference solids and the neighbouring
// assemblies. EMPTY = interference-free; NON-EMPTY = material/contact present. Contact is not
// proof of retention, fit or load capacity. Every probe covers BOTH stations.
//
// Run: openscad --export-format binstl -o /dev/null -D 'check="name"' antenna_checks.scad
// then look for "Current top level object is empty".
include <enclosure_common.scad>
use <antenna_mount.scad>
use <enclosure_body.scad>
use <enclosure_lid.scad>
use <hitch_arm_layer.scad>
use <pushrod.scad>
use <camera_stack.scad>
use <assemblies.scad>                 // ref_hitch_plate, ref_servo_at
use <throttle_77603/assemblies.scad>  // throttle_station

check = "grip_contact";
probe_tube_d = 25;   // grip probes: 25 and 40 bound the design range

module ant_grips(d) { for (s = [-1, 1], i = [0, 1]) antenna_grip_placed(i, s, d); }
module ant_posts() { for (s = [-1, 1]) antenna_post_placed(s); }
module ant_tubes(d) { for (s = [-1, 1]) antenna_ref(s, d); }
// One station's outside hardware: full-length Ø40 envelope, post, grips at both ends of the
// range, bulkhead top.
module ant_station(s) {
    antenna_ref(s, env = true);
    antenna_post_placed(s);
    for (i = [0, 1], d = [25, 40]) antenna_grip_placed(i, s, d);
    antenna_bulkhead_top_ref(s);
}
module ant_envelope() { for (s = [-1, 1]) ant_station(s); }
// Every purchased part under the lid: both N bulkheads + pigtails and the GNSS SMA jack.
module ant_under_parts() {
    for (s = [-1, 1]) { antenna_bulkhead_under_ref(s); antenna_pigtail_ref(s); }
    gnss_jack_under_ref();
}
// GNSS port room on the lid top: a 10 mm ring round the washer up to 6 mm above the nut (no
// printed dam, nut-driver room), the mated antenna-cable plug, and 20 mm above it where the
// cable turns away (ESTIMATE).
function gn_ring_r() = gn_washer_d_mm() / 2 + 10;
module gn_outside_room() {
    translate([gn_xy_mm()[0], gn_xy_mm()[1], ant_lid_top_z() + 0.01]) {
        cylinder(r = gn_ring_r(), h = gn_nut_top_z() - ant_lid_top_z() + 6, $fn = 72);
        cylinder(r = 15, h = gn_top_h_mm() + 20, $fn = 48);
    }
    gnss_plug_ref();
}
// Room round each N bulkhead: the flange ring on the lid top and the rib cut-back under it.
module ant_n_spanner_room() {
    for (s = [-1, 1]) translate([ant_axis_xy(s)[0], ant_axis_xy(s)[1], ant_lid_top_z() + 0.01])
        cylinder(r = ant_top_d_mm() / 2 + 10, h = ant_top_t_mm() + 6, $fn = 72);
    antenna_lid_rib_clear();
}
// Annular slab dz thick at z0 (body), radii r0..r1 round the GNSS axis.
module gn_ring(r0, r1, z0, dz) {
    translate([gn_xy_mm()[0], gn_xy_mm()[1], z0]) difference() {
        cylinder(r = r1, h = dz, $fn = 72);
        translate([0, 0, -1]) cylinder(r = r0, h = dz + 2, $fn = 72);
    }
}
// Steering servos and adapters over the analysed face range, both pushrods at neutral, and
// each rod's servo-end segment carried with its crank pin (as fit_checks.scad sweeps it).
module ant_steering_sweep() {
    for (side = [-1, 1]) {
        rod_between_points(servo_crank_pin(side), lapbar_pin(side), rod_nominal_setting());
        for (face = [servo_face_forward_check_deg:15:servo_face_reverse_deg]) {
            ref_servo_at(side, face);
            translate(servo_crank_pin_at(side, face) - servo_crank_pin(side)) intersection() {
                rod_between_points(servo_crank_pin(side), lapbar_pin(side), rod_nominal_setting());
                translate(servo_crank_pin(side) + [0, 60, 0]) cube([200, 260, 200], center = true);
            }
        }
    }
}
// Ring round each bulkhead top, 10 mm wide and 1 mm higher than the flange: nothing printed
// may stand there (no dam that holds water round the N joint).
module ant_flange_ring() {
    for (s = [-1, 1]) translate([ant_axis_xy(s)[0], ant_axis_xy(s)[1], ant_lid_top_z() + 0.01])
        difference() {
            cylinder(r = ant_top_d_mm() / 2 + 10, h = ant_top_t_mm() + 1, $fn = 72);
            translate([0, 0, -1]) cylinder(r = ant_top_d_mm() / 2 + 0.01, h = ant_top_t_mm() + 3, $fn = 72);
        }
}
// Driver path to each of the 12 lid screws: head + washer, then the shaft, to 400 mm up.
module ant_lid_screw_access() {
    for (p = lid_screw_points) translate([p[0], p[1], ant_lid_top_z() + 0.01]) {
        cylinder(d = 10, h = 6);
        cylinder(d = 6, h = 400);
    }
}

// ---- Grips ----
// Crush ribs land on the tube at both ends of the range (all four grips).
if (check == "grip_contact") intersection() { ant_grips(probe_tube_d); ant_tubes(probe_tube_d); }
// Held off the ribs toward the open band, each tube slides axially through its grips.
else if (check == "grip_insertion_clear") intersection() {
    ant_grips(probe_tube_d);
    translate([0, -ant_insert_shift_mm(), 0]) ant_tubes(probe_tube_d);
}
// Grips (25 and 40) slide on the post rails without touching (sampled 0.05 above the ledges) ...
else if (check == "grip_post_clear")
    intersection() { ant_posts(); translate([0, 0, 0.05]) { ant_grips(25); ant_grips(40); } }
// ... and sit on the ledges (0.05 sample below each grip).
else if (check == "grip_ledge_contact") intersection() {
    ant_posts();
    translate([0, 0, -0.05]) for (s = [-1, 1]) antenna_grip_placed(0, s, probe_tube_d);
}
else if (check == "grip_upper_ledge_contact") intersection() {
    ant_posts();
    translate([0, 0, -0.05]) for (s = [-1, 1]) antenna_grip_placed(1, s, probe_tube_d);
}
// ---- Post lock ----
// Seated posts (sampled 0.05 above the lid top) clear the socket, recess and lid ...
else if (check == "post_socket_clear")
    intersection() { enclosure_lid_closed(); translate([0, 0, 0.05]) ant_posts(); }
// ... the column foot bears on the lid top ...
else if (check == "post_seat_contact")
    intersection() { enclosure_lid_closed(); translate([0, 0, -0.05]) ant_posts(); }
// ... lifting a post 0.5 mm drives its snap tooth into the socket recess's top face ...
else if (check == "post_lock_retains")
    intersection() { enclosure_lid_closed(); translate([0, 0, ant_post_lift_mm()]) ant_posts(); }
// ... and the free leaf, pressed in by tooth + clearance, clears the socket wall and the post
// core (the post goes in and comes out).
else if (check == "post_snap_deflect_clear") intersection() {
    union() { enclosure_lid_closed(); for (s = [-1, 1]) at_station(s) antenna_post_core(); }
    for (s = [-1, 1]) at_station(s) intersection() {
        translate([ant_snap_deflect_mm(), 0, 0]) antenna_post_snap();
        translate([-100, -100, ant_snap_free_z()]) cube([200, 200, 100]);
    }
}
// ---- Port ----
// Seal lands present on the lid top (under the flange/nut) and the inner face (under the
// nut/shoulder): 0.1 samples.
else if (check == "bulkhead_land_contact")
    intersection() { enclosure_lid_closed(); translate([0, 0, -0.1]) for (s = [-1, 1]) antenna_bulkhead_top_ref(s); }
else if (check == "bulkhead_inner_land_contact")
    intersection() { enclosure_lid_closed(); translate([0, 0, 0.1]) for (s = [-1, 1]) antenna_bulkhead_under_ref(s); }
// Jack, nut and D-flat body fit the D-hole and lands (0.05 samples off the lands).
else if (check == "bulkhead_fit") intersection() {
    enclosure_lid_closed();
    union() for (s = [-1, 1]) {
        translate([0, 0, 0.05]) antenna_bulkhead_top_ref(s);
        translate([0, 0, -0.05]) antenna_bulkhead_under_ref(s);
    }
}
// Pigtails clear the lid's ribs and socket bosses ...
else if (check == "pigtail_clear_lid")
    intersection() { enclosure_lid_closed(); for (s = [-1, 1]) antenna_pigtail_ref(s); }
// ... and the nuts, jacks, pigtails and socket bosses clear the body's walls, flange and seal.
else if (check == "port_clear_body")
    intersection() { enclosure_body(); union() { ant_under_parts(); antenna_lid_add(); } }
// Everything under the lid stays inside the envelope agreed with control_77603.
else if (check == "underlid_in_keepout") difference() {
    intersection() { ant_under_parts(); translate([-200, -200, 0]) cube([400, 400, box_top]); }
    antenna_lid_keepout();
}
// Nothing printed stands in a 10 mm ring round either flange.
else if (check == "no_pool_ring")
    intersection() { ant_flange_ring(); union() { ant_posts(); ant_grips(25); ant_grips(40); } }
// The 12 lid screws stay reachable with both antennas, posts and grips fitted.
else if (check == "lid_screw_access") intersection() { ant_lid_screw_access(); ant_envelope(); }
// ---- Antenna envelope clearances (full 1143 mm, Ø40, plus posts and grips) ----
else if (check == "antenna_vs_antenna") intersection() { ant_station(-1); ant_station(1); }
else if (check == "env_vs_body") intersection() { ant_envelope(); enclosure_body(); }
else if (check == "env_vs_arm_layer") intersection() { ant_envelope(); arm_layer_assembly(); }
else if (check == "env_vs_steering") intersection() { ant_envelope(); ant_steering_sweep(); }
else if (check == "env_vs_throttle") intersection() { ant_envelope(); throttle_station(); }
else if (check == "env_vs_tower") intersection() { ant_envelope(); camera_stack_assembly(); }
else if (check == "env_vs_hitch_plate") intersection() { ant_envelope(); ref_hitch_plate(); }
// ---- GNSS port (SMA bulkhead jack near the compute tower) ----
// Jack, D body, hex, heat-shrink, washer and nut fit the land and hole (0.05 samples off the
// lid top and the land) ...
else if (check == "gnss_jack_fit") intersection() {
    enclosure_lid_closed();
    union() { translate([0, 0, 0.05]) gnss_jack_top_ref(); translate([0, 0, -0.05]) gnss_jack_under_ref(); }
}
// ... the body hex, lifted 0.05, bears on the land (the hole is a counterbored land, not open) ...
else if (check == "gnss_hex_bears")
    intersection() { enclosure_lid_closed(); translate([0, 0, 0.05]) gnss_jack_under_ref(); }
// ... and every point of the hex top face outside the hole (r 3.3..3.9) is backed by the land.
else if (check == "gnss_land_backed")
    difference() { gn_ring(3.3, 3.9, gn_land_z0() + 0.001, 0.1); enclosure_lid_closed(); }
// The D flat keys the body: turned 10 degrees in the land, the body meets the hole's flat.
else if (check == "gnss_dflat_keys")
    intersection() { enclosure_lid_closed(); gnss_dbody(gn_land_z0() + 0.2, 2.6, 10); }
// Outside seat: the washer, lowered 0.05, lands on the lid top ...
else if (check == "gnss_seat_contact")
    intersection() { enclosure_lid_closed(); translate([0, 0, -0.05]) gnss_nut_washer_ref(); }
// ... the whole washer annulus outside the hole is backed by the flat top (0.1 slab under it) ...
else if (check == "gnss_seat_backed")
    difference() { gn_ring(3.3, gn_washer_d_mm() / 2, ant_lid_top_z() - 0.1, 0.099); enclosure_lid_closed(); }
// ... and nothing printed (lid, posts, grips) stands in the 10 mm ring round it (no water dam).
else if (check == "gnss_no_pool") intersection() {
    gn_ring(gn_washer_d_mm() / 2, gn_ring_r(), ant_lid_top_z() + 0.01, 8);
    union() { enclosure_lid_closed(); ant_posts(); ant_grips(25); ant_grips(40); }
}
// The mated antenna-cable plug clears the panel nut, washer and lid (thread left for its nut).
else if (check == "gnss_plug_clear")
    intersection() { gnss_plug_ref(); union() { enclosure_lid_closed(); gnss_nut_washer_ref(); } }
// Under the lid the jack envelope (hex, heat-shrink, spanner room) clears the lid's ribs and
// skirt (sampled 0.01 under the lid's inner face) and the body's walls and flange.
else if (check == "gnss_underlid_clear")
    intersection() {
        translate([0, 0, -0.01]) gnss_lid_keepout();
        union() { enclosure_lid_closed(); enclosure_body(); }
    }
// The GNSS port, outside and under the lid, clears both N stations: antennas (full length,
// Ø40), posts, grips, bulkheads, pigtails, their under-lid envelopes and spanner rooms.
else if (check == "gnss_clear_stations") intersection() {
    union() { gn_outside_room(); gnss_lid_keepout(); }
    union() {
        ant_envelope();
        for (s = [-1, 1]) { antenna_bulkhead_under_ref(s); antenna_pigtail_ref(s); }
        halow_lid_keepout();
        ant_n_spanner_room();
    }
}
// The 12 lid screws stay reachable round the GNSS port room.
else if (check == "gnss_lid_screw_access") intersection() { ant_lid_screw_access(); gn_outside_room(); }
else assert(false, str("unknown check: ", check));
