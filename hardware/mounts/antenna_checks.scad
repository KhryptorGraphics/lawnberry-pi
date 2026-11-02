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
module ant_under_parts() { for (s = [-1, 1]) { antenna_bulkhead_under_ref(s); antenna_pigtail_ref(s); } }
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
else assert(false, str("unknown check: ", check));
