// Adjustable throttle-servo mounting fixture, not a Toro panel bolt pattern.
// Photo-derived SIX-hole stationary rear bracket; steel through-fasteners required.
// ASA; print base down. Verify reaction load, panel stiffness and all travel.
// A powered control mount must be metal or independently load-qualified.
include <common.scad>
throttle_base_l = 110;
throttle_base_w = 72;
throttle_base_t = 8;
throttle_face_t = 6;
throttle_pattern_z = 40;

module throttle_servo_mount() {
    difference() {
        union() {
            plate(throttle_base_l, throttle_base_w, throttle_base_t, 6);
            translate([-23, 0, throttle_base_t - 0.1]) cube([46, throttle_face_t, 60.1]);
            for (x = [-23, 19]) translate([x, 0, 0]) rotate([90, 0, 90])
                linear_extrude(height = 4)
                    polygon([[-25, throttle_base_t - 0.1], [0.1, throttle_base_t - 0.1], [0.1, 64]]);
        }
        for (x = [-38, 38], y = [-20, 20])
            translate([x, y, -1]) slot(m6_clear, 8, throttle_base_t + 2);
        // Shared rear axes Y/Z map to fixture X/Z; all six slots are through-Y.
        // Back face Y=0 is free for OD6 steel washers, AF5 nuts and an OD8 tool.
        translate([0,-1,throttle_pattern_z]) rotate([0,0,90]) {
            servo_rear_hole_cut(throttle_face_t+2);
            servo_rear_opening_cut(throttle_face_t+2);
        }
    }
}
throttle_servo_mount();
