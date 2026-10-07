// Adjustable throttle-servo mounting fixture, not a Toro panel bolt pattern.
// RDS51150 stationary SIDE: 24 x 24 mm M2.5; metal horn/linkage required.
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
        for (x = [-12, 12], z = [-12, 12])
            translate([x, -1, throttle_pattern_z + z]) rotate([-90, 0, 0])
                cylinder(d = servo_side_hole_clear_d, h = throttle_face_t + 2);
    }
}
throttle_servo_mount();
