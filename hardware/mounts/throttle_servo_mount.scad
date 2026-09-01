// Throttle servo mount (control-panel side).
//
// The Toro throttle is a continuous-variable lever on the control panel
// (manual, form 3465-589). This bracket bolts to a flat panel face or a
// small tube and carries the third RDS51150SG.
//
// NOTE ON SCOPE: the machine also has a MANUAL CHOKE, which is NOT
// actuated. Cold starting stays a human action in every phase of this
// build -- do not read a throttle servo as "autonomous engine start".
//
//   openscad -o throttle_servo_mount.stl throttle_servo_mount.scad

include <common.scad>

base_l        = 86;
base_w        = 52;
base_t        = 6;
face_h        = 58;   // height of the servo face above the base
face_t        = 7;
panel_bolts_x = 66;   // MEASURE: your panel bolt spacing
panel_bolts_y = 34;

module base_plate() {
    difference() {
        hull()
            for (x = [-1, 1], y = [-1, 1])
                translate([x * (base_l / 2 - 6), y * (base_w / 2 - 6), 0])
                    cylinder(h = base_t, r = 6);
        // Panel fixings, slotted for adjustment
        for (x = [-1, 1], y = [-1, 1])
            translate([x * panel_bolts_x / 2, y * panel_bolts_y / 2, -1])
                slot(m5_clear, 6, base_t + 2);
    }
}

module servo_face() {
    difference() {
        cube([face_t, base_w, face_h]);
        // Servo U-bracket bolt slots
        for (y = [-1, 1], z = [-1, 1])
            translate([-1,
                       base_w / 2 + y * servo_bracket_hole_spacing_y / 2,
                       face_h / 2 + z * servo_bracket_hole_spacing_x / 2])
                rotate([0, 90, 0])
                    slot(m5_clear, 6, face_t + 2);
        // Cable/heat relief
        translate([-1, base_w / 2, face_h * 0.5])
            rotate([0, 90, 0])
                slot(14, 18, face_t + 2);
    }
}

module throttle_servo_mount() {
    union() {
        base_plate();
        translate([-base_l / 2 + 4, -base_w / 2, base_t]) servo_face();
        // Ribs
        for (y = [-1, 1])
            translate([-base_l / 2 + 4 + face_t, y * (base_w / 2 - wall / 2),
                       base_t])
                rotate([90, 0, 90])
                    rotate([0, 0, -90])
                        linear_extrude(height = wall, center = true)
                            polygon([[0, 0], [face_h * 0.8, 0],
                                     [0, base_l * 0.42]]);
    }
}

throttle_servo_mount();
