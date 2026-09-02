// Throttle servo mount (control-panel side).
//
// The Toro throttle is a continuous-variable lever on the control panel
// (manual form 3465-589). This L-bracket bolts to a flat panel face and
// carries the third RDS51150SG.
//
// SCOPE NOTE: the machine also has a MANUAL CHOKE, which is not actuated.
// Cold starting stays a human action in every phase of this build -- a
// throttle servo does not imply autonomous engine start. Per the manual the
// engine will not even crank unless the levers are outboard in PARK, which
// no actuator here can reach.
//
//   openscad -o throttle_servo_mount.stl throttle_servo_mount.scad

include <common.scad>

base_l        = 92;    // panel-mounting foot
base_w        = 56;
base_t        = 7;
face_l        = 76;    // servo face, across
face_h        = 62;    // servo face, standing up
face_t        = 8;
panel_bolts_x = 70;    // MEASURE: your panel bolt spacing
panel_bolts_y = 36;
gusset_reach  = 30;

module base_plate() {
    difference() {
        plate(base_l, base_w, base_t, r = 7);
        for (x = [-1, 1], y = [-1, 1])
            translate([x * panel_bolts_x / 2, y * panel_bolts_y / 2, -1])
                slot(m5_clear, 7, base_t + 2);
    }
}

// Vertical plate in the XZ plane, thickness along Y, rising from Z=0.
module servo_face() {
    difference() {
        translate([0, 0, face_h / 2])
            cube([face_l, face_t, face_h], center = true);
        for (x = [-1, 1], z = [-1, 1])
            translate([x * servo_bracket_hole_spacing_x / 2,
                       face_t / 2 + 1,
                       face_h / 2 + z * servo_bracket_hole_spacing_y / 2])
                rotate([90, 0, 0])
                    slot(m5_clear, 8, face_t + 2);
        translate([0, face_t / 2 + 1, face_h / 2])
            rotate([90, 0, 0])
                slot(15, face_l * 0.28, face_t + 2);
    }
}

module gusset() {
    rotate([0, -90, 0])
        linear_extrude(height = wall)
            polygon([[0, 0], [face_h * 0.7, 0], [0, gusset_reach]]);
}

module throttle_servo_mount() {
    // Face stands at the back edge of the foot; gussets brace forward over it.
    face_y = -base_w / 2 + face_t / 2 + 2;
    union() {
        base_plate();
        translate([0, face_y, base_t]) servo_face();
        for (x = [-1, 1])
            translate([x * (face_l / 2 - 8) + wall / 2,
                       face_y + face_t / 2, base_t])
                gusset();
    }
}

throttle_servo_mount();
