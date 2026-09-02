// E-stop panel bracket.
//
// Carries a 22 mm mushroom emergency-stop button. Seats on the 3" x 1.5"
// rectangular frame rail and is held by two U-bolts, same as the servo mount.
//
// Position it so it can be struck by someone standing OFF the machine,
// without reaching over the deck or the operator station. Verify by walking
// up to the parked mower and hitting it without leaning in.
//
//   openscad -o estop_bracket.stl estop_bracket.scad

include <common.scad>

button_hole  = 22.5;   // 22 mm button + fit
face_l       = 66;     // along the rail
face_h       = 66;     // standing up
face_t       = 7;
saddle_l     = 104;
saddle_t     = 10;
ubolt_span   = 70;
gusset_reach = 26;

// Vertical plate standing in the XZ plane, thickness along Y, rising from Z=0.
module button_face() {
    difference() {
        translate([0, 0, face_h / 2])
            cube([face_l, face_t, face_h], center = true);
        // Button bore, through Y
        translate([0, face_t / 2 + 1, face_h / 2])
            rotate([90, 0, 0])
                cylinder(h = face_t + 2, d = button_hole);
        // Anti-rotation notch. Deliberately overlaps into the bore -- a notch
        // that merely touches the bore wall leaves a tangent edge and a
        // non-manifold mesh.
        translate([-2.1, face_t / 2 + 1, face_h / 2 + button_hole / 2 - 2])
            rotate([90, 0, 0])
                cube([4.2, 5.2, face_t + 2]);
    }
}

module gusset() {
    rotate([0, -90, 0])
        linear_extrude(height = wall)
            polygon([[0, 0], [face_h * 0.7, 0], [0, gusset_reach]]);
}

module estop_bracket() {
    difference() {
        union() {
            frame_saddle(saddle_l, saddle_t, ubolt_span);
            translate([0, 0, saddle_t]) button_face();
            for (x = [-1, 1])
                translate([x * (face_l / 2 - 8) + wall / 2,
                           face_t / 2, saddle_t])
                    gusset();
        }
        frame_ubolt_slots(saddle_t, ubolt_span);
    }
}

estop_bracket();
