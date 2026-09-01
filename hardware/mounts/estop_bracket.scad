// E-stop panel bracket.
//
// Carries a 22 mm mushroom emergency-stop button on a tube or flat face,
// positioned so it can be struck from beside the machine without reaching
// over the deck or the operator station.
//
// The button must be reachable by someone standing OFF the machine. Mount
// it high and outboard; verify by walking up to the parked mower and
// hitting it without leaning in.
//
//   openscad -o estop_bracket.stl estop_bracket.scad

include <common.scad>

button_hole = 22.5;   // 22 mm button + fit
face_l      = 62;
face_w      = 62;
face_t      = 6;
riser       = 46;     // face height above the clamp centreline
clamp_width = 34;

module button_face() {
    difference() {
        hull()
            for (x = [-1, 1], y = [-1, 1])
                translate([x * (face_l / 2 - 8), y * (face_w / 2 - 8), 0])
                    cylinder(h = face_t, r = 8);
        translate([0, 0, -1]) cylinder(h = face_t + 2, d = button_hole);
        // Anti-rotation notch, typical of 22 mm switch bodies.
        // Deliberately overlaps into the bore -- a notch that merely
        // touches the bore wall leaves a tangent edge and a non-manifold
        // mesh.
        translate([-2.1, button_hole / 2 - 2, -1])
            cube([4.2, 5.2, face_t + 2], center = false);
    }
}

module estop_bracket() {
    union() {
        tube_clamp_half(frame_tube_od, clamp_width, half = 0);
        translate([riser, 0, 0]) rotate([0, 90, 0]) button_face();
        // Spine + ribs
        translate([(frame_tube_od / 2 + clamp_wall + riser) / 2, 0, 0])
            cube([riser - frame_tube_od / 2, wall * 1.6, clamp_width],
                 center = true);
        for (z = [-1, 1])
            translate([0, 0, z * (clamp_width / 2 - wall / 2)])
                rotate([90, 0, 0])
                    linear_extrude(height = wall, center = true)
                        polygon([[frame_tube_od / 2 + clamp_wall - 1, 0],
                                 [riser + face_t / 2, 0],
                                 [frame_tube_od / 2 + clamp_wall - 1,
                                  face_l / 2 - 4]]);
    }
}

estop_bracket();
