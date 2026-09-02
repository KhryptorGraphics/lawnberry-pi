// Printed weatherproof enclosure -- LID.
//
// Pairs with enclosure_body.scad. Compresses a 3 mm silicone O-ring cord
// sitting in the body's flange groove.
//
// The skirt around the edge is a DRIP LIP, not decoration: it overhangs the
// body's flange so rain running off the lid is thrown clear of the seam
// instead of running into it. On a horizontal-seam box this matters more than
// the gasket does, because a gasket under standing water eventually wicks.
//
// PRINT NOTES
//   * Print OUTSIDE FACE DOWN on the bed. That gives a smooth, water-shedding
//     outer surface and leaves the skirt and ribs pointing up, so no supports
//     are needed. Flip it in the slicer.
//   * 5+ perimeters. A lid that flexes between screws will not compress the
//     gasket evenly -- err thick, and do not reduce the ribs.
//   * PETG or ASA, same as the body.
//   * Footprint 209 x 164 mm -- fits a 220 x 220 bed.
//
//   openscad -o enclosure_lid.stl enclosure_lid.scad

include <common.scad>

// These MUST match enclosure_body.scad.
inner_l      = 173;
inner_w      = 128;
e_wall       = 4.0;
flange_w     = 10.0;
outer_l      = inner_l + 2 * e_wall;
outer_w      = inner_w + 2 * e_wall;
flange_l     = outer_l + 2 * flange_w;
flange_ow    = outer_w + 2 * flange_w;
screw_inset  = flange_w / 2;
lid_screw    = m4_clear;

lid_t        = 5.0;    // lid plate
skirt_out    = 4.0;    // how far the drip lip stands proud of the flange
skirt_drop   = 6.0;    // how far it hangs down past the flange face
rib_h        = 7.0;    // stiffening rib, keeps the lid flat between screws
rib_w        = 5.0;

lid_l        = flange_l + 2 * skirt_out;
lid_w        = flange_ow + 2 * skirt_out;

module rounded_ring(l, w, r, h) {
    hull()
        for (x = [-1, 1], y = [-1, 1])
            translate([x * (l / 2 - r), y * (w / 2 - r), 0])
                cylinder(h = h, r = r);
}

module screw_holes(h) {
    for (fx = [-1, -0.34, 0.34, 1], fy = [-1, 1])
        translate([fx * (outer_l / 2 + screw_inset),
                   fy * (outer_w / 2 + screw_inset), -1])
            cylinder(h = h + 2, d = lid_screw);
    for (fy = [-0.34, 0.34], fx = [-1, 1])
        translate([fx * (outer_l / 2 + screw_inset),
                   fy * outer_w / 2, -1])
            cylinder(h = h + 2, d = lid_screw);
}

module enclosure_lid() {
    difference() {
        union() {
            // Lid plate
            rounded_ring(lid_l, lid_w, 12, lid_t);
            // Drip skirt, hanging down past the flange
            translate([0, 0, -skirt_drop])
                difference() {
                    rounded_ring(lid_l, lid_w, 12, skirt_drop + 0.1);
                    translate([0, 0, -1])
                        rounded_ring(lid_l - 2 * skirt_out * 0.9,
                                     lid_w - 2 * skirt_out * 0.9, 11,
                                     skirt_drop + 3);
                }
            // Stiffening ribs, on the INSIDE face. Ribs on the outside would
            // form channels that hold water and grit on top of the seam --
            // the outer face must stay smooth so it sheds.
            // center = true: an uncentred cube here grows the lid by its own
            // length in +X/+Y instead of spanning it.
            for (x = [-0.55, 0, 0.55])
                translate([x * inner_l * 0.34, 0, -rib_h / 2])
                    cube([rib_w, inner_w * 0.9, rib_h], center = true);
            translate([0, 0, -rib_h / 2])
                cube([inner_l * 0.9, rib_w, rib_h], center = true);
        }
        screw_holes(lid_t);
        // Countersinks so screw heads sit flush and shed water
        for (fx = [-1, -0.34, 0.34, 1], fy = [-1, 1])
            translate([fx * (outer_l / 2 + screw_inset),
                       fy * (outer_w / 2 + screw_inset), lid_t - 2.2])
                cylinder(h = 2.4, d1 = lid_screw, d2 = lid_screw + 4.4);
        for (fy = [-0.34, 0.34], fx = [-1, 1])
            translate([fx * (outer_l / 2 + screw_inset),
                       fy * outer_w / 2, lid_t - 2.2])
                cylinder(h = 2.4, d1 = lid_screw, d2 = lid_screw + 4.4);
    }
}

enclosure_lid();
