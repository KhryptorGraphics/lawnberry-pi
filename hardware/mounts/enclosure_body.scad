// Printed weatherproof enclosure -- BODY.
//
// Replaces the purchased IP65 box. Holds electronics_tray.scad.
//
// HOW A PRINTED BOX IS MADE TO SEAL
// Printed walls are porous along layer lines and the mating faces are never
// flat enough to seal on their own, so this design does not try. Instead:
//   * A 3 mm silicone O-ring cord sits in a groove in the top flange, and the
//     lid compresses it. The gasket seals, not the plastic.
//   * Lid screws sit OUTBOARD of the gasket groove, so tightening squeezes
//     the cord rather than bowing the lid away from it.
//   * The flange is wide (12 mm) so the groove has real material either side.
//     A groove cut into a 3-4 mm wall leaves paper-thin lips that split.
//   * Cable glands enter through the FLOOR, on thickened bosses. Mounting
//     ears on the back wall hold the box off its mounting surface so those
//     glands have clearance and cables can drip-loop below.
//   * The breather vent goes in a SIDE wall, low. Never the top face.
//
// PRINT NOTES
//   * Open side up, no supports needed. 5+ perimeters, >=40% infill.
//   * PETG or ASA. Not PLA -- it warps in a hot machine and UV-degrades.
//   * Walls MUST be watertight: print hot enough and slow enough that layers
//     fuse. A gasket cannot save a porous wall. Consider a wipe of epoxy or
//     acrylic conformal coat on the inside if in doubt.
//   * Footprint is 216 x 166 mm -- needs a 220 x 220 bed or larger.
//
//   openscad -o enclosure_body.stl enclosure_body.scad

include <common.scad>

// Interior sized around electronics_tray.scad (195 x 145) with clearance.
inner_l      = 173;
inner_w      = 128;
inner_h      = 85;

e_wall       = 4.0;    // side wall
e_floor      = 4.5;    // floor
flange_w     = 10.0;   // outward flange carrying gasket + screws
flange_t     = 6.0;

gasket_cord  = 3.0;    // silicone O-ring cord diameter
groove_w     = gasket_cord + 0.3;
groove_d     = gasket_cord * 0.72;   // ~28% compression when closed

lid_screw    = m4_clear;
insert_od    = 5.7;    // M4 heat-set insert pocket; tap or self-tap instead
insert_h     = 8.0;

gland_hole   = 15.2;   // PG9
vent_hole    = 12.5;   // M12 x 1.5 breather
gland_boss   = 24;     // boss OD around each gland
gland_boss_h = 3.0;    // extra floor thickness at the gland

tray_l       = 165;    // must match electronics_tray.scad
tray_w       = 120;

outer_l      = inner_l + 2 * e_wall;
outer_w      = inner_w + 2 * e_wall;
flange_l     = outer_l + 2 * flange_w;
flange_ow    = outer_w + 2 * flange_w;

// Lid screw positions, outboard of the gasket groove.
screw_x      = [-1, -0.34, 0.34, 1];
screw_y      = [-1, 1];
screw_inset  = flange_w / 2;

function screw_pos_x(f) = f * (outer_l / 2 + screw_inset);
function screw_pos_y(f) = f * (outer_w / 2 + screw_inset);

// Gasket groove centre-line, midway through the flange.
groove_off   = flange_w * 0.28;

module rounded_ring(l, w, r, h) {
    hull()
        for (x = [-1, 1], y = [-1, 1])
            translate([x * (l / 2 - r), y * (w / 2 - r), 0])
                cylinder(h = h, r = r);
}

// Closed path for the gasket groove, following the wall outline.
module gasket_groove() {
    difference() {
        rounded_ring(outer_l + 2 * groove_off + groove_w,
                     outer_w + 2 * groove_off + groove_w, 10, groove_d + 1);
        translate([0, 0, -0.5])
            rounded_ring(outer_l + 2 * groove_off - groove_w,
                         outer_w + 2 * groove_off - groove_w, 10,
                         groove_d + 2);
    }
}

module shell() {
    union() {
        // Walls + floor
        difference() {
            rounded_ring(outer_l, outer_w, 8, inner_h + e_floor);
            translate([0, 0, e_floor])
                rounded_ring(inner_l, inner_w, 6, inner_h + 1);
        }
        // Top flange
        translate([0, 0, inner_h + e_floor - flange_t])
            difference() {
                rounded_ring(flange_l, flange_ow, 10, flange_t);
                translate([0, 0, -1])
                    rounded_ring(inner_l, inner_w, 6, flange_t + 2);
            }
        // Gland bosses, thickening the floor locally
        for (i = [-1, 0, 1])
            translate([i * 44, -inner_w / 2 + 22, 0])
                cylinder(h = e_floor + gland_boss_h, d = gland_boss);
        // Tray mounting bosses
        for (x = [-1, 1], y = [-1, 1])
            translate([x * (tray_l / 2 - 18), y * (tray_w / 2 - 11), 0])
                cylinder(h = e_floor + 6, d = 11);
        // Mounting ears on the back wall -- hold the box off its surface so
        // the floor glands have clearance for a drip loop.
        for (x = [-1, 1])
            translate([x * (outer_l / 2 - 26), outer_w / 2 + 7, 0])
                hull() {
                    translate([0, -8, 0]) cube([26, 1, 22], center = true);
                    translate([0, 6, 0]) cube([26, 1, 22], center = true);
                }
    }
}

module enclosure_body() {
    difference() {
        shell();
        // Gasket groove in the flange top
        translate([0, 0, inner_h + e_floor - groove_d]) gasket_groove();
        // Lid screw inserts, outboard of the groove
        for (fx = screw_x, fy = screw_y)
            translate([screw_pos_x(fx), screw_pos_y(fy),
                       inner_h + e_floor - insert_h])
                cylinder(h = insert_h + 1, d = insert_od);
        for (fy = [-0.34, 0.34])
            for (fx = [-1, 1])
                translate([fx * (outer_l / 2 + screw_inset),
                           fy * outer_w / 2 * 1.0,
                           inner_h + e_floor - insert_h])
                    cylinder(h = insert_h + 1, d = insert_od);
        // Cable glands through the floor
        for (i = [-1, 0, 1])
            translate([i * 44, -inner_w / 2 + 22, -1])
                cylinder(h = e_floor + gland_boss_h + 2, d = gland_hole);
        // Breather vent, low in a side wall
        translate([outer_l / 2 + 1, 34, e_floor + 16])
            rotate([0, -90, 0])
                cylinder(h = e_wall + 2, d = vent_hole);
        // Tray fixing holes
        for (x = [-1, 1], y = [-1, 1])
            translate([x * (tray_l / 2 - 18), y * (tray_w / 2 - 11), e_floor])
                cylinder(h = 8, d = 3.2);
        // Mounting-ear bolt holes
        for (x = [-1, 1])
            translate([x * (outer_l / 2 - 26), outer_w / 2 + 7, 0])
                rotate([90, 0, 0])
                    cylinder(h = 40, d = m6_clear, center = true);
    }
}

enclosure_body();
