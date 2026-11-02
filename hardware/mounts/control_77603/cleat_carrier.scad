// Carrier family: the CARRIER side of the bolt-free French cleat, and the shared helpers.
//
// Every electronics carrier in this package hangs on one molded wall strip
// (../electronics_rail.scad wall_cleat()) through the same three features:
//   * a 45-degree HOOK that drops into the V between the strip's top bevel and the wall,
//     with KEY RIBS (2.0 wide) that sit in the strip's 2.4 mm notches so it cannot slide;
//   * a LATCH tongue freed from the plate by two slots; its tooth snaps up under the
//     strip's 2 mm catch land, so the carrier cannot lift off; a lip at the free end is the
//     release (pull it 2 mm away from the wall, then lift);
//   * two FEET at the plate bottom that bear on the wall under the carrier's own moment.
// All drawn in the cleat local frame (parameters.scad), carrier x from x0 to x1 along
// the strip, at the NOMINAL pose (0.4 mm off every strip face).
//
// Printability: carriers print UPRIGHT as installed. The hook's lower face, the key ribs
// and the tooth's lead-in are 45-degree faces; the tooth's catch face faces up; the feet
// sit on the bed. The latch strain is asserted in parameters.scad (<= 1.5 % PETG).
include <parameters.scad>

module box3(p0, p1) { translate(p0) cube(p1 - p0); }

// A (y, z) profile extruded along x from x0 to x1.
module yz_prism(x0, x1) {
    translate([x0, 0, 0]) rotate([90, 0, 90]) linear_extrude(height = x1 - x0) children();
}
// An (x, z) profile extruded along y from y0 to y1.
module xz_prism(y0, y1) {
    translate([0, y1, 0]) rotate([90, 0, 0]) linear_extrude(height = y1 - y0) children();
}

// ---- Hook, key ribs, latch, feet -------------------------------------------------------
module cr_hook_block(x0, x1) {
    yz_prism(x0, x1) polygon([
        [cr_hook_wall, -cr_hook_wall + cr_hook_off],
        [cr_plate_back, -cr_plate_back + cr_hook_off],
        [cr_plate_back - 0.01, cr_hook_top],
        [cr_hook_wall, cr_hook_top],
    ]);
}
// Band under the hook face down to 0.4 above the notch floor.
module cr_key_rib(xc) {
    yz_prism(xc - cr_rib_w / 2, xc + cr_rib_w / 2) polygon([
        [cr_hook_wall, -cr_hook_wall + cr_hook_off + 0.01],
        [cr_plate_back, -cr_plate_back + cr_hook_off + 0.01],
        [cr_plate_back, -cr_plate_back - cr_rib_below_bevel * sqrt(2)],
        [cr_hook_wall, -cr_hook_wall - cr_rib_below_bevel * sqrt(2)],
    ]);
}
function cr_tongue_x(x0, x1) = (x0 + x1) / 2;
function cr_gap(x0, x1) =
    [cr_tongue_x(x0, x1) - cr_tongue_w / 2 - cr_tongue_slot,
     cr_tongue_x(x0, x1) + cr_tongue_w / 2 + cr_tongue_slot];
// Notch centres (strip phase: carriers hang at strip local x = 0) inside [a, b].
function cr_rib_xs(a, b) =
    [for (x = [cleat_notch_pitch() / 2 : cleat_notch_pitch() : 400])
        if (x - cr_rib_w / 2 >= a + 0.5 && x + cr_rib_w / 2 <= b - 0.5) x];

// The full hanger: plate x [x0, x1], z [z_bot, cr_plate_top], hook, ribs, latch, feet.
// `feet_x` gives the two feet's left edges (default: the plate's two ends).
module cr_hanger(x0, x1, z_bot, feet_x = undef) {
    xc = cr_tongue_x(x0, x1);
    g = cr_gap(x0, x1);
    fx = is_undef(feet_x) ? [x0, x1 - cr_foot_w] : feet_x;
    tf = cr_plate_back - cr_tongue_t;
    lip_y = cr_plate_front - cr_lip;
    assert(z_bot + cr_foot_h <= cr_foot_max_top, "Carrier feet must sit below the strip");
    difference() {
        box3([x0, cr_plate_front, z_bot], [x1, cr_plate_back, cr_plate_top]);
        // Slots that free the tongue: two sides and the free end.
        box3([g[0], cr_plate_front - 1, cr_tongue_free - cr_tongue_slot],
             [xc - cr_tongue_w / 2, cr_plate_back + 1, cr_tongue_root]);
        box3([xc + cr_tongue_w / 2, cr_plate_front - 1, cr_tongue_free - cr_tongue_slot],
             [g[1], cr_plate_back + 1, cr_tongue_root]);
        box3([g[0], cr_plate_front - 1, cr_tongue_free - cr_tongue_slot],
             [g[1], cr_plate_back + 1, cr_tongue_free]);
        // The tongue is thinner than the plate (front recess), back face flush. The recess
        // stops 1 mm under the root so no face is coplanar with the slot tops.
        box3([xc - cr_tongue_w / 2 - 0.01, cr_plate_front - 1, cr_tongue_free - 0.01],
             [xc + cr_tongue_w / 2 + 0.01, tf, cr_tongue_root - 1]);
    }
    // Hook in two segments either side of the tongue, ribs on the segments.
    cr_hook_block(x0, g[0]);
    cr_hook_block(g[1], x1);
    for (x = concat(cr_rib_xs(x0, g[0]), cr_rib_xs(g[1], x1))) cr_key_rib(x);
    // Latch tooth on the tongue back, 45-degree lead-in below its catch face.
    yz_prism(xc - cr_tongue_w / 2 + 0.5, xc + cr_tongue_w / 2 - 0.5) polygon([
        [cr_plate_back - 0.01, cr_tooth_top],
        [cr_plate_back + cr_tooth_d, cr_tooth_top],
        [cr_plate_back + cr_tooth_d, cr_tooth_top - cr_tooth_flat],
        [cr_plate_back - 0.01, cr_tooth_bottom],
    ]);
    // Release lip on the tongue front at its free end (45-degree faces both ways).
    yz_prism(xc - cr_tongue_w / 2, xc + cr_tongue_w / 2) polygon([
        [tf + 0.01, cr_tongue_free],
        [lip_y, cr_tongue_free + (tf - lip_y)],
        [lip_y, cr_tongue_free + (tf - lip_y) + 1.5],
        [tf + 0.01, cr_tongue_free + 2 * (tf - lip_y) + 1.5],
    ]);
    // Feet bear on the wall below the strip.
    for (x = fx)
        box3([x, cr_plate_back - 0.01, z_bot], [x + cr_foot_w, cr_hook_wall, z_bot + cr_foot_h]);
}

// Probe zones in the cleat frame (used by control_checks.scad).
module cr_hook_zone(x0, x1) { box3([x0, -cleat_t() - 0.01, -3], [x1, 0.01, cr_hook_top]); }
module cr_land_zone(x0, x1) {
    box3([x0, cleat_land_y0() - 0.01, cleat_land_z() - 2],
         [x1, cleat_land_y1(), cleat_land_z() + 0.01]);
}
