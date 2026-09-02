// Shared parameters and helper modules for the LawnBerry Pi zero-turn
// actuation mounts (Toro TimeCutter 50" conversion).
//
// DESIGN BASIS -- researched, not guessed:
//   * Frame is 3" x 1.5" x 0.120" wall RECTANGULAR steel tube (Toro published
//     spec for the TimeCutter/TimeCutter MAX carrier frame). It is NOT round,
//     so frame mounts use a saddle plate + U-bolts, not a round clamp.
//   * Lap-bar tube OD is NOT published by Toro. Universal ZTR accessory
//     brackets quote a 0.8"-2" fit range, so the lap-bar saddle is
//     parametric and U-bolted rather than a fixed-diameter split clamp.
//
// U-bolts do the clamping everywhere. That is deliberate: a printed clamp ear
// carrying the preload of a bolt is the weakest possible arrangement, and
// U-bolts also absorb the dimensional uncertainty above. Printed parts here
// only locate and spread load; steel takes the tension.

$fn = 64;

// ------------------------------------------------------------ machine (known)
// Toro published carrier-frame spec, converted from 3" x 1.5" x 0.120".
frame_rail_w      = 76.2;   // 3"
frame_rail_h      = 38.1;   // 1.5"

// MEASURE: lap-bar tube OD. 25.4 (1") is the common ZTR size and a sane
// starting guess, but Toro does not publish it -- calipers, then re-render.
lapbar_tube_od    = 25.4;

// ----------------------------------------------------------------- servo
// DSSERVO RDS51150SG, manufacturer spec: 65 x 30 x 48 mm, 165 kg.cm @ 12V,
// 0.21 s/60deg. Ships with U-shaped aluminium holders; these mounts bolt to
// those holders rather than gripping the servo body.
servo_body_l      = 65;
servo_body_w      = 30;
servo_body_h      = 48;
// MEASURE: hole spacing on the supplied U-bracket. Slots absorb error.
servo_bracket_hole_spacing_x = 50;
servo_bracket_hole_spacing_y = 20;

// --------------------------------------------------------------- fixings
m3_clear          = 3.4;
m4_clear          = 4.5;
m5_clear          = 5.5;
m6_clear          = 6.4;
m8_clear          = 8.4;

// U-bolt leg diameter. 5/16" (7.94) is typical for a 3" square U-bolt;
// 1/4" (6.35) for a 1" round one.
ubolt_leg_frame   = 8.4;    // 5/16" + clearance
ubolt_leg_lapbar  = 6.8;    // 1/4"  + clearance

// ----------------------------------------------------------------- print
wall              = 5.0;    // structural wall; do not thin below 4 mm
saddle_wall       = 8.0;    // under a U-bolt, where load concentrates
clearance         = 0.4;    // printer fit clearance, tune per machine

// ---------------------------------------------------------------- helpers

// Slotted hole -- absorbs mounting error without a reprint.
module slot(d, len, h) {
    hull() {
        translate([-len / 2, 0, 0]) cylinder(h = h, d = d);
        translate([ len / 2, 0, 0]) cylinder(h = h, d = d);
    }
}

// Rounded plate, centred on the origin, growing in +Z.
module plate(l, w, t, r = 6) {
    hull()
        for (x = [-1, 1], y = [-1, 1])
            translate([x * (l / 2 - r), y * (w / 2 - r), 0])
                cylinder(h = t, r = r);
}

// Centre-line offset of a U-bolt leg from the tube axis: half the tube, half
// the leg, fit clearance, plus a little standoff so the leg does not scrub
// the tube corner.
function ubolt_leg_offset(tube_across, leg) =
    tube_across / 2 + leg / 2 + clearance + 1.5;

// Minimum saddle width that keeps `wall` of material outboard of the U-bolt
// legs. Computed, not hardcoded -- getting this wrong breaks the holes out
// through the edge of the part.
function saddle_width(tube_across, leg) =
    2 * (ubolt_leg_offset(tube_across, leg) + leg / 2 + wall);

// Saddle that seats against a flat face of the rectangular frame rail and is
// held by two U-bolts wrapping the rail. Solid; subtract frame_ubolt_slots()
// and add your own boss.
//   span = distance between the two U-bolt centre lines
module frame_saddle(l, t, span, leg = ubolt_leg_frame) {
    difference() {
        plate(l, saddle_width(frame_rail_h, leg), t, r = 8);
        // Shallow relief so the saddle beds onto the rail face rather than
        // rocking on a flat-to-flat contact.
        translate([0, 0, -0.01])
            cube([l + 2, frame_rail_h + clearance * 2, 1.2], center = true);
    }
}

// The two U-bolt leg pairs for a frame saddle. Subtract from the saddle.
module frame_ubolt_slots(t, span, leg = ubolt_leg_frame) {
    for (x = [-1, 1], y = [-1, 1])
        translate([x * span / 2,
                   y * ubolt_leg_offset(frame_rail_h, leg),
                   -1])
            cylinder(h = t + 2, d = leg);
}

// Curved seat matching a round tube, for the lap-bar saddle.
module tube_seat(od, len, depth_frac = 0.5) {
    r = od / 2 + clearance;
    intersection() {
        rotate([0, 90, 0]) cylinder(h = len, r = r, center = true);
        translate([0, 0, -r * (1 - depth_frac) - r])
            cube([len + 2, od + 4, r * 2], center = true);
    }
}
