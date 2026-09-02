// Lap-bar pushrod saddle.
//
// Seats on the motion-control (lap bar) tube, held by two U-bolts, and gives
// the pushrod a single ball-joint anchor. Print TWO.
//
// WHY A U-BOLT SADDLE AND NOT A SPLIT CLAMP
// Toro does not publish the lap-bar tube OD, and universal ZTR accessory
// brackets advertise a 0.8"-2" fit range, so a fixed-diameter printed clamp
// would be a guess with no tolerance. A U-bolt saddle takes whatever the tube
// actually measures (set lapbar_tube_od in common.scad), and puts the clamping
// preload through steel instead of printed ears.
//
// CRITICAL GEOMETRY CONSTRAINT
// Per the Toro operator's manual (form 3465-589) the levers move on TWO axes:
//   * fore/aft  -- Fast / Slow / NEUTRAL / Reverse  (what the servo drives)
//   * outboard  -- PARK, which sets the parking brake AND is required by the
//                  safety-interlock system before the engine will crank.
// The linkage must NEVER obstruct the outboard swing. Two features protect it:
//   1. anchor_offset places the ball joint INBOARD, out of the swing arc.
//   2. Ball joints at both rod ends plus a quick-release clevis pin here, so
//      the linkage unpins in seconds to restore full manual PARK travel.
// Verify the full outboard sweep by hand, linkage fitted, before running.
//
//   openscad -o lapbar_pushrod_clamp.stl lapbar_pushrod_clamp.scad

include <common.scad>

saddle_l      = 78;    // along the lap-bar tube
// Computed so the U-bolt legs keep `wall` of material outboard of them --
// see saddle_width() in common.scad. Do not hardcode this.
saddle_w      = saddle_width(lapbar_tube_od, ubolt_leg_lapbar);
body_t        = 20;    // saddle body height before the seat is cut
ubolt_span    = 52;    // between U-bolt centres
// Must clear the saddle body, which is itself computed -- an offset smaller
// than saddle_w/2 buries the eye inside the block instead of standing it off.
anchor_reach  = 22;    // how far the eye stands proud of the saddle edge
anchor_offset = saddle_w / 2 + anchor_reach;
anchor_thk    = 12;
eye_bore      = m6_clear;   // M6 ball joint / clevis pin
eye_od        = 20;

// Arm reaching inboard from the saddle, ending in a vertical-axis eye.
// The pin axis is deliberately perpendicular to both the lap-bar tube and the
// arm, so the rod end swings in the plane the lever travels in.
module anchor_arm() {
    difference() {
        hull() {
            translate([0, -saddle_w / 2 + 3, 0])
                cube([eye_od * 1.15, 6, anchor_thk], center = true);
            translate([0, -anchor_offset, 0])
                cylinder(h = anchor_thk, d = eye_od, center = true);
        }
        translate([0, -anchor_offset, 0])
            cylinder(h = anchor_thk + 2, d = eye_bore, center = true);
    }
}

module lapbar_pushrod_clamp() {
    seat_z = body_t - lapbar_tube_od / 2;
    difference() {
        union() {
            // Saddle body
            translate([0, 0, body_t / 2])
                cube([saddle_l, saddle_w, body_t], center = true);
            // Arm out to the ball-joint eye, inboard and clear of the swing
            translate([0, 0, anchor_thk / 2]) anchor_arm();
        }
        // Tube seat along the saddle's long axis
        translate([0, 0, body_t])
            rotate([0, 0, 0])
                rotate([0, 90, 0])
                    cylinder(h = saddle_l + 2, d = lapbar_tube_od + clearance * 2,
                             center = true);
        // Clear everything above the tube centreline so the saddle is a cradle
        translate([0, 0, body_t + (lapbar_tube_od + 20) / 2])
            cube([saddle_l + 2, saddle_w + 2, lapbar_tube_od + 20],
                 center = true);
        // U-bolt legs, straddling the tube
        for (x = [-1, 1], y = [-1, 1])
            translate([x * ubolt_span / 2,
                       y * ubolt_leg_offset(lapbar_tube_od, ubolt_leg_lapbar),
                       -1])
                cylinder(h = body_t + 2, d = ubolt_leg_lapbar);
    }
}

lapbar_pushrod_clamp();
