// Lap-bar pushrod clamp.
//
// Grips the motion-control (lap bar) tube and gives the pushrod a single
// ball-joint anchor point. Print TWO (mirror is not required -- the part
// is symmetric; set anchor_side per lever).
//
// CRITICAL GEOMETRY CONSTRAINT
// Per the Toro operator's manual (form 3465-589), the motion-control
// levers move on TWO axes:
//   * fore/aft  -- Fast / Slow / NEUTRAL / Reverse  (what the servo drives)
//   * outboard  -- PARK, which engages the parking brake AND is required
//                  by the safety-interlock system before the engine will
//                  crank.
// The servo linkage must NEVER obstruct the outboard swing. Two features
// here protect that:
//   1. anchor_offset pushes the ball joint INBOARD of the lever, out of
//      the outboard swing arc.
//   2. The pushrod uses ball/heim joints at both ends plus a quick-release
//      clevis pin at this end, so the linkage can be unpinned in seconds
//      to restore full manual PARK travel.
// Verify the full outboard swing by hand, with the linkage fitted, before
// running the engine.
//
//   openscad -o lapbar_pushrod_clamp.stl lapbar_pushrod_clamp.scad

include <common.scad>

clamp_width    = 30;    // along the lap-bar tube
anchor_offset  = 22;    // tube centre -> ball-joint eye, INBOARD
anchor_thk     = 10;    // eye thickness
eye_bore       = m6_clear;   // M6 ball joint / clevis pin
eye_od         = 18;

module anchor_eye() {
    difference() {
        hull() {
            cylinder(h = anchor_thk, d = eye_od, center = true);
            translate([-anchor_offset * 0.6, 0, 0])
                cube([1, eye_od, anchor_thk], center = true);
        }
        cylinder(h = anchor_thk + 2, d = eye_bore, center = true);
    }
}

module lapbar_pushrod_clamp() {
    r_out = lapbar_tube_od / 2 + clamp_wall;
    union() {
        tube_clamp_half(lapbar_tube_od, clamp_width, ear_bolt = m5_clear,
                        ear_len = 13, half = 0);
        // Arm carrying the eye, offset inboard and clear of the swing
        translate([anchor_offset, 0, 0]) anchor_eye();
        hull() {
            translate([r_out - 2, 0, 0])
                cube([1, eye_od * 0.9, clamp_width], center = true);
            translate([anchor_offset, 0, 0])
                cylinder(h = anchor_thk, d = eye_od * 0.9, center = true);
        }
    }
}

lapbar_pushrod_clamp();
