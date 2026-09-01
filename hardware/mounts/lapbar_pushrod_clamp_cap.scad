// Closing half of the lap-bar clamp used by lapbar_pushrod_clamp.scad.
// Print TWO. Captive M5 nuts.
//
//   openscad -o lapbar_pushrod_clamp_cap.stl lapbar_pushrod_clamp_cap.scad

include <common.scad>

clamp_width = 30;   // must match lapbar_pushrod_clamp.scad

rotate([0, 180, 0])
    tube_clamp_half(lapbar_tube_od, clamp_width, ear_bolt = m5_clear,
                    ear_len = 13, half = 1);
