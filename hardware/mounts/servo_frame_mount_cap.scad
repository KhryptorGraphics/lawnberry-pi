// Closing half of the frame-tube clamp used by servo_frame_mount.scad.
// Print TWO (one per servo mount). Captive M6 nuts drop into the pockets;
// bolts pass through from the mount side.
//
//   openscad -o servo_frame_mount_cap.stl servo_frame_mount_cap.scad

include <common.scad>

clamp_width = 40;   // must match servo_frame_mount.scad

rotate([0, 180, 0])
    tube_clamp_half(frame_tube_od, clamp_width, half = 1);
