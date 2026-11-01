// Toro 77603 THROTTLE arm head: the right arm head that carries the throttle MG995 BELOW the
// raised right steering servo.
//
// WHAT THIS PART IS
//   The legacy right arm head -- raised, with the left head, by servo_station_raise_mm
//   (../enclosure_common.scad) -- plus a shelf grown from the rib's outboard face under the
//   steering servo, a 45-degree corbel that carries the shelf back to the rib, and the MG995
//   cradle walls standing on the shelf. It REPLACES arm_head_right. One printed piece: the
//   shelf is the cradle's floor, so there is no seat joint between the head and the cradle.
//
// WHAT IS PRESERVED, unchanged from ../hitch_arm_layer.scad's arm_head():
//   * the joint flange, its four M6 passages and the spigot pocket;
//   * the upright rib;
//   * the servo-face land with its photo-derived six-slot pattern and rear opening, so the
//     steering servo bolts to the same land with the same through-bolts.
//   The shelf and corbel only add material; a probe and a unit test subtract this part from
//   arm_head() and require nothing to remain.
//
// WHY THE SHELF IS SHAPED THIS WAY
//   * The throttle stack (cradle, servo, horn interface and the rod's servo end) stands under
//     the steering servo's case, which hangs below the raised station; the raise was chosen
//     so the stack clears the case, its moving bracket and the steering adapter's sweep.
//     The probes in fit_checks.scad check those clearances on the real solids.
//   * The servo's shaft stands throttle_seat_x_local_mm outboard of the joint, so every horn
//     interface and rod servo section swept +/-30 degrees clears the rib.
//   * The corbel's underside rises at exactly 45 degrees from its root on the rib to the
//     shelf's outboard edge, and the shelf's wings beyond the corbel's width fall at 45
//     degrees to it, so the part still prints mating-face down without support. The corbel
//     is narrower than the shelf and the wings start high, which leaves tool room above the
//     joint's two outboard M6 bolts (probed).
//
// GEOMETRY-DRIVEN, NOT LOAD-DERIVED
//   The shelf and corbel are sized by the clearance envelope and by printability. Their
//   STRENGTH is not qualified: no throttle-actuation load has been stated, so nothing here is
//   a load case.
//
// Unverified: fit to the real mower (inherited 77502 datum) and joint capacity under load.
use <../hitch_arm_layer.scad>    // modules only: the file has its own entry call
use <mg995_throttle_mount.scad>  // cradle_walls(); modules only
include <parameters.scad>

// The shelf slab under the cradle, in the head's local frame.
module shelf_slab(half_w, t) {
    translate([shelf_x0_local_mm, throttle_seat_y_local_mm - half_w, throttle_seat_local_mm - t])
        cube([shelf_x1_local_mm - shelf_x0_local_mm, 2 * half_w, t]);
}

// The shelf: a CORBEL at the corbel's width, rooted on the rib's face, and two WINGS out to
// the cradle's length. The root strip sits 1 mm inside the rib and the slab reaches 1 mm into
// it, so they fuse. Every downward face is at most 45 degrees from vertical in this frame.
module throttle_shelf() {
    corbel_w = corbel_half_w_mm;
    shelf_w = shelf_y1_local_mm - throttle_seat_y_local_mm;
    // Corbel: the slab at the corbel's width and the root strip on the rib.
    hull() {
        shelf_slab(corbel_w, shelf_t_mm);
        translate([shelf_x0_local_mm, throttle_seat_y_local_mm - corbel_w, corbel_z0_local_mm])
            cube([rib_face_local_mm - shelf_x0_local_mm, 2 * corbel_w,
                  throttle_seat_local_mm - corbel_z0_local_mm]);
    }
    // Wings: the full slab and the corbel's part that reaches the wings' depth. That part
    // ends where the corbel's 45-degree underside rises to it, so the wings' outboard
    // underside continues the corbel's slope instead of hanging below it.
    hull() {
        shelf_slab(shelf_w, shelf_t_mm);
        translate([shelf_x0_local_mm, throttle_seat_y_local_mm - corbel_w,
                   throttle_seat_local_mm - shelf_t_mm - shelf_wing_drop_mm])
            cube([shelf_x1_local_mm - shelf_wing_drop_mm - shelf_x0_local_mm, 2 * corbel_w,
                  shelf_t_mm + shelf_wing_drop_mm]);
    }
}

module throttle_arm_head() {
    union() {
        arm_head();
        throttle_shelf();
        // The walls root into the slab, so they fuse with the shelf rather than butt on it.
        translate([throttle_seat_x_local_mm, throttle_seat_y_local_mm, throttle_seat_local_mm])
            cradle_walls(root = shelf_t_mm);
    }
}

// Print orientation matches the legacy head (mating face down), so the piece is bed-zero.
module throttle_arm_head_print() {
    translate([-arm_head_min_x(), arm_joint_flange / 2, 0])
        rotate([0, -arm_rake_deg, 0]) throttle_arm_head();
}

throttle_arm_head_print();
