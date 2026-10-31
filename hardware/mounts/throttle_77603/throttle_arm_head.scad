// Toro 77603 THROTTLE arm head: a derivative of the existing right arm head.
//
// This is not a new bracket beside the design -- it IS the right arm head with the throttle
// station grown out of it, which is what the requirement asks for ("vertically extend and
// structurally reinforce the existing right steering servo bracket").
//
// How it differs from ../hitch_arm_layer.scad's arm_head():
//   * everything from arm_head() is kept unchanged: the joint flange, the rib, the servo-face
//     land with its photo-derived six-hole slot pattern, the spigot pocket and the M6 bolt
//     holes. The original 150 kg steering servo bolts to exactly the same land as before;
//   * the servo-face land is thickened by `reinforcement_t_mm` on its inboard side, so the
//     head is structurally reinforced rather than merely extended;
//   * a gusseted column rises from that reinforced land, in the Y band clear of the steering
//     servo case, up to the MG995 cradle at the anchor declared in parameters.scad.
//
// Unverified: the fit of the real head to the real mower (inherited 77502 datum), and whether
// the added column loads the joint within its capacity. No load qualification exists.
include <../enclosure_common.scad>
use <../hitch_arm_layer.scad>   // modules only: the file has its own entry call
include <parameters.scad>

arm_head_part = "print";

// ---- Throttle column grown from the head's own rib ---------------------------------
// Built in the head's LOCAL frame, because that is the frame throttle_arm_head() lives in and
// the print transform is applied outside it. The rib there is
//   translate([0, -arm_col_w/2, 0]) cube([servo_face_dx()-servo_face_t, arm_col_w, h])
// i.e. x 0..16.25, y -35..35. Two earlier attempts used the PRINT frame instead and produced
// two shells, because print-frame constants were then rotated a second time by the wrapper.
rib_len = servo_face_dx() - servo_face_t;   // 16.25
throttle_post_z = arm_head_rise() + throttle_above_servo_mm;
post_over = 2.0;   // overlap into the rib so the solids fuse, never coplanar

module throttle_column() {
    // Off the shaft centreline, toward one end of the rib band, clear of the servo case.
    y_post = arm_col_w / 2 - post_w_mm - 2;
    difference() {
        union() {
            // Post starts INSIDE the rib (x < rib_len) so it is one solid with the head.
            translate([rib_len - post_over, y_post - post_w_mm / 2, arm_head_rise() - 30])
                cube([post_w_mm, post_w_mm, throttle_post_z - arm_head_rise() + 30]);
            // Outboard arm carrying the MG995 cradle.
            translate([rib_len - post_w_mm - post_over, y_post - post_w_mm / 2,
                       throttle_post_z - foot_t_mm])
                cube([tower_offset_x, post_w_mm, foot_t_mm]);
            // Gusset tying the arm back down the post.
            for (side = [-1, 1])
                translate([rib_len - post_w_mm - post_over,
                           y_post + side * (post_w_mm / 2 - 1.5),
                           throttle_post_z - foot_t_mm])
                    linear_extrude(height = 3)
                        polygon([[0, 0], [tower_offset_x * 0.5, 0], [0, -50]]);
        }
        // Lightening window, clear of the rib below and the arm above.
        translate([rib_len - post_w_mm - post_over - 1, y_post - 5,
                   arm_head_rise() + 25])
            cube([post_w_mm + 2, 10, throttle_post_z - arm_head_rise() - 80]);
    }
}

// ---- Reinforced arm head -----------------------------------------------------------
module throttle_arm_head() {
    difference() {
        union() {
            // The ORIGINAL head, unchanged.
            arm_head();
            // Reinforcement web across the rib/land junction: overlaps the rib's tail so it
            // fuses with the existing head.
            translate([rib_len - reinforcement_t_mm - 1, -arm_col_w / 2, 0])
                cube([reinforcement_t_mm + 1, arm_col_w, arm_head_rise() + 10]);
            throttle_column();
        }
        // Keep the original servo bolt passages and shaft opening clear through the web.
        translate([rib_len - reinforcement_t_mm - 2, 0, arm_head_rise()])
            rotate([0, 90, 0]) servo_rear_hole_cut(reinforcement_t_mm + 4);
        translate([rib_len - reinforcement_t_mm - 2, 0, arm_head_rise()])
            rotate([0, 90, 0]) servo_rear_opening_cut(reinforcement_t_mm + 4);
    }
}

// Print orientation matches the legacy head (mating face down), so the piece is bed-zero.
module throttle_arm_head_print() {
    translate([-arm_head_min_x(), arm_joint_flange / 2, 0])
        rotate([0, -arm_rake_deg, 0]) throttle_arm_head();
}

throttle_arm_head_print();
