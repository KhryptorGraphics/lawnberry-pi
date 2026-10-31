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
//   * a buttressed column rises out of the rib itself, in the Y band clear of the steering
//     servo case, up to the MG995 cradle at the anchor declared in parameters.scad. It is
//     buried 2 mm into the rib, so the head is structurally reinforced rather than merely
//     extended: the exported mesh is 141,100 mm3 (+47.1%) heavier than the original
//     arm_head_right (440,689 vs 299,589 mm3, measured from both exports). That is ADDED
//     MATERIAL, not a strength calculation: no load qualification exists for the extended arm,
//     and the buttress dimensions were chosen for manufacturability, not for the bending
//     moment of the cantilever.
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
// GEOMETRY-DRIVEN, NOT LOAD-DERIVED. These two were chosen so the profile stays a simple
// polygon that triangulates reproducibly; nothing here sized them for the bending moment of a
// 362 mm cantilever. They are not a structural calculation and must not be cited as one.
// Sizing this buttress properly needs a stated throttle-actuation load, which does not exist
// yet, so any value here is provisional scaffolding around the real question.
gusset_run_mm = 55;    // geometry-driven: buttress foot along the arm's underside
gusset_drop_mm = 90;   // geometry-driven: buttress root down the post from the arm

module throttle_column() {
    // The post, the outboard arm and the buttress are ONE extruded profile. Modelling them as
    // separate cubes in a union left coplanar faces wherever they met, and OpenSCAD's
    // triangulation of those coincident faces is not reproducible: about three renders in four
    // failed the manifold gate with edges shared by four and six triangles. A single simple
    // polygon has no internal faces at all, so it is manifold by construction.
    //
    // The profile's bottom edge sits at arm_head_rise(), inside the head's rib, so the column
    // fuses with the head instead of butting against it.
    y_post = arm_col_w / 2 - post_w_mm - 2;   // off the shaft centreline, clear of the case
    x_arm0 = rib_len - post_w_mm - post_over; // arm's inboard end, inside the rib
    x_arm1 = x_arm0 + tower_offset_x;         // arm's outboard end
    x_post0 = rib_len - post_over;            // post inboard face, inside the rib
    x_post1 = x_post0 + post_w_mm;            // post outboard face
    z_arm0 = arm_head_rise();                 // bottom of the profile, buried in the rib
    z_top = throttle_post_z;
    z_bot = z_top - foot_t_mm;                // arm underside
    x_toe = x_post1 + gusset_run_mm;          // buttress foot, along the arm underside
    z_toe = z_bot - gusset_drop_mm;           // buttress root, down the post

    // The profile is drawn in (X, Z) and must be extruded along Y, the head's width axis.
    // linear_extrude() always extrudes along +Z from an XY polygon, so an unrotated call
    // silently builds the profile in the WRONG plane: the column ended up 16 mm thick in Z
    // with its 362 mm length lying in Y, which also pushed Y past the head's own 88 mm.
    // rotate([90,0,0]) maps polygon-y to +Z and the extrusion height to -Y.
    translate([0, y_post + post_w_mm / 2, 0])
        rotate([90, 0, 0])
            linear_extrude(height = post_w_mm)
                polygon([
                    [x_arm0, z_arm0],
                    [x_arm0, z_top],
                    [x_arm1, z_top],
                    [x_arm1, z_bot],
                    [x_toe,  z_bot],
                    [x_post1, z_toe],
                    [x_post1, z_arm0],
                ]);
}

// ---- Reinforced arm head -----------------------------------------------------------
module throttle_arm_head() {
    // The ORIGINAL right arm head with the throttle column grown out of its rib.
    //
    // There is deliberately no "reinforcement web" here. An earlier version added a cube at
    // x 9.25..16.25, y -35..35, z 0..159.75, which is ENTIRELY INSIDE the rib
    // (x 0..16.25, y -35..35, z 0..179.75): it thickened nothing while sharing three large
    // coplanar face planes with the rib's y=+/-35 sides and its z=0 bottom. Those coincident
    // faces produced edges shared by four and six triangles, and the resulting
    // non-reproducible triangulation failed the manifold gate in roughly three renders out of
    // four. The reinforcement that actually exists is the column itself, which is buried
    // post_over into the rib and, with it, makes the exported mesh 141,100 mm3 (+47.1%)
    // heavier than arm_head_right.
    union() {
        arm_head();
        throttle_column();
    }
}

// Print orientation matches the legacy head (mating face down), so the piece is bed-zero.
module throttle_arm_head_print() {
    translate([-arm_head_min_x(), arm_joint_flange / 2, 0])
        rotate([0, -arm_rake_deg, 0]) throttle_arm_head();
}

throttle_arm_head_print();
