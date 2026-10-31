// Toro 77603 THROTTLE arm head: the existing right arm head plus an integral column that
// carries the throttle MG995 directly above the steering station.
//
// WHAT THIS PART IS
//   The legacy right arm head with one addition: a column that grows out of the head's own
//   servo-face land, in the forward band that the retained steering case and moving bracket
//   leave free, and rises to carry the throttle servo. It REPLACES arm_head_right. It is not
//   a bracket beside the design.
//
// WHAT IS PRESERVED, unchanged from ../hitch_arm_layer.scad's arm_head():
//   * the joint flange, its four M6 passages and the spigot pocket;
//   * the upright rib;
//   * the servo-face land with its photo-derived six-slot pattern and rear opening, so the
//     original steering servo bolts to exactly the same land with the same through-bolts.
//   No steering joint is drilled and no steering datum moves.
//
// WHY THE COLUMN IS SHAPED THIS WAY
//   The retained steering gear, measured from the real legacy solids (not assumed):
//     * servo case                world X 218..266, Y 108..138, Z 378..443
//     * moving bracket at 0/45/90 deg   X 216..268, Y 94..144, Z 409..459
//     * steering horn sweep       about X 273..303, Z 375..485
//     * steering pushrod          world X 274..370, Z 465..505
//   The column:
//     * sits in the land's forward band, world Y 145..157, which is forward of the moving
//       bracket's swept Y (144) and clear of the servo case band (138);
//     * stays inboard of world X 234, so it never enters the horn's X band (273) or the
//       steering pushrod's X band (274);
//     * rises vertically, which is also how it prints: the head prints mating-face down, and
//       a vertical column needs no support. An earlier revision grew a 190 mm horizontal
//       cantilever out of this part, which left it floating above the bed and outside any
//       reasonable overhang length; the reach is now short because the throttle servo sits
//       above the steering station rather than outboard of it.
//
// GEOMETRY-DRIVEN, NOT LOAD-DERIVED
//   The column's section is sized by the clearance envelope above and by printability. Its
//   STRENGTH is not qualified: no throttle-actuation load has been stated, so nothing here is
//   a load case, and the printed stack above it carries the same caveat.
//
// Unverified: fit to the real mower (inherited 77502 datum) and joint capacity under load.
include <../enclosure_common.scad>
use <../hitch_arm_layer.scad>   // modules only: the file has its own entry call
use <printed_joints.scad>       // pin bore helper
include <parameters.scad>

arm_head_part = "print";

// ---- Head-local frame ---------------------------------------------------------------
// The head's local frame: X runs outboard along the arm, Y across it, Z up the rib. The
// rib is translate([0,-arm_col_w/2,0]) cube([rib_len,arm_col_w,h]), i.e. x 0..16.25,
// y -35..35; the servo-face land is the last 6 mm of X, full 70 mm width.
rib_len = servo_face_dx() - servo_face_t;
land_top_local = arm_tip_z - arm_joint_z + servo_face_h / 2;   // 179.75
land_face_local = servo_face_dx();                              // 22.25

// The column: outboard of the land's face, in the forward band, rising to the cradle seat.
// It overlaps the rib and the land by 1 mm so the solids fuse rather than butt.
// The column's footprint comes from the shared parameters, so the head and the mount cannot
// drift apart: an earlier revision had the two files compute it differently and the column
// silently lost 6 mm of its rib root.
column_x0 = column_x0_local_mm;
column_x1 = column_x1_local_mm;
column_y0 = tower_offset_y - column_w_y_mm / 2;
column_y1 = tower_offset_y + column_w_y_mm / 2;
column_z0 = column_z0_local_mm;
column_z1 = column_z1_local_mm;

module throttle_support_column() {
    // ONE difference over the whole column. An earlier version built the column as a union
    // and then appended a separate difference() block for the seat socket and the pin bore:
    // that made the socket cut a SECOND solid rather than a subtraction, so the column was
    // never socketed, never bored, and interfered with the cradle's tongue.
    difference() {
        union() {
            // Narrow foot: rises from the land in the forward band, clear of the steering
            // servo's moving bracket at every angle.
            translate([column_x0, column_y0, column_z0])
                cube([column_width_mm, column_y1 - column_y0,
                      column_widen_local_mm - column_z0]);
            // Wide head: above the bracket's swept band (world Z 458.8) the column carries
            // the whole cradle instead of letting it overhang a 12 mm stem.
            translate([column_x0 + column_overhang_mm, tower_offset_y - cradle_len_mm / 2,
                       column_widen_local_mm])
                cube([column_width_mm, cradle_len_mm,
                      column_z1 - column_widen_local_mm]);
        }
        // The dowels locate the cradle; a transverse printed pin captures them.
        for (dx = [-seat_dowel_x_offset_mm, seat_dowel_x_offset_mm])
            translate([seat_x_local_mm + dx, seat_y_local_mm + seat_dowel_y_offset_mm,
                       column_z1 - seat_dowel_depth_mm])
                cylinder(d = seat_dowel_d_mm + 2 * seat_dowel_clearance_mm,
                         h = seat_dowel_depth_mm + 1);
        translate([seat_x_local_mm, seat_y_local_mm + seat_dowel_y_offset_mm,
                   column_z1 - seat_dowel_depth_mm / 2])
            rotate([0, 90, 0])
                cylinder(d = seat_cross_hole_d_mm, h = seat_cross_pin_len_mm + 2,
                         center = true);
    }
}

module throttle_support_gusset() {
    // Triangular buttress on the inboard face of the column, tying it into the rib. Drawn in
    // (X, Z) and extruded along Y; rotate([-90,0,0]) maps polygon-y to -Z, so the profile's
    // height is negated to make the gusset rise toward the rib.
    y0 = column_y0 + 1;
    translate([column_x0, y0, column_z0 + column_gusset_h_mm])
        rotate([-90, 0, 0])
            linear_extrude(height = column_w_y_mm - 2)
                polygon([[0, 0],
                         [0, -column_gusset_h_mm],
                         [-column_gusset_run_mm, 0]]);
}

module throttle_arm_head() {
    union() {
        arm_head();
        throttle_support_column();
        throttle_support_gusset();
    }
}

// Print orientation matches the legacy head (mating face down), so the piece is bed-zero.
module throttle_arm_head_print() {
    translate([-arm_head_min_x(), arm_joint_flange / 2, 0])
        rotate([0, -arm_rake_deg, 0]) throttle_arm_head();
}

throttle_arm_head_print();
