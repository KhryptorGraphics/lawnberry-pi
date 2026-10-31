// Projected 2D geometry for the annotated assembly drawings.
//
// The builder renders these as SVG through OpenSCAD's 2D projection, then overlays the
// dimension/label/watermark annotations. A drawing therefore contains real projected
// geometry, not a screenshot of a 3D view.
include <parameters.scad>
use <mg995_throttle_mount.scad>
use <throttle_pushrod.scad>
use <throttle_u_fitting.scad>
use <throttle_horn_variants.scad>

assembly = "throttle_a";
projection = "top";

module throttle_u_outline() {
    // Nominal opening outline with the owner's dimensions called out by the annotation pass.
    difference() {
        translate([-throttle_u_inner_width_mm / 2 - throttle_u_wall_mm,
                   -throttle_u_inner_depth_mm / 2 - throttle_u_wall_mm])
            square([throttle_u_inner_width_mm + 2 * throttle_u_wall_mm,
                    throttle_u_inner_depth_mm + 2 * throttle_u_wall_mm]);
        square([throttle_u_inner_width_mm, throttle_u_inner_depth_mm], center = true);
    }
}

module saddle_section_outline() {
    x_in = saddle_x_in();
    y_out = saddle_y_out();
    translate([x_in, -y_out - saddle_side_w_mm])
        square([saddle_back_t_mm, 2 * (y_out + saddle_side_w_mm)]);
    square([2 * x_in + saddle_back_t_mm, saddle_side_w_mm]);
}

module rod_section_outline() {
    difference() {
        square([rods_w_mm, rods_w_mm], center = true);
        square([rods_bore_mm, rods_bore_mm], center = true);
    }
}

if (projection == "top") {
    // Plan view of the throttle end: opening, wall, and the rod section.
    translate([0, -30]) throttle_u_outline();
    translate([0, 0]) rod_section_outline();
} else if (projection == "section") {
    // Cross-section of the saddle wrap around the legacy column envelope.
    saddle_section_outline();
    color([0.9, 0.9, 0.9]) square([arm_col_d, arm_col_w], center = false);
} else {
    assert(false, str("unknown projection: ", projection));
}
