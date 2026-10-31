// Assembly scenes for the 77603 throttle station.
//
// Scenes are for views and drawings only. A scene is NEVER a printable part: the builder
// validates individual pieces from the catalog, not these compositions. fit_checks.scad
// reuses the placement functions and modules below for the linkage probes.
//
// ALL GEOMETRY IS PROTOTYPE GEOMETRY. The throttle path below is a DEMONSTRATION: the
// lever pivot, arc, cross-section and height are unmeasured, and no scene here proves fit,
// retention or load capacity. Nothing here modifies the legacy steering geometry.
//
// Placement authority (the scene never re-derives the station):
//   * mg995_throttle_mount.scad publishes the servo base and the horn underside datum;
//     the horn frame sits on the demonstration shaft position mg995_shaft_xy_mm.
//   * throttle_horn_variants.scad fixes the horn frame (origin on the shaft axis, z = 0 at
//     the horn underside) and the rod eye's bore centre per variant.
//   * throttle_pushrod.scad's rod-local frame has its servo eye bore at the origin and
//     both eyes in the x = 0 plane; the rod is placed with rotate([0, -90, 0]) plus a yaw.
//   * throttle_u_fitting.scad publishes u_post_eye_z_mm(), the rod eye's height on the
//     U neck's pivot post.
//
// Neutral pose: the horn arm (shaft to rod eye) is PERPENDICULAR to the rod. An eye on
// the line from the shaft to the throttle would be the toggle point, where small horn
// rotations move the rod least.
include <parameters.scad>
use <mg995_throttle_mount.scad>
use <throttle_pushrod.scad>
use <throttle_u_fitting.scad>
use <throttle_horn_variants.scad>
use <../hitch_arm_layer.scad>   // legacy arm head, as unmodified context

assembly = "throttle_station";
exploded_mm = 0;

// ---- Demonstration geometry ----------------------------------------------------------
// The throttle eye's plan position. X is the owner's idle-pose datum (parameters.scad);
// Y is DERIVED so the station-to-throttle distance equals rods_nominal_span_mm. The real
// lever pivot, arc and height are TO MEASURE.
lever_x_mm = throttle_idle_offset_x_mm;
function throttle_pin_mm() =
    [lever_x_mm,
     tower_world_y() + sqrt(max(0, pow(rods_nominal_span_mm, 2)
                                 - pow(lever_x_mm - tower_world_x(), 2)))];

// The servo output shaft: horn datum plus the demonstration shaft offset in the body.
function shaft_axis_mm() =
    throttle_horn_base_mm() + [mg995_shaft_xy_mm[0], mg995_shaft_xy_mm[1], 0];

// Which side of the shaft each variant's eye sits on at neutral (+1 or -1), chosen so the
// rod's eye plate leaves away from that variant's own hardware.
function neck_side(letter) = letter == "D" ? -1 : 1;

function eye_radius_mm(letter) = norm(variant_rod_eye_xy(letter));
function pin_from_shaft(letter) =
    [throttle_pin_mm()[0] - shaft_axis_mm()[0], throttle_pin_mm()[1] - shaft_axis_mm()[1]];
// Rod setting (servo eye to throttle eye) with the horn arm perpendicular to the rod.
function rod_setting_mm(letter) =
    sqrt(pow(norm(pin_from_shaft(letter)), 2) - pow(eye_radius_mm(letter), 2));
// Plan direction of the rod, from the servo eye toward the throttle eye.
function run_angle_deg(letter) =
    atan2(pin_from_shaft(letter)[1], pin_from_shaft(letter)[0])
    + atan2(neck_side(letter) * eye_radius_mm(letter), rod_setting_mm(letter));
// Plan direction from the shaft to the servo eye at neutral.
function eye_angle_deg(letter) = run_angle_deg(letter) - neck_side(letter) * 90;
// Horn mounting phase that puts the variant's eye at that direction.
function horn_phase_deg(letter) =
    eye_angle_deg(letter)
    - atan2(variant_rod_eye_xy(letter)[1], variant_rod_eye_xy(letter)[0]);
// The servo eye's bore centre; `sweep` rotates the horn about the shaft.
function eye_world_mm(letter, sweep = 0) =
    shaft_axis_mm()
    + [eye_radius_mm(letter) * cos(eye_angle_deg(letter) + sweep),
       eye_radius_mm(letter) * sin(eye_angle_deg(letter) + sweep),
       variant_rod_eye_z(letter)];
// rotate([0, -90, 0]) sends rod-local +Z to -X, so the yaw adds 180 degrees.
function rod_yaw_deg(letter) = run_angle_deg(letter) - 180;
// The U fitting's frame origin, so its post's eye height equals the rod's eye plane.
function u_origin_mm(letter) =
    [throttle_pin_mm()[0], throttle_pin_mm()[1],
     eye_world_mm(letter)[2] - u_post_eye_z_mm()];

// ---- Reference servo ---------------------------------------------------------------
module mg995_reference() {
    // The purchased MG995 as a plain body-envelope placeholder (the real outline, shaft
    // boss and lug shapes are TO MEASURE), centred in the cradle it occupies.
    color([0.92, 0.70, 0.25, 0.55])
        translate(throttle_servo_base_mm()
                  + [-mg995_body_width_mm / 2, -mg995_body_length_mm / 2, 0])
            cube([mg995_body_width_mm, mg995_body_length_mm, mg995_body_height_mm]);
}

module horn_reference(radius_mm = x_horn_arm_radius_mm) {
    // The retained X horn as a DRAWING PLACEHOLDER: a disc carrying the four arm-hole
    // stations the variants use (45 + 90*k phase) with an open hub. It is NOT a real
    // MG995 X horn; radius, arm width, hole diameter and thickness are TO MEASURE.
    color([0.75, 0.76, 0.78, 0.6])
        linear_extrude(height = x_horn_thickness_mm) difference() {
            circle(d = 2 * radius_mm);
            circle(d = 8);
            for (k = [0:3]) rotate([0, 0, 45 + k * 90])
                translate([radius_mm, 0]) circle(d = x_horn_hole_d_mm);
        }
}

module throttle_lever_reference(letter = "E") {
    // Illustrative lever stub under the U. Pivot, arc and the lever's cross-section are
    // UNMEASURED: this section is smaller than the owner's 31.75 x 12.70 inside opening so
    // it demonstrably seats, and it stops just under the U's closed roof.
    translate([throttle_pin_mm()[0] - (throttle_u_inner_width_mm - 6) / 2,
               throttle_pin_mm()[1] - (throttle_u_inner_depth_mm - 3) / 2,
               u_origin_mm(letter)[2] + throttle_u_engagement_mm - 161])
        color([0.42, 0.44, 0.46, 0.55])
            cube([throttle_u_inner_width_mm - 6, throttle_u_inner_depth_mm - 3, 160]);
}

// ---- Linkage, at the published frames -----------------------------------------------
module horn_interface(letter = "E", sweep = 0) {
    translate(shaft_axis_mm())
        rotate([0, 0, horn_phase_deg(letter) + sweep]) {
            horn_reference();
            throttle_horn_variant(letter, "assembly");
        }
}

module rod_place(letter = "E", sweep = 0) {
    translate(eye_world_mm(letter, sweep))
        rotate([0, 0, rod_yaw_deg(letter)])
            rotate([0, -90, 0])
                children();
}

module rod_run(letter = "E") {
    rod_place(letter) throttle_pushrod("assembly", length_mm = rod_setting_mm(letter));
}

// The rod's servo section and its moving throttle end, for the linkage probes.
module rod_servo_end(letter = "E", sweep = 0) {
    rod_place(letter, sweep) throttle_pushrod("servo_eye");
}

module rod_throttle_end_placed(letter = "E") {
    rod_place(letter) throttle_pushrod("throttle_end", length_mm = rod_setting_mm(letter));
}

module throttle_end(letter = "E") {
    // The U fitting drops over the demonstration lever, unrotated: the owner's opening is
    // specified left-right x front-back. Its post carries the rod's throttle eye.
    translate(u_origin_mm(letter)) throttle_u_fitting("assembly");
}

// ---- Complete throttle station -----------------------------------------------------
module throttle_station() {
    // Mount is SELF-PLACED on the head's integral column seat: no extra transform.
    mg995_throttle_mount("assembly");
    mg995_reference();
    // Retained factory steering arm context (unmodified by this work).
    color([0.16, 0.55, 0.60, 0.35]) arm_head_assembly();
    // Linkage: horn interface (station variant E), adjustable rod, throttle-end fitting.
    horn_interface("E");
    rod_run("E");
    throttle_end("E");
    throttle_lever_reference("E");
}

module throttle_mount_service() {
    mg995_throttle_mount("assembly");
    mg995_reference();
}

module throttle_variant_scene(letter = "A", expl = 0) {
    // A variant view shows the COMPLETE throttle station with that horn interface. `expl`
    // separates the stack along the vertical pivot axes: 0 renders assembled; >0 lifts the
    // same parts apart.
    gap = expl;
    mg995_throttle_mount("assembly");
    translate([0, 0, gap * 0.5]) mg995_reference();
    translate([0, 0, gap]) horn_interface(letter);
    translate([0, 0, gap * 1.6]) rod_run(letter);
    // The U lowers away from the rod's throttle eye; the lever stays where it is.
    translate([0, 0, -gap * 0.6]) throttle_end(letter);
    throttle_lever_reference(letter);
    color([0.16, 0.55, 0.60, 0.30]) arm_head_assembly();
}

if (assembly == "throttle_station") throttle_station();
else if (assembly == "throttle_mount_service") throttle_mount_service();
else if (assembly == "throttle_a") throttle_variant_scene("A", exploded_mm);
else if (assembly == "throttle_b") throttle_variant_scene("B", exploded_mm);
else if (assembly == "throttle_c") throttle_variant_scene("C", exploded_mm);
else if (assembly == "throttle_d") throttle_variant_scene("D", exploded_mm);
else if (assembly == "throttle_e") throttle_variant_scene("E", exploded_mm);
else assert(false, str("unknown assembly: ", assembly));
