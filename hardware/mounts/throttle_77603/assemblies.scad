// Assembly scenes for the 77603 throttle station.
//
// Scenes are for views and drawings only. A scene is NEVER a printable part: the builder
// validates individual pieces from the catalog, not these compositions.
//
// Steering geometry is pulled from the legacy datum for context. Nothing here modifies it.
include <parameters.scad>
use <mg995_throttle_mount.scad>
use <throttle_pushrod.scad>
use <throttle_u_fitting.scad>
use <throttle_horn_variants.scad>
use <../hitch_arm_layer.scad>   // legacy arm head, as unmodified context

assembly = "throttle_station";
exploded_mm = 0;

// ---- Reference servo ---------------------------------------------------------------
module mg995_reference() {
    // The purchased MG995 as context, placed in the cradle it actually occupies: at the tower
    // axis, on top of the riser (foot + riser + cradle floor).
    color([0.92, 0.70, 0.25, 0.55])
        translate([servo_station_x + tower_offset_x, servo_shaft_y() + tower_offset_y,
                   arm_tip_z + foot_t_mm + riser_z_mm + cradle_t_mm])
            mg995_body_envelope();
}

module horn_reference(radius_mm = x_horn_arm_radius_mm) {
    // The retained X horn: four arms, central hub, factory retainer land.
    color([0.75, 0.76, 0.78, 0.6])
        linear_extrude(height = x_horn_thickness_mm) difference() {
            circle(d = 2 * radius_mm);
            circle(d = 8);
            for (k = [0:3]) rotate([0, 0, 45 + k * 90])
                translate([radius_mm, 0]) circle(d = x_horn_hole_d_mm);
        }
}

module throttle_lever_reference() {
    // Illustrative lever stub at the derived world X. Pivot, arc and the lever's actual
    // cross-section are UNMEASURED: this section is deliberately smaller than the owner's
    // 31.75 (left-right) x 12.70 (front-back) inside opening so it demonstrably seats, and it
    // is vertical so the U drops over it from above.
    color([0.42, 0.44, 0.46, 0.55])
        translate([throttle_idle_offset_x_mm - (throttle_u_inner_width_mm - 6) / 2,
                   tower_offset_y - (throttle_u_inner_depth_mm - 3) / 2,
                   arm_tip_z - 40])
            cube([throttle_u_inner_width_mm - 6, throttle_u_inner_depth_mm - 3, 160]);
}

// ---- Complete throttle station -----------------------------------------------------
module throttle_station() {
    // Mount is SELF-PLACED at the right steering servo land: no extra transform.
    mg995_throttle_mount("assembly");
    mg995_reference();
    // Retained factory steering arm context (unmodified by this work).
    color([0.16, 0.55, 0.60, 0.35]) arm_head_assembly();
    // Linkage: horn interface, adjustable rod, throttle-end fitting.
    translate([servo_station_x + tower_offset_x, servo_shaft_y() + tower_offset_y,
               arm_tip_z + foot_t_mm + riser_z_mm + cradle_t_mm
               + mg995_body_height_mm + mg995_shaft_projection_mm])
        throttle_horn_variant("E", "assembly");
    translate([servo_station_x + tower_offset_x,
               servo_shaft_y() + tower_offset_y,
               arm_tip_z + foot_t_mm + riser_z_mm + cradle_t_mm
               + mg995_body_height_mm + mg995_shaft_projection_mm])
        throttle_pushrod("assembly");
    translate([throttle_idle_offset_x_mm, tower_offset_y,
               arm_tip_z + foot_t_mm + riser_z_mm + cradle_t_mm + mg995_body_height_mm])
        throttle_u_fitting("body");
    throttle_lever_reference();
}

module throttle_mount_service() {
    mg995_throttle_mount("assembly");
    mg995_reference();
}

module throttle_variant_scene(letter = "A", expl = 0) {
    // A variant view shows the COMPLETE throttle station with that horn interface, so the
    // owner can judge whether it fits. `expl` separates the stack along its assembly axis:
    // 0 renders assembled; >0 renders the same parts pulled apart (an "exploded companion"
    // that merely drew extra duplicates looked identical to the assembled view, which is
    // exactly the bug this replaces).
    horn_z = arm_tip_z + foot_t_mm + riser_z_mm + cradle_t_mm
             + mg995_body_height_mm + mg995_shaft_projection_mm;
    gap = expl;

    // Mount + servo. In exploded view the servo lifts off its cradle.
    mg995_throttle_mount("assembly");
    translate([0, 0, gap * 0.5]) mg995_reference();

    // Horn interface, lifted further on the shaft axis.
    translate([servo_station_x + tower_offset_x, servo_shaft_y() + tower_offset_y,
               horn_z + gap]) throttle_horn_variant(letter, "assembly");

    // Rod: lifted with the horn so the pin joint stays legible.
    translate([servo_station_x + tower_offset_x,
               servo_shaft_y() + tower_offset_y,
               horn_z + gap * 0.6]) throttle_pushrod("assembly");

    // U fitting, dropped away from the rod end.
    translate([throttle_idle_offset_x_mm, tower_offset_y,
               horn_z - rods_nominal_span_mm - gap])
        throttle_u_fitting("body");

    // Legacy steering arm head + the retained mower lever as context.
    color([0.16, 0.55, 0.60, 0.30]) arm_head_assembly();
    throttle_lever_reference();
}

if (assembly == "throttle_station") throttle_station();
else if (assembly == "throttle_mount_service") throttle_mount_service();
else if (assembly == "throttle_a") throttle_variant_scene("A", exploded_mm);
else if (assembly == "throttle_b") throttle_variant_scene("B", exploded_mm);
else if (assembly == "throttle_c") throttle_variant_scene("C", exploded_mm);
else if (assembly == "throttle_d") throttle_variant_scene("D", exploded_mm);
else if (assembly == "throttle_e") throttle_variant_scene("E", exploded_mm);
else assert(false, str("unknown assembly: ", assembly));
