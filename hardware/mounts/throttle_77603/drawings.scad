// 2D drawing geometry for the throttle horn variants.
//
// Each drawing is a genuine orthographic cross-section of THAT variant's horn interface, taken
// through the pin plane, plus drawn dimension lines. An earlier revision ignored the variant
// entirely and emitted one fixed schematic outline, so all five SVGs were byte-identical --
// useless for comparing variants. Dimensions and the watermark are overlaid in the builder,
// which keeps this file pure geometry.
include <parameters.scad>
use <throttle_horn_variants.scad>

// Selector names are deliberately distinct from the included module's parameters so that
// -D horn_variant="C" cannot be shadowed by a signature default.
horn_variant = "A";
radius_mm = horn_radius_default_mm;

// ---- Dimension primitives (2D) -----------------------------------------------------
module dim_arrow(len = 4, w = 1.2) {
    polygon([[0, 0], [len, w / 2], [len, -w / 2]]);
}

module dim_h(x0, x1, y, label_side = 1) {
    // Horizontal dimension line with ticks and inward arrowheads.
    translate([x0, y]) rotate([0, 0, 0]) dim_arrow();
    translate([x1, y]) rotate([0, 0, 180]) dim_arrow();
    translate([x0, y]) square([x1 - x0, 0.35]);
    translate([x0 - 0.2, y - 2]) square([0.4, 4]);
    translate([x1 - 0.2, y - 2]) square([0.4, 4]);
}

module dim_v(y0, y1, x) {
    translate([x, y0]) rotate([0, 0, 90]) dim_arrow();
    translate([x, y1]) rotate([0, 0, 270]) dim_arrow();
    translate([x, y0]) square([0.35, y1 - y0]);
    translate([x - 2, y0 - 0.2]) square([4, 0.4]);
    translate([x - 2, y1 - 0.2]) square([4, 0.4]);
}

// ---- Variant cross-sections --------------------------------------------------------
// The horn interface is carried on the MG995 shaft; sections are taken through the pin plane
// so the retained features (eye bore, carriage slot, cage, clevis gap) are all visible.
module variant_section(letter = horn_variant) {
    projection(cut = true) rotate([0, 0, 0])
        throttle_horn_variant(letter, "assembly", radius_mm = radius_mm);
}

// ---- Owner-specified opening, for reference on every sheet -------------------------
module u_opening_reference() {
    difference() {
        translate([-throttle_u_inner_width_mm / 2 - throttle_u_wall_mm,
                   -throttle_u_inner_depth_mm / 2 - throttle_u_wall_mm])
            square([throttle_u_inner_width_mm + 2 * throttle_u_wall_mm,
                    throttle_u_inner_depth_mm + 2 * throttle_u_wall_mm]);
        square([throttle_u_inner_width_mm, throttle_u_inner_depth_mm], center = true);
    }
}

// ---- Sheet -------------------------------------------------------------------------
// Two views on one sheet: the variant section (left) and the lever-opening reference
// (right), separated so dimension lines do not overlap the section.
translate([-40, 0]) variant_section(horn_variant);

// Overall width/height of the section are dimensioned; these are the numbers the owner needs
// to compare against the real horn and lever.
dim_h(-40 - horn_radius_min_mm, -40 + horn_radius_max_mm, -horn_radius_max_mm - 8);

translate([30, 0]) {
    u_opening_reference();
    dim_h(-throttle_u_inner_width_mm / 2, throttle_u_inner_width_mm / 2,
          -throttle_u_inner_depth_mm / 2 - 8);
    dim_v(-throttle_u_inner_depth_mm / 2, throttle_u_inner_depth_mm / 2,
          throttle_u_inner_width_mm / 2 + 8);
}
