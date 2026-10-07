// Frame-mounted E-stop panel; FDM ASA, 0.2 mm layers. Not a rated safety enclosure.
// HB2-ES545 listing drawing: 22 mm panel bore, 40 mm mushroom, 42 mm rear depth,
// contact envelope 30 x 29 mm. Source and applicability: README.md.
// Print base down; support the horizontal button bore and rear-facing gussets.
// The 3 mm panel is a design choice, NOT a sourced maximum panel thickness.
include <common.scad>

estop_bore = 22.5;                 // 0.5 mm diametral print allowance
estop_panel_t = 3;
estop_panel_w = 96;
estop_panel_h = 80;
estop_axis_z = 54;
estop_panel_y = -26;              // rear face; mushroom faces negative Y
estop_base_l = 148;
estop_base_t = 10;
estop_ubolt_span = 124;
estop_cover_x = 35;
estop_cover_z = 27;

module estop_panel() {
    difference() {
        plate(estop_panel_w, estop_panel_h, estop_panel_t, 6);
        translate([0, 0, -1]) cylinder(h = estop_panel_t + 2, d = estop_bore);
        for (x = [-1, 1], y = [-1, 1])
            translate([x * estop_cover_x, y * estop_cover_z, -1])
                cylinder(h = estop_panel_t + 2, d = m3_clear);
    }
}

module estop_panel_placed() {
    translate([0, estop_panel_y, estop_axis_z])
        rotate([90, 0, 0]) estop_panel();
}

module estop_bracket() {
    difference() {
        union() {
            frame_saddle(estop_base_l, estop_base_t, estop_ubolt_span);
            estop_panel_placed();
            // Continuous toe joins the panel bottom to the base; contacts sit above it.
            translate([-estop_panel_w / 2 + 6, estop_panel_y - estop_panel_t, 8])
                cube([estop_panel_w - 12, 7, 10]);
            // Ribs remain outside the removable cover (x +/-40).
            for (x = [-46, 42])
                translate([x, 0, 0]) rotate([90, 0, 90])
                    linear_extrude(height = 4)
                        polygon([[estop_panel_y, 8], [27, 8],
                                 [estop_panel_y, estop_axis_z + 36]]);
        }
        frame_ubolt_slots(estop_base_t, estop_ubolt_span);
    }
}

// Verify with selected switch fitted: front projection, retention screws,
// service access, and a direct palm strike from off the parked mower.
estop_bracket();
