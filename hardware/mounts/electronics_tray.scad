// Electronics tray for the IP65 enclosure (Zulkit 220 x 170 x 110 mm).
//
// Carries the Raspberry Pi 5, Adafruit PCA9685, Pololu D24V50F5 regulator and
// the opto-isolated relay module.
//
// MOUNTING STRATEGY -- deliberate, after checking what is actually published:
//   * Raspberry Pi 5   -- dedicated standoffs. Its 58 x 49 mm hole pattern is
//                         a published, stable spec.
//   * Pololu D24V50F5  -- dedicated standoffs. Pololu publishes the two-hole
//                         pattern as 0.53" x 0.63" (13.5 x 16.0 mm), on
//                         opposite corners, for #2/M2 screws.
//   * Everything else  -- a generic 10 mm M3 grid. Adafruit does not publish
//                         the PCA9685's mounting-hole spacing, and generic
//                         relay modules vary between suppliers. Rather than
//                         invent a pattern that would be wrong, use nylon
//                         standoffs anywhere on the grid.
//
// This is the only part in the set with no structural safety role.
//
//   openscad -o electronics_tray.stl electronics_tray.scad

include <common.scad>

tray_l      = 195;   // fits the Zulkit box's internal floor with margin
tray_w      = 145;
tray_t      = 4;
rim         = 6;
post_h      = 7;     // standoff height (airflow under boards)
post_od      = 8;

// Published, verified patterns (centre-referenced).
pi5_holes    = [[-29, -24.5], [29, -24.5], [-29, 24.5], [29, 24.5]];
pololu_holes = [[-6.75, -8.0], [6.75, 8.0]];   // 0.53" x 0.63", opposite corners

pi5_pos      = [-46,  28, 0];
pololu_pos   = [ 60, -44, 0];

// Generic mounting grid for boards whose patterns are not published.
grid_pitch   = 10;
grid_x       = [10 : grid_pitch : 88];    // right-hand bay
grid_y       = [-30 : grid_pitch : 60];

module standoff(d = m3_clear) {
    difference() {
        cylinder(h = post_h, d = post_od);
        translate([0, 0, -1]) cylinder(h = post_h + 2, d = d);
    }
}

module board_posts(holes, pos, d = m3_clear) {
    translate([pos[0], pos[1], tray_t]) rotate([0, 0, pos[2]])
        for (h = holes) translate([h[0], h[1], 0]) standoff(d);
}

module tray_base() {
    difference() {
        union() {
            plate(tray_l, tray_w, tray_t, r = 8);
            // Low rim for stiffness
            difference() {
                plate(tray_l, tray_w, rim, r = 8);
                translate([0, 0, -1])
                    plate(tray_l - wall, tray_w - wall, rim + 2, r = 8);
            }
        }
        // Enclosure fixing holes -- MEASURE your box's boss positions
        for (x = [-1, 1], y = [-1, 1])
            translate([x * (tray_l / 2 - 12), y * (tray_w / 2 - 12), -1])
                cylinder(h = tray_t + 2, d = m4_clear);
        // Generic M3 mounting grid
        for (x = grid_x, y = grid_y)
            translate([x, y, -1]) cylinder(h = tray_t + 2, d = m3_clear);
        // Ventilation + cable routing, kept clear of the grid and the boards
        for (x = [-2 : 0], y = [-1 : 1])
            translate([x * 34 - 20, y * 42, -1])
                slot(9, 16, tray_t + 2);
    }
}

module electronics_tray() {
    union() {
        tray_base();
        board_posts(pi5_holes,    pi5_pos);
        board_posts(pololu_holes, pololu_pos, 2.4);   // M2
    }
}

electronics_tray();
