// Electronics tray for the IP65 enclosure (Zulkit 220 x 170 x 110 mm).
//
// Drops into the enclosure and carries: Raspberry Pi 5, Adafruit PCA9685,
// Pololu D24V50F5 regulator, and the opto-isolated relay module. Slotted
// mounting so board positions can shift without a reprint.
//
// This is the one part in this set with no structural safety role -- print
// it in whatever you like. The clamps and pushrod parts are different; see
// README.md.
//
//   openscad -o electronics_tray.stl electronics_tray.scad

include <common.scad>

tray_l      = 195;   // MEASURE your enclosure's internal floor
tray_w      = 145;
tray_t      = 4;
rim         = 6;
post_h      = 7;     // standoff height (airflow under boards)
post_od     = 8;

// Board hole patterns (centre-referenced), from published dimensions.
pi5_holes    = [[-29, -24.5], [29, -24.5], [-29, 24.5], [29, 24.5]];
pca_holes    = [[-27.5, -9.5], [27.5, -9.5], [-27.5, 9.5], [27.5, 9.5]];
pololu_holes = [[-6.5, 0], [6.5, 0]];
relay_holes  = [[-22, -13], [22, -13], [-22, 13], [22, 13]];

// Board placements on the tray (x, y, rotation)
pi5_pos    = [-45,  30, 0];
pca_pos    = [ 52,  40, 0];
pololu_pos = [ 55, -10, 0];
relay_pos  = [-40, -38, 0];

module standoff(d = m3_clear) {
    difference() {
        cylinder(h = post_h, d = post_od);
        translate([0, 0, -1]) cylinder(h = post_h + 2, d = d);
    }
}

module board_posts(holes, pos) {
    translate([pos[0], pos[1], tray_t]) rotate([0, 0, pos[2]])
        for (h = holes) translate([h[0], h[1], 0]) standoff();
}

module tray_base() {
    difference() {
        union() {
            // Floor
            hull()
                for (x = [-1, 1], y = [-1, 1])
                    translate([x * (tray_l / 2 - 8), y * (tray_w / 2 - 8), 0])
                        cylinder(h = tray_t, r = 8);
            // Low rim for stiffness
            difference() {
                hull()
                    for (x = [-1, 1], y = [-1, 1])
                        translate([x * (tray_l / 2 - 8),
                                   y * (tray_w / 2 - 8), 0])
                            cylinder(h = rim, r = 8);
                translate([0, 0, -1])
                    hull()
                        for (x = [-1, 1], y = [-1, 1])
                            translate([x * (tray_l / 2 - 8 - wall / 2),
                                       y * (tray_w / 2 - 8 - wall / 2), 0])
                                cylinder(h = rim + 2, r = 8);
            }
        }
        // Enclosure fixing holes (corners) -- MEASURE your box
        for (x = [-1, 1], y = [-1, 1])
            translate([x * (tray_l / 2 - 12), y * (tray_w / 2 - 12), -1])
                cylinder(h = tray_t + 2, d = m4_clear);
        // Ventilation + cable routing
        for (x = [-2 : 2], y = [-1 : 1])
            translate([x * 34, y * 40, -1])
                slot(9, 16, tray_t + 2);
    }
}

module electronics_tray() {
    union() {
        tray_base();
        board_posts(pi5_holes,    pi5_pos);
        board_posts(pca_holes,    pca_pos);
        board_posts(pololu_holes, pololu_pos);
        board_posts(relay_holes,  relay_pos);
    }
}

electronics_tray();
