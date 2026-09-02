// Electronics tray for the IP65 enclosure (Zulkit 220 x 170 x 110 mm).
//
// Carries the Raspberry Pi 5, Adafruit PCA9685, Pololu D24V50F5 regulator and
// the opto-isolated relay module.
//
// WEATHER STRATEGY -- the tray is one part of it, not the whole thing
//
// The sealed box does the sealing; this tray assumes water WILL eventually get
// in and makes that survivable. Outdoor enclosures on engine-powered machines
// rarely fail by bulk ingress through the gasket -- they fail by
//   (a) water tracking in along a cable, and
//   (b) condensation, because a sealed box heated by the engine and an
//       afternoon sun and then cooled overnight pumps moist air in and out
//       and condenses it on the coldest surface inside.
// Neither is solved by a better gasket. They are solved by downward-facing
// cable glands with a drip loop, a breather vent, and keeping the boards up
// off the floor -- which is what this tray is shaped for.
//
// Consequently:
//   * The tray stands on feet, so any liquid pools on the ENCLOSURE floor,
//     below the electronics, not around them.
//   * Perimeter drain slots let anything landing on the tray run off the edge
//     rather than puddling under a board.
//   * Boards sit on tall standoffs, clear of the tray floor.
//   * Cable tie-down slots take the strain at the tray, so cable movement
//     never works the gland seal or a connector loose.
//
// MOUNT THE ENCLOSURE WITH THE GLANDS FACING DOWN, and leave a drip loop in
// every cable below its gland. A gland on a top or side face is an invitation.
//
// MOUNTING STRATEGY -- after checking what is actually published:
//   * Raspberry Pi 5   -- dedicated standoffs; 58 x 49 mm is a published spec.
//   * Pololu D24V50F5  -- dedicated standoffs; Pololu publishes 0.53" x 0.63"
//                         (13.5 x 16.0 mm) on opposite corners, #2/M2 screws.
//   * Everything else  -- a generic 10 mm M3 grid. Adafruit does not publish
//                         the PCA9685's hole spacing and generic relay modules
//                         vary by supplier; an invented pattern would fit
//                         nothing.
//
// This is the only part in the set with no structural safety role.
//
//   openscad -o electronics_tray.stl electronics_tray.scad

include <common.scad>

tray_l       = 195;   // fits the Zulkit box's internal floor with margin
tray_w       = 145;
tray_t       = 4;
rim          = 7;
foot_h       = 9;     // lifts the tray off the enclosure floor
foot_od      = 14;
post_h       = 12;    // board standoffs -- tall, to clear any film of water
post_od      = 8;

// Published, verified patterns (centre-referenced).
pi5_holes    = [[-29, -24.5], [29, -24.5], [-29, 24.5], [29, 24.5]];
pololu_holes = [[-6.75, -8.0], [6.75, 8.0]];   // 0.53" x 0.63", opposite corners

pi5_pos      = [-46,  26, 0];
pololu_pos   = [ 62, -46, 0];

// Generic mounting grid for boards whose patterns are not published.
grid_pitch   = 10;
grid_x       = [12 : grid_pitch : 82];
grid_y       = [-26 : grid_pitch : 54];

// Cable entry edge. Ties here take strain before it reaches a gland.
tie_x        = [-70 : 24 : 74];

module standoff(d = m3_clear, h = post_h) {
    difference() {
        cylinder(h = h, d = post_od);
        translate([0, 0, -1]) cylinder(h = h + 2, d = d);
    }
}

module board_posts(holes, pos, d = m3_clear) {
    translate([pos[0], pos[1], tray_t]) rotate([0, 0, pos[2]])
        for (h = holes) translate([h[0], h[1], 0]) standoff(d);
}

// Feet raising the whole tray off the enclosure floor.
module feet() {
    for (x = [-1, 0, 1], y = [-1, 1])
        translate([x * (tray_l / 2 - 20), y * (tray_w / 2 - 16), -foot_h])
            difference() {
                cylinder(h = foot_h + 0.1, d = foot_od);
                translate([0, 0, -1]) cylinder(h = foot_h + 3, d = m4_clear);
            }
}

module tray_base() {
    difference() {
        union() {
            plate(tray_l, tray_w, tray_t, r = 8);
            // Low rim, stiffens the plate and stops splash crossing the edge
            difference() {
                plate(tray_l, tray_w, rim, r = 8);
                translate([0, 0, -1])
                    plate(tray_l - wall * 2, tray_w - wall * 2, rim + 2, r = 8);
            }
        }
        // Enclosure fixing holes -- MEASURE your box's boss positions
        for (x = [-1, 1], y = [-1, 1])
            translate([x * (tray_l / 2 - 12), y * (tray_w / 2 - 12), -1])
                cylinder(h = tray_t + 2, d = m4_clear);
        // Generic M3 mounting grid
        for (x = grid_x, y = grid_y)
            translate([x, y, -1]) cylinder(h = tray_t + 2, d = m3_clear);
        // Perimeter drain slots -- anything landing on the tray leaves it
        for (x = [-2 : 2])
            for (y = [-1, 1])
                translate([x * 36, y * (tray_w / 2 - 10), -1])
                    rotate([0, 0, 90]) slot(6, 12, tray_t + 2);
        // Corner drains, the low points if the machine is parked off-level
        for (x = [-1, 1], y = [-1, 1])
            translate([x * (tray_l / 2 - 26), y * (tray_w / 2 - 22), -1])
                cylinder(h = tray_t + 2, d = 9);
        // Cable tie-down slot pairs along the entry edge
        for (x = tie_x)
            for (dx = [-4, 4])
                translate([x + dx, -tray_w / 2 + 15, -1])
                    rotate([0, 0, 90]) slot(3.4, 7, tray_t + 2);
    }
}

module electronics_tray() {
    union() {
        tray_base();
        feet();
        board_posts(pi5_holes,    pi5_pos);
        board_posts(pololu_holes, pololu_pos, 2.4);   // M2
    }
}

electronics_tray();
