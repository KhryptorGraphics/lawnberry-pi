// Removable rear-contact cover for estop_bracket.scad. ASA; not IP-rated.
// Switch drawing: 42 mm rear projection, 30 x 29 mm contact envelope.
// Free internal envelope: 55 x 57 x 48.5 mm; cable exit faces down in assembly.
// Print closed back down (default), support four flange ears. Four M3 nuts
// are installed from the open face before the switch is mounted.
$fn = 64;
estop_cover_depth = 51;
estop_cover_wall = 2.5;
estop_cover_nut_af = 5.8;          // M3 nut + FDM allowance; verify your nuts

module estop_cover_outline(w, h, t, r = 4) {
    hull() for (x = [-1, 1], y = [-1, 1])
        translate([x * (w / 2 - r), y * (h / 2 - r), 0]) cylinder(r = r, h = t);
}

// Assembly coordinates: open mating plane z=0, closed rear z=51.
module estop_contact_cover() {
    difference() {
        union() {
            estop_cover_outline(60, 62, estop_cover_depth);
            estop_cover_outline(80, 70, 3);
            for (x = [-35, 35], y = [-27, 27])
                translate([x, y, 0]) cylinder(d = 9, h = 6);
        }
        translate([0, 0, -1])
            estop_cover_outline(55, 57, estop_cover_depth - estop_cover_wall + 1, 2);
        for (x = [-35, 35], y = [-27, 27]) {
            translate([x, y, -1]) cylinder(d = 3.4, h = 8);
            translate([x, y, 3.2])
                cylinder(d = estop_cover_nut_af / cos(30), h = 4, $fn = 6);
        }
        // Downward, unsealed wire opening; route a drip loop outside this guard.
        translate([-7, 27, 10]) cube([14, 8, 12]);
        // Paired tie slots support the exiting cable bundle, not the terminals.
        for (x = [-12, 9]) translate([x, 27, 11]) cube([3, 8, 5]);
    }
}

module estop_cover_placed() {
    translate([0, -26, 54]) rotate([-90, 0, 0]) estop_contact_cover();
}

translate([0, 0, estop_cover_depth]) rotate([180, 0, 0]) estop_contact_cover();
