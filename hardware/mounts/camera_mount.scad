// Raspberry Pi Camera Module v2.1 carrier + tilt foot. FDM ASA.
// Official RPI-CAM-V2_1 drawing: PCB 23.862 x 25; holes diameter 2.2,
// hole centres (2,2),(14.5,2),(2,23),(14.5,23), lens centre (14.4,12.5).
// NOT compatible by assumption with ELP stereo, Camera v3 or a camera enclosure.
// No optical hood/window: these need actual lens FOV and weather qualification.
include <common.scad>
camera_part = "carrier";           // carrier | foot | assembly
camera_angle = 90;                 // assembly pitch only; 90 looks along -Y
camera_pcb_gap = 4.5;
camera_carrier_t = 3;
camera_hinge_gap = 8.6;             // 8 mm lug + 0.3 mm each side
camera_hole_d = 2.4;                // M2 screws, not M2.5

module camera_holes(h = 12) {
    for (x = [-6.25, 6.25], y = [-10.5, 10.5])
        translate([x, y, -1]) cylinder(d = camera_hole_d, h = h);
}

module camera_carrier() {
    difference() {
        union() {
            plate(36, 34, camera_carrier_t, 3);
            translate([-4, -26, 0]) cube([8, 13, camera_carrier_t]);
            translate([-4, -25, 8]) rotate([0, 90, 0]) cylinder(r = 8, h = 8);
            for (x = [-6.25, 6.25], y = [-10.5, 10.5])
                translate([x, y, camera_carrier_t - 0.1])
                    cylinder(d = 4.4, h = camera_pcb_gap + 0.1);
        }
        camera_holes();
        // Open the connector side; cable leaves right edge without folding over a lip.
        translate([11, -13, -1]) cube([12, 26, camera_carrier_t + 2]);
        translate([-6, -25, 8]) rotate([0, 90, 0]) cylinder(d = m4_clear, h = 12);
    }
}

module camera_foot() {
    difference() {
        union() {
            plate(60, 40, 4, 4);
            for (x = [-camera_hinge_gap / 2 - 4, camera_hinge_gap / 2]) {
                translate([x, -8, 3]) cube([4, 16, 13]);
                translate([x, 0, 16]) rotate([0, 90, 0]) cylinder(r = 8, h = 4);
            }
        }
        translate([-12, 0, 16]) rotate([0, 90, 0]) cylinder(d = m4_clear, h = 24);
        for (x = [-22, 22]) translate([x, 0, -1]) rotate([0, 0, 90]) slot(m4_clear, 12, 6);
    }
}

module camera_carrier_placed() {
    translate([0, 0, 16]) rotate([camera_angle, 0, 0])
        translate([0, 25, -8]) camera_carrier();
}

if (camera_part == "carrier") camera_carrier();
else if (camera_part == "foot") camera_foot();
else if (camera_part == "assembly") {
    camera_foot();
    camera_carrier_placed();
} else assert(false, "camera_part must be carrier, foot or assembly");
