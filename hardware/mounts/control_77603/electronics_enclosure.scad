// Removable electronics sidecar for one Pi 5, one Libre Alta, two PCA9685 boards, one
// assembled Pololu 2806, two HiLetGo modules, two steering converters, two USB-C
// converters and the unverified automotive harness.
//
// Capacity-based: every board that lacks a verified outline is an edge-capture bay, and the
// printed parts carry no invented hole pattern or connector position. No IP, thermal or
// structural qualification is claimed, and the mower attachment is unmeasured.
include <parameters.scad>
use <board_carriers.scad>

module enclosure_shell(h = box_h_mm) {
    difference() {
        linear_extrude(height = h) rounded_plate_2d(outer_l_mm, outer_w_mm, corner_r_mm);
        translate([0, 0, floor_mm])
            linear_extrude(height = h - floor_mm + 1)
                rounded_plate_2d(box_l_mm, box_w_mm, corner_r_mm - wall_mm);
    }
}

module gasket_groove() {
    // Groove for a printed gasket. Not an IP-rated seal; a printed gasket is a prototype.
    translate([0, 0, box_h_mm - gasket_d_mm])
        difference() {
            linear_extrude(height = gasket_d_mm + 0.5)
                rounded_plate_2d(outer_l_mm - 2 * flange_mm, outer_w_mm - 2 * flange_mm,
                                 corner_r_mm);
            translate([0, 0, -1])
                linear_extrude(height = gasket_d_mm + 2)
                    rounded_plate_2d(outer_l_mm - 2 * flange_mm - 2 * gasket_w_mm,
                                     outer_w_mm - 2 * flange_mm - 2 * gasket_w_mm,
                                     corner_r_mm - gasket_w_mm);
        }
}

module enclosure_body(part = "body") {
    if (part == "body") {
        union() {
            difference() {
                enclosure_shell();
                gasket_groove();
                // Blind gland plate openings; actual glands are unmeasured.
                for (x = [-60, 0, 60])
                    translate([x, -box_w_mm / 2 - 1, 30])
                        rotate([90, 0, 0]) cylinder(d = 16, h = 10);
                // Corner bosses for printed lid pins.
                for (x = [-1, 1], y = [-1, 1])
                    translate([x * (box_l_mm / 2 - 12), y * (box_w_mm / 2 - 12), floor_mm])
                        difference() {
                            cylinder(d = 12, h = 40);
                            translate([0, 0, -1]) cylinder(d = pin_d_mm + 0.6, h = 42);
                        }
            }
            // Tiered shelves are tied to the wall by printed gussets; unsupported shelves
            // were separate solids and failed the one-solid gate.
            shelf_gussets(signal_shelf_z_mm);
            shelf_gussets(converter_bay_z_mm);
            translate([0, 0, signal_shelf_z_mm])
                difference() {
                    linear_extrude(height = shelf_t_mm)
                        rounded_plate_2d(box_l_mm - 4, box_w_mm - 4, 6);
                    for (x = [-1, 1], y = [-1, 1])
                        translate([x * (box_l_mm / 2 - 22), y * (box_w_mm / 2 - 22), -1])
                            cylinder(d = 14, h = shelf_t_mm + 2);
                    // Vertical cable corridor, kept clear of both PCB stacks.
                    translate([-box_l_mm / 2 + 30, -14, -1])
                        cube([44, 28, shelf_t_mm + 2]);
                }
            translate([0, 0, converter_bay_z_mm])
                difference() {
                    linear_extrude(height = shelf_t_mm)
                        rounded_plate_2d(box_l_mm - 4, box_w_mm - 4, 6);
                    for (x = [-1, 1], y = [-1, 1])
                        translate([x * (box_l_mm / 2 - 22), y * (box_w_mm / 2 - 22), -1])
                            cylinder(d = 14, h = shelf_t_mm + 2);
                    for (x = [-1, 1], y = [-1, 1])
                        translate([x * 70, y * 45, -1])
                            cylinder(d = 30, h = shelf_t_mm + 2);
                }
        }
    } else if (part == "lid") {
        // Printed upside down on its outer face. The lowest feature is the skirt base at
        // (box_h_mm - lid_t_mm), so shift by that to land exactly on the bed.
        translate([0, 0, -(box_h_mm - lid_t_mm)]) union() {
            translate([0, 0, box_h_mm]) enclosure_lid_plate();
            // Skirt.
            translate([0, 0, box_h_mm - lid_t_mm]) difference() {
                linear_extrude(height = lid_skirt_mm + lid_t_mm)
                    rounded_plate_2d(outer_l_mm + 2, outer_w_mm + 2, corner_r_mm + 1);
                translate([0, 0, -1])
                    linear_extrude(height = lid_skirt_mm + lid_t_mm + 2)
                        rounded_plate_2d(box_l_mm + 0.6, box_w_mm + 0.6, corner_r_mm - wall_mm);
            }
        }
    } else if (part == "gasket") {
        // Printed prototype gasket as a closed LOOP extruded from a ring profile: extruding
        // an outer plate then subtracting an inner plate of the same height left two shells.
        linear_extrude(height = gasket_d_mm)
            difference() {
                rounded_plate_2d(outer_l_mm - 2 * flange_mm, outer_w_mm - 2 * flange_mm,
                                 corner_r_mm);
                rounded_plate_2d(outer_l_mm - 2 * flange_mm - 2 * gasket_w_mm,
                                 outer_w_mm - 2 * flange_mm - 2 * gasket_w_mm,
                                 corner_r_mm - gasket_w_mm);
            }
    } else {
        assert(false, str("enclosure: unknown part ", part));
    }
}

module shelf_gussets(z) {
    // Four short ribs bridging the shelf edge into the side walls.
    for (sx = [-1, 1])
        translate([sx * (box_l_mm / 2 - wall_mm / 2 - 1), -box_w_mm / 2 + 20, z - 26])
            cube([wall_mm + 2, 8, 26]);
}

module enclosure_lid_plate() {
    difference() {
        linear_extrude(height = lid_t_mm)
            rounded_plate_2d(outer_l_mm + 2, outer_w_mm + 2, corner_r_mm + 1);
        // Lid pin bores align with the body bosses.
        for (x = [-1, 1], y = [-1, 1])
            translate([x * (box_l_mm / 2 - 12), y * (box_w_mm / 2 - 12), -1])
                cylinder(d = pin_d_mm + joint_clearance_mm, h = lid_t_mm + 2);
    }
}

enclosure_part = "body";

module control_enclosure(part = enclosure_part) { enclosure_body(part); }

control_enclosure(enclosure_part);
