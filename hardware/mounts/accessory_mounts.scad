// Outdoor accessory carriers, FDM ASA. These are strap/drill mounting interfaces,
// not asserted fits for unpublished sensor, antenna, fuse or relay housing SKUs.
// Never clamp a bare PCB, crush coax or obstruct an optical/RF aperture.
include <common.scad>
accessory_part = "harness_saddle";  // harness_saddle | equipment_saddle | antenna_deck
accessory_strap_w = 5.5;            // accommodates up to 4.8 mm wide ties
accessory_strap_t = 2.5;            // clear slot height; verify tie head routing
accessory_equipment_w = 36;         // flat housing support width, not a measured relay width
accessory_equipment_l = 55;
antenna_deck_size = 120;
antenna_metal_size = 100;           // OPTIONAL free-design metal ground plane; RF suitability unverified

module accessory_harness_saddle() {
    difference() {
        union() {
            plate(40, 24, 4, 4);
            translate([0, 0, 3.9]) plate(18, 22, 5.1, 3);
        }
        for (x = [-15, 15]) translate([x, 0, -1]) cylinder(d = m4_clear, h = 6);
        // Raised transverse tunnel lets a tie pass under an installed cable.
        translate([-6, -13, 4]) cube([12, 26, accessory_strap_t]);
        // Shallow cable seat only locates; the external tie supplies retention.
        translate([0, -13, 13]) rotate([-90, 0, 0]) cylinder(d = 12, h = 26);
    }
}

module accessory_equipment_saddle() {
    difference() {
        union() {
            plate(accessory_equipment_l + 22, accessory_equipment_w + 12, 4, 4);
            translate([0, 0, 3.9])
                plate(accessory_equipment_l, accessory_equipment_w, 5.1, 3);
        }
        for (x = [-1, 1])
            translate([x * (accessory_equipment_l / 2 + 6), 0, -1])
                rotate([0, 0, 90]) slot(m4_clear, 18, 6);
        // Two independent ties retain a sealed relay body or inline fuse holder.
        for (x = [-accessory_equipment_l / 4, accessory_equipment_l / 4])
            translate([x - accessory_strap_w / 2, -accessory_equipment_w / 2 - 8, 4])
                cube([accessory_strap_w, accessory_equipment_w + 16, accessory_strap_t]);
    }
}

module accessory_antenna_deck() {
    assert(antenna_deck_size >= antenna_metal_size + 20,
           "Leave attachment and tie access outside any metal ground plane");
    difference() {
        plate(antenna_deck_size, antenna_deck_size, 4, 6);
        for (x = [-1, 1], y = [-1, 1])
            translate([x * (antenna_deck_size / 2 - 7), y * (antenna_deck_size / 2 - 7), -1])
                cylinder(d = m4_clear, h = 6);
        // Four edge slots: hold the housing/metal plate, leave the sky aperture clear.
        for (a = [0, 90, 180, 270]) rotate([0, 0, a])
            translate([0, antenna_deck_size / 2 - 6, -1])
                slot(accessory_strap_w, 22, 6);
    }
}

if (accessory_part == "harness_saddle") accessory_harness_saddle();
else if (accessory_part == "equipment_saddle") accessory_equipment_saddle();
else if (accessory_part == "antenna_deck") accessory_antenna_deck();
else assert(false, "Unknown accessory_part");
