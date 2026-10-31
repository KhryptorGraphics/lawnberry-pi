// Packaging assembly scene: every owned component placed on its carrier/shelf.
//
// This is a placement study, not a mower installation drawing. No board outline or connector
// position is asserted where the manufacturer data is unavailable; those appear as
// capacity blocks labelled DESIGN CAPACITY.
include <parameters.scad>
use <board_carriers.scad>
use <electronics_enclosure.scad>

module capacity_block(sz, label_color) {
    color(label_color) translate([-sz[0] / 2, -sz[1] / 2, 0]) cube(sz);
}

module electronics_packaging(exploded_mm = 0) {
    // Shell as context (transparent so the interior layout reads).
    color([0.16, 0.55, 0.60, 0.22]) control_enclosure("body");

    // Lower compute zone: Pi 5 (documented outline) and the Alta (capacity).
    color([0.16, 0.55, 0.60]) translate([-70, 20, floor_mm]) carrier_pi5("body");
    color([0.14, 0.50, 0.18, 0.85])
        translate([-70, 20, floor_mm + shelf_t_mm]) capacity_block(
            [pi5_pcb_mm[0], pi5_pcb_mm[1], 3], [0.14, 0.50, 0.18]);
    color([0.16, 0.55, 0.60]) translate([70, 20, floor_mm]) carrier_alta("body");
    capacity_block(alta_capacity_mm, [0.92, 0.70, 0.25]);
    translate([70, 20, floor_mm + shelf_t_mm]) capacity_block(alta_capacity_mm,
                                                             [0.92, 0.70, 0.25, 0.5]);

    // Signal shelf: two PCA9685 carriers, the Pololu on its riser, relay carriers.
    signal_z = signal_shelf_z_mm + shelf_t_mm;
    for (x = [-90, -20]) {
        color([0.25, 0.50, 0.80]) translate([x, -70, signal_z]) carrier_pca9685("body");
        translate([x, -70, signal_z + 10]) capacity_block(pca_capacity_mm,
                                                          [0.25, 0.50, 0.80, 0.45]);
    }
    // The single purchased assembled mux: steering signal selection only.
    color([0.85, 0.42, 0.10]) translate([50, -70, signal_z])
        carrier_pololu_2806("body");
    translate([50, -70, signal_z + 6 + 9]) capacity_block(
        [pololu_pcb_mm[0], pololu_pcb_mm[1], 4], [0.85, 0.42, 0.10, 0.9]);
    for (x = [95, 130]) {
        color([0.42, 0.44, 0.46]) translate([x, -60, signal_z]) carrier_hiletgo("body");
        translate([x, -60, signal_z + 12]) capacity_block(hiletgo_capacity_mm,
                                                          [0.42, 0.44, 0.46, 0.45]);
    }

    // Converter bay: two steering converters and two USB-C converters.
    conv_z = converter_bay_z_mm + shelf_t_mm;
    for (x = [-1, 1])
        for (k = [-1, 1]) {
            translate([k * 70, x * 45, conv_z])
                color([0.30, 0.30, 0.32]) carrier_converter("body");
            translate([k * 70, x * 45, conv_z + 12]) capacity_block(
                steering_converter_capacity_mm, [0.55, 0.35, 0.20, 0.45]);
        }
    for (k = [-1, 1])
        for (y = [-15, 15])
            translate([k * 105, y, conv_z + 62])
                capacity_block(usb_converter_capacity_mm, [0.20, 0.35, 0.55, 0.45]);

    // Unverified automotive harness: clamped, with relay bodies NOT asserted as present.
    color([0.30, 0.30, 0.32]) translate([0, 90, box_h_mm - 40])
        carrier_harness_clamp("body");

    // Lid, lifted clear so the interior is visible in both views.
    color([0.16, 0.55, 0.60, 0.35]) translate([0, 0, exploded_mm / 2])
        control_enclosure("lid");
}

assembly = "electronics_packaging";

if (assembly == "electronics_packaging") electronics_packaging(0);
else assert(false, str("unknown control assembly: ", assembly));
