// Electronics sidecar parameter contract for the Pi 5 + Libre Alta + peripherals package.
//
// Capacities, not manufacturer dimensions. Unknown board outlines are edge-capture slots so
// no invented hole pattern or connector position is modelled. Nothing here claims IP
// rating, thermal margin or a mower attachment that has been measured.
include <../enclosure_common.scad>

control_geometry_verified = false;
control_watermark = "PROTOTYPE - FIT / LOAD / ELECTRICAL VALIDATION REQUIRED";

// ---- Printed joint scalars used by this package -------------------------------------
// Duplicated deliberately: this package does not (and should not) include the throttle
// parameter file. `use` shares modules/functions only, never variables, so a shared
// variable across packages would be an "Ignoring unknown variable" warning at render time.
pin_d_mm = 5;
joint_clearance_mm = 0.4;

// ---- Enclosure ---------------------------------------------------------------------
box_l_mm = 310;                    // INTERNAL, design capacity
box_w_mm = 240;
box_h_mm = 260;
wall_mm = 4;
floor_mm = 4;
flange_mm = 15;
lid_t_mm = 5;
lid_skirt_mm = 6;
gasket_w_mm = 4;
gasket_d_mm = 2.4;
corner_r_mm = 10;
shelf_t_mm = 4;
shelf_clearance_mm = 10;
service_gap_mm = 10;

outer_l_mm = box_l_mm + 2 * wall_mm;
outer_w_mm = box_w_mm + 2 * wall_mm;

// ---- Board capacities (unknown outlines are capacities, not measurements) ----------
pi5_pcb_mm = [85, 56];             // official reference board outline
pi5_holes_mm = [58, 49];           // official reference hole pitch
pi5_hole_d_mm = 2.7;
pi5_stack_h_mm = 80;               // reserved cooler/Hailo/cable envelope, NOT measured

pololu_pcb_mm = [24.638, 24.638];  // 0.97 in square, from the manufacturer drawing
pololu_hole_d_mm = 2.1844;         // 0.086 in; hole CENTRES are not published
pololu_header_pitch_mm = 2.54;
pololu_overhead_mm = 20;           // headers, plugs, FAILMODE/LEARN access

alta_capacity_mm = [150, 120, 60];
pca_capacity_mm = [70, 45, 25];
hiletgo_capacity_mm = [100, 70, 35];
steering_converter_capacity_mm = [120, 65, 50];
usb_converter_capacity_mm = [80, 55, 35];
harness_clamp_mm = [90, 30, 24];

// ---- Zones -------------------------------------------------------------------------
compute_zone_z_mm = floor_mm;
compute_card_l_mm = 200;
compute_card_w_mm = 160;
compute_card_h_mm = 90;
signal_shelf_z_mm = compute_zone_z_mm + compute_card_h_mm + service_gap_mm;
converter_bay_z_mm = signal_shelf_z_mm + shelf_t_mm + service_gap_mm + 40;

assert(box_h_mm >= converter_bay_z_mm + max(usb_converter_capacity_mm[2],
                                            steering_converter_capacity_mm[2])
       + service_gap_mm + lid_t_mm,
       "Enclosure height cannot host the converter bays");
assert(outer_l_mm <= 420 - 20 && outer_w_mm <= 420 - 20,
       "Enclosure footprint exceeds the usable bed after margins");
assert(pi5_holes_mm[0] < pi5_pcb_mm[0] && pi5_holes_mm[1] < pi5_pcb_mm[1],
       "Pi hole pitch must lie inside the board outline");
