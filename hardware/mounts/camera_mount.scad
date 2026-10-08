// Raspberry Pi Camera Module v2.1 carrier + tilt foot. FDM ASA.
// Official RPI-CAM-V2_1 drawing: PCB 23.862 x 25; holes diameter 2.2,
// hole centres (2,2),(14.5,2),(2,23),(14.5,23), lens centre (14.4,12.5).
// NOT compatible by assumption with ELP stereo, Camera v3 or a camera enclosure.
// No optical hood/window: these need actual lens FOV and weather qualification.
include <common.scad>
include <enclosure_common.scad>   // tower_od/bore/y and tower_cap_top_z() for the stereo bracket
camera_part = "carrier";           // carrier | foot | stereo | assembly
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

// ---- dual-lens (stereo) camera bracket: clamps the column below the Pi Camera station ----------
// The Pi Camera Module v2.1 keeps the cap; this carries a dual-lens board camera about 12 in lower
// so both optical axes stay parallel and see the same ground. Nothing here depends on the stereo
// camera's own dimensions: the LENS APERTURE is one wide window so both M12 barrels clear whatever
// the baseline turns out to be, and the body holes are horizontal SLOTS so a range of PCB patterns
// fits. Its bolt pair crosses the COLUMN, so the column must be drilled: the bracket's two holes
// are the drill guide for whichever height you set, and the column's existing brace bores are the
// only pre-drilled alternative. MEASURE the baseline before fitting: the window is wide, not
// unlimited, and it does not set the baseline for you.
stereo_drop_mm = 304.8;             // 12 in below the cap's camera station
function stereo_drop_mm_val() = stereo_drop_mm;
function stereo_bolt_y_val() = stereo_bolt_y;
function stereo_reach_mm_val() = stereo_reach_mm;
stereo_collar_t = 10;
stereo_collar_clearance = 0.4;
stereo_bolt_y = 18;                 // collar bolt pair, fore-aft
stereo_reach_mm = 34;               // column face to camera plate
stereo_plate_w = 46;
stereo_plate_h = 58;
stereo_plate_t = 4;
stereo_window_w = 34;
stereo_window_h = 22;
stereo_slot_pitch_y = 24;
stereo_slot_len = 12;

module stereo_camera_bracket() {
    wall = 8;                                 // collar wall thickness
    id2 = tower_od/2+stereo_collar_clearance;  // column half-size plus clearance
    od2 = id2+wall;                           // collar half-size
    t = stereo_collar_t;
    af = 7.2;                                 // M4 nut across flats + allowance
    difference() {
        union() {
            translate([-od2,-od2,-t/2]) cube([2*od2,2*od2,t]);
            // web from the collar's front wall out to the camera plate; it must reach INTO the
            // wall, or what the collar cut removes leaves the plate as a separate body.
            translate([-stereo_plate_w/2,-stereo_reach_mm,-t/2])
                cube([stereo_plate_w,stereo_reach_mm-id2+2,t]);
            translate([-stereo_plate_w/2,-stereo_reach_mm-stereo_plate_t,-stereo_plate_h/2])
                cube([stereo_plate_w,stereo_plate_t,stereo_plate_h]);
            // diagonal brace: collar's lower front wall down to the plate's lower half
            hull() {
                translate([-stereo_plate_w/2+2,-id2+2,-t/2]) cube([stereo_plate_t,stereo_plate_t,t]);
                translate([-stereo_plate_w/2+2,-stereo_reach_mm+2,-t/2-18])
                    cube([stereo_plate_t,stereo_plate_t,stereo_plate_t]);
            }
        }
        // the column passes through this collar
        translate([-id2,-id2,-t-1]) cube([2*id2,2*id2,2*t+2]);
        // collar bolts across the column, nut flats on both outer faces
        for (y = [-stereo_bolt_y, stereo_bolt_y]) {
            translate([-od2-1,y,0]) rotate([0,90,0]) cylinder(d = m4_clear, h = 2*od2+2);
            translate([od2-stereo_plate_t+1,y,0]) rotate([0,90,0]) cylinder(d = af/cos(30), h = 3, $fn = 6);
        }
        // one aperture for both M12 barrels
        translate([-stereo_window_w/2,-stereo_reach_mm-stereo_plate_t-1,-stereo_window_h/2])
            cube([stereo_window_w,stereo_plate_t+2,stereo_window_h]);
        // body slots, horizontal, for a range of hole pitches
        for (y = [-stereo_slot_pitch_y/2, stereo_slot_pitch_y/2])
            translate([0,-stereo_reach_mm-stereo_plate_t/2,y]) rotate([0,90,0])
                slot(camera_hole_d, stereo_slot_len, stereo_plate_t+2);
    }
}

// Placement lives in the files that already know the cap's z (assemblies.scad, fit_checks.scad):
// tower_cap_top_z() is defined in camera_tower.scad, and calling it from HERE silently produced an
// undefined translate - a vacuous check that reported success while testing nothing.

if (camera_part == "carrier") camera_carrier();
else if (camera_part == "foot") camera_foot();
else if (camera_part == "stereo") translate([0,0,stereo_plate_h/2]) stereo_camera_bracket();
else if (camera_part == "assembly") {
    camera_foot();
    camera_carrier_placed();
} else assert(false, "camera_part must be carrier, foot, stereo or assembly");
