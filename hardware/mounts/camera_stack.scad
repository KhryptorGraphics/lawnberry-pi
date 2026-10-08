// Pi2 LOWER,304.8mm extension ABOVE it, stereo UPPER, closed mast cap.
// Only board outlines and Pi drawing coordinates are verified. Lens/barrel/projection
// geometry shown here is DESIGN CAPACITY at a selected jig-fit pose, not exact camera CAD.
include <enclosure_common.scad>
use <camera_tower.scad>
use <camera_mount.scad>

assert(abs(camera_extension_mm-304.8)<0.001,"Camera-to-camera tower span must be12 inches");
assert(camera_upper_z()>camera_lower_z(),"Stereo housing must be ABOVE the Pi housing");
assert(abs(camera_lens_z("stereo")-camera_lens_z("pi")-(camera_housing_h+304.8))<0.001,
       "Optical separation includes lower housing height plus12-inch tube");

function camera_reference_y_shift_mm(kind) = kind == "pi" ? -11.8 : 1;
function camera_reference_pcb_t_mm() = 1; // unknown actual thickness; groove/pad fitting uses the real PCB
function camera_reference_projection_mm(kind) =
    camera_board_front_y_mm(kind,camera_reference_y_shift_mm(kind))-camera_lens_front_plane_mm(kind);

module camera_station_structure(kind = "pi") {
    camera_shell(kind);
    camera_bottom_tray(kind);
    camera_lens_inserts_assembly(kind);
    camera_seals_assembly(kind);
}
module camera_reference_board(kind="pi") {
    b = camera_board_bounds_xz(kind);
    y = camera_board_front_y_mm(kind,camera_reference_y_shift_mm(kind));
    difference() {
        if (kind == "pi") {
            // PublishedR2 PCB corners; thickness is a selected clearance reference only.
            translate([b[0],y,b[3]]) rotate([-90,0,0]) linear_extrude(camera_reference_pcb_t_mm())
                offset(r=2) translate([2,2]) square([23.862-4,25-4]);
        } else translate([b[0],y,b[2]]) cube([80,camera_reference_pcb_t_mm(),16.5]);
        if (kind == "pi") for (x = [-12.4,0.1],z = [29.5,50.5])
            translate([x,y-1,z]) rotate([-90,0,0]) cylinder(d=2,h=4);
    }
}
module camera_reference_lenses(kind="pi") {
    y = camera_board_front_y_mm(kind,camera_reference_y_shift_mm(kind));
    projection = camera_reference_projection_mm(kind);
    ps = kind == "pi" ? [[0,40]] : camera_lens_centres_xz(kind);
    base = kind == "pi" ? [8.5,8.5] : [14,13];
    // Neither base depth nor barrelOD is sourced. Chosen values illustrate the adjustable interface.
    base_depth = kind == "pi" ? 1.2 : 3;
    for (p = ps) {
        color([0.12,0.12,0.12]) translate([p[0]-base[0]/2,y-base_depth,p[1]-base[1]/2])
            cube([base[0],base_depth,base[1]]);
        color([0.06,0.06,0.07]) translate([p[0],y-base_depth,p[1]]) rotate([90,0,0])
            cylinder(d=camera_insert_bore_d_mm(kind)-0.6,h=projection-base_depth);
        color([0.15,0.28,0.38]) translate([p[0],camera_lens_front_plane_mm(kind)-0.03,p[1]])
            rotate([90,0,0]) cylinder(d=kind == "pi" ? 3 : 5,h=0.04);
    }
}
module camera_station_assembly(kind = "pi") {
    color([0.16,0.55,0.60]) camera_shell(kind);
    color([0.25,0.50,0.80]) camera_bottom_tray(kind);
    color([0.36,0.40,0.44]) camera_carrier_assembly(kind,camera_reference_y_shift_mm(kind));
    color([0.10,0.13,0.16]) camera_lens_inserts_assembly(kind);
    color([0.18,0.18,0.20]) camera_seals_assembly(kind);
    color(kind == "pi" ? [0.14,0.50,0.18] : [0.92,0.70,0.25]) camera_reference_board(kind);
    camera_reference_lenses(kind);
}
module camera_stack_structure() {
    camera_tower_assembly();
    for (kind = ["pi","stereo"])
        translate([0,tower_y,kind == "pi" ? camera_lower_z() : camera_upper_z()])
            camera_station_structure(kind);
}
module camera_stack_assembly() {
    color([0.16,0.55,0.60]) camera_tower_assembly();
    for (kind = ["pi","stereo"])
        translate([0,tower_y,kind == "pi" ? camera_lower_z() : camera_upper_z()])
            camera_station_assembly(kind);
}

camera_stack_assembly();
