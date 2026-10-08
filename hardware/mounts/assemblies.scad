// Inspection/install views only. The mower is schematic; no Toro geometry or locations are known.
// Grey = schematic mower parts; amber = measured-interface placeholders; teal = printed parts;
// silver = steel fasteners; red = E-stop; blue = tray/lid. Views communicate arrangement, not fit.
include <enclosure_common.scad>
use <enclosure_body.scad>
use <enclosure_lid.scad>
use <electronics_tray.scad>
use <enclosure_board_sled.scad>
use <hitch_arm_layer.scad>
use <lapbar_four_bolt_clamp.scad>
use <pushrod.scad>
use <camera_tower.scad>
use <estop_bracket.scad>
use <estop_contact_cover.scad>
use <camera_mount.scad>
use <sensor_carrier.scad>
use <arm_support.scad>
assembly = "steering_arms";
// Scenes: steering_arms, arm_layer, lapbar_connector, rod_adjustment, enclosure_mounted,
// enclosure, tower, tower_base, estop_station, estop, camera, sensor, overview.

c_print  = [0.16, 0.55, 0.60];
c_steel  = [0.75, 0.76, 0.78];
c_frame  = [0.42, 0.44, 0.46];
c_hold   = [0.92, 0.70, 0.25, 0.55];
c_red    = [0.85, 0.12, 0.10];
c_yellow = [0.95, 0.80, 0.10];
c_blue   = [0.25, 0.50, 0.80];

// Servo shaft axis is LATERAL (+/-X): the metal crank sweeps the YZ plane and each rod
// runs FORWARD to its drive lever. Both pin axes are X, so each rod moves in a
// fore-and-aft plane; the lap bar's extra outboard offset is built into the rod.
lapbar_on_bar = [[0,0,1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]];
// The servo station is MEASURED. The crank and lap-bar clamp pin are the Toro 77502
// ESTIMATES in enclosure_common.scad; measure both before printing rods.
function lapbar_bar_centre(side) = lapbar_pin(side)+[0,clamp_pin_local()[1],0];
lapbar_pivot_drop = 320;   // schematic lever stub below the clamp
lapbar_top_rise = 180;

module ref_rail(x0, x1) {
    color(c_frame) translate([x0, -frame_rail_h/2, -frame_rail_w]) cube([x1-x0, frame_rail_h, frame_rail_w]);
}
module ref_frame_ubolts(span, saddle_t) {
    color(c_steel) for(x=[-1,1], y=[-1,1])
        translate([x*span/2, y*frame_ubolt_center_pitch/2, -frame_rail_w-12]) cylinder(d=8, h=frame_rail_w+12+saddle_t+14);
    color(c_steel) for(x=[-1,1])
        translate([x*span/2-4, -frame_ubolt_center_pitch/2, -frame_rail_w-12]) cube([8, frame_ubolt_center_pitch, 8]);
}
// Servo on its lateral face. The kit's metal output arm points UP at neutral, square to
// the level rod, so the servo starts with full leverage rather than near dead centre.
module ref_servo_at(side) {
    pin = servo_crank_pin(side);
    color(c_hold) servo_body_envelope(side);
    color(c_steel) servo_crank_arm(side,0);
    // M6 crank pin: arm, washer stack and rod eye in single shear.
    pin_len = servo_crank_t+servo_crank_washer+clamp_rod_eye_t_mm+12;
    color(c_steel)
        translate([side > 0 ? servo_station_x+servo_crank_face_x-6
                            : pin[0]-clamp_rod_eye_t_mm/2-6,pin[1],pin[2]])
            rotate([0,90,0]) cylinder(d=6,h=pin_len);
}
module ref_lapbar(side) {
    c = lapbar_bar_centre(side);
    color(c_hold) translate([c[0],c[1],c[2]-lapbar_pivot_drop])
        cylinder(d=clamp_display_bar_mm(),h=lapbar_pivot_drop+lapbar_top_rise);
    color(c_frame) translate([c[0]-45,c[1],c[2]-lapbar_pivot_drop])
        rotate([0,90,0]) cylinder(d=12,h=90);
}
module ref_estop_mushroom(x) {
    color(c_yellow) translate([x,-29,54]) rotate([90,0,0]) cylinder(d=60,h=3);
    color(c_red) translate([x,-32,54]) rotate([90,0,0]) cylinder(d=40,h=16);
}
module ref_hitch_plate() {
    // Generic hitch plate width/depth are placeholders; mower geometry is unknown.
    color(c_frame) translate([-hitch_plate_width_mm/2,
                              tongue_bolt_y()-hitch_plate_depth_mm/2,
                              -layer_t-hitch_plate_t_mm])
        cube([hitch_plate_width_mm,hitch_plate_depth_mm,hitch_plate_t_mm]);
    color(c_steel) translate([0,tongue_bolt_y(),-layer_t-hitch_plate_t_mm-4])
        cylinder(d=hitch_hole_d*0.85,h=tongue_t+layer_t+hitch_plate_t_mm+8);
}
module ref_glands() {
    for(p=gland_centres) {
        color(c_frame) translate([p[0],p[1],-22]) cylinder(d=19,h=22);
        color([0.1,0.1,0.1]) translate([p[0],p[1],-60]) cylinder(d=7,h=40);
    }
}
module ref_servo_pair() {
    for(side=[-1,1]) ref_servo_at(side);
}
module clamp_on_bar(side) {
    translate(lapbar_bar_centre(side))
        multmatrix(lapbar_on_bar) {
            if (side < 0) clamp_assembly();
            else mirror([0,0,1]) clamp_assembly();
        }
}
module ref_clamp_pivot_pin(side) {
    p = lapbar_pin(side);
    color(c_steel) {
        translate([p[0]-clamp_half_height_mm()-2,p[1],p[2]])
            rotate([0,90,0]) cylinder(d=6,h=55);
        translate([p[0]-clamp_half_height_mm()-6,p[1],p[2]])
            rotate([0,90,0]) cylinder(d=10,h=4,$fn=6);
        translate([p[0]-clamp_half_height_mm()-2,p[1],p[2]])
            rotate([0,90,0]) cylinder(d=12,h=2);
        translate([p[0]+clamp_half_height_mm(),p[1],p[2]])
            rotate([0,90,0]) cylinder(d=12,h=2);
        translate([p[0]+clamp_half_height_mm()+2,p[1],p[2]])
            rotate([0,90,0]) cylinder(d=10,h=5,$fn=6);
    }
}
module ref_bars_and_clamps() {
    for(side=[-1,1]) {
        ref_lapbar(side);
        clamp_on_bar(side);
    }
}
// Both rods run FORWARD from their servo crank to the drive lever. The left rod is the
// same printed rod turned 180 degrees about its own axis by rod_between_points().
module steering_linkages() {
    for(side=[-1,1])
        rod_between_points(servo_crank_pin(side),lapbar_pin(side),rod_nominal_setting());
}
module hitch_layer_arms() {
    color(c_print) arm_layer_assembly();
    color(c_steel) for(side=[-1,1],p=arm_root_bolt_points())
        translate([side*p[0],p[1],arm_foot_z-1])
            cylinder(d=5.5,h=arm_foot_h+layer_t+2);
    color(c_steel) translate([0,tongue_bolt_y(),-layer_t-hitch_plate_t_mm-2])
        cylinder(d=hitch_hole_d*0.88,h=tongue_t+layer_t+hitch_plate_t_mm+4);
    // Sway bars: one set per side, tying each arm section to the camera tower column
    // that the camera base mount bolts to the electronics box.
    for(side=[-1,1]) color(c_print) arm_brace_bars_side(side);
    // Factory plate's extra anti-rotation feature is unverified and deliberately
    // not drawn as present. A central ball-hole bolt alone is NOT approved for
    // powered steering; install positive torque restraint verified on the mower.
}
module steering_assembly() {
    ref_hitch_plate();
    color(c_frame) enclosure_body();
    color(c_frame) enclosure_lid_closed();
    hitch_layer_arms();
    ref_servo_pair();
    ref_bars_and_clamps();
    for(side=[-1,1]) ref_clamp_pivot_pin(side);
    color(c_print) steering_linkages();
}
module camera_on_tower() {
    translate([0,tower_y,tower_cap_top_z()]) {
        color(c_frame) camera_foot();
        color(c_print) camera_carrier_placed();
    }
}
module enclosure_mounted() {
    ref_hitch_plate();
    color(c_frame) enclosure_body();
    color(c_frame) enclosure_lid_closed();
    hitch_layer_arms();
    ref_servo_pair();
    ref_bars_and_clamps();
    for(side=[-1,1]) ref_clamp_pivot_pin(side);
    color(c_print) steering_linkages();
    color(c_print) camera_tower_assembly();
    camera_on_tower();
}
module estop_station() {
    ref_rail(-120,120);
    color(c_print) estop_bracket();
    ref_frame_ubolts(124,10);
    color(c_blue) estop_cover_placed();
    ref_estop_mushroom(0);
}

if (assembly == "steering_arms") steering_assembly();
else if (assembly == "arm_layer") {
    ref_hitch_plate();
    color(c_frame) enclosure_body();
    hitch_layer_arms();
    ref_servo_pair();
}
else if (assembly == "lapbar_connector") {
    p = lapbar_pin(-1);
    intersection() {
        ref_lapbar(-1);
        translate(lapbar_bar_centre(-1)) cube([120,120,180],center=true);
    }
    clamp_on_bar(-1);
    ref_clamp_pivot_pin(-1);
    color(c_print) intersection() {
        rod_between_points(servo_crank_pin(-1),p,rod_nominal_setting());
        translate(p+[0,-60,0]) cube([120,140,120],center=true);
    }
}
else if (assembly == "rod_adjustment") {
    translate([0,-45,0]) color(c_print) rod_assembly(0);
    translate([0,45,0]) color(c_print) rod_assembly(rod_last_setting());
}
else if (assembly == "estop_station") estop_station();
else if (assembly == "enclosure_mounted") enclosure_mounted();
else if (assembly == "tower") {
    color(c_print) camera_tower_assembly();
    camera_on_tower();
    color(c_frame) translate([-tongue_w/2,box_inner_w/2,0]) cube([tongue_w,tongue_end_y()-box_inner_w/2,tongue_t]);
    color(c_frame) translate([-60,box_inner_w/2,0]) cube([120,box_wall,box_top]);
}
else if (assembly == "tower_base") {
    color(c_print) tower_base_assembly();
    color(c_print) translate([0,tower_y,tower_base_top_z+25]) tower_segment_stub();
    color(c_frame) translate([-tongue_w/2,box_inner_w/2,0]) cube([tongue_w,tongue_end_y()-box_inner_w/2,tongue_t]);
}
else if (assembly == "overview") enclosure_mounted();
else if (assembly == "enclosure") {
    color(c_frame) difference() { enclosure_body(); translate([-150,-150,box_floor+0.1]) cube([300,150,box_top+10]); }
    color(c_blue) enclosure_tray_assembly();
    color(c_print) enclosure_sleds_assembly();
    color(c_hold) translate([-73,-39.5,pi_pcb_z]) cube([56,85,54]);
    for(p=utility_anchors) color(c_steel) translate([p[0],p[1],tray_top_z]) cylinder(d=3,h=118);
    for(p=relay_anchors) color(c_steel) translate([p[0],p[1],tray_top_z]) cylinder(d=3,h=128);
    color(c_frame) translate([0,0,45]) enclosure_lid_closed();
}
else if (assembly == "estop") { color(c_print) estop_bracket(); color(c_blue) estop_cover_placed(); }
else if (assembly == "camera") { color(c_frame) camera_foot(); color(c_print) camera_carrier_placed(); }
else if (assembly == "sensor") { color(c_frame) camera_foot(); color(c_print) sensor_carrier_placed(); }
else assert(false,"Unknown assembly");
