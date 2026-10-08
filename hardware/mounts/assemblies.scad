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
use <servo_pushrod_adapter.scad>
use <camera_stack.scad>
use <sensor_carrier.scad>
use <arm_support.scad>
assembly = "steering_arms";
// Scenes: steering_arms, arm_layer, servo_adapter, lapbar_connector, rod_adjustment,
// enclosure_mounted, enclosure, tower, tower_base, estop_station, estop, camera, camera_stack,
// camera_pi_exploded, camera_stereo_exploded, sensor, overview.

c_print  = [0.16, 0.55, 0.60];
c_steel  = [0.75, 0.76, 0.78];
c_frame  = [0.42, 0.44, 0.46];
c_hold   = [0.92, 0.70, 0.25, 0.55];
c_red    = [0.85, 0.12, 0.10];
c_yellow = [0.95, 0.80, 0.10];
c_blue   = [0.25, 0.50, 0.80];
servo_view_face_deg = servo_face_neutral_deg; // physical moving plate; not a PWM command
servo_ghost_alpha = 0.3;                      // servo_adapter scene: full-reverse pose overlay

// Servo shaft axis is LATERAL (+/-X): the moving eight-hole crossplate and its printed
// adapter sweep the YZ plane and each rod runs FORWARD to its drive lever. Both pin axes
// are X, so each rod moves in a fore-and-aft plane; the lap bar's extra outboard offset
// is built into the rod.
lapbar_on_bar = [[0,0,1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]];
// The servo station is MEASURED. The adapter pin radius and lap-bar clamp pin are the
// Toro 77502 ESTIMATES in enclosure_common.scad; measure both before printing rods.
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
// Moving crossplate, printed adapter and its X-axis M6 pivot in DOUBLE shear, with free
// outer head/nut access. Face45 = neutral (pin straight up), face90 = full reverse.
module ref_servo_moving(side,face_angle_deg=servo_view_face_deg,alpha=1) {
    pin = servo_crank_pin_at(side,face_angle_deg);
    pin_len = servo_adapter_gap_mm+2*servo_adapter_ear_t_mm+12;
    color(c_steel,alpha) servo_moving_bracket(side,face_angle_deg);
    color(c_print,alpha) servo_pushrod_adapter_at(side,face_angle_deg);
    color(c_steel,alpha)
        translate([pin[0]-pin_len/2,pin[1],pin[2]])
            rotate([0,90,0]) cylinder(d=6,h=pin_len);
}
module ref_servo_at(side,face_angle_deg=servo_view_face_deg) {
    color(c_hold) servo_body_envelope(side);
    ref_servo_moving(side,face_angle_deg);
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
// Bottom servicing leaves the shell and mast stationary. The tray, its integral closing
// face, groove cassette, removable PCB-edge retainer and front-fitted lens inserts travel
// together; here they are separated so every insert and gasket is visible.
module camera_station_exploded(kind="pi") {
    ports = [0:len(camera_insert_centres(kind))-1];
    color(c_print) camera_shell(kind);
    // Floor gasket sits in the tray recess, between shell and tray.
    color([0.18,0.18,0.20]) translate([0,0,-50]) camera_floor_gasket(kind);
    translate([0,0,-105]) {
        color(c_blue) camera_bottom_tray(kind);
        color([0.36,0.40,0.44]) camera_carrier_assembly(kind,camera_reference_y_shift_mm(kind));
        color(kind == "pi" ? [0.14,0.50,0.18] : [0.92,0.70,0.25]) camera_reference_board(kind);
        camera_reference_lenses(kind);
        // Front seal stack, pulled forward (-Y) in fitting order: face gasket,
        // per-port insert gasket, per-lens rim gasket, opaque lens-only insert.
        color([0.18,0.18,0.20]) translate([0,-10,0]) camera_face_gasket(kind);
        for (i = ports) {
            color([0.18,0.18,0.20]) translate([0,-22,0]) camera_insert_gasket(kind,i);
            color([0.30,0.30,0.32]) translate([0,-34,0]) camera_lens_rim_gasket(kind,i);
            color([0.10,0.13,0.16]) translate([0,-48,0]) camera_lens_insert(kind,i);
        }
    }
}
module camera_stack_close() {
    translate([0,-tower_y,-camera_lower_z()])
        intersection() {
            camera_stack_assembly();
            translate([-150,tower_y-150,camera_lower_z()-25])
                cube([300,300,tower_upper_cap_z()+tower_cap_t-camera_lower_z()+50]);
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
    camera_stack_assembly();
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
else if (assembly == "servo_adapter") {
    // One right-hand servo: neutral face45 solid, full-reverse face90 ghosted.
    ref_servo_at(1,servo_face_neutral_deg);
    ref_servo_moving(1,servo_face_reverse_deg,servo_ghost_alpha);
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
    camera_stack_assembly();
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
else if (assembly == "camera") camera_station_assembly("pi");
else if (assembly == "camera_stack") camera_stack_close();
else if (assembly == "camera_pi_exploded") camera_station_exploded("pi");
else if (assembly == "camera_stereo_exploded") camera_station_exploded("stereo");
else if (assembly == "sensor") { color(c_frame) camera_foot(); color(c_print) sensor_carrier_placed(); }
else assert(false,"Unknown assembly");
