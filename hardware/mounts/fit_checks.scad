// Negative-volume regression probes. Every selected check must render EMPTY.
// Built/source geometry is exercised, not just parameter arithmetic.
include <enclosure_common.scad>
use <enclosure_body.scad>
use <enclosure_lid.scad>
use <electronics_tray.scad>
use <enclosure_board_sled.scad>
use <hitch_arm_layer.scad>
use <throttle_servo_mount.scad>
use <lapbar_four_bolt_clamp.scad>
use <pushrod.scad>
use <estop_bracket.scad>
use <estop_contact_cover.scad>
use <camera_mount.scad>
use <sensor_carrier.scad>
use <camera_tower.scad>
use <arm_support.scad>
use <camera_stack.scad>
use <servo_pushrod_adapter.scad>
check = "body_lid";
contact_station = 0;

// Linkage at the Toro 77502 ESTIMATE in enclosure_common.scad: measured servo station,
// estimated crank and lap-bar clamp pins. Both pin axes are global X.
fit_lapbar_on_bar = [[0,0,1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]];
function fit_bar_centre(side) = lapbar_pin(side)+[0,clamp_pin_local()[1],0];
// The assembled rod starts at face 45 (neutral). Move its servo-end segment
// with the moving crossplate face, not with the obsolete zero-centred horn angle.
module fit_rod_servo_end(side, face, crank_r=servo_crank_r) {
    translate(servo_crank_pin_at(side,face,crank_r)-servo_crank_pin(side)) intersection() {
        rod_between_points(servo_crank_pin(side),lapbar_pin(side),rod_nominal_setting());
        translate(servo_crank_pin(side)+[0,60,0]) cube([200,260,200],center=true);
    }
}
// The sway pads AS BUILT, not a re-derived slab. A probe re-derived from the same
// formula as the bar agrees with the bar whether or not that formula matches the real
// pad, which is how a bar could run straight through a pad (or stop short of it) while
// the check stayed empty. Intersecting the bars against the real pad solids is the
// only version of this check that means anything.
// Placement must mirror arm_lower_local exactly: arm_pose() plus arm_column_brace_pad(t).
// The pad module already translates along the arm axis by t, so wrapping it in
// arm_vertical(t) as well would apply t twice and put the probe nowhere near the pad.
module arm_brace_pads_real() {
    arm_pose() for(z=arm_column_brace_z) arm_column_brace_pad(arm_brace_t(z));
}
// One station's pad, for a contact probe.
module arm_brace_pad_at(dz) {
    arm_pose() arm_column_brace_pad(arm_brace_t(dz));
}
module check_enclosure_prints() {
    enclosure_body(); enclosure_lid_closed(); enclosure_tray_assembly(); enclosure_sleds_assembly();
}
// Use the SAME assembled frame for the head and independent fastener gauges.
module fit_servo_pose(side) {
    if(side>0) translate(arm_point(arm_joint_t)) children();
    else mirror([1,0,0]) translate(arm_point(arm_joint_t)) children();
}
function fit_servo_faces() = [for(face=[servo_face_forward_check_deg:15:servo_face_reverse_deg]) face];
function fit_servo_offsets(i) = let(t=servo_rear_hole_travel_mm(i))
    [[0,0],for(y=[-t[0]/2,t[0]/2],z=[-t[1]/2,t[1]/2]) [y,z]];
module fit_ring(od,id,h) {
    difference() {
        cylinder(d=od,h=h);
        translate([0,0,-0.1]) cylinder(d=id,h=h+0.2);
    }
}
module fit_servo_shank(i, y, z, d=servo_mount_bolt_d_mm) {
    p=servo_rear_hole_points_mm()[i];
    translate([-0.5,p[0]+y,arm_head_rise()+p[1]+z]) rotate([0,90,0])
        cylinder(d=d,h=servo_face_dx()+1.5);
}
function fit_camera_kind(index) = index == 0 ? "pi" : "stereo";
function fit_camera_z(kind) = kind == "pi" ? camera_lower_z() : camera_upper_z();
function fit_camera_bottom_points(kind) =
    [for(x=[-(camera_shell_width(kind)/2+5),camera_shell_width(kind)/2+5],
         y=[-93,-45]) [x,y]];
function fit_camera_flange_points() = [for(x=[-28,28],y=[-28,28]) [x,y]];
module fit_camera_fixed_stack() {
    camera_tower_assembly();
    for(kind=["pi","stereo"])
        translate([0,tower_y,fit_camera_z(kind)]) camera_shell(kind);
}
module fit_camera_service_load(kind) {
    camera_bottom_tray(kind);
    camera_carrier_assembly(kind);
    camera_lens_inserts_assembly(kind);
}
module fit_camera_joint_lower(i) {
    if(i==0) translate([0,0,-tower_seg_len()]) tower_segment();
    else if(i==1) translate([0,0,-80]) camera_shell("pi");
    else if(i==2) translate([0,0,-camera_extension_mm]) tower_camera_extension();
    else translate([0,0,-80]) camera_shell("stereo");
}
module fit_camera_joint_upper(i) {
    if(i==0) camera_shell("pi");
    else if(i==1) tower_camera_extension();
    else if(i==2) camera_shell("stereo");
    else tower_cap();
}
module fit_camera_joint_solids(i) {
    fit_camera_joint_lower(i);
    fit_camera_joint_upper(i);
}
module fit_camera_usb_plug(angle=0) {
    rotate([angle,0,0]) cube([24.06,14.06,45.06],center=true);
}
module fit_camera_usb_port_path() {
    hull() {
        translate([0,0,40]) fit_camera_usb_plug();
        translate([0,-9,40]) fit_camera_usb_plug();
    }
    translate([0,-9,40]) for(angle=[0:5:85]) hull() {
        fit_camera_usb_plug(angle);
        fit_camera_usb_plug(angle+5);
    }
    hull() {
        translate([0,-9,40]) fit_camera_usb_plug(90);
        translate([0,-70,40]) fit_camera_usb_plug(90);
    }
}
module fit_camera_usb_base_pose(angle) {
    translate([2.5,-9*sin(angle),143-19*sin(angle)]) fit_camera_usb_plug(angle);
}
module fit_camera_usb_base_path() {
    for(angle=[0:3:87]) hull() {
        fit_camera_usb_base_pose(angle);
        fit_camera_usb_base_pose(angle+3);
    }
    hull() {
        fit_camera_usb_base_pose(90);
        translate([2.5,-9,122]) fit_camera_usb_plug(90);
    }
    hull() {
        translate([2.5,-9,122]) fit_camera_usb_plug(90);
        translate([2.5,-tower_y,122]) fit_camera_usb_plug(90);
    }
}
module fit_camera_fov_lens(kind,p) {
    reach=19.01;
    half=2+reach*tan(camera_fov_capacity_deg(kind)/2);
    hull() {
        translate([p[0],camera_lens_front_plane_mm(kind)-0.01,p[1]])
            cube([3,0.002,3],center=true);
        translate([p[0],camera_lens_front_plane_mm(kind)-reach,p[1]])
            cube([2*half,0.002,2*half],center=true);
    }
}
if(check == "body_lid") intersection() { enclosure_body(); translate([0,0,0.02]) enclosure_lid_closed(); }
else if(check == "body_tray") intersection() { enclosure_body(); translate([0,0,0.02]) enclosure_tray_assembly(); }
else if(check == "sled_clearance") intersection() {
    enclosure_sleds_assembly();
    union() { enclosure_body(); enclosure_lid_closed(); }
}
else if(check == "pi_bores") intersection() {
    electronics_tray();
    // Independent expected transformed 58x49 pattern, M2.5 fasteners.
    union() for(x=[-69.5,-20.5],y=[-36,22]) translate([x,y,-1]) cylinder(d=2.5,h=14);
}
else if(check == "tray_fixings") intersection() {
    union() { enclosure_body(); enclosure_tray_assembly(); }
    union() for(x=[-62,62],y=[-52,52]) translate([x,y,-1]) cylinder(d=4,h=22);
}
else if(check == "pi_keepout") intersection() {
    check_enclosure_prints();
    translate([-73,-39.5,pi_pcb_z+0.05]) cube([56,85,53.95]);
}
else if(check == "stack_columns") intersection() {
    union() { enclosure_tray_assembly(); enclosure_sleds_assembly(); }
    union() {
        for(p=utility_anchors) translate([p[0],p[1],tray_bottom_z-1]) cylinder(d=3,h=120);
        for(p=relay_anchors) translate([p[0],p[1],tray_bottom_z-1]) cylinder(d=3,h=130);
    }
}
else if(check == "stack_payloads") intersection() {
    check_enclosure_prints();
    union() {
        for(z=relay_deck_z) translate([relay_pos[0]-23,relay_pos[1]-38,z+sled_t+8]) cube([46,76,25]);
        for(z=utility_deck_z) translate([utility_pos[0]-30,utility_pos[1]-37,z+sled_t+8]) cube([60,74,26]);
    }
}
else if(check == "lid_nut_access") intersection() {
    enclosure_body();
    union() for(p=lid_screw_points)
        translate([p[0],p[1],box_top-box_flange_t-20]) cylinder(d=8.2,h=20+3.5,$fn=6);
}
else if(check == "throttle_bores") intersection() {
    throttle_servo_mount();
    union() for(p=servo_rear_hole_points_mm())
        translate([p[0],-2,40+p[1]]) rotate([-90,0,0])
            cylinder(d=servo_mount_bolt_d_mm,h=10);
}
else if(check == "frame_bores") intersection() {
    estop_bracket();
    union() for(x=[-62,62],y=[-24,24]) translate([x,y,-1]) cylinder(d=8,h=12);
}
else if(check == "frame_nut_access") intersection() {
    estop_bracket();
    union() for(x=[-62,62],y=[-24,24]) translate([x,y,10.02]) cylinder(d=20,h=30);
}
// The coupon must contain three holes at the tongue-pattern coordinates, not
// partial edge cutouts on a plate displaced from its bores. Require corner
// material too: a missing or moved coupon cannot pass the empty-bore probe.
else if(check == "arm_layer_gauge_bores") union() {
    intersection() {
        arm_layer_gauge();
        for(x=[-40,0,40])
            translate([x,0,-1]) cylinder(d=x==0 ? 20 : 6,h=6);
    }
    difference() {
        for(x=[-46,46],y=[-14,14])
            translate([x-1,y-1,0.1]) cube([2,2,0.5]);
        arm_layer_gauge();
    }
}
// Sandwich plate matches the box flange floor footprint and preserves the
// independent pass-throughs for tray fasteners, glands and tower feet.
else if(check == "arm_layer_gland_paths") intersection() {
    arm_layer_center();
    union() for(x=[-40,0,40]) translate([x,-49,-1]) cylinder(d=18,h=10);
}
else if(check == "arm_layer_box_fasteners") intersection() {
    arm_layer_center();
    union() for(x=[-62,62],y=[-52,52]) translate([x,y,-1]) cylinder(d=3.8,h=10);
}
else if(check == "arm_layer_tray_clearance") intersection() {
    arm_layer_assembly();
    translate([0,0,0.02]) enclosure_body();
}
else if(check == "arm_tower_clearance") intersection() {
    arm_layer_assembly();
    tower_base_assembly();
}
// The foot seats on the layer TOP. A 0.1 mm top-face tolerance removes only
// coincident-face noise; any actual intrusion into the 8 mm panel fails.
else if(check == "arm_layer_root_clearance") intersection() {
    union() for(s=[-1,1]) arm_assembly_side(s);
    intersection() {
        translate([0,0,-layer_t]) arm_layer_center();
        translate([-500,-500,-layer_t])
            cube([1000,1000,layer_t-0.1]);
    }
}
// A thin extension of the real layer over z=0 must meet EACH arm foot.
else if(check == "arm_root_seat_contact") intersection() {
    if(contact_station == 0) arm_lower_assembly();
    else mirror([1,0,0]) arm_lower_assembly();
    translate([0,0,-layer_t+0.3]) arm_layer_center();
}
// Both mirrored heads have six full-length M2.5 passages and an open central boss relief.
else if(check == "arm_servo_passages") union() for(side=[-1,1]) intersection() {
    fit_servo_pose(side) arm_head();
    fit_servo_pose(side) union() {
        for(i=[0:len(servo_rear_hole_points_mm())-1],o=fit_servo_offsets(i))
            fit_servo_shank(i,o[0],o[1]);
        translate([-0.5,0,arm_head_rise()]) rotate([0,90,0])
            cylinder(d=servo_rear_clearance_d_mm-0.4,h=servo_face_dx()+1.5);
    }
}
// Washer, nut and slim driver approach the exposed inboard bearing plane.
else if(check == "arm_servo_hardware_access") union() for(side=[-1,1]) intersection() {
    fit_servo_pose(side) arm_head();
    fit_servo_pose(side) for(i=[0:len(servo_rear_hole_points_mm())-1],
                               o=fit_servo_offsets(i)) {
        p=servo_rear_hole_points_mm()[i];
        translate([-20,p[0]+o[0],arm_head_rise()+p[1]+o[1]])
            rotate([0,90,0]) cylinder(d=servo_mount_tool_d_mm,h=19.98);
    }
}
// Positive-material gauge: a missing/misplaced face must NOT pass empty-bore checks.
else if(check == "arm_servo_registration") union() for(side=[-1,1],
                                                       p=servo_rear_hole_points_mm()) difference() {
    fit_servo_pose(side)
        translate([servo_face_dx()-0.5,p[0],arm_head_rise()+p[1]])
            rotate([0,90,0]) fit_ring(6,4.8,0.3);
    fit_servo_pose(side) arm_head();
}
else if(check == "arm_head_joint_clearance") union() for(side=[-1,1]) intersection() {
    if(side>0) arm_lower_assembly(); else mirror([1,0,0]) arm_lower_assembly();
    if(side>0) translate([0.02*arm_sin(),0,0.02*arm_cos()]) arm_head_assembly();
    else mirror([1,0,0]) translate([0.02*arm_sin(),0,0.02*arm_cos()]) arm_head_assembly();
}
// Non-overlap alone can pass a missing or floating head. Require material just
// below and above each joint's four flange corners, plus spigot and socket roof.
else if(check == "arm_head_joint_seating_material")
    union() for(side=[-1,1]) {
        difference() {
            fit_servo_pose(side) rotate([0,arm_rake_deg,0]) union() {
                for(x=[-36,36],y=[-36,36])
                    translate([x-2,y-2,0.05]) cube([4,4,0.3]);
                translate([-4,-4,arm_joint_spigot_h+0.3]) cube([8,8,0.3]);
            }
            // Test the actual placement wrapper, not a head rebuilt in the gauge frame.
            if(side>0) arm_head_assembly(); else mirror([1,0,0]) arm_head_assembly();
        }
        difference() {
            fit_servo_pose(side) rotate([0,arm_rake_deg,0]) union() {
                for(x=[-36,36],y=[-36,36])
                    translate([x-2,y-2,-0.35]) cube([4,4,0.3]);
                translate([-4,-4,arm_joint_spigot_h-0.4]) cube([8,8,0.3]);
            }
            if(side>0) arm_lower_assembly(); else mirror([1,0,0]) arm_lower_assembly();
        }
    }
// Eight independently indexed flange contacts. Lift ONLY the lower part by
// 0.1 mm along its joint normal to give a face-contact probe finite volume.
else if(check == "arm_head_joint_seat_contact") intersection() {
    side=contact_station<4 ? 1 : -1;
    corner=contact_station%4;
    x=corner<2 ? -36 : 36;
    y=corner%2==0 ? -36 : 36;
    fit_servo_pose(side) rotate([0,arm_rake_deg,0])
        translate([x-2,y-2,0]) cube([4,4,0.15]);
    if(side>0) arm_head_assembly(); else mirror([1,0,0]) arm_head_assembly();
    if(side>0)
        translate([0.1*arm_sin(),0,0.1*arm_cos()]) arm_lower_assembly();
    else mirror([1,0,0])
        translate([0.1*arm_sin(),0,0.1*arm_cos()]) arm_lower_assembly();
}
// Four M6 pass upward through the layer and root into side-access
// captured nuts, not into inaccessible roofed material above the nuts.
else if(check == "arm_joint_bores") intersection() {
    arm_layer_assembly();
    union() for(s=[-1,1],p=arm_root_bolt_points())
        translate([s*p[0],p[1],-layer_t-0.5])
            cylinder(d=arm_joint_bore_d-0.5,
                     h=layer_t+arm_root_nut_z+arm_root_nut_h+0.5);
}
// The splayed arms must clear the schematic hitch plate, the tongue (with its
// gussets) and the enclosure. Only the arm solids are tested: the sandwich layer is
// itself the part that sits between the plate and the tongue, so including it here
// would always intersect.
else if(check == "arm_plate_clearance") intersection() {
    union() for(s=[-1,1]) arm_assembly_side(s);
    union() {
        translate([-hitch_plate_width_mm/2,
                   tongue_bolt_y()-hitch_plate_depth_mm/2,
                   -layer_t-hitch_plate_t_mm])
            cube([hitch_plate_width_mm,hitch_plate_depth_mm,hitch_plate_t_mm]);
        translate([-tongue_w/2,box_outer_w/2,0]) cube([tongue_w,tongue_reach,tongue_t]);
        enclosure_body();
    }
}
// Include the REAL base webs/flange and the segment's bottom flange; the
// latter stands from z=160..166 and can catch a bar that clears the base.
// The tongue gusset envelope and enclosure must also remain disjoint.
else if(check == "arm_brace_stations") intersection() {
    union() for(side=[-1,1]) arm_brace_bars_side(side);
    union() {
        translate([0,tower_y,0]) cylinder(d=tower_od-0.4,h=arm_tip_z);
        tower_base_assembly();
        translate([0,tower_y,tower_flange_z(0)]) tower_segment();
        translate([-tongue_w/2,box_outer_w/2,0])
            cube([tongue_w,tongue_end_y()-box_outer_w/2,tongue_gusset_h]);
        enclosure_body();
    }
}
// Handed halves must not occupy the same tower station.
else if(check == "arm_brace_pair_clearance") intersection() {
    arm_brace_bars_side(-1);
    arm_brace_bars_side(1);
}
// A continuous +Y approach, not sparse snapshots. The 0.02 mm X/Z thickness
// makes this a conservative sweep without a zero-volume Minkowski operand.
else if(check == "arm_brace_insertion") intersection() {
    minkowski() {
        union() for(side=[-1,1]) arm_brace_bars_side(side);
        cube([0.02,arm_brace_collar_od,0.02]);
    }
    union() {
        tower_base_assembly();
        translate([0,tower_y,tower_flange_z(0)]) tower_segment();
        enclosure_body();
        for(side=[-1,1]) arm_assembly_side(side);
    }
}
// Each bar's far end lands ON a pad mate face, so bar-vs-arm must have no volume:
// the faces meet, they do not interfere. Built against the REAL pad solids, not a slab
// re-derived from the same formula the bar uses -- that version agrees with the bar
// whether or not it matches the real pad.
else if(check == "arm_brace_bar_fit") intersection() {
    arm_brace_bars_assembly();
    union() for(s=[-1,1]) arm_assembly_side(s);
}
// The two M5 screw axes on each built pad must remain open through BOTH
// printed pieces. The ray uses the pad's real cut frame; its offset is
// independently projected into the bar by arm_brace_bolt_z().
else if(check == "arm_brace_bolt_bores") intersection() {
    union() { arm_brace_bars_assembly(); arm_lower_assembly(); }
    union() for(i=[0:2],
                z_rel=[-arm_brace_pad_bolt_pitch/2,arm_brace_pad_bolt_pitch/2])
        arm_pose() arm_pad_cut_frame(arm_brace_t(arm_brace_z[i]),z_rel)
            translate([-arm_brace_gusset_t()-1,0,0]) rotate([0,90,0])
                cylinder(d=m5_clear-0.6,
                         h=arm_brace_gusset_t()+arm_brace_pad_bolt_wall+
                           arm_brace_nut_t+2);
}
// Contact is checked for EACH station separately, against its built pad.
// A union of all pads inside intersection() would require a bar to touch
// every disjoint pad simultaneously; a missing station would otherwise pass
// if any other one touched. The 0.5 mm probe covers the 0.2 mm assembly gap.
else if(check == "arm_brace_bar_contact") intersection() {
    arm_brace_bar_at(contact_station);
    minkowski() {
        arm_brace_pad_at(arm_brace_z[contact_station]);
        sphere(0.5,$fn=8);
    }
}
else if(check == "clamp_four_bolts") intersection() {
    clamp_assembly();
    union() for(x=[-clamp_bolt_x_mm_value(),clamp_bolt_x_mm_value()],
                 y=[-clamp_bolt_y_mm_value(),clamp_bolt_y_mm_value()])
        translate([x,y,-clamp_half_height_mm()-1]) cylinder(d=5,h=2*clamp_half_height_mm()+2);
}
else if(check == "clamp_pin_bore") intersection() {
    clamp_assembly();
    translate([0,clamp_pin_local()[1],-clamp_half_height_mm()-1])
        cylinder(d=6.0,h=2*clamp_half_height_mm()+2);
}
// The sleeve must not touch the clamp body it is fitted into: body bore minus sleeve OD is
// the working clearance, and it is what lets the clamp close the sleeve onto the bar. This
// replaced a Ø25 rod probe through the body - the bar no longer touches the body at all.
else if(check == "clamp_tube_fit") intersection() {
    union() {
        translate([0,0,-clamp_half_height_mm()]) clamp_anchor_print();
        clamp_cap_print();
    }
    // Nudged off the split plane: the sleeve's flat faces are coplanar with the body halves'
    // mating faces, which alone reads as an intersection.
    translate([0,0,-0.05]) clamp_inserts_assembly();
}
// Each sleeve's own bar must pass it with the printed diametral clearance, so the clamp closes
// the sleeve onto the bar instead of seizing on it. Pairwise, sleeve i against bar i: a union of
// every bar against every sleeve would always fail, since the set spans different sizes.
else if(check == "clamp_insert_clear") union() for(i=[0:clamp_insert_count()-1]) intersection() {
    clamp_inserts_assembly(i);
    translate([-100,0,0]) rotate([0,90,0])
        cylinder(d=clamp_insert_bar_mm(i),h=clamp_length_mm()+200);
}
// Each sleeve must close on a bar a fraction oversize: contact, not a rattle. Indexed over the
// sleeve set by contact_station, like the brace-pad and root-seat contact checks.
else if(check == "clamp_insert_grip") intersection() {
    clamp_inserts_assembly(contact_station);
    translate([-100,0,0]) rotate([0,90,0])
        cylinder(d=clamp_insert_bore_mm(contact_station)+0.2,h=clamp_length_mm()+200);
}
// Sandwich-layer interface: independent hitch, auxiliary, gland and box-bolt
// probes against hardcoded design coordinates in the current first article.
else if(check == "arm_layer_hitch_bore") intersection() {
    arm_layer_center();
    translate([0,168,-1]) cylinder(d=19.4,h=10);
}
else if(check == "arm_layer_aux_bores") intersection() {
    arm_layer_center();
    union() for(x=[-40,40]) translate([x,168,-1]) cylinder(d=5.8,h=10);
}
// Tongue: hitch hole and optional anti-twist holes open through the full slab.
// Hardcoded expectations: bolt at y=68+100=168, slab 16 thick, holes 20 / 6.4.
else if(check == "tongue_bore") intersection() {
    enclosure_body();
    translate([0,168,-1]) cylinder(d=19.4,h=18);
}
else if(check == "tongue_aux_bores") intersection() {
    enclosure_body();
    union() for(x=[-40,40]) translate([x,168,-1]) cylinder(d=5.8,h=18);
}
// Tower base on the box: six M5 through the base flange AND the box wall, the
// two foot bolts through the base foot AND the tongue, and the 24 mm cable port
// continuous from the column bore to the box interior. All in the assembled position.
else if(check == "tower_wall_bores") intersection() {
    union() { enclosure_body(); tower_base_assembly(); }
    union() for(x=[-40,40],z=[40,90,140]) translate([x,60,z]) rotate([-90,0,0]) cylinder(d=4.9,h=20);
}
else if(check == "tower_foot_bores") intersection() {
    union() { enclosure_body(); tower_base_assembly(); }
    union() for(x=[-35,35]) translate([x,93,-1]) cylinder(d=4.9,h=27);
}
else if(check == "tower_port") intersection() {
    union() { enclosure_body(); tower_base_assembly(); }
    translate([0,60,110]) rotate([-90,0,0]) cylinder(d=22,h=58);
}
// Base touches the wall (y) and the tongue (z) on faces only: shift it 0.02 off both.
else if(check == "tower_body_clearance") intersection() {
    enclosure_body();
    translate([0,0.02,0.02]) tower_base_assembly();
}
else if(check == "tower_lid_clearance") intersection() {
    camera_tower_assembly();
    enclosure_lid_closed();
}
// First segment on the base: flanges meet face to face (shift 0.02), spigot
// inside the bore with clearance -- no volume overlap allowed.
else if(check == "tower_joint") intersection() {
    tower_base_assembly();
    translate([0,tower_y,tower_base_top_z+0.02]) tower_segment();
}
// Full 32.19 mm square connector throat across BOTH camera spines and the 12-inch
// extension, stopping beneath the terminal cap's closed roof.
else if(check == "tower_cable_path") intersection() {
    camera_stack_structure();
    translate([-16.095,tower_y-16.095,26])
        cube([32.19,32.19,tower_upper_cap_z()+17.9-26]);
}
else if(check == "tower_cap_bores") intersection() {
    tower_cap();
    for(p=fit_camera_flange_points())
        translate([p[0],p[1],-1]) cylinder(d=4,h=tower_cap_t+2);
}
else if(check == "tower_cap_socket") intersection() {
    tower_cap();
    translate([0,0,-15]) tower_spigot();
}
// Both pin bores stay open along the global pin axis at every lock setting.
else if(check == "rod_end_bores") intersection() {
    union() for(i=[0:rod_last_setting()]) translate([0,i*80,0]) rod_assembly(i);
    union() for(i=[0:rod_last_setting()], x=[0,rod_pin_length(i)])
        translate([x,i*80,0]) rotate([0,0,-rod_skew()]) rotate([90,0,0])
            cylinder(d=6.0,h=60,center=true);
}
// Both steel M6 lock bolts pass through sleeve and matching inner-bar holes at the
// shortest and longest settings, and all splice bolts pass tube and spigot.
else if(check == "rod_lock_bores") intersection() {
    rod_assembly(0);
    for(s=[0:1]) translate([rod_lock_x(s),0,-25]) cylinder(d=5.8,h=50);
}
else if(check == "rod_lock_bores_max") intersection() {
    rod_assembly(rod_last_setting());
    union() for(s=[0:1]) translate([rod_lock_x(s),0,-25]) cylinder(d=5.8,h=50);
}
// Paired clamp yoke ears and the rod eye share the X pin axis at the estimated
// lap-bar point without solid-on-solid contact, on both sides.
else if(check == "rod_clamp_interface") intersection() {
    union() for(side=[-1,1])
        rod_between_points(servo_crank_pin(side),lapbar_pin(side),rod_nominal_setting());
    union() for(side=[-1,1]) translate(fit_bar_centre(side)) multmatrix(fit_lapbar_on_bar)
        if (side < 0) clamp_assembly(); else mirror([0,0,1]) clamp_assembly();
}
// The eye turns inside the yoke through a full circle in its own fitting frame.
else if(check == "rod_clamp_sweep") {
    for(angle=[0:45:315]) intersection() {
        clamp_assembly();
        translate([0,clamp_pin_local()[1],0]) rotate([0,0,angle]) rotate([90,0,0])
            intersection() {
                rod_eye_fitting();
                translate([-clamp_rod_eye_length_mm/2,-50,-50])
                    cube([clamp_rod_eye_length_mm,100,100]);
            }
    }
}
// Face 45 is neutral and 90 full reverse; 0 is a forward analysis bound only.
else if(check == "rod_servo_sweep") union() for(side=[-1,1],face=fit_servo_faces()) intersection() {
    fit_rod_servo_end(side,face);
    union() { servo_body_envelope(side); servo_moving_bracket(side,face);
              servo_pushrod_adapter_at(side,face); }
}
// Cover all geometric extrema of the unmeasured pin radius/disc envelope,
// both handed assemblies and both travel ends (plus the neutral witness).
// The nominal continuous 0..90 face sweep is checked separately above.
else if(check == "rod_servo_sweep_band") union()
    for(side=[-1,1],radius=servo_crank_r_band_mm,
        disc_d=[26,34],
        face=[servo_face_forward_check_deg,servo_face_neutral_deg,servo_face_reverse_deg])
        intersection() {
            fit_rod_servo_end(side,face,radius);
            union() { servo_body_envelope(side,disc_d); servo_moving_bracket(side,face);
                      servo_pushrod_adapter_at(side,face,radius); }
        }
// Moving plate angles are ABSOLUTE: 45 neutral, 90 full reverse.
else if(check == "servo_adapter_bolt_passages") intersection() {
    servo_pushrod_adapter();
    for(p=servo_moving_hole_points_mm())
        translate([p[0],p[1],-1])
            cylinder(d=servo_adapter_bolt_d_mm,h=servo_adapter_pad_t_mm+2);
}
else if(check == "servo_adapter_hardware_access") intersection() {
    servo_pushrod_adapter();
    for(p=servo_moving_hole_points_mm())
        translate([p[0],p[1],servo_adapter_pad_t_mm+0.02])
            cylinder(d=servo_adapter_washer_od_mm,h=12);
}
else if(check == "servo_adapter_sweep_clearance") union()
    for(side=[-1,1],face=fit_servo_faces()) intersection() {
        servo_pushrod_adapter_at(side,face);
        union() {
            servo_body_envelope(side);
            if(side>0) arm_head_assembly(); else mirror([1,0,0]) arm_head_assembly();
        }
    }
else if(check == "servo_adapter_eye_fit") intersection() {
    servo_pushrod_adapter();
    pin=servo_adapter_pin_local();
    translate([servo_adapter_ear_inner_mm+1,pin[1]-10,pin[2]-10])
        cube([servo_adapter_gap_mm-2,20,20]);
}
// Every splice position must be BOLT-ABLE: at each ROW joint, the bolt pair has to pass through both
// pieces once the joint is shifted to that position. Probed at both extremes rather than every step -
// the row is one uniform pitch, so a mis-cut row shows at its ends, and all 28 combinations cost
// minutes of CSG per build for no extra discrimination. The inner-feeding joint is single-position
// and is covered at j=0, which is also why rod_lock_bores no longer probes splices: it was using one
// bolt list for every joint, and the two joints now have different offsets.
else if(check == "rod_splice_bores") {
    for(k=[0:rod_outer_count()-3],
        j=[0, rod_splice_positions_count()-1],
        s=rod_splice_offsets())
        intersection() {
            rod_outer_piece(k);
            translate([j*rod_splice_step_mm(),0,0]) rod_outer_piece(k+1);
            translate([rod_outer_boundary(k+1)+s+j*rod_splice_step_mm(),0,-60])
                cylinder(d=rod_bolt_bore_mm(),h=120);
        }
}
// The sway bars' M4 screws must pass through the COMPLETE column, both walls, at every brace
// station, so a collar pair and the column are one bolted joint rather than a collar gripping
// nothing. Stations land on the base (station 0) and inside the first segment (the rest).
else if(check == "tower_brace_bores") union() {
    // Station 0 is on the base, in absolute coordinates.
    intersection() {
        tower_base_assembly();
        for (z = [for (zz = arm_brace_z) if (zz <= tower_base_top_z) zz],
             y = [-arm_brace_collar_bolt_y, arm_brace_collar_bolt_y])
            translate([-tower_od / 2 - 2, tower_y + y, z]) rotate([0, 90, 0])
                cylinder(d = m4_clear - 0.6, h = tower_od + 4);
    }
    // The rest fall inside the first segment, in that segment's local frame.
    intersection() {
        tower_segment();
        for (z = tower_brace_seg_z(),
             y = [-arm_brace_collar_bolt_y, arm_brace_collar_bolt_y])
            translate([-tower_od / 2 - 2, y, z]) rotate([0, 90, 0])
                cylinder(d = m4_clear - 0.6, h = tower_od + 4);
    }
}
// Exclude the intended coplanar mounting contact, as with the tray/lid checks.
else if(check == "estop_cover") intersection() { estop_bracket(); translate([0,0.02,0]) estop_cover_placed(); }
else if(check == "estop_contacts") intersection() {
    union() { estop_bracket(); estop_cover_placed(); }
    translate([-15,-25.95,39.5]) cube([30,41.95,29]);
}
else if(check == "estop_bore") intersection() {
    estop_bracket(); translate([0,-31,54]) rotate([-90,0,0]) cylinder(d=22,h=8);
}
// Both opaque lens-only bays lift in from below with no protruding-lens collision.
else if(check == "camera_bottom_clearance") union() for(kind=["pi","stereo"]) intersection() {
    camera_shell(kind);
    translate([0,0,-0.02]) fit_camera_service_load(kind);
}
else if(check == "camera_bottom_service") union() for(kind=["pi","stereo"]) intersection() {
    fit_camera_fixed_stack();
    translate([0,tower_y,fit_camera_z(kind)-0.05]) minkowski() {
        fit_camera_service_load(kind);
        translate([-0.001,-0.001,-84]) cube([0.002,0.002,84]);
    }
}
else if(check == "camera_carrier_fixings") union() for(kind=["pi","stereo"]) intersection() {
    union() { camera_bottom_tray(kind); camera_carrier_assembly(kind); }
    for(p=camera_carrier_screw_points(kind)) {
        translate([p[0],p[1],-5]) cylinder(d=3,h=13);
        translate([p[0],p[1],8.02]) cylinder(d=7,h=14);
    }
}
else if(check == "camera_carrier_travel") union() for(kind=["pi","stereo"]) intersection() {
    union() { camera_shell(kind); camera_bottom_tray(kind); }
    for(dy=camera_carrier_y_travel_mm(kind))
        translate([0,0,0.02]) camera_carrier_assembly(kind,dy);
}
else if(check == "camera_cassette_insertion") union() for(kind=["pi","stereo"]) intersection() {
    camera_shell(kind);
    translate([0,0,-0.05]) minkowski() {
        camera_carrier_assembly(kind);
        translate([-0.001,-0.001,-84]) cube([0.002,0.002,84]);
    }
}
else if(check == "camera_pcb_groove_capacity") union() for(kind=["pi","stereo"]) intersection() {
    b=camera_board_bounds_xz(kind);
    camera_carrier_assembly(kind);
    translate([b[0]+0.05,camera_board_front_y_mm(kind),b[2]+0.05])
        cube([b[1]-b[0]-0.1,camera_pcb_thickness_capacity_mm()[1],b[3]-b[2]-0.1]);
}
else if(check == "camera_bottom_fixings") union() for(kind=["pi","stereo"]) intersection() {
    union() { camera_shell(kind); camera_bottom_tray(kind); }
    for(p=fit_camera_bottom_points(kind)) {
        translate([p[0],p[1],-5]) cylinder(d=3,h=18);
        translate([p[0],p[1],-20]) cylinder(d=6.5,h=15.98);
    }
}
else if(check == "camera_flange_bores") union() for(i=[0:3]) intersection() {
    fit_camera_joint_solids(i);
    for(p=fit_camera_flange_points())
        translate([p[0],p[1],-7]) cylinder(d=4,h=32);
}
else if(check == "camera_flange_access") union() for(i=[0:3]) intersection() {
    fit_camera_joint_solids(i);
    for(p=fit_camera_flange_points()) {
        translate([p[0],p[1],-20]) cylinder(d=8,h=13.98);
        translate([p[0],p[1],i==3 ? 24.02 : 6.02]) cylinder(d=8,h=14);
    }
}
else if(check == "camera_stack_joints") {
    assert(abs(camera_extension_mm-304.8)<0.001);
    assert(abs(camera_lens_z("stereo")-camera_lens_z("pi")-384.8)<0.001);
    assert(camera_throat_mm()>=32.2-0.000001);
    for(i=[0:3]) intersection() {
        fit_camera_joint_lower(i);
        translate([0,0,0.02]) fit_camera_joint_upper(i);
    }
}
else if(check == "camera_usb_housing_bends") union() for(kind=["pi","stereo"]) intersection() {
    camera_stack_structure();
    translate([0,tower_y,fit_camera_z(kind)]) fit_camera_usb_port_path();
}
else if(check == "camera_usb_base_bend") intersection() {
    union() {
        camera_stack_structure(); check_enclosure_prints();
        translate([0,box_inner_w/2,90]) rotate([90,0,0]) tower_backing();
        for(p=concat(utility_anchors,relay_anchors))
            translate([p[0],p[1],tray_bottom_z]) cylinder(d=3,h=160-tray_bottom_z);
        for(y=[-14,14])
            translate([-arm_brace_collar_od/2,tower_y+y,110]) rotate([0,90,0])
                cylinder(d=4,h=arm_brace_collar_od);
        for(z=relay_deck_z)
            translate([relay_pos[0]-23,relay_pos[1]-38,z+sled_t+8]) cube([46,76,25]);
        for(z=utility_deck_z)
            translate([utility_pos[0]-30,utility_pos[1]-37,z+sled_t+8]) cube([60,74,26]);
    }
    translate([0,tower_y,0]) fit_camera_usb_base_path();
}
else if(check == "camera_fov") union() for(kind=["pi","stereo"]) intersection() {
    union() { camera_shell(kind); camera_bottom_tray(kind); camera_lens_inserts_assembly(kind); }
    for(p=camera_lens_centres_xz(kind)) fit_camera_fov_lens(kind,p);
}
else if(check == "camera_material_gauges") union() for(kind=["pi","stereo"]) {
    assert(camera_board_outline_mm(kind)==(kind=="pi" ? [23.862,25] : [80,16.5]));
    assert(len(camera_insert_centres(kind))==(kind=="pi" ? 1 : 2));
    difference() {
        // Roof, side walls and central opaque divider must remain solid.
        translate([-2,-104,camera_housing_h-2]) cube([4,3,1]);
        camera_shell(kind);
    }
    difference() {
        if(kind=="pi") translate([-19,camera_insert_front_y_mm(kind)+0.4,39])
            cube([2,0.4,2]);
        else translate([-2,-107.6,39]) cube([4,0.4,2]);
        if(kind=="pi") camera_lens_inserts_assembly(kind); else camera_bottom_tray(kind);
    }
}
// Contact checks exercise distinct gasket lands and fastener material, not a union
// in which one surviving station could mask a missing side.
else if(check == "camera_bottom_nut_land_contact") intersection() {
    kind=fit_camera_kind(floor(contact_station/4));
    p=fit_camera_bottom_points(kind)[contact_station%4];
    camera_shell(kind);
    translate([p[0],p[1],5.4]) fit_ring(6,3.8,0.3);
}
else if(check == "camera_carrier_nut_land_contact") intersection() {
    kind=fit_camera_kind(floor(contact_station/2));
    p=camera_carrier_screw_points(kind)[contact_station%2];
    camera_carrier_assembly(kind);
    translate([p[0],p[1],5.3]) fit_ring(7,3.8,0.3);
}
else if(check == "camera_flange_land_contact") intersection() {
    i=floor(contact_station/4);
    p=fit_camera_flange_points()[contact_station%4];
    fit_camera_joint_upper(i);
    translate([p[0],p[1],0.5]) fit_ring(8,4.8,0.3);
}
else if(check == "camera_floor_gasket_contact") intersection() {
    kind=fit_camera_kind(floor(contact_station/2));
    s=contact_station%2==0 ? -1 : 1;
    camera_floor_gasket(kind);
    translate([s*(camera_shell_width(kind)/2-1.5)-0.5,-75,-0.7])
        cube([1,2,0.5]);
}
else if(check == "camera_face_gasket_contact") intersection() {
    kind=fit_camera_kind(floor(contact_station/2));
    s=contact_station%2==0 ? -1 : 1;
    camera_face_gasket(kind);
    translate([s*(camera_shell_width(kind)/2-1.5)-0.5,-105.5,65])
        cube([1,0.6,1]);
}
else if(check == "camera_insert_gasket_contact") intersection() {
    kind=contact_station<2 ? "pi" : "stereo";
    lens=kind=="pi" ? 0 : floor((contact_station-2)/2);
    top=contact_station%2;
    camera_insert_gasket(kind,lens);
    translate([camera_insert_centres(kind)[lens]-2,-108.5,top==0 ? 5 : 75])
        cube([4,0.6,1]);
}
else if(check == "camera_rim_gasket_contact") intersection() {
    kind=contact_station==0 ? "pi" : "stereo";
    lens=contact_station==0 ? 0 : contact_station-1;
    p=camera_lens_centres_xz(kind)[lens];
    camera_lens_rim_gasket(kind,lens);
    translate([p[0]+camera_insert_bore_d_mm(kind)/2-0.1,
               camera_lens_rim_seal_y_mm(kind)-0.1,p[1]-0.5])
        cube([1,1,1]);
}
else if(check == "arm_servo_washer_land_contact") intersection() {
    side=contact_station<6 ? 1 : -1;
    i=contact_station%6;
    p=servo_rear_hole_points_mm()[i];
    fit_servo_pose(side) arm_head();
    fit_servo_pose(side) translate([0.1,p[0],arm_head_rise()+p[1]])
        rotate([0,90,0]) fit_ring(servo_mount_washer_od_mm,4.8,0.3);
}
else if(check == "servo_adapter_seat_contact") intersection() {
    side=contact_station==0 ? 1 : -1;
    servo_moving_bracket(side,servo_face_neutral_deg);
    servo_adapter_pose(side,servo_face_neutral_deg)
        translate([0,0,-0.2]) servo_pushrod_adapter();
}
else if(check == "sensor_pitch") {
    for(a=[60,90,120]) intersection() { camera_foot(); sensor_carrier_placed(a); }
}
else assert(false,"Unknown geometry check");
