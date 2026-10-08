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
check = "body_lid";
contact_station = 0;

// Linkage at the Toro 77502 ESTIMATE in enclosure_common.scad: measured servo station,
// estimated crank and lap-bar clamp pins. Both pin axes are global X.
fit_lapbar_on_bar = [[0,0,1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]];
function fit_bar_centre(side) = lapbar_pin(side)+[0,clamp_pin_local()[1],0];
// The servo end of the rod, cropped near the servo, carried round by the crank.
module fit_rod_servo_end(side, angle, crank_r=servo_crank_r) {
    translate(servo_crank_pin_at(side,angle,crank_r)-servo_crank_pin(side,crank_r)) intersection() {
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
    union() for(x=[-12,12],z=[28,52]) translate([x,-2,z]) rotate([-90,0,0]) cylinder(d=2.5,h=10);
}
else if(check == "frame_bores") intersection() {
    estop_bracket();
    union() for(x=[-62,62],y=[-24,24]) translate([x,y,-1]) cylinder(d=8,h=12);
}
else if(check == "frame_nut_access") intersection() {
    estop_bracket();
    union() for(x=[-62,62],y=[-24,24]) translate([x,y,10.02]) cylinder(d=20,h=30);
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
// Negative probe: a ray down each servo-bore centreline must find no material in the
// 6 mm servo plate. It is cut to the nominal bore diameter, so it stays empty only if
// the four M2.5 bores are actually open and in the right place. Probed in the head's
// ASSEMBLY frame (arm_vertical at the joint), not the print pose, which carries a bed
// lift that would shift the rays off the bores.
// The probe must be framed exactly like the part. Earlier this probe was written in bare
// absolute coordinates while only the head carried arm_vertical, so it sat ~190 mm away
// and intersected nothing - a vacuous pass, the very trap the previous comment claimed
// to have closed. Widening the probe to Ø14 still returned empty, which is how it was
// caught. Both children below are framed; keep them that way.
else if(check == "arm_servo_pattern") intersection() {
    arm_vertical(arm_joint_t) arm_head();
    arm_vertical(arm_joint_t) {
        union() for(y=[-servo_side_hole_pitch/2,servo_side_hole_pitch/2],
                     z=[arm_head_rise()-servo_side_hole_pitch/2,
                        arm_head_rise()+servo_side_hole_pitch/2])
            translate([servo_face_dx()-servo_face_t,y,z]) rotate([0,90,0])
                cylinder(d=servo_side_hole_clear_d,h=servo_face_t);
    }
}
// The four M2.5 holder screws enter from the arm's INBOARD side: free air, through the
// head's rib, then the 6 mm servo plate, into the metal holder. A screw is only usable if
// that whole line is open. Open plate bores alone are not enough: the rib that braces the
// plate sealed them from behind, so the holder bolted to nothing a driver could reach.
else if(check == "arm_servo_access") intersection() {
    arm_vertical(arm_joint_t) arm_head();
    arm_vertical(arm_joint_t) {
        union() {
            // Screw shank, from open air inboard right out through the 6 mm plate.
            for(y=[-servo_side_hole_pitch/2,servo_side_hole_pitch/2],
                z=[arm_head_rise()-servo_side_hole_pitch/2,
                   arm_head_rise()+servo_side_hole_pitch/2])
                translate([-2,y,z]) rotate([0,90,0])
                    cylinder(d=servo_side_hole_clear_d,h=servo_face_dx()+1);
            // Socket head and driver, up to - but not through - the plate it bears on.
            for(y=[-servo_side_hole_pitch/2,servo_side_hole_pitch/2],
                z=[arm_head_rise()-servo_side_hole_pitch/2,
                   arm_head_rise()+servo_side_hole_pitch/2])
                translate([-2,y,z]) rotate([0,90,0])
                    cylinder(d=servo_access_clear(),h=servo_face_dx()-servo_face_t+0.5);
        }
    }
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
// Cable path: a 14 mm square rod down the whole column, from the base sump to
// the cap socket, through every flange and spigot -- must be fully open.
else if(check == "tower_cable_path") intersection() {
    camera_tower_assembly();
    translate([-7,tower_y-7,26]) cube([14,14,tower_flange_z(tower_seg_count())+17-26]);
}
// Cap: camera-foot bolts through the roof at +-22, spigot fits the socket.
else if(check == "tower_cap_bores") intersection() {
    tower_cap();
    // Holes start at the socket ceiling (z=18) and run out the roof (z=30).
    union() for(x=[-22,22]) translate([x,0,19]) cylinder(d=3.8,h=13);
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
// Through the full crank travel the rod's servo end clears the servo body and disc,
// and the crank arm never meets the rod beyond its washer gap.
else if(check == "rod_servo_sweep") {
    for(side=[-1,1], angle=[-servo_crank_travel_deg:servo_crank_travel_deg/5:
                            servo_crank_travel_deg])
        intersection() {
            fit_rod_servo_end(side,angle);
            union() { servo_body_envelope(side); servo_crank_arm(side,angle); }
        }
}
// The kit crank's pin radius and the disc's OD are not measured on the real part, so this proves
// the rod's servo-end transition clears the WHOLE plausible band, not just the assumed 55 mm and
// Ø30: 40..70 mm pin radius, Ø26..34 disc, full travel, both sides. It says nothing about the
// rod LENGTH that a different radius implies - that is trim, and the docs state its limit.
else if(check == "rod_servo_sweep_band") {
    for(side=[-1,1], crank_r=[servo_crank_r_band_mm[0]:15:servo_crank_r_band_mm[1]],
        disc_d=[26,30,34],
        angle=[-servo_crank_travel_deg:servo_crank_travel_deg/5:servo_crank_travel_deg])
        intersection() {
            fit_rod_servo_end(side,angle,crank_r);
            union() { servo_body_envelope(side,disc_d); servo_crank_arm(side,angle,crank_r); }
        }
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
else if(check == "camera_bores") intersection() {
    camera_carrier();
    union() for(x=[-6.25,6.25],y=[-10.5,10.5]) translate([x,y,-1]) cylinder(d=2,h=11);
}
else if(check == "camera_pitch") {
    for(a=[60,90,120]) intersection() {
        camera_foot(); translate([0,0,16]) rotate([a,0,0]) translate([0,25,-8]) camera_carrier();
    }
}
else if(check == "sensor_pitch") {
    for(a=[60,90,120]) intersection() { camera_foot(); sensor_carrier_placed(a); }
}
else assert(false,"Unknown geometry check");
