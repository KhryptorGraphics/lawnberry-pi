// Center sandwich layer and detachable left/right servo arms.
// FIRST-ARTICLE FIT/LOAD-TEST PROTOTYPE ONLY; not powered-steering certified.
// Printer: 420 x 420 x 500. Center layer matches the enclosure flange footprint
// and continues to the hitch tongue. Arms print separately (long axis X); they
// have a closed rectangular beam section with a central web and a solid bolt-root.
//
// Servo interface: RDS51150 stationary-holder SIDE, 4-M2.5 on 24 x 24 mm from the
// ANNIMOS listing drawing. Verify the RDS51150SG revision and actual holder before use.
// Hitch interface: 20 mm hole/aux M6 centres mirror the *enclosure* tongue, not a
// verified mower hitch drawing. A single central bolt does not establish a torque
// path for opposed servos. Use the aux holes only when the actual hitch plate can be
// drilled to match, or provide another positively keyed attachment to the mower frame.
// Lap-bar tube OD and bar/servo coordinates remain unmeasured; motor/rod placement in
// assemblies.scad is an illustrative layout, not a fit guarantee.
include <enclosure_common.scad>

arm_part = "center";                // center | left | right | gauge

echo(str("hitch_arm_layer: box footprint=", box_flange_l, " x ", box_flange_w,
         " mm; full center layer bbox=", box_flange_l+2*arm_layer_lug_out_mm,
         " x ", tongue_end_y()+box_flange_w/2, " x ", layer_t,
         " mm; hitch bore=", hitch_bore_d, " mm; gland pass-throughs=", gland_pass_d,
         " mm; servo station x=+/-", servo_station_x, " mm at z=", arm_tip_z,
         " mm; arm rake=", arm_rake_deg, " deg from vertical; lower arm=", arm_joint_t,
         " x ", arm_col_d, " x ", arm_col_w, " mm; head=", arm_head_len, " mm"));
module arm_rounded_plate(l,w,t,r) {
    hull() for(x=[-1,1],y=[-1,1])
        translate([x*(l/2-r),y*(w/2-r),0]) cylinder(r=r,h=t);
}
// The layer prints underside-down; in assembly it spans z=-8..0. Arm
// feet now seat ON its top surface rather than sharing the layer volume.
module layer_outline() {
    union() {
        arm_rounded_plate(box_flange_l,box_flange_w,layer_t,layer_corner_r);
        // Box-footprint center panel with outboard bolt ears for the two arms.
        translate([box_flange_l/2-1,arm_ear_y0,0])
            cube([arm_layer_lug_out_mm+1,arm_ear_y1-arm_ear_y0,layer_t]);
        translate([-box_flange_l/2-arm_layer_lug_out_mm,arm_ear_y0,0])
            cube([arm_layer_lug_out_mm+1,arm_ear_y1-arm_ear_y0,layer_t]);
        // Widened rear shoulder aligns to the existing tongue.
        translate([-box_flange_l/2,box_flange_w/2-layer_corner_r,0])
            cube([box_flange_l,45+layer_corner_r,layer_t]);
        // Tongue-width continuation to the ball-hole end.
        translate([-tongue_w/2,box_flange_w/2+35,0])
            cube([tongue_w,tongue_end_y()-(box_flange_w/2+35),layer_t]);
    }
}
module arm_layer_center() {
    assert(max([box_flange_l+2*arm_layer_lug_out_mm,
                tongue_end_y()+box_flange_w/2])<=420,
           "Center sandwich layer exceeds 420 mm printer bed");
    assert(arm_root_x+arm_root_bolt_x_offset()+max(joint_x)+6.25
           <= box_flange_l/2+arm_layer_lug_out_mm-3,
           "Arm root bolts/washers do not fit on the sandwich-layer arm ear");
    assert(assembly_arm_y+max([for (y=joint_y) abs(y)])+6.25 <= arm_ear_y1,
           "Arm root bolts/washers run off the rear of the sandwich-layer ear");
    difference() {
        layer_outline();
        // Existing gland and tray fastener paths pass through this added layer.
        for(p=gland_centres) translate([p[0],p[1],-1]) cylinder(d=gland_pass_d,h=layer_t+2);
        for(p=tray_fixings) translate([p[0],p[1],-1]) cylinder(d=tray_fixing_d+0.8,h=layer_t+2);
        // Existing hitch tongue centre/auxiliary bolt pattern.
        translate([0,tongue_bolt_y(),-1]) cylinder(d=hitch_bore_d,h=layer_t+2);
        for(x=[-tongue_aux_bolt_x,tongue_aux_bolt_x])
            translate([x,tongue_bolt_y(),-1]) cylinder(d=aux_bore_d,h=layer_t+2);
        // Camera tower foot bolt paths through the sandwich layer.
        for(p=tower_foot_bolts)
            translate([p[0],p[1],-1]) cylinder(d=tower_wall_bolt_d,h=layer_t+2);
        // Four vertical bolts per arm, through the ear into the arm foot.
        for(s=[-1,1],p=arm_root_bolt_points())
            translate([s*p[0],p[1],-1])
                cylinder(d=arm_joint_bore_d,h=layer_t+2);
    }
}

// Half-spaces for trimming. Each keeps everything on one side of a plane, with a deep
// slab BEHIND the plane and only a thin lip past it -- so the cut stays correct however
// far the geometry is from the plane, instead of only inside a fixed 400 mm box.
module arm_half_space_below(rake, plane_z) {
    rotate([0,rake,0]) translate([-2000,-2000,plane_z-2000]) cube([4000,4000,2000]);
}
module arm_half_space_above(rake, plane_z) {
    rotate([0,rake,0]) translate([-2000,-2000,plane_z]) cube([4000,4000,2000]);
}
// Horizontal seat and root top, expressed in the arm-local frame. The outer
// arm_pose() rotation cancels -rake and leaves assembly Z planes at 0 and 50.
module arm_seat_keep() {
    arm_half_space_above(-arm_rake_deg,arm_seat_z-arm_foot_z);
}
module arm_foot_top_keep() {
    arm_half_space_below(-arm_rake_deg,arm_seat_z-arm_foot_z+arm_foot_h);
}
// Above the head's mating plane, in an upright frame.
module arm_mate_keep() { arm_half_space_above(-arm_rake_deg,0); }

// Four vertical M6 enter from UNDER the free side ears (clear of the hitch
// plate). Each terminates in a steel nut dropped through a side-entry slot
// at z=16..22.5: the column above the root otherwise roofs over the bolts,
// making a through-fastener impossible to assemble after printing.
module arm_root_bores() {
    rotate([0,-arm_rake_deg,0])
        for(x=joint_x,y=joint_y)
            translate([arm_root_bolt_x_offset()+x,y,
                       arm_seat_z-arm_foot_z-layer_t-1])
                cylinder(d=arm_joint_bore_d,
                         h=layer_t+arm_root_nut_z+arm_root_nut_h+2);
}
module arm_root_nut_slots() {
    rotate([0,-arm_rake_deg,0])
        for(x=joint_x,y=joint_y)
            translate([arm_root_bolt_x_offset()+x-arm_root_nut_af/2,
                       y>0 ? y-arm_root_nut_af/2 : -arm_col_w/2-1,
                       arm_seat_z-arm_foot_z+arm_root_nut_z])
                cube([arm_root_nut_af,
                      arm_col_w/2-abs(y)+arm_root_nut_af/2+1,
                      arm_root_nut_h]);
}
module arm_foot_solid() {
    intersection() {
        translate([-arm_root_foot_d/2,-arm_col_w/2,0])
            cube([arm_root_foot_d,arm_col_w,
                  arm_seat_local_z()+arm_foot_h/arm_cos()+arm_col_d]);
        arm_seat_keep();
        arm_foot_top_keep();
    }
}
// ---- Sway-bar pads on the arms ----
// A pad is a boss standing proud of the arm's INBOARD face, so its exposed mate face
// -- the one the flat bar butts against -- looks back toward the tower. Built as an
// explicit polygon in the arm's local XZ plane: the inner edge follows the column face
// x=-arm_col_d/2, the outer edge follows the mate plane, offset arm_brace_pad_x along
// that plane's normal (cos(rake),0,sin(rake)). The sign matters: a pad standing
// outboard would leave the bar passing straight through it.
function arm_pad_c0() = arm_brace_mate_x();
function arm_pad_x_at(z_rel) =
    (arm_pad_c0()-sin(arm_rake_deg)*z_rel)/cos(arm_rake_deg);
// The pad's inner edge runs to x=0, the column's mid-plane, NOT to the column's
// inboard face at -arm_col_d/2. Stopping at the face would leave the pad meeting the
// column on a single plane of zero volume, which renders as a detached body. Buried to
// the mid-plane the pad has real overlap with the solid band behind it; the visible
// proud portion is unchanged.
//
// Built by hull()ing the two XZ edges at the pad's own Y width, rather than by
// linear_extrude of an XZ polygon: that extrude ran the polygon's X into the pad's Z
// and produced a solid tower beside the arm instead of a boss on its inboard face.
module arm_column_brace_pad(t) {
    translate([0,-arm_brace_pad_w/2,t]) hull() {
        for(z=[-arm_brace_pad_h/2,arm_brace_pad_h/2]) {
            translate([0,0,z]) cube([1,arm_brace_pad_w,1]);
            translate([arm_pad_x_at(z),0,z]) cube([1,arm_brace_pad_w,1]);
        }
    }
}
// Frame on the real mate plane: +X goes INTO the solid pad and Y is the
// bar/tower centreline, not the shifted arm axis. The nut lies 6 mm behind
// the face and slides in from the pad's +Y edge before fitting the bar.
module arm_pad_cut_frame(t, z_rel) {
    translate([arm_pad_x_at(z_rel),arm_brace_y-assembly_arm_y,t+z_rel])
        rotate([0,-arm_rake_deg,0]) children();
}
module arm_column_brace_pad_cuts(t) {
    for(z_rel=[-arm_brace_pad_bolt_pitch/2,arm_brace_pad_bolt_pitch/2])
        arm_pad_cut_frame(t,z_rel) {
            translate([arm_brace_pad_bolt_wall,-arm_brace_nut_af/2,
                       -arm_brace_nut_af/2])
                cube([arm_brace_nut_t,
                      arm_brace_pad_w/2-(arm_brace_y-assembly_arm_y)+
                      arm_brace_nut_af/2+1,arm_brace_nut_af]);
            translate([-1,0,0]) rotate([0,90,0])
                cylinder(d=m5_clear,
                         h=arm_brace_pad_bolt_wall+arm_brace_nut_t+2);
        }
}
// There is no head pad: a bar reaches its pad from inboard, and the head's own joint
// flange flares inboard across that path, so a head pad is unreachable by a straight
// bar. Every station is a column pad instead (see arm_brace_z).
// Sloped stations of the three column sway pads.
function arm_column_band_t() = [for (z = arm_brace_z) arm_brace_t(z)];
function arm_band_half_h() = arm_brace_pad_h/2+arm_col_wall;
// The column is ONE through-channel cut from end to end, with the sway bands unioned
// back in afterwards.
//
// This matters for more than looks. Cutting the channel as separate per-gap boxes
// left each gap enclosed on all six sides, so every one became a sealed internal
// void: a second connected solid, a vapour trap during printing, and the reason the
// arm rendered as 3 and then 5 disconnected bodies. A single channel that breaks
// out at both ends can never enclose anything, whatever the band positions are.
function arm_column_channel() = [arm_seat_local_z()-2, arm_joint_t+2];
module arm_column_channel_cut(y) {
    z = arm_column_channel();
    translate([-arm_col_d/2+arm_col_wall, y, z[0]])
        cube([arm_col_d-2*arm_col_wall,
              (arm_col_w-2*arm_col_wall-6)/2,
              z[1]-z[0]]);
}
// The channel must stop AT the inner faces of both 6 mm walls, not past them. The
// earlier +0.2 mm overshoot reached x=+19.2 through a wall whose inner face is at
// x=+19, which severed the column.
module arm_column_box_only() {
    translate([-arm_col_d/2,-arm_col_w/2,arm_seat_local_z()-1])
        cube([arm_col_d,arm_col_w,arm_joint_t-arm_seat_local_z()+1]);
}
// One uninterrupted through-channel, and NOTHING filled back across it.
// The horizontal seat clip prevents any column material from entering the
// sandwich layer, while its first 50 mm still overlap the solid root.
module arm_column_solid() {
    intersection() {
        difference() {
            arm_column_box_only();
            arm_column_channel_cut(-arm_col_w/2+arm_col_wall);
            arm_column_channel_cut(3);
        }
        arm_seat_keep();
    }
}
// The solid root and joint flange close the two formerly through-open
// channel lobes. Vent EACH lobe just above the root; otherwise both are
// sealed printing voids and the STL contains three disconnected shells.
// Keep the openings clear when hand-fitting; they are not cable ports.
module arm_column_vents() {
    for(side=[-1,1])
        translate([0,side*(arm_col_w/2+1),
                   arm_brace_t(arm_seat_z+arm_foot_h+5)])
            rotate([side*90,0,0])
                cylinder(d=8,h=arm_col_wall+4);
}
// Each sway pad (arm_brace_pad_h tall, centred on its station) must sit wholly on the
// column between the seat and the joint face. The bands that used to require these
// checks are gone; the pads are solid wedges in their own right.
assert(min([for (t = arm_column_band_t()) t-arm_brace_pad_h/2])
       >= arm_seat_local_z(),
       "Lowest sway pad runs into the arm foot; raise arm_brace_z");
assert(max([for (t = arm_column_band_t()) t+arm_brace_pad_h/2])
       <= arm_joint_t,
       "Highest sway pad runs past the joint face; lower arm_brace_z");
module arm_joint_flange_local(t) {
    difference() {
        translate([-arm_joint_flange/2,-arm_joint_flange/2,t])
            cube([arm_joint_flange,arm_joint_flange,arm_joint_flange_t]);
        for(x=[-arm_joint_bolt,arm_joint_bolt],y=[-arm_joint_bolt,arm_joint_bolt])
            translate([x,y,t-1]) cylinder(d=m6_clear,h=arm_joint_flange_t+2);
    }
}

// ---- lower arm: foot + hollow column + joint flange, printed lying flat ----
// All three sway stations sit on the column (see arm_brace_z); the head has no pad,
// because its joint flange blocks a bar approaching from inboard.
arm_column_brace_z = arm_brace_z;
module arm_lower_local() {
    difference() {
        union() {
            arm_foot_solid();
            arm_column_solid();
            arm_joint_flange_local(arm_joint_t-arm_joint_flange_t);
            // spigot locates the head so the four M6 only preload the joint
            translate([-arm_joint_spigot/2,-arm_joint_spigot/2,arm_joint_t])
                cube([arm_joint_spigot,arm_joint_spigot,arm_joint_spigot_h]);
            for(z=arm_column_brace_z) arm_column_brace_pad(arm_brace_t(z));
        }
        arm_root_bores();
        arm_root_nut_slots();
        arm_column_vents();
        for(z=arm_column_brace_z) arm_column_brace_pad_cuts(arm_brace_t(z));
    }
}

// ---- head: flange, rib and the lateral servo face, printed mating-face down ----
module arm_head() {
    difference() {
        intersection() {
            union() {
                rotate([0,-arm_rake_deg,0])
                    translate([-arm_joint_flange/2,-arm_joint_flange/2,0])
                        cube([arm_joint_flange,arm_joint_flange,arm_joint_flange_t]);
                // rib, upright, outboard of the mating plane
                translate([0,-arm_col_w/2,0])
                    cube([servo_face_dx()-servo_face_t,arm_col_w,arm_head_rise()+servo_face_h/2]);
                // servo face: normal along the machine's X, so the horn sweeps YZ
                translate([servo_face_dx()-servo_face_t,-servo_face_w/2,arm_head_rise()-servo_face_h/2])
                    cube([servo_face_t,servo_face_w,servo_face_h]);
            }
            arm_mate_keep();
        }
        for(x=[-arm_joint_bolt,arm_joint_bolt],y=[-arm_joint_bolt,arm_joint_bolt])
            rotate([0,-arm_rake_deg,0]) translate([x,y,-1]) cylinder(d=m6_clear,h=arm_joint_flange_t+2);
        // spigot pocket
        rotate([0,-arm_rake_deg,0]) translate([-arm_joint_spigot/2,-arm_joint_spigot/2,-1])
            cube([arm_joint_spigot+0.4,arm_joint_spigot+0.4,arm_joint_spigot_h+0.2]);
        arm_servo_holes();
    }
}
function servo_face_dx() = servo_station_x-arm_joint_x;   // head's outboard reach
function arm_head_rise() = arm_tip_z-arm_joint_z;        // head's rise, upright
module arm_servo_holes() {
    for(y=[-servo_side_hole_pitch/2,servo_side_hole_pitch/2],
        z=[-servo_side_hole_pitch/2,servo_side_hole_pitch/2])
        translate([servo_face_dx()-servo_face_t-1,y,arm_head_rise()+z])
            rotate([0,90,0]) cylinder(d=servo_side_hole_clear_d,h=servo_face_t+2);
}
// Print transforms. The lower arm's rotate([90,0,90]) sends local (x,y,z) -> bed
// (z,x,y), so the arm axis lies along bed X and the hollow section lies flat on the
// bed; the head prints mating-face down. The offsets are the local-frame minima, so
// every part is bed-zero without hand-tuned guesses: the sway pad is the extreme on
// the lower arm, the tilted joint flange on the head.
function arm_lower_min_x() = arm_pad_x_at(arm_brace_pad_h/2);
function arm_head_min_x() =
    -(arm_joint_flange/2*arm_cos()+arm_joint_flange_t*arm_sin());
function arm_head_min_z() = -(arm_joint_flange/2*arm_sin());
module arm_lower_print() {
    translate([0,-arm_lower_min_x(),arm_joint_flange/2]) rotate([90,0,90]) arm_lower_local();
}
module arm_head_print() {
    translate([-arm_head_min_x(),arm_joint_flange/2,-arm_head_min_z()]) arm_head();
}
// Print width of the head in X, from the tilted joint flange's projection: its lower
// corner sits T*sin(rake) inboard of its upper one.
function arm_head_print_w() = arm_joint_flange*arm_cos()+arm_joint_flange_t*arm_sin();

module arm_lower_assembly() { arm_pose() arm_lower_local(); }
module arm_head_assembly() { arm_pose() translate([0,0,arm_joint_t]) arm_head(); }
module arm_assembly_side(side) {
    if (side < 0) { arm_lower_assembly(); arm_head_assembly(); }
    else mirror([1,0,0]) { arm_lower_assembly(); arm_head_assembly(); }
}
// The lower arm prints lying flat with its axis along +X; the right-hand copy is
// mirrored into the same print quadrant so both start at the bed origin.
module arm_layer_left() { arm_lower_print(); }
module arm_layer_right() {
    translate([arm_joint_t+arm_joint_spigot_h,0,0]) mirror([1,0,0]) arm_lower_print();
}
module arm_head_left() { arm_head_print(); }
module arm_head_right() { translate([arm_head_print_w(),0,0]) mirror([1,0,0]) arm_head_print(); }

module arm_layer_gauge() {
    difference() {
        translate([-50,-18,0]) arm_rounded_plate(100,36,4,4);
        for(x=[-tongue_aux_bolt_x,0,tongue_aux_bolt_x])
            translate([x,0,-1]) cylinder(d=x==0 ? hitch_bore_d : aux_bore_d,h=6);
    }
}

// The solid feet seat on the layer TOP; the sloping beams rise outboard of
// the measured hitch plate, forward of the enclosure flange.
module arm_layer_assembly() {
    translate([0,0,-layer_t]) arm_layer_center();
    for(s=[-1,1]) arm_assembly_side(s);
}

if(arm_part=="center") arm_layer_center();
else if(arm_part=="left") arm_layer_left();
else if(arm_part=="right") arm_layer_right();
else if(arm_part=="head_left") arm_head_left();
else if(arm_part=="head_right") arm_head_right();
else if(arm_part=="gauge") arm_layer_gauge();
else assert(false,"arm_part must be center, left, right, head_left, head_right or gauge");
