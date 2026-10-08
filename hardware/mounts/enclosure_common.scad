// One enclosure dimensional contract, mm. ASA, 0.2 mm FDM; NOT IP-rated.
// Pi5 RP-008347-DS-1: 85x56 PCB, holes58x49 at3.5 edge offsets (reference drawing).
// All other PCB envelopes below are DESIGN CAPACITIES, not inferred vendor dimensions.
// Printer: 420 x 420 x 500 (user-stated). Body prints bed-down at ~211 x 285 x 185.
include <common.scad>
box_inner_l = 169;
box_inner_w = 128;
box_inner_h = 180;
box_wall = 4;
box_floor = 4.5;
box_flange = 17;
box_flange_t = 7;
box_corner = 8;
box_outer_l = box_inner_l + 2 * box_wall;
box_outer_w = box_inner_w + 2 * box_wall;
box_flange_l = box_outer_l + 2 * box_flange;
box_flange_w = box_outer_w + 2 * box_flange;
box_top = box_floor + box_inner_h;
seal_cord = 3;
seal_width = 3.8;
seal_depth = 2.4;
seal_offset = 3.5;
seal_fill_ratio = PI * pow(seal_cord / 2, 2) / (seal_width * seal_depth);
lid_screw_d = m4_clear;
lid_screw_offset = 11;
lid_thickness = 5;
lid_skirt_clearance = 0.6;
lid_skirt_wall = 3.4;
lid_skirt_drop = 7;
lid_skirt_out = lid_skirt_clearance + lid_skirt_wall;
lid_rib_h = 4;
lid_rib_w = 4;
lid_length = box_flange_l + 2 * lid_skirt_out;
lid_width = box_flange_w + 2 * lid_skirt_out;
lid_nut_pocket_d = 8.5;             // M4 hex 7mm AF + FDM allowance
lid_nut_pocket_depth = 3.6;          // accessible from flange underside
lid_screw_points = concat(
    [for (x = [-70, -24, 24, 70], s = [-1, 1]) [x, s * (box_outer_w / 2 + lid_screw_offset)]],
    [for (y = [-26, 26], s = [-1, 1]) [s * (box_outer_l / 2 + lid_screw_offset), y]]
);
tray_length = 167;
tray_width = 120;
tray_thickness = 3.5;
tray_lift = 10;
tray_bottom_z = box_floor + tray_lift;
tray_top_z = tray_bottom_z + tray_thickness;
tray_support_d = 12;
tray_fixing_d = m4_clear;
tray_fixings = [[-62,-52], [62,-52], [-62,52], [62,52]];
// ---- Rear hitch tongue (integral to enclosure_body.scad) ----
// A thick slab off the bottom of the +Y wall rests on the mower's horizontal
// hitch plate and bolts through its existing ball hole ("reverse trailer hitch,
// no ball"). The box stands vertical BESIDE the plate, hung from the tongue as a
// cantilever -- so the tongue is thick, wide and gusseted. Printed bed-down, its
// bending stress runs along the filament (in-plane), not across layers.
// Load check in enclosure_body.scad. The print bed is z=0 = floor bottom = tongue bottom.
tongue_w = 140;
tongue_t = 16;
tongue_reach = 100;                // wall face to bolt centre; MEASURE the plate depth first
tongue_end_margin = 32;            // material beyond the bolt centre
tongue_gusset_h = 90;
tongue_gusset_t = 8;
tongue_aux_bolt_x = 40;            // optional M6 anti-twist holes, only if the plate may be drilled
// UNVERIFIED: no Toro-specific or hitch-ball-standard source was confirmed for this
// mower. 20 mm is a generic starting point (roughly a 3/4in shank hole); print
// enclosure_body.scad's -D body_part="hitch_gauge" and measure the actual
// ball/bolt/hole before printing the body.
hitch_hole_d = 20;
function tongue_bolt_y() = box_outer_w / 2 + tongue_reach;
function tongue_end_y() = tongue_bolt_y() + tongue_end_margin;
// ---- Servo arm sandwich layer (hitch_arm_layer.scad) ----
// The sandwich layer shares the enclosure flange footprint, then runs along the
// hitch tongue to the ball hole. Motor arms bolt to outboard shoulder stations.
layer_t = 8;
layer_y_min = -box_flange_w / 2;
layer_y_max = tongue_end_y();
layer_corner_r = box_corner + box_flange;
hitch_bore_d = 20.8;               // provisional FDM clearance about hitch_hole_d
aux_bore_d = m6_clear;
gland_pass_d = 24;                 // provisional around locknuts; measure actual glands first
arm_joint_bore_d = m6_clear;
joint_x = [-12, 12];                // 24 mm pitch, four M6 per side, centred on the arm axis
joint_y = [-24, 24];                // 48 mm transverse pitch, within the box-width footprint
arm_layer_lug_out_mm = 42;          // keeps a 3 mm edge beyond the shifted root washer
arm_ear_y0 = 58;
arm_ear_y1 = 160;                   // root clears box flange forward of y=85
hitch_plate_width_mm = 160;          // SCHEMATIC ONLY: measure actual mower plate width
hitch_plate_depth_mm = 120;          // SCHEMATIC ONLY: measure actual plate depth
hitch_plate_t_mm = 8;                // SCHEMATIC ONLY: measure actual plate thickness
// ---- Servo arm: splayed column carrying a measured servo station ----
// MEASURED on this mower: the servo station sits 8-1/4 in (209.55 mm) left and
// right of the centre ball hole, reached by a 1 ft 6 in (457.2 mm) arm. The splay
// angle is not a free choice -- it is what those two numbers give once the root
// is pushed outboard far enough to clear the hitch plate, the tongue and the box.
//
// DATUM: the 1 ft 6 in is from the ball-hole PLATE TOP to the servo face.
// arm_foot_z is the extrapolated arm-axis origin BELOW the printed part.
// The solid root seats on TOP of the sandwich layer (z=0), avoiding any
// overlap with its eight-millimetre ear; the measured servo datum is unchanged.
servo_station_x = 209.55;              // MEASURED 8-1/4 in from the centre ball hole
arm_len_mm = 457.2;                    // MEASURED 1 ft 6 in, plate-top datum to servo face
arm_root_x = 110;                      // arm axis at the foot; clears plate/tongue/box
arm_rake_deg = asin((servo_station_x-arm_root_x)/arm_len_mm);
arm_rise_mm = arm_len_mm*cos(arm_rake_deg);
arm_foot_h = 50;                       // solid root height ABOVE the layer seat
arm_foot_datum_z = -layer_t-hitch_plate_t_mm;
arm_col_w = 70;
arm_col_d = 50;
arm_col_wall = 6;
arm_root_foot_d = 60;                // solid root wider than the hollow column
arm_root_nut_z = 16;                 // steel M6 nut slots above the layer seat
arm_root_nut_h = 6.5;                // verify purchased nut thickness
arm_root_nut_af = 12;                // square slide-in pocket, across-corners clearance
arm_foot_z = arm_foot_datum_z-arm_foot_h; // extrapolated arm-axis origin
arm_seat_z = 0;                        // top of the sandwich-layer ear
arm_tip_z = arm_foot_datum_z+arm_rise_mm;
assembly_arm_y = 123;                  // entire foot clears box flange (y<=85)
arm_joint_t = 355;                     // joint clears the upper pad; bed-limited
arm_joint_z = arm_foot_z+arm_joint_t*cos(arm_rake_deg);
arm_joint_x = arm_root_x+arm_joint_t*sin(arm_rake_deg);
// Local arm length: the measured 1 ft 6 in is from the datum, so the modelled part is
// that plus the foot's drop back down to the datum, measured along the raked axis.
arm_tip_local_z = arm_len_mm+arm_foot_h/arm_cos();
arm_head_len = arm_tip_local_z-arm_joint_t;
arm_joint_flange_t = 12;
arm_joint_flange = 88;
arm_joint_bolt = 26;                   // half pitch of the four M6 joint bolts
arm_joint_spigot = 30;
arm_joint_spigot_h = 5;
// Servo face is a LATERAL plate: the output shaft runs across the machine, so the
// horn sweeps the YZ plane and the rod pushes the drive handle fore-and-aft.
// Vendor ANNIMOS B0C69W2QP7 dimensioned drawing (RDS51150): case 65 x 30 x 48 with the
// output shaft along the 48 mm dimension. The mounting flange is the 65 x 30 face
// perpendicular to the shaft; it carries 4-M2.5 on 24 x 24 and is the printed head's
// mate. The kit's round disc sits on the output spline; case + disc is the drawing's
// 61.4 mm overall axial envelope, and the metal crank's inner face lands on the disc.
servo_case_l = 65;                     // case length, along Y (across the arm)
servo_case_w = 30;                     // case width, along Z (up)
servo_case_axial = 48;                 // case depth along the output shaft
servo_axial_env = 61.4;                // drawing: case + disc, overall axial envelope
servo_disc_t = servo_axial_env-servo_case_axial;   // disc proud of the flange: 13.4
servo_disc_d = 30;                     // MEASURE: the drawing does not dimension the disc;
                                       // ~30 mm scaled off its side view (= the case width)
servo_shaft_offset_l = 0;              // Output position along the case (Y) from the 4-M2.5
                                       // pattern centre = 0, i.e. the pattern is CONCENTRIC
                                       // with the output, which is what the mount is drilled
                                       // for. Established from the user's photos of the metal
                                       // boss, not the drawing: the plate carries the Ø26.5 mm
                                       // output opening and its screw holes on one face, the
                                       // nearest holes only ~23.6 mm out. A 19 mm offset - what
                                       // a naive read of the dimensioned drawing suggests -
                                       // would bury a hole inside that opening, so the drawing
                                       // view was a composite of the two ends. Setting this
                                       // nonzero moves the shaft, envelope, crank and pin.
function servo_shaft_y() = assembly_arm_y+servo_shaft_offset_l;
servo_face_t = 6;
servo_face_w = 70;                     // along Y, across the arm; covers the 65 mm flange
servo_face_h = 60;                     // up
servo_face_out_x = arm_col_d/2;       // outboard face of the column; the head's reach to
                                      // the servo plate is derived, never assumed
// ---- Drive-lever linkage: Toro TimeCutter MAX 50 in MyRIDE, model 77502 ----
// ESTIMATES, not measurements. Toro publishes no lever coordinates; these come
// from the 77502 specification/product images and the measured servo station.
// See TORO_77502_LINKAGE.md, then measure the lapbar_pin_* and crank values.
// The crank is the RDS51150SG kit's metal output arm/holder, crank-up at neutral.
servo_crank_r = 55;          // MEASURE the kit arm's pin radius; rod_servo_sweep proves the
                             // rod transition clears the case and disc over +/-35 deg
servo_crank_t = 4;           // metal arm thickness at that pin; MEASURE
servo_crank_face_x = servo_axial_env;  // crank inner face = disc outer face; the drawing's
                             // 61.4 mm overall envelope, NOT a 30 mm body plus a 3 mm disc
servo_crank_washer = 2.5;    // washer stack between arm and rod eye; absorbs trim
servo_crank_travel_deg = 35; // each way from vertical neutral; swept in fit_checks
lapbar_pin_fwd_mm = 889;     // ESTIMATE 35 in: hitch-hole centre forward to clamp pin
lapbar_pin_x_mm = 355.6;     // ESTIMATE 14 in: lap-bar clamp pin from mower centreline
lapbar_pin_dz_mm = 0;        // clamp pin above crank pin at neutral; 0 keeps rod level
function servo_crank_eye_x() = servo_station_x+servo_crank_face_x+servo_crank_t
                               +servo_crank_washer+clamp_rod_eye_t_mm/2;
function servo_crank_pin_at(side, angle) =
    [side*servo_crank_eye_x(), servo_shaft_y()+servo_crank_r*sin(angle),
     arm_tip_z+servo_crank_r*cos(angle)];
function servo_crank_pin(side) = servo_crank_pin_at(side, 0);
function lapbar_pin(side) =
    [side*lapbar_pin_x_mm, tongue_bolt_y()+lapbar_pin_fwd_mm,
     arm_tip_z+servo_crank_r+lapbar_pin_dz_mm];
// Vendor case 65 x 30 x 48 outboard of the mounting flange, then the kit's disc on the
// output spline, out to the drawing's 61.4 mm overall axial envelope.
module servo_body_envelope(side) {
    translate([side > 0 ? servo_station_x : -servo_station_x-servo_case_axial,
               servo_shaft_y()-servo_case_l/2, arm_tip_z-servo_case_w/2])
        cube([servo_case_axial,servo_case_l,servo_case_w]);
    translate([side > 0 ? servo_station_x+servo_case_axial
                        : -servo_station_x-servo_case_axial-servo_disc_t,
               servo_shaft_y(), arm_tip_z])
        rotate([0,90,0]) cylinder(d=servo_disc_d,h=servo_disc_t);
}
// Metal output arm at a crank angle (0 = up), inner face on the servo disc.
module servo_crank_arm(side, angle) {
    shaft = [0,servo_shaft_y(),arm_tip_z];
    pin = servo_crank_pin_at(side, angle);
    hull() for(p=[shaft,pin])
        translate([side > 0 ? servo_station_x+servo_crank_face_x
                            : -servo_station_x-servo_crank_face_x-servo_crank_t,p[1],p[2]])
            rotate([0,90,0]) cylinder(d=14,h=servo_crank_t);
}
// Printed sway bars tie each arm section back to the camera base mount's column.
// Each bar lies in a vertical plane at the tower column centreline, so it lands square
// on both the tower's +X face and a wedge pad on the arm's inboard face.
// The bar/collar dimensions need tower_od and tower_y, so they follow that block.
//
// All three pads sit on the lower arm's column. Station 0 anchors to the tower
// base below its joint flange; stations 1/2 sit above the base. Each bar is
// laterally fitted, not slid over the tower's flanged joints.
arm_brace_z = [110, 190, 250];
arm_brace_pad_x = 20;
arm_brace_pad_w = 46;
arm_brace_pad_h = 50;
arm_brace_pad_bolt_pitch = 34;
arm_brace_nut_af = 9;                    // M5 nut square pocket
arm_brace_nut_t = 5;
arm_brace_pad_bolt_wall = 6;          // solid pad material ahead of an M5 nut
// Pad mate plane shared with the arm's wedge profile.
// Pad mate plane in assembly X: arm_pose() rotates the sloping column +rake.
// In arm-local XZ, the wedge's outer edge satisfies
// x*cos(rake) + z*sin(rake) = -(column_depth/2)*cos(rake) - pad_reach.
// Rotation cancels z, leaving this offset from arm_point(t)[0].
function arm_brace_mate_x() = -arm_col_d/2*arm_cos()-arm_brace_pad_x;
// Four vertical M6 per foot, centred on the arm axis at the seat height.
function arm_root_bolt_x_offset() = (arm_seat_z-arm_foot_z)*tan(arm_rake_deg);
function arm_root_bolt_points() =
    [for (x = joint_x, y = joint_y)
        [arm_root_x + arm_root_bolt_x_offset() + x, assembly_arm_y + y]];
// ---- Shared splayed-arm frames ----
// Local arm frame: origin on the extrapolated axis below the printed foot,
// +Z along the raked axis, +X outboard and +Y lateral. arm_pose() places
// the arm; arm_vertical(t) gives an upright station frame.
function arm_sin() = sin(arm_rake_deg);
function arm_cos() = cos(arm_rake_deg);
function arm_seat_local_z() = (arm_seat_z-arm_foot_z)/arm_cos();
function arm_point(t) = [arm_root_x+t*arm_sin(), assembly_arm_y, arm_foot_z+t*arm_cos()];
module arm_pose() { translate([arm_root_x, assembly_arm_y, arm_foot_z]) rotate([0,arm_rake_deg,0]) children(); }
module arm_vertical(t) { translate(arm_point(t)) rotate([0,-arm_rake_deg,0]) children(); }
// Sloped station on the arm axis that lands at absolute z: the inverse of arm_point's
// z. arm_support.scad uses it to find each sway pad's mate face.
function arm_brace_t(z) = (z-arm_foot_z)/arm_cos();
assembly_layer_z = -layer_t;


// ---- Camera tower (camera_tower.scad) on the +Y wall, standing on the tongue ----
// Hollow square column; the bore routes a USB camera cable down into the box
// through a wall port between the two electronics stacks. Split into flanged
// segments for a 420 x 420 x 500 printer; segments print LYING FLAT so bending
// stress runs along the filament (ASA interlayer is ~4x weaker than in-plane).
tower_od = 50;
tower_wall = 5;
tower_flange = 70;                 // square joint flange side
tower_flange_t = 6;
tower_flange_bolt = 28;            // half pitch of the 4 M4 joint bolts
tower_spigot_len = 15;
tower_spigot_clear = 0.4;          // per side; spigot slides into the next bore
tower_base_w = 110;                // wall flange width (X); fits between the tongue gussets
tower_base_h = 150;                // wall flange height above the tongue top
tower_base_t = 8;
tower_base_bolts = [for (x = [-40, 40], z = [40, 90, 140]) [x, z]];   // wall XZ, z from box floor
tower_wall_bolt_d = m5_clear;
tower_port_d = 24;                 // passes a USB-A plug; needs a split grommet/sealant, not a gland
tower_port_z = 110;                // between relay decks 98/138 and the two stack columns at x=0
tower_foot_bolts = [[-35, box_outer_w / 2 + 25], [35, box_outer_w / 2 + 25]];  // XY through tongue
tower_y = box_outer_w / 2 + tower_base_t + 15 + tower_od / 2;   // tube centre: clears the lid skirt
tower_base_top_z = 160;            // absolute z of the base's joint flange top
tower_cap_t = 30;                  // socket 18 + nut trap + 8 mm roof
tower_height = 900;                // tongue top (z=0) to camera-foot surface; MEASURE on the mower
tower_seg_max = 400;               // lying flat on a 420 bed, spigot included
cam_foot_bolt_x = 22;              // camera_mount.scad camera_foot() slot centres
function tower_seg_span() = tower_height - tower_base_top_z - tower_cap_t;
function tower_seg_count() = ceil(tower_seg_span() / tower_seg_max);
function tower_seg_len() = tower_seg_span() / tower_seg_count();
function tower_flange_z(i) = tower_base_top_z + i * tower_seg_len();   // joint i, i=0 is base top
// Sway bars reach the camera base mount, whose column is defined above.
arm_brace_y = tower_y;                   // camera tower column centreline
arm_brace_bar_t = 12;
arm_brace_bar_w = 34;
arm_brace_collar_t = 12;
arm_brace_collar_od = tower_od+20;
arm_brace_collar_id = tower_od+2;      // 1 mm per side keeps the base webs clear
arm_brace_collar_bolt_y = 14;          // +/-Y in the collar wall; not +/-Z
arm_brace_pair_gap = 0.4;             // gap between the two handed jaw halves
assert(arm_brace_z[0] > tongue_t+tower_base_t,
       "Lowest sway station lands inside the camera base foot, not on its column");
assert(arm_brace_y+tower_od/2 <= box_outer_w/2+box_flange_w/2,
       "Sway bar collar fouls the camera tower column");
assert(tower_seg_len() >= 100, "Tower segments would be shorter than 100 mm; lower tower_height");
assert(tower_seg_len() + tower_spigot_len <= 420, "Tower segment exceeds the 420 mm bed lying flat");
assert(tower_y - tower_od / 2 >= lid_width / 2 + 2, "Tower tube fouls the lid skirt");
assert(tongue_bolt_y() - tower_y - tower_od / 2 >= 18, "Hitch bolt head fouls the tower tube");
assert(tower_base_w + 2 * tongue_gusset_t + 4 <= tongue_w, "Tower base does not fit between gussets");
assert(box_flange_w / 2 + tongue_end_y() <= 420 && box_flange_l <= 420, "Body exceeds the 420 bed");
gland_cutout_d = 15.5;              // PG9 provisional print clearance; verify actual thread/locknut
vent_cutout_d = 12.5;               // M12 provisional print clearance; verify purchased fitting
// Front cable corridor, kept outside Pi PCB and between supports.
gland_centres = [[-40,-49], [0,-49], [40,-49]];
pi_pos = [-45,3];                  // PCB long axis rotated into Y
pi_screw_d = 2.9;                  // M2.5, never M3 through the Pi PCB
pi_post_h = 8;
pi_post_d = 6;
// Rotate the published lower-left-coordinate hole pattern 90deg about PCB centre.
pi_holes = [for (u = [3.5,61.5], v = [3.5,52.5])
    [pi_pos[0] - (v - 28), pi_pos[1] + (u - 42.5)]];
pi_pcb_z = tray_top_z + pi_post_h;
pi_hardware_top = pi_pcb_z + 54;    // reserved Pi+cooler+Hailo envelope, not actual stack measurement
utility_pos = [-45,3];
relay_pos = [47,3];
relay_sled_l = 62;
relay_sled_w = 100;
utility_sled_l = 77;
utility_sled_w = 100;
sled_t = 3;
relay_anchors = [for(x=[-25,25],y=[-42,42]) [relay_pos[0]+x,relay_pos[1]+y]];
utility_anchors = [for(x=[-33.5,33.5],y=[-40,40]) [utility_pos[0]+x,utility_pos[1]+y]];
utility_deck_z = [90,130];          // absolute assembly z; threaded rods OUTSIDE Pi PCB
relay_deck_z = [tray_top_z, tray_top_z+40, tray_top_z+80, tray_top_z+120];
// Four printed relay sleds, one PCB per level; 8mm insulating spacers + PCB/component envelope25mm.
relay_payload_height = 25;
utility_payload_height = 26;
assert(lid_length <= 420 && lid_width <= 420, "Lid exceeds the 420 bed");
assert(seal_fill_ratio < 0.85 && seal_fill_ratio > 0.7);
assert(lid_screw_offset - seal_offset - seal_width/2 - lid_nut_pocket_d/2 >= 1,
       "Nut pocket/gasket ligament too thin");
assert(box_flange - lid_screw_offset - lid_nut_pocket_d/2 >= 0.7);
assert(pi_hardware_top + 8 <= utility_deck_z[0], "Pi stack collides with utility deck");
assert(utility_deck_z[1]+sled_t+8+utility_payload_height < box_top-lid_rib_h);
assert(relay_deck_z[3]+sled_t+8+relay_payload_height < box_top-lid_rib_h);

module enclosure_rounded_prism(l,w,h,r) { plate(l,w,h,r); }
module enclosure_seal_path(h, width=seal_width) {
    difference() {
        plate(box_outer_l+2*seal_offset+width, box_outer_w+2*seal_offset+width, h,
              box_corner+seal_offset+width/2);
        translate([0,0,-0.5])
            plate(box_outer_l+2*seal_offset-width, box_outer_w+2*seal_offset-width, h+1,
                  box_corner+seal_offset-width/2);
    }
}
module enclosure_lid_transform() {
    translate([0,0,box_top+lid_thickness]) rotate([180,0,0]) children();
}
