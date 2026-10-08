// Shared dimensions in mm; FDM fit allowances are design choices, not tolerances
// guaranteed by vendors. Exact mower model/serial and printer remain unknown.
// 76.2 x 38.1 rectangular rail is an UNVERIFIED selected envelope, not a
// specification for all TimeCutters (Toro also sells C-channel frames).
$fn = 64;
frame_rail_w = 76.2;                // depth wrapped by square U-bolt
frame_rail_h = 38.1;                // contact face width between U-bolt legs
frame_ubolt_inside_w = 40;
frame_ubolt_leg_nominal_d = 8;
frame_ubolt_center_pitch = frame_ubolt_inside_w + frame_ubolt_leg_nominal_d;
frame_ubolt_leg_hole_d = 9;
frame_ubolt_backing_t = 3;
frame_ubolt_washer_t = 2.4;
frame_ubolt_nut_h = 8;
frame_ubolt_thread_projection = 3;  // beyond full nut, not a second nut engagement
// Four-bolt printed steering-bar clamp. The bar size is set by the sleeve set below, so nothing
// here needs a caliper reading; see clamp_bar_od_mm.
clamp_tube_clearance_mm = 0.4;
clamp_shell_mm = 10;
clamp_bolt_x_mm = 18;
clamp_bolt_clear_mm = 5.5;          // four M5 clamp bolts
clamp_nut_af_mm = 8.2;
clamp_nut_depth_mm = 4.8;
clamp_pin_bore_mm = 6.6;
clamp_rod_eye_length_mm = 32;
clamp_rod_eye_width_mm = 20;
clamp_rod_eye_t_mm = 8;             // printed single-eye lug thickness
clamp_rod_gap_mm = 9;               // yoke clearance around the eye, nominally 0.5 mm per side
clamp_yoke_span_mm = 42;            // ear width clears the eye through its planar sweep
clamp_yoke_reach_mm = 42;           // total outboard yoke extension from the clamp body
clamp_yoke_root_overlap_mm = 4;     // fuse each printed ear into its half beyond the bar opening
clamp_yoke_sweep_clearance_mm = 1;
// ---- One clamp body, a SET of sleeves: choose the size by trial, not by calipers ----------
// The bore is FIXED (clamp_body_bore_mm) and every sleeve is turned to that one OD, so the body,
// bolts, yoke and pin are printed once per side and never depend on the bar. Which bar the clamp
// takes is set by which sleeve is fitted; the set below spans the plausible range for a mower
// lever, and each sleeve doubles as its own coupon - the bore is a slip fit over that bar OD - so
// the size is chosen by trying them on the bar. Nothing has to be measured. A bar that falls
// between entries needs one more sleeve printed from this list, and the wall assert refuses an
// entry so close to the fixed bore that the sleeve would be too thin to clamp.
clamp_bar_od_mm = [20, 22.2, 25.4, 28.6, 30];   // 13/16 to 1-3/16 in; extend as needed
function clamp_insert_count() = len(clamp_bar_od_mm);
function clamp_insert_bar_mm(index) = clamp_bar_od_mm[index];
function clamp_insert_bore_mm(index) = clamp_bar_od_mm[index] + clamp_tube_clearance_mm;
clamp_insert_clearance_mm = 0.3;    // sleeve OD to clamp bore, per side
clamp_insert_flange_t_mm = 2.5;     // end flange thickness; bears on the clamp end faces
clamp_insert_flange_grip_mm = 2.2;  // flange radial grip beyond the sleeve OD
clamp_insert_protrusion_mm = 3;     // sleeve tube projects past the body's end faces
clamp_insert_flange_gap_mm = 0.1;   // flange to body end face: retention with just fit clearance
// The clamp bore fits the largest sleeve plus clearance; the BAR never touches the body.
clamp_body_bore_mm = 36.4;
clamp_insert_od_mm = clamp_body_bore_mm - 2*clamp_insert_clearance_mm;
function clamp_insert_wall_at_mm(index) = (clamp_insert_od_mm-clamp_insert_bore_mm(index))/2;
function clamp_insert_flange_od_mm(index) = clamp_insert_od_mm + 2*clamp_insert_flange_grip_mm;
clamp_r_mm = clamp_body_bore_mm / 2;
clamp_half_h_mm = clamp_r_mm + clamp_shell_mm;
clamp_bolt_y_mm = clamp_r_mm + 7;
clamp_w_mm = 2 * (clamp_bolt_y_mm + clamp_shell_mm);
clamp_l_mm = 2 * (clamp_bolt_x_mm + 7);

function clamp_rod_eye_sweep_radius_mm() =
    sqrt(pow(clamp_rod_eye_length_mm/2, 2) + pow(clamp_rod_eye_width_mm/2, 2));

// Photo aa8b43: narrow stationary SIX-hole rear bracket, origin at rear opening.
// Y spans the bracket; Z is up. Photo-derived centres, NOT factory tolerances or
// measured threads. Selected M2.5 through-bolts require real steel nuts/washers.
servo_mount_bolt_d_mm = 2.5;
servo_mount_clear_d_mm = 2.9;
servo_mount_washer_od_mm = 6;        // selected M2.5 washer capacity; check actual hardware
servo_mount_washer_t_mm = 1;
servo_mount_nut_af_mm = 5;
servo_mount_nut_h_mm = 2.5;
servo_mount_tool_d_mm = 8;           // selected slim socket envelope, check actual tool
servo_rear_clearance_d_mm = 13.5;    // capacity for photo opening estimate 12 +/- 1.5
function servo_rear_hole_points_mm() =
    [[-10,18.5],[10,18.5],[-8,8],[8,8],[-8,-7.5],[8,-7.5]];
// Cutter-centre travel +/-0.4 in each axis; M2.5 shank centres can reach +/-0.6
// with diametral clearance. Preserves bore-side web; not full photo uncertainty.
function servo_rear_hole_travel_mm(index) = [0.8,0.8];
// Full-X rounded rectangular slots; h is the complete required passage length.
module servo_rear_hole_cut(h) {
    for(i=[0:len(servo_rear_hole_points_mm())-1]) {
        p = servo_rear_hole_points_mm()[i];
        travel = servo_rear_hole_travel_mm(i);
        hull() for(y=[-travel[0]/2,travel[0]/2],z=[-travel[1]/2,travel[1]/2])
            translate([0,p[0]+y,p[1]+z]) rotate([0,90,0])
                cylinder(d=servo_mount_clear_d_mm,h=h);
    }
}
module servo_rear_opening_cut(h) {
    rotate([0,90,0]) cylinder(d=servo_rear_clearance_d_mm,h=h);
}
m3_clear = 3.4;
m4_clear = 4.5;
m5_clear = 5.5;
m6_clear = 6.4;
m8_clear = 8.4;
wall = 5;
clearance = 0.4;

function frame_ubolt_required_leg_length(saddle_t) =
    frame_rail_w + saddle_t + frame_ubolt_backing_t + frame_ubolt_washer_t +
    frame_ubolt_nut_h + frame_ubolt_thread_projection;
function saddle_width(pitch, bore) = pitch + bore + 2 * wall;


module slot(d, len, h) {
    hull() for (x = [-len / 2, len / 2]) translate([x, 0, 0]) cylinder(h = h, d = d);
}
module plate(l, w, t, r = 6) {
    assert(l >= 2 * r && w >= 2 * r && t > 0);
    hull() for (x = [-1, 1], y = [-1, 1])
        translate([x * (l / 2 - r), y * (w / 2 - r), 0]) cylinder(h = t, r = r);
}
module frame_saddle(l, t, span, leg = frame_ubolt_leg_hole_d) {
    assert(l / 2 - span / 2 - leg / 2 >= 5, "U-bolt hole too near saddle end");
    plate(l, saddle_width(frame_ubolt_center_pitch, leg), t, 8);
}
module frame_ubolt_slots(t, span, leg = frame_ubolt_leg_hole_d) {
    for (x = [-1, 1], y = [-1, 1])
        translate([x * span / 2, y * frame_ubolt_center_pitch / 2, -1]) cylinder(h = t + 2, d = leg);
}
