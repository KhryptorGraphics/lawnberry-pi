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
// Four-bolt printed steering-bar clamp. The 25.4 mm tube is only a coupon default;
// measure the actual lap-bar tube and tune its fit before printing working clamps.
clamp_tube_od_mm = 25.4;
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
// ---- One clamp body, interchangeable split sleeves -------------------------------------
// The bore is FIXED at clamp_body_bore_mm and every sleeve is turned to that one OD, so the
// body, bolts, yoke and pin are printed once per side and never depend on the bar. The bar size
// is set by fitting a sleeve: the three gauge bores (tight/nominal/loose about the measured OD)
// ARE the installed sleeves, so the coupon that checked the bar is the coupon that gets fitted.
// Sleeve wall absorbs the difference, so a different bar - even one far from 25.4 mm - costs one
// small reprint, and the wall assert refuses a bar so close to the bore that the sleeve would be
// too thin to clamp.
clamp_sleeve_step_mm = 0.2;         // bore step between sleeves (matches the coupon family)
clamp_insert_clearance_mm = 0.3;    // sleeve OD to clamp bore, per side
clamp_insert_flange_t_mm = 2.5;     // end flange thickness; bears on the clamp end faces
clamp_insert_flange_grip_mm = 2.2;  // flange radial grip beyond the sleeve OD
clamp_insert_protrusion_mm = 3;     // sleeve tube projects past the body's end faces
clamp_insert_flange_gap_mm = 0.1;   // flange to body end face: retention with just fit clearance
// The clamp bore is FIXED and every sleeve is made to that one outside diameter, so the body,
// bolts, yoke and pin never depend on the bar: change clamp_tube_od_mm for a different bar and
// only the sleeve is reprinted, with the wall absorbing the difference. Raise the bore only if
// the wall assert fires - a bar too near the bore leaves a sleeve too thin to clamp.
clamp_body_bore_mm = 36.4;
clamp_insert_od_mm = clamp_body_bore_mm - 2*clamp_insert_clearance_mm;
function clamp_insert_bore_mm(index) =
    clamp_tube_od_mm + clamp_tube_clearance_mm + (index-1)*clamp_sleeve_step_mm;
function clamp_insert_wall_at_mm(index) = (clamp_insert_od_mm-clamp_insert_bore_mm(index))/2;
function clamp_insert_flange_od_mm(index) =
    clamp_insert_od_mm + 2*clamp_insert_flange_grip_mm;
function clamp_insert_reach_mm() = clamp_insert_flange_od_mm(0)/2;
clamp_r_mm = clamp_body_bore_mm / 2;
clamp_half_h_mm = clamp_r_mm + clamp_shell_mm;
clamp_bolt_y_mm = clamp_r_mm + 7;
clamp_w_mm = 2 * (clamp_bolt_y_mm + clamp_shell_mm);
clamp_l_mm = 2 * (clamp_bolt_x_mm + 7);
clamp_gauge_index = 1;

function clamp_rod_eye_sweep_radius_mm() =
    sqrt(pow(clamp_rod_eye_length_mm/2, 2) + pow(clamp_rod_eye_width_mm/2, 2));

// Exact listing's RDS51150 stationary-holder SIDE drawing, not bottom holes.
// RDS51150 vs purchased RDS51150SG revision must be confirmed with the coupon.
servo_side_hole_pitch = 24;
servo_side_hole_clear_d = 2.9;       // M2.5 with 0.4 mm diametral allowance
servo_side_tool_bore_d = 6;
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
