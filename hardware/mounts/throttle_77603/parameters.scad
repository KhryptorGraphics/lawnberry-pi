// Toro 77603 throttle-station dimensional contract.
//
// ALL GEOMETRY IS PROTOTYPE GEOMETRY. Values marked TO MEASURE are owner-supplied or
// designed capacities, not verified dimensions, and geometry_evidence_verified stays false
// until the physical machine is measured. Nothing here asserts Toro fit.
//
// Coordinate frame is the repository's shared assembly frame: +X right, +Y forward, +Z up.
// The legacy steering datum lives in ../enclosure_common.scad and is included, not copied.
//
// Include path note: the builder passes no -I, so legacy sources are reached as
// <../enclosure_common.scad> (which itself includes ../common.scad). The arm constants this
// file depends on -- arm_col_d, arm_col_w, arm_root_foot_d, arm_joint_flange, arm_joint_t,
// arm_joint_flange_t, arm_tip_z, servo_station_x -- are defined there and are NOT copied.
include <../enclosure_common.scad>

geometry_evidence_verified = false;
prototype_watermark = "PROTOTYPE - FIT / LOAD / ELECTRICAL VALIDATION REQUIRED";

// ---- MG995 envelope: owner provisional (inches converted to mm) ----------------------
mg995_body_height_mm = 38.10;      // 1.50 in, vertical body height
mg995_body_width_mm = 12.70;       // 0.50 in, left-right thickness
mg995_body_length_mm = 38.10;      // 1.50 in, front-back length
mg995_shaft_projection_mm = 12.70; // ~0.50 in above housing; approximate
mg995_mount_pitch_fb_mm = 50.80;   // 2.00 in front-back hole centres
mg995_mount_height_mm = 28.575;    // 1.125 in above body base
// TO MEASURE: shaft position, lug thickness, hole diameter, horn radius and horn holes.
mg995_shaft_xy_mm = [0, 9.525];    // DEMONSTRATION: centred laterally, 3/8 in back
mg995_lug_thickness_mm = 2.0;      // design capacity
mg995_mount_hole_d_mm = 2.7;       // design capacity

// ---- Existing X-horn interface: capacities, not measurements ------------------------
x_horn_arm_radius_mm = 18;         // TO MEASURE
x_horn_hole_d_mm = 3;              // TO MEASURE
x_horn_thickness_mm = 2;           // TO MEASURE
x_horn_hub_clear_d_mm = 12;        // clearance around the factory shaft retainer

// ---- Throttle-end inverted-U fitting: owner-selected INSIDE opening -----------------
throttle_u_inner_depth_mm = 12.70; // 0.50 in front-back, inside
throttle_u_inner_width_mm = 31.75; // 1.25 in left-right, inside
fit_clearance_front_mm = 0;        // refinement only, applied after measuring the lever
fit_clearance_side_mm = 0;
throttle_u_wall_mm = 4;
throttle_u_engagement_mm = 18;     // prototype design choice; lever engagement TO MEASURE
throttle_u_leadin_mm = 1.5;
throttle_u_fillet_mm = 2;
// Signed and UNMEASURED. 0 is a demonstration pose, not a claim that the lever is vertical.
throttle_lean_right_deg = 0;
throttle_yaw_deg = 0;
throttle_pitch_deg = 0;

// ---- Print envelope ----------------------------------------------------------------
print_xy_margin_mm = 10;           // per XY edge; usable 400
print_z_margin_mm = 10;            // usable 490 of the 420 x 420 x 500 machine
print_part_max_len_mm = 380;       // no exported piece longer than this, spigots included
build_volume_mm = [420, 420, 500];

// ---- Rod and lock skeleton ---------------------------------------------------------
sliding_clearance_mm = 0.4;        // per side, prototype FDM allowance (coupon first)
joint_clearance_mm = 0.4;
rods_w_mm = 30;                    // outer square tube across flats
rods_wall_mm = 4;
rods_bore_mm = rods_w_mm - 2 * rods_wall_mm;      // 22
rods_slider_mm = rods_bore_mm - 2 * sliding_clearance_mm; // 21.2
rods_min_overlap_mm = 60;
rods_coarse_travel_mm = 240;
rods_fine_travel_mm = 20;
// Demonstration rod length: the planar horn-to-lever distance for the anchor above
// (~150 mm), not the 1066.8 mm span that belonged to the column-mounted design. The owner's
// 40/44-inch figures have undefined datums and are NOT used as a stroke or a rod length.
rods_nominal_span_mm = 150;
rods_span_min_mm = 110;
rods_span_max_mm = 210;

// Fine-trim printed screw (fully printed; fractional turns give continuous adjustment).
fine_screw_d_mm = 16;
fine_screw_pitch_mm = 2;
fine_nut_height_mm = 10;
fine_anti_backdrive_gap_mm = 2.5;
fine_jam_collar_h_mm = 6;

// Coarse lock capacities.
wedge_taper_deg = 8;
wedge_len_mm = 90;
collet_slit_mm = 3;
collet_collar_pitch_mm = 2;
leadscrew_anti_backdrive_lip_mm = 8;

// ---- Printed-joint capacities ------------------------------------------------------
pin_d_mm = 5;                      // captive printed hinge pin
pin_head_d_mm = 8;
bushing_od_mm = 9;
bushing_id_mm = 5.4;               // running clearance on pin_d
bushing_len_mm = 8;
clevis_ear_t_mm = 6;
clevis_gap_mm = 10;                // rod eye sits in double shear inside this
keeper_t_mm = 4;                   // secondary printed retainer
ball_d_mm = 10;
ball_socket_wall_mm = 4;
joint_articulation_deg = 30;       // declared capacity, NOT a measured mower requirement

// ---- Saddle section (shared: referenced by mount, drawings and fit checks) ----------
// These live here, not in mg995_throttle_mount.scad, because the builder's other entry
// points reach this file with `use`, which imports modules/functions but NOT variables.
saddle_back_t_mm = 8;
saddle_side_w_mm = 8;
saddle_wall_mm = 6;
saddle_dovetail_w_mm = 10;
saddle_dovetail_h_mm = 4;
saddle_split_gap_mm = 0.4;
saddle_skin_clear_mm = 0.4;
saddle_side_clear_mm = 0.4;
saddle_vent_relief_d_mm = 12;

// ---- Throttle station placement, anchored to the RIGHT STEERING SERVO -----------------
// DERIVED clearance, not a mower measurement. The throttle MG995 is carried by an extension
// of the existing right steering servo bracket (owner requirement), NOT by the lower column:
//   * the steering servo case occupies world X 211.6..273.0, Y 108..138, Z 378.2..445.2
//   * its horn sweeps a radius ~55 mm about (Y 123, Z 430)
// so the throttle servo base is placed OUTBOARD in X and ABOVE the case top. The earlier
// column-mounted design could not clear this: the column rakes outboard with Z, so raising a
// riser moved the cradle further into the servo case.
// Outboard distance is derived from the steering horn sweep, not guessed: the horn reaches
// world X = servo_station_x + servo_bracket_outer_span_mm/2 + servo_crank_r
// = 209.55 + 32.75 + 55 = 297.3 mm. The throttle body (12.70 wide) must sit fully outboard
// of that, so its centre needs X >= 297.3 + 6.35 + clearance.
// The steering pushrod sweeps X 273.8..370.1 at Z 465..505, so an outboard tower inside that
// band collides with it. Clearing the rod in X needs the tower axis beyond ~395 mm, i.e.
// ~186 mm outboard of the servo station. That is a long cantilever, which is why a foot is
// required and why the arm is gusseted.
throttle_outboard_mm = 180;     // DERIVED: clears the steering horn sweep AND pushrod in X
// The steering pushrod sweeps X 273.8..370.1 at Z 465.2..505.2, and the outboard tower sits
// inside that corridor. The cradle must therefore clear ABOVE the rod band, not merely above
// the servo case (Z 482.2). 180 mm above arm_tip_z puts the cradle at Z ~610.
throttle_above_servo_mm = 180;  // DERIVED: clears servo case AND steering pushrod sweep
riser_z_mm = 120;               // extension height above the bracket anchor (printable split)
saddle_ledge_w_mm = 5;          // root-top shoulder land
saddle_flange_overhang_mm = 19; // outboard flange overhang land

// ---- Riser / cradle section (shared with fit_checks.scad) ----------------------------
// Declared here, not in mg995_throttle_mount.scad, because `use` imports only modules and
// functions: file-local scalars are invisible to sibling probes and silently render as
// undefined geometry.
// Foot that reaches inboard to the existing right steering bracket's servo land. Without it
// the outboard tower is unsupported: the steering bracket's outer edge is at
// servo_station_x + servo_bracket_outer_span_mm/2 = 242.3 mm, while an outboard-mounted
// throttle station begins near 290 mm -- a ~48 mm cantilever with nothing under it.
// The tower sits ~186 mm outboard of the servo station to clear the pushrod, so the foot must
// span all the way back to the ARM-HEAD JOINT FLANGE, which is the only structure actually
// there: world X 143.3..231.3 (arm_joint_x 187.3 +/- arm_joint_flange/2 44). The servo land
// itself is under the servo case (Y 108..138), so it cannot be the attachment; the flange
// band Y 141..157 is clear.
foot_reach_mm = 253;     // tower axis -> flange inner edge (143.3), DERIVED
tower_offset_x = mg995_body_width_mm / 2 + throttle_outboard_mm;  // land -> tower axis
// The steering servo case occupies Y 108..138 and the arm-head land is 70 mm wide
// (Y 88..158), so the tower post is offset in Y to a band that clears the case while still
// landing on the head. 16 mm wide at Y centre 149 spans 141..157: 3 mm clear of the case and
// inside the land edge.
tower_offset_y = 26;    // forward of shaft centre, DERIVED for case clearance
post_w_mm = 16;
foot_t_mm = 8;
foot_w_mm = 16;   // matches post_w_mm: the clear attachment band width
riser_wall_mm = 6;
riser_w_mm = 46;
riser_d_mm = 40;
cradle_t_mm = 8;
cradle_wall_mm = 5;
cradle_w_mm = mg995_body_width_mm + 14;
cradle_len_mm = mg995_body_length_mm + 24;

// ---- Variant C continuous radius ---------------------------------------------------
horn_radius_min_mm = 12;
horn_radius_max_mm = 30;
horn_radius_default_mm = 18;

// ---- Demonstration throttle path ---------------------------------------------------
// The lever pivot/arc and the 40/44-inch datum are unmeasured; these are illustrative only.
// The owner's rightward offsets are measured from the RIGHT STEERING BAR, so the lever's
// world X starts at the legacy lap-bar clamp estimate and adds the offset. The previous
// values (76.2 / 101.6) were absolute X from the abandoned column-mounted design and put the
// lever ~320 mm from the servo, which is not where the mower's throttle control is.
throttle_idle_offset_x_mm = lapbar_pin_x_mm + 76.2;   // bar + centre of the 63.5-88.9 band
throttle_full_offset_x_mm = lapbar_pin_x_mm + 101.6;  // bar + centre of the 88.9-114.3 band
throttle_pose_angles = [-30, -15, 0, 15, 30]; // illustrative horn sweep

function throttle_u_inner_opening_mm() =
    [throttle_u_inner_depth_mm + fit_clearance_front_mm,
     throttle_u_inner_width_mm + fit_clearance_side_mm];

assert(throttle_u_inner_depth_mm == 12.70 && throttle_u_inner_width_mm == 31.75,
       "Owner-selected U opening must remain 12.70 x 31.75 mm INSIDE");
assert(throttle_u_wall_mm >= 2.5, "U wall too thin to print reliably");
assert(rods_slider_mm > 0 && rods_slider_mm < rods_bore_mm, "Rod slider must fit its bore");
assert(rods_min_overlap_mm > 2 * pin_d_mm, "Minimum overlap must hold the end-stop capture");
assert(print_part_max_len_mm <= build_volume_mm[0] - print_xy_margin_mm,
       "Declared part length exceeds the usable bed");
assert(horn_radius_min_mm < horn_radius_default_mm
       && horn_radius_default_mm < horn_radius_max_mm,
       "Default horn radius must lie inside the continuous range");
