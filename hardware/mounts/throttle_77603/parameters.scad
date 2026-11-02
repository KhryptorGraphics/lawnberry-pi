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
// file depends on -- arm_joint_x, arm_joint_z, arm_joint_bolt, arm_joint_flange_t,
// servo_station_x, servo_station_z, servo_face_t and the steering case envelope -- are defined
// there and are NOT copied.
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

// ---- MG995 mounting flange -----------------------------------------------------------
// The owner gave the mounting holes as FRONT-BACK hole centres, 2.00 in (50.80 mm) apart,
// 1.125 in (28.575 mm) above the body base. That is the standard MG995 arrangement: the case
// carries a flange with an EAR on each side at the front and back, so the two holes on one
// side are 50.80 mm apart in Y and the ear reaches SIDEWAYS out past the case.
//
// The first revision of the cradle read the owner's pitch as a LEFT-RIGHT span and cut a
// 50.80 mm slot along X through both side walls at the shaft's own Y. That slot ran through
// material the servo never occupies: it retained nothing, it blocked the drop-in, and the
// keeper that was supposed to bear on the ears had no ears to bear on.
//
// The values below are DESIGN CAPACITIES on the owner's numbers. They are NOT measurements.
mg995_flange_ear_offset_mm = mg995_mount_pitch_fb_mm / 2;  // 25.4: the ear rows' Y
mg995_flange_ear_w_mm = 8;        // the ear's front-back width (TO MEASURE)
mg995_flange_ear_out_mm = 11.0;   // the flange's outer edge in X. It must stay INSIDE the
                                  // cradle's outer side-wall face, so the keeper's legs can
                                  // drop beside the walls without fouling the flange (TO MEASURE)
mg995_shaft_d_mm = 6;             // shaft diameter, for the reference body only (TO MEASURE)

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
// Illustrative only: the owner's endpoint-distance datums and the actual lever arc are unknown.
rods_min_overlap_mm = 60;
rods_coarse_travel_mm = 240;      // continuous slide, held by the print-in-place cam lever
rods_fine_travel_mm = 20;         // print-in-place thumbwheel trim on the throttle fork
rods_nominal_span_mm = 1066.8;
rods_span_min_mm = 940;
rods_span_max_mm = 1200;

// Print-in-place locks (design capacities, NOT qualified values). One lever design serves
// the coarse cam lock and both section joints: its flat lands rod_lock_preload_mm past
// seating the clamped member, and its corner reaches rod_lock_bump_mm further (the
// over-centre bump that keeps it shut).
rod_lock_preload_mm = 0.3;
rod_lock_bump_mm = 0.3;
// Axial holding each lock should reach: 2 x the MG995 stall force at the 12 mm minimum horn
// radius (assignment estimate ~1.1 N*m at 6 V -> ~92 N; datasheet value TO VERIFY).
// DESIGN TARGET ONLY: unqualified until the bench coupons are pulled.
rod_holding_target_n = 185;
// Thumbwheel fine trim: a 45-degree-flank thread on the slider's tip; the wheel clicks
// rod_thread_pitch_mm / rod_wheel_detents = 0.25 mm per detent.
rod_thread_pitch_mm = 4;
rod_wheel_detents = 16;
rod_helix_gap_mm = 0.45;           // 0.4 nominal + 0.05 for facet chords on helices

// printed_joints.scad's thread primitives read these defaults; the rod no longer uses them.
fine_screw_d_mm = 16;
fine_screw_pitch_mm = 2;
fine_nut_height_mm = 10;
fine_nut_window_mm = 8;           // the printed nut's four relief windows

// ---- Printed-joint capacities ------------------------------------------------------
pin_d_mm = 5;                      // captive printed hinge pin
pin_head_d_mm = 8;
bushing_od_mm = 9;
bushing_id_mm = 5.4;               // running clearance on pin_d
bushing_len_mm = 8;
keeper_t_mm = 4;                   // secondary printed retainer
ball_d_mm = 10;
joint_articulation_deg = 30;       // declared capacity, NOT a measured mower requirement
// Pivot post shared by horn variants A-D and the U fitting's upper joint. The rod eye's
// 6.6 mm cut bore runs on the 6.2 mm spigot, and the 4.4 mm spigot gives the 4 mm eye
// 0.2 mm axial play per side. These are as-printed fits and require coupons.
pivot_flange_d_mm = 7;
pivot_flange_t_mm = 1.5;
pivot_spigot_d_mm = 6.2;
pivot_spigot_len_mm = 4.4;
pivot_cap_seat_d_mm = 7.6;
pivot_cap_seat_len_mm = 4;
pivot_lip_d_mm = 8.6;
pivot_lip_t_mm = 1.5;
pivot_cap_od_mm = 10;
pivot_cap_id_mm = 7.7;
pivot_cap_h_mm = 4;

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

// ---- Throttle station: ONE derivation ---------------------------------------------------
// The throttle MG995 stands BELOW the right steering servo, on a shelf grown from the rib of
// the derivative right arm head (throttle_arm_head.scad). Both steering servo stations are
// raised by servo_station_raise_mm (../enclosure_common.scad), so the whole throttle stack --
// cradle, servo, horn interface and the rod's servo end -- sits under the steering servo's
// case. The left head is raised by the same amount and carries no throttle parts.
//
// Every value below derives from a leaf constant, and the block sits at the END of the file:
// OpenSCAD evaluates top to bottom, so a value declared before its input is silently undef,
// and undef geometry looks exactly like geometry that rendered. The asserts at the end fail
// at parse time instead.
//
// Frames. Head-LOCAL: X outboard from the arm joint, Y across, Z up the rib; world X is
// arm_joint_x + x, world Y is assembly_arm_y + y, world Z is arm_joint_z + z. CRADLE: origin
// on the servo's base plane (the shelf's top face) at the cradle's centre, same axes.
//
// These are design capacities, not mower measurements, and not a physical safety proof.

// (1) Leaf values ---------------------------------------------------------------------
// The shaft stands far enough outboard that every horn interface swept +/-30 degrees, and
// every rod's servo section, clears the rib's outboard face (probed, not assumed).
throttle_seat_x_local_mm = 66;
throttle_seat_y_local_mm = 0;   // centred under the steering servo's shaft
// The servo's base plane, head-local Z: the lowest seat whose 45-degree corbel still roots
// on the rib above the joint flange. The steering stations were raised to clear the stack.
throttle_seat_local_mm = 92;
shelf_t_mm = 4;                 // the shelf slab, at its outboard edge
shelf_margin_mm = 2;            // shelf beyond the cradle's outboard wall
// The corbel's root is narrower than the shelf, so the joint's two outboard M6 bolts (at
// +/-arm_joint_bolt across the flange) keep tool room above them.
corbel_half_w_mm = 14;

// (2) The cradle: walls on the shelf, which is its floor ----------------------------------
cradle_wall_mm = 5;
cradle_w_mm = mg995_body_width_mm + 2 * joint_clearance_mm + 2 * cradle_wall_mm;
// The walls stop at the provisional flange UNDERSIDE; the keeper bears on the provisional
// lugs from above. Whether the real lugs also bear on the wall tops depends on their
// unmeasured width.
cradle_wall_h_mm = mg995_mount_height_mm;                          // 28.575
cradle_end_y0_mm = mg995_body_length_mm / 2 + joint_clearance_mm; // 19.45: the case's end
cradle_len_mm = 68;           // capacity around the 50.80 mm MG995 lug pitch; TO MEASURE
// The MG995's flange in the cradle frame.
cradle_flange_z0_mm = mg995_mount_height_mm;                        // 28.575: underside
cradle_flange_z1_mm = cradle_flange_z0_mm + mg995_lug_thickness_mm; // 30.575: top face

// (3) The snap keeper -------------------------------------------------------------------
// ONE printed keeper, no pin, ring or purchased fastener: a frame bears on the MG995's ears
// and four spring fingers drop beside the cradle's side walls; each finger's barb snaps into
// a window in the wall's outer face, under the window's flat roof. Two crush ribs per ear
// row stand proud of the frame's underside, so the seated keeper presses the ears onto the
// wall tops instead of rattling. Prying the four finger tails outward releases it.
// DESIGN CAPACITIES on the provisional ear geometry: fit, snap force, rib crush, hold-down
// against vibration and fatigue are bench/coupon items, not established.
keeper_finger_t_mm = 2;       // finger thickness, in X, beside the side wall
keeper_finger_w_mm = 10;      // finger width, in Y
// Finger centres: inside the end walls, over the ear rows (25.4 mm), so the pull is local.
keeper_finger_y_mm = cradle_len_mm / 2 - cradle_wall_mm - keeper_finger_w_mm / 2;   // 24
keeper_finger_x_mm = cradle_w_mm / 2 + joint_clearance_mm;   // finger inner face: 12.15
keeper_barb_d_mm = 1.0;       // barb protrusion past the finger's inner face
keeper_barb_land_mm = 0.5;   // flat land under the catch face
keeper_catch_play_mm = 0.2;   // seated (pads on the ears): catch face below the roof
keeper_window_depth_mm = 1.2; // window depth into the wall's outer face
keeper_window_roof_mm = 10;   // cradle frame: the window's flat roof
keeper_finger_tail_mm = 2.5; // finger below the barb, to pry it free
keeper_rib_h_mm = 0.4;        // crush ribs: designed interference with the ear tops
keeper_rib_x0_mm = mg995_body_width_mm / 2 + joint_clearance_mm + 0.6; // clear of the body
// Probe pose: each finger opened this far at its barb, so the barb tip stands 0.1 mm
// outside the wall's outer face.
keeper_snap_spread_mm = keeper_barb_d_mm - joint_clearance_mm + 0.1;
// Cradle frame heights of the barb: the catch face (seated) and the barb's lowest point.
keeper_catch_z_mm = keeper_window_roof_mm - keeper_catch_play_mm;
keeper_barb_z0_mm = keeper_catch_z_mm - keeper_barb_land_mm - keeper_barb_d_mm;
// The horn variants use these provisional index-pin diameters (unchanged by the snap keeper).
keeper_post_pin_d_mm = 2.4;
keeper_post_pin_bore_d_mm = 2.8;

// (4) The shelf and its corbel, sized from the cradle --------------------------------------
// The shelf reaches back 1 mm into the RIB (x 0..rib face): that is where it fuses with the
// head and where the load path runs.
rib_face_local_mm = servo_station_x - arm_joint_x - servo_face_t;   // 16.25
shelf_x0_local_mm = rib_face_local_mm - 1;
shelf_x1_local_mm = throttle_seat_x_local_mm + cradle_w_mm / 2 + shelf_margin_mm;
shelf_y0_local_mm = throttle_seat_y_local_mm - cradle_len_mm / 2;
shelf_y1_local_mm = throttle_seat_y_local_mm + cradle_len_mm / 2;
// The corbel's underside rises at 45 degrees from its root on the rib's face to the shelf's
// outboard edge, so it prints without support once the head is laid mating-face down.
corbel_z0_local_mm = throttle_seat_local_mm - shelf_t_mm
                     - (shelf_x1_local_mm - rib_face_local_mm);
// Beyond the corbel's width the shelf's wings fall at 45 degrees from the shelf's edges to
// the corbel's width, high above the joint's outboard bolts.
shelf_wing_drop_mm = shelf_y1_local_mm - throttle_seat_y_local_mm - corbel_half_w_mm;

// ---- Parse-time consistency ---------------------------------------------------------
assert(is_num(shelf_x1_local_mm) && is_num(shelf_y0_local_mm) && is_num(shelf_y1_local_mm),
       "the shelf's footprint must be defined");
assert(is_num(corbel_z0_local_mm) && is_num(keeper_barb_z0_mm),
       "the corbel and the keeper's barbs must be defined");
assert(corbel_z0_local_mm > arm_joint_flange_t,
       "the corbel must root on the rib above the joint flange");
assert(throttle_seat_x_local_mm - cradle_w_mm / 2 > rib_face_local_mm,
       "the cradle must stand outboard of the rib");
assert(corbel_half_w_mm < arm_joint_bolt - 10,
       "the corbel must leave tool room above the joint's outboard bolts");
assert(cradle_w_mm > mg995_body_width_mm + 2 * cradle_wall_mm,
       "the cradle must be wider than the servo by a printable wall each side");
assert(mg995_flange_ear_out_mm < cradle_w_mm / 2,
       "the MG995 flange must stay inside the cradle's outer wall faces");
assert(throttle_seat_local_mm + mg995_body_height_mm < servo_station_z - arm_joint_z
       - servo_case_axis_offset_z_mm - servo_case_l / 2,
       "the throttle servo must stand below the steering servo's case");
assert(mg995_flange_ear_out_mm < keeper_finger_x_mm - keeper_barb_d_mm,
       "the keeper's barb tips must pass outside the MG995's provisional ears");
assert(keeper_window_depth_mm - (keeper_barb_d_mm - joint_clearance_mm) >= joint_clearance_mm,
       "the seated barb must clear its window's floor");
assert(keeper_barb_z0_mm - keeper_finger_tail_mm > 1,
       "the keeper's finger tails must stand clear of the shelf");
assert(keeper_finger_y_mm + keeper_finger_w_mm / 2 + joint_clearance_mm
       < cradle_len_mm / 2 - cradle_wall_mm / 2,
       "the keeper's fingers must stay inside the cradle's ends");
