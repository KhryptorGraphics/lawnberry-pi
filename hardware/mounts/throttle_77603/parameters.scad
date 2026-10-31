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
rods_coarse_travel_mm = 240;
rods_fine_travel_mm = 20;
rods_nominal_span_mm = 1066.8;
rods_span_min_mm = 940;
rods_span_max_mm = 1200;

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

// ---- Integral support column (part of the derivative arm head) ---------------------
// The head prints mating-face down, which fixes its orientation: anything it grows beyond
// the flange ends up floating high above the bed. A vertical column is therefore the only
// shape that can be added to this part and still print.
//
// All values are in the head's local frame (X outboard, Y across, Z up the rib).
column_reach_mm = 24;          // outboard of the land's face: carries the whole cradle
column_w_y_mm = 12;            // narrow foot: forward band width
// The column widens above the steering servo's moving bracket, whose highest swept point is
// world Z 458.8. Below that height the column must stay in the land's narrow forward band
// (world Y 145..157) to clear the bracket's swept Y; above it there is nothing to avoid, so
// the column can carry the whole 68 mm cradle instead of letting it overhang.
column_widen_world_z_mm = 470;
column_widen_local_mm = column_widen_world_z_mm - arm_joint_z;
column_gusset_run_mm = 14;     // geometry-driven: buttress depth into the rib
column_gusset_h_mm = 50;       // geometry-driven: buttress height
// The column's outboard face, in the head's local frame.

// The keeper: a printed crossbar with two legs that drop beside the cradle's walls and bear
// on the MG995's mounting ears, plus its own locating pins.



// ---- Throttle station placement ------------------------------------------------------
// The throttle MG995 stands DIRECTLY ABOVE the retained steering station, carried by the
// integral column on the derivative arm head. There is no outboard cantilever and no riser
// stack: an earlier revision reached 180 mm outboard to clear the steering pushrod, which
// left the head printing a long horizontal arm above the bed and the servo unsupported.
//
// The clearance that decided this was measured from the real legacy solids:
//   servo case Y 108..138, moving bracket Y 94..144 (0/45/90 deg), horn sweep X 273..303,
//   steering pushrod X 274..370 at Z 465..505. The column sits in the land's forward band
//   (world Y 145..157), inboard of world X 234, and rises to a seat above the horn's sweep,
//   so it clears all of them with margin.
//
// These are design capacities, not mower measurements, and not a physical safety proof.


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

// ---- Station, column and seat: ONE derivation -----------------------------------------
// Every value below is derived from a leaf constant declared above, and the whole block
// sits at the END of the file so that OpenSCAD's top-down evaluation cannot see an
// undefined input. Earlier revisions scattered these across the file and each reorder
// silently produced undef geometry; the asserts at the bottom now fail loudly instead.
//
// Frame: the head's LOCAL frame. X runs outboard along the arm, Y across it, Z up the rib.
// World X of a local X is arm_joint_x + local_x; world Z is arm_joint_z + local_z.

// ---- The throttle station, in one dependency order ------------------------------------
//
// 1. LEAF VALUES. Nothing below depends on these.
// 2. THE CRADLE, the printed part that holds the MG995.
// 3. THE SEAT, the way the cradle locates on the column.
// 4. THE COLUMN, sized from the seat.
// 5. THE STATION, centred on the column.
//
// OpenSCAD evaluates this file top to bottom, so the order above is not cosmetic: a derived
// value declared before its input is silently undefined, and undefined geometry looks
// exactly like geometry that rendered correctly. The asserts at the end of this file exist to
// catch that at parse time.

// (1) Leaf values ---------------------------------------------------------------------
column_seat_local_mm = 240;    // the column's top face, head-local Z
column_foot_mm = 16;           // how far the column's root reaches down into the land
column_overhang_mm = 2;        // the column reaches this far inboard of its seat
seat_y_local_mm = 29;          // the forward band: world Y 145..157

// (2) The cradle ----------------------------------------------------------------------
cradle_wall_mm = 5;
cradle_w_mm = mg995_body_width_mm + 2 * joint_clearance_mm + 2 * cradle_wall_mm;
// The side walls stop at the provisional flange UNDERSIDE. The body base rests on the floor;
// the keeper frame bears on the provisional lugs from above. Whether the real lugs also
// bear on the wall tops depends on their unmeasured width.
cradle_wall_h_mm = mg995_mount_height_mm;                         // 28.575
// The end walls capture the case's ends and stop at the same height, so the flange passes
// over them rather than fouling them.
cradle_end_h_mm = mg995_mount_height_mm;                          // 28.575
cradle_end_y0_mm = mg995_body_length_mm / 2 + joint_clearance_mm; // 19.45: the case's end
cradle_end_y1_mm = cradle_end_y0_mm + cradle_wall_mm;              // 24.45
cradle_len_mm = 68;           // capacity around the 50.80 mm MG995 lug pitch; TO MEASURE
cradle_t_mm = 8;
// The MG995's flange, in the cradle's own frame (z = 0 at the seat face).
cradle_flange_z0_mm = cradle_t_mm + mg995_mount_height_mm;          // 36.575: underside
cradle_flange_z1_mm = cradle_flange_z0_mm + mg995_lug_thickness_mm; // 38.575: top face
// The notch clearance, per side, between the flange's edge and the cradle's relief.
cradle_notch_clear_mm = 0.2;

// (3) The seat: how the cradle locates on the column ---------------------------------
//
// SIMPLE BY DESIGN. The cradle's floor plate rests on the column's top face, two printed
// DOWELS on its underside plug into blind holes in that face, and the keeper -- which bears
// on the servo's mounting ears -- retains the servo in the cradle.
//
// There is deliberately no socket, no skirt and no tongue. An earlier revision built a
// socket-and-tongue joint whose mating faces were coplanar; that made the BOOLEAN
// non-manifold and cost a long series of render cycles before the cause was found. The
// coplanarity was an artefact of probing two touching solids, not a defect in the parts.
//
// A transverse printed pin captures the two dowels. Its strength, clip retention and
// vibration resistance are unverified; this is not a safety qualification.
seat_dowel_d_mm = 6;
seat_dowel_depth_mm = 8;       // into the column's top face
seat_dowel_clearance_mm = 0.2; // per side, dowel to hole
seat_dowel_x_offset_mm = 7;      // +/- about the seat's centre, inboard of the walls
seat_dowel_y_offset_mm = 0;
seat_cross_pin_d_mm = 3.2;
seat_cross_hole_d_mm = 3.6;
seat_cross_band_d_mm = 2.4;
seat_cross_ring_bore_d_mm = 2.6;
seat_cross_ring_d_mm = 5.8;
seat_cross_ring_t_mm = 1.8;
keeper_pin_d_mm = 4;            // the keeper's own locating pins, into the cradle
keeper_pin_depth_mm = 6;
keeper_pin_clearance_mm = 0.2;
keeper_pin_x_offset_mm = (cradle_w_mm - cradle_wall_mm) / 2; // side-wall centres
keeper_pin_y_offset_mm = 0;
// The horn variants also use the provisional index-pin diameters below.
keeper_leg_t_mm = 2;               // leg thickness, in X, beside the cradle's side walls
keeper_leg_drop_mm = cradle_len_mm;   // full-length legs carry the transverse pins
keeper_post_t_mm = 4;
keeper_post_pin_d_mm = 2.4;        // provisional horn index pin
keeper_post_pin_bore_d_mm = 2.8;   // provisional horn index bore
// Two transverse printed pins pass through both keeper legs and both cradle side walls
// beyond the servo's ends and below its lugs, so the keeper cannot lift off. They are
// geometry only: fit, clip retention and vibration life are unverified.
keeper_cross_pin_y_mm = (cradle_end_y0_mm + cradle_len_mm / 2 - cradle_wall_mm) / 2;
keeper_cross_pin_z_mm = cradle_t_mm + 8;   // cradle frame
keeper_cross_pin_x0_mm = -(cradle_w_mm / 2 + joint_clearance_mm + keeper_leg_t_mm + 0.5
                           + seat_cross_ring_t_mm);
keeper_cross_pin_len_mm = 2 * (cradle_w_mm / 2 + joint_clearance_mm + keeper_leg_t_mm
                               + 0.5) + seat_cross_ring_t_mm + 2;

// (4) The column ----------------------------------------------------------------------
// The column's root must reach back into the RIB (x 0..rib_len): that is where it fuses
// with the head and where the load path runs. Stopping at the land's 6 mm face would hang
// the whole servo on that thin land.
column_x0_local_mm = servo_station_x - arm_joint_x - servo_face_t - 1;   // rib_len - 1
column_width_mm = cradle_w_mm + 8;    // wide enough to carry the whole cradle
column_x1_local_mm = column_x0_local_mm + column_width_mm;
seat_cross_pin_len_mm = column_width_mm + 8;
// Pin start in the cradle frame: the keeper ring clears the column's inboard face by
// 2.2 mm and the head clears its outboard face by 2 mm.
seat_cross_pin_x0_mm = column_overhang_mm - column_width_mm / 2 - 4;
// The column's vertical extent, head-local. The root reaches into the land (local Z
// 119.75..179.75) so the column fuses with it.
column_z0_local_mm = arm_tip_z - arm_joint_z - servo_face_h / 2 - column_foot_mm;
column_z1_local_mm = column_seat_local_mm;

// (5) The station, centred on the column -----------------------------------------------
seat_x_local_mm = column_x0_local_mm + column_width_mm / 2;
throttle_servo_x_offset_mm = seat_x_local_mm;
tower_offset_x = seat_x_local_mm;
tower_offset_y = seat_y_local_mm;
throttle_body_base_offset_mm = column_seat_local_mm + cradle_t_mm;
throttle_cradle_servo_offset_x_mm = seat_x_local_mm;
throttle_cradle_servo_offset_y_mm = 0;
// The cradle's floor sits ON the column's top face.
cradle_seat_local_mm = column_seat_local_mm;

// ---- Parse-time consistency ---------------------------------------------------------
// OpenSCAD evaluates this file top to bottom, so a value derived from a later declaration
// is silently undefined and only shows up as missing geometry. These asserts fail at parse
// time, on the first render, instead of producing an undef part.
assert(is_num(column_x0_local_mm) && is_num(column_x1_local_mm),
       "the column's footprint must be defined");
assert(is_num(column_width_mm) && is_num(column_z0_local_mm) && is_num(column_z1_local_mm),
       "the column's section must be defined");
assert(is_num(seat_x_local_mm) && is_num(seat_y_local_mm),
       "the seat datum must be defined");
assert(is_num(throttle_servo_x_offset_mm) && is_num(throttle_body_base_offset_mm),
       "the station must be defined");
assert(column_x1_local_mm - column_x0_local_mm == column_width_mm,
       "the column's declared span must match its declared width");
assert(column_x0_local_mm < servo_station_x - arm_joint_x,
       "the column's root must reach into the rib, not stop at the land's face");
assert(column_z0_local_mm < arm_tip_z - arm_joint_z + servo_face_h / 2,
       "the column's root must reach into the land");
assert(column_z1_local_mm == column_seat_local_mm,
       "the column's top must be its seat");
assert(cradle_seat_local_mm == column_seat_local_mm,
       "the cradle must sit on the column's seat face");
assert(throttle_servo_x_offset_mm - mg995_body_width_mm / 2 >= column_x0_local_mm
       && throttle_servo_x_offset_mm + mg995_body_width_mm / 2 <= column_x1_local_mm,
       "the servo must stand over the column, not off its edge");
assert(column_width_mm >= cradle_w_mm,
       "the column must be at least as wide as the cradle it carries");
assert(cradle_w_mm > mg995_body_width_mm + 2 * cradle_wall_mm,
       "the cradle must be wider than the servo by a printable wall each side");
assert(column_x0_local_mm + column_width_mm / 2 == seat_x_local_mm,
       "the station must be centred on the column");
assert(seat_dowel_x_offset_mm + seat_dowel_d_mm / 2 < cradle_w_mm / 2,
       "the dowels must land inside the cradle's footprint");
