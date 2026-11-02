// Segmented, fully printed 77603 throttle pushrod with NO purchased fasteners: print-in-place
// levers lock the section joints and the coarse slide, and a print-in-place thumbwheel gives
// detented fine trim. The lever and thumbwheel primitives are shared with the steering rods
// in ../pip_rod_joints.scad (documented there).
//
// PROVISIONAL DESIGN. The 940-1200 mm setting range is the requested illustrative adjustment
// range. It is NOT a measured mower span, NOT a measured throttle stroke and NOT a verified
// lever path. Nothing in this file proves fit, clamp force, holding force, detent torque,
// wear life or vibration resistance. Those remain coupon, bench and mower tests.
//
// Printed parts (four prints, nine bodies, zero loose pieces):
//   rod_segment_lower    servo section: servo eye, tube, socket boss, joint lever  (2 bodies)
//   rod_segment_middle   middle section: spigot, tube, socket boss, joint lever    (2 bodies)
//   rod_capture_assembly sleeve (spigot, tube, cam housing), slider, cam lever     (3 bodies)
//   rod_fork_assembly    throttle-eye fork with its captive thumbwheel             (2 bodies)
//
// Mechanisms
// ----------
// * LEVER (pip_rod_joints.scad, used three times). A rigid lever prints in place on two
//   D-section pins standing inward from two cheeks on the housing and turns 90 degrees. As printed
//   (open) its arm sticks out radially and lies on the bed. Closing, its lobe's corner
//   passes the clamped member about 15 degrees before closed, then a flat lands
//   rod_lock_preload_mm past seating the member (after the pivot's 0.4 mm play), the corner
//   having reached rod_lock_bump_mm further: the closed lever sits in that over-centre valley,
//   its arm on the housing face. Clamp force and holding are UNQUALIFIED.
// * COARSE LOCK: the cam lever on the sleeve's tip clamps the square slider anywhere in its
//   240 mm travel (continuous friction). V ticks on the slider's +X flat are read against the
//   index notch in the sleeve's tip face. An integral 45-degree lug in the slider's closed -X
//   groove is an overtravel catch only.
// * SECTION JOINTS: each socket boss carries a joint lever. Closed, its lobe clamps the
//   inserted square spigot (the square keys it against rotation) and a TOOTH on its arm drops
//   through the boss wall into a pocket in the spigot. The tooth's flanks and the pocket's
//   walls are arcs about the pivot, so the tooth swings in without sweeping the pocket and a
//   withdrawal load presses through the pivot: it cannot cam the lever open. Withdrawal is
//   blocked positively (probed), not by friction.
// * FINE TRIM: the slider's tip is a round 4 mm-pitch, 45-degree-flank thread. The fork's
//   thumbwheel is printed captive in the fork and screws along that thread, carrying the fork
//   and its throttle eye. Two flexure pawls drop into 16 rim notches: 0.25 mm of eye travel
//   per click over the 20 mm fine range (80 clicks). With its eye on the U post the fork
//   cannot turn with the wheel. Detent torque and thread holding are UNQUALIFIED.
//
// Holding target (UNQUALIFIED, bench coupon test required): rod_holding_target_n (185 N)
// axially for every lock: twice the ~92 N an MG995 stalling at ~1.1 N*m (assignment estimate,
// datasheet value TO VERIFY) pushes through the 12 mm minimum horn radius. Coupons:
// rod_cam_coupon, rod_joint_coupon + rod_joint_coupon_spigot.
//
// Axes and mating interfaces (unchanged)
// --------------------------------------
// * Rod-local +Z runs from the servo pin toward the throttle pin; the servo pin centre is
//   rod-local (0,0,0). Both eyes are horizontal plates 4 mm thick in X, 12 mm wide in Y, with a
//   6.6 mm cut bore along X, in the x = 0 plane. The tube axis is cranked rod_eye_crank_mm
//   above that plane (+X). Every member is modelled in the TUBE-AXIS frame (tube axis on z,
//   servo bore at x = -crank) at its installed station; throttle_pushrod() adds the crank.
// * The levers are on +Y (world: beside the rod), the ticks on +X (world: on top), the
//   overtravel lug and groove on -X. The scene places the rod with rotate([0, -90, 0]) and a
//   yaw: rod-local X -> world +Z.
include <parameters.scad>
use <../pip_rod_joints.scad>

// ---- Section --------------------------------------------------------------------------
rod_w = rods_w_mm;                           // 30 mm outer square tube
rod_bore = rods_bore_mm;                     // 22 mm square bore: the slider's passage
rod_slider = rods_slider_mm;                 // 21.2 mm square slider
rod_gap = joint_clearance_mm;                // 0.4 print-in-place gap
rod_coarse_travel = rods_coarse_travel_mm;   // 240
rod_fine_travel = rods_fine_travel_mm;       // 20

// ---- Eyes (the horn and U-post interface; unchanged) ------------------------------------
rod_eye_plate_t = 4;
rod_eye_plate_w = 12;
rod_eye_bore_d = 5.8;                        // cut bore 5.8 + 2 x 0.4 = 6.6
rod_eye_crank_mm = rod_w / 2 + rod_eye_plate_t / 2 - 1;   // 16
rod_eye_reach_mm = 18;                       // servo bore to the tube's start
rod_eye_fuse_len = 30;

// ---- Chain layout (rod z, tube-axis frame; print-plan lengths, not mower measurements) ----
rod_joint_mouth_z = [330, 600];              // socket mouths: servo/middle, middle/sleeve
rod_tip_z = 896;                             // sleeve tip face (cam housing)
rod_slider_len = 360;
rod_thread_len = 35;                         // slider's round threaded tip
rod_sq_len = rod_slider_len - rod_thread_len;   // 325: square, clamped body
rod_fork_offset = 8;                         // throttle eye ahead of the slider tip at f = 0
rod_slider_start_min = rods_span_min_mm - rod_fork_offset - rod_slider_len;   // 572
rod_lug_z = 850;                             // the sleeve's integral overtravel lug
rod_groove_z0 = 30;                          // slider-local closed groove ends (-X face)
rod_groove_z1 = 323;
rod_groove_w = 6;
rod_groove_depth = 4.6;

// ---- Section joints -------------------------------------------------------------------
rod_spigot_w = 34;                           // square spigot: keyed against rotation
rod_socket_w = 35.2;                         // 0.6 mm per side
rod_boss_w = 42;
rod_boss_len = 42;
rod_socket_straight = 33.4;                  // then a 45-degree pyramid down to the bore
rod_spigot_insert = 33.8;                    // spigot end below the mouth when seated
rod_spigot_flare = 2;                        // 34 -> 30 just outside the mouth

// ---- Levers (pip_rod_joints.scad) ------------------------------------------------------------
// The cam lock clamps the slider through the sleeve's +Y wall; each joint lever clamps a spigot
// through its boss's +Y wall and drops a 4.5 mm tooth on a 20 mm radius into its pocket.
function rod_cam_spec() =
    pip_lever_spec(rod_w / 2, rod_bore / 2, rod_slider / 2, pivot_out = 3, arm = 32,
                   width = 12, pin_r = 3, gap = rod_gap, preload = rod_lock_preload_mm,
                   bump = rod_lock_bump_mm);
function rod_joint_spec() =
    pip_lever_spec(rod_boss_w / 2, rod_socket_w / 2, rod_spigot_w / 2, pivot_out = 2.4,
                   arm = 26, width = 12, pin_r = 3, gap = rod_gap,
                   preload = rod_lock_preload_mm, bump = rod_lock_bump_mm,
                   tooth_lt = 20, tooth_w = 4.5, pocket_d = 2.8);
rod_lever_side = 90;                         // levers on +Y

// ---- Thumbwheel and fork (fork frame: wheel's servo face z = 0, rod +Z up) -----------------
function rod_wheel_spec() =
    pip_wheel_spec(rod_slider / 2, depth = 1.35, pitch = rod_thread_pitch_mm, wheel_r = 16,
                   wheel_h = 12, detents = rod_wheel_detents, gap = rod_gap,
                   hgap = rod_helix_gap_mm);
rod_wheel_tip_back = 34;                     // wheel servo face behind the tip, at f = 0
rod_fork_top = 35;
rod_fork_eye_z = rod_wheel_tip_back + rod_fork_offset;   // 42
rod_fork_floor_x = -rod_eye_crank_mm - rod_eye_plate_t / 2;   // -18: the eye's underside
// Slider-local centre of the thread ridge's first turn. The wheel then sits exactly 8 notches
// from its print pose at zero fine extension, so every 0.25 mm of fine extension is a seat.
rod_ridge_z0 = rod_sq_len + 3;
function rod_ridge_a1() =
    (rod_slider_len - pip_get(rod_wheel_spec(), "b") - 0.3 - rod_ridge_z0)
    / rod_thread_pitch_mm * 360;

// ---- Adjustment mapping -----------------------------------------------------------------
// Coarse travel is used first; the final 20 mm is fine trim. Deterministic for the scenes,
// not installer advice.
function rod_coarse_extension_mm(length_mm) =
    min(rod_coarse_travel, max(0, length_mm - rods_span_min_mm));
function rod_fine_extension_mm(length_mm) =
    min(rod_fine_travel, max(0, length_mm - rods_span_min_mm - rod_coarse_travel));
function rod_slider_start_z(length_mm) =
    rod_slider_start_min + rod_coarse_extension_mm(length_mm);
function rod_slider_tip_z(length_mm) = rod_slider_start_z(length_mm) + rod_slider_len;
function rod_throttle_pin_z(length_mm) =
    rod_slider_tip_z(length_mm) + rod_fork_offset + rod_fine_extension_mm(length_mm);
// The slider-local station read at the sleeve's index for a setting.
function rod_tick_station(length_mm) = rod_tip_z - rod_slider_start_z(length_mm);
// The wheel's detent seat (degrees from its print pose) at fine extension f.
function rod_wheel_seat(f) =
    pip_wheel_seat_turn(rod_wheel_spec(),
                        rod_slider_len - rod_wheel_tip_back + f - rod_ridge_z0);
// The slider in the capture print pose: its tip flush with the sleeve's tip.
function rod_print_start_z() = rod_tip_z - rod_slider_len;

// Accessors for files that `use` this one (they cannot see its variables).
function rod_mouth_z(j) = rod_joint_mouth_z[j];
function rod_tip() = rod_tip_z;
function rod_slider_length() = rod_slider_len;
function rod_lug_station() = rod_lug_z;
function rod_groove_ends() = [rod_groove_z0, rod_groove_z1];
function rod_tip_back() = rod_wheel_tip_back;
function rod_side_deg() = rod_lever_side;
function rod_crank() = rod_eye_crank_mm;
function rod_height_lower() = rod_joint_mouth_z[0] + 6;
function rod_height_middle() =
    rod_joint_mouth_z[1] - (rod_joint_mouth_z[0] - rod_spigot_insert);
function rod_height_capture() =
    max(rod_slider_len, rod_tip_z - (rod_joint_mouth_z[1] - rod_spigot_insert));
function rod_height_fork() = rod_fork_eye_z + 6;

assert(rod_eye_crank_mm == 16, "The eye crank is a published interface");
assert(abs(rod_throttle_pin_z(rods_span_min_mm) - rods_span_min_mm) < 1e-6,
       "The shortest setting must place the throttle pin exactly at 940 mm");
assert(abs(rod_throttle_pin_z(rods_span_max_mm) - rods_span_max_mm) < 1e-6,
       "The longest setting must place the throttle pin exactly at 1200 mm");
assert(abs(rod_throttle_pin_z(rods_nominal_span_mm) - rods_nominal_span_mm) < 1e-6,
       "The nominal setting must place the throttle pin exactly at its published span");
assert(rod_coarse_travel + rod_fine_travel == rods_span_max_mm - rods_span_min_mm,
       "Coarse plus fine travel must cover the 940-1200 mm range");
assert(max(rod_height_lower(), rod_height_middle(), rod_height_capture(), rod_height_fork())
       <= print_part_max_len_mm, "Every rod print must fit the declared print height");
assert(rod_tip_z - rod_slider_start_z(rods_span_max_mm) >= rods_min_overlap_mm,
       "The slider must overlap the sleeve by the minimum at the longest setting");
assert(rod_slider_start_z(rods_span_min_mm) + rod_sq_len > rod_tip_z,
       "At 940 mm the slider's thread must lie beyond the sleeve's tip");
assert(rod_tip_z - pip_get(rod_cam_spec(), "rb") - pip_get(rod_cam_spec(), "hf") - 1
       > rod_slider_start_z(rods_span_max_mm) + 20,
       "At 1200 mm the cam must bear on the slider well inside its servo end");
assert(rod_lug_z - rod_slider_start_z(rods_span_max_mm) > rod_groove_z0
       && rod_lug_z + 7 - rod_print_start_z() < rod_groove_z1,
       "The capture lug must stay inside the slider groove at every setting and as printed");
assert(rod_groove_z1 < rod_sq_len - 1.5, "The groove must end on the square body");
assert(rod_lug_z + 7 < rod_tip_z - pip_get(rod_cam_spec(), "rb")
       - pip_get(rod_cam_spec(), "hf") - pip_get(rod_cam_spec(), "roof") - 2,
       "The lug must stand clear of the cam window");
assert(rod_tick_station(rods_span_min_mm) < rod_sq_len - 0.5
       && rod_tick_station(rods_span_min_mm + rod_coarse_travel) > 0,
       "Ticks must lie on the slider's square body");
assert(rod_ridge_z0 - pip_get(rod_wheel_spec(), "b") - 0.1 > rod_sq_len,
       "The first ridge must start on the round tip");
assert(rod_slider_len - rod_wheel_tip_back - rod_sq_len >= 1,
       "At zero fine extension the wheel must clear the square body's end");
assert(rod_wheel_tip_back - rod_fine_travel >= pip_get(rod_wheel_spec(), "wheel_h"),
       "At full fine extension the wheel must still engage its whole height");
assert(rod_fork_eye_z - (rod_wheel_tip_back + 5.5) >= 2,
       "The slider tip must stay clear of the throttle eye's cap");
assert(rod_fork_eye_z - rod_fork_top >= 6, "The fork body must stop below the eye's cap");
assert(abs((360 * (rod_slider_len - rod_wheel_tip_back - rod_ridge_z0) / rod_thread_pitch_mm)
           % (360 / rod_wheel_detents)) < 1e-6,
       "Zero fine extension must mesh the thread exactly at a detent seat");

// ---- Fixed members (rod frame, installed positions) ----------------------------------------
module rod_at_cam() {
    pip_at_lever(rod_cam_spec(), rod_tip_z, rod_lever_side) children();
}
module rod_at_joint(j) {
    pip_at_lever(rod_joint_spec(), rod_joint_mouth_z[j], rod_lever_side) children();
}
// Pocket cut into joint j's spigot. The spigot is modelled centred; the closed lever seats it
// across its side play, away from the lever, so the pocket is cut where it lies once seated.
module rod_spigot_pocket(j) {
    s = rod_joint_spec();
    translate(pip_get(s, "clr") * [cos(rod_lever_side), sin(rod_lever_side), 0])
        rod_at_joint(j) pip_lever_pocket(s);
}

module rod_servo_body() {
    m = rod_joint_mouth_z[0];
    eye_x0 = -rod_eye_crank_mm - rod_eye_plate_t / 2;
    difference() {
        union() {
            translate([eye_x0, -rod_eye_plate_w / 2, -6])
                cube([rod_eye_plate_t, rod_eye_plate_w,
                      6 + rod_eye_reach_mm + rod_eye_fuse_len]);
            pip_square(rod_eye_reach_mm, m - rod_boss_len, rod_w);
            pip_socket_boss(m, rod_boss_w, rod_boss_len, rod_w);
        }
        translate([eye_x0 - 1, 0, 0]) rotate([0, 90, 0])
            cylinder(d = rod_eye_bore_d + 2 * rod_gap, h = rod_eye_plate_t + 2, $fn = 48);
        pip_square(rod_eye_reach_mm - 1, m, rod_bore);
        pip_socket_cavity(m, rod_socket_w, rod_socket_straight, rod_bore);
        rod_at_joint(0) pip_lever_housing_cuts(rod_joint_spec());
    }
    rod_at_joint(0) pip_lever_housing_adds(rod_joint_spec());
}

module rod_middle_body() {
    m0 = rod_joint_mouth_z[0];
    m1 = rod_joint_mouth_z[1];
    difference() {
        union() {
            pip_spigot(m0, rod_spigot_w, rod_spigot_insert, rod_spigot_flare, rod_w);
            pip_square(m0 + rod_spigot_flare - 0.01, m1 - rod_boss_len + 0.01, rod_w);
            pip_socket_boss(m1, rod_boss_w, rod_boss_len, rod_w);
        }
        pip_square(m0 - rod_spigot_insert - 1, m1, rod_bore);
        pip_socket_cavity(m1, rod_socket_w, rod_socket_straight, rod_bore);
        rod_spigot_pocket(0);
        rod_at_joint(1) pip_lever_housing_cuts(rod_joint_spec());
    }
    rod_at_joint(1) pip_lever_housing_adds(rod_joint_spec());
}

// A self-supporting lug rooted 0.4 mm into the sleeve's -X wall, 45-degree face toward the
// tip (down as printed), in the slider's -X groove with the gap all round.
module rod_capture_lug() {
    width = rod_groove_w - 2 * rod_gap;
    depth = rod_groove_depth + rod_gap;
    translate([-rod_bore / 2 - rod_gap, width / 2, rod_lug_z])
        rotate([90, 0, 0]) linear_extrude(height = width)
            polygon([[0, 0], [depth, 0], [depth, 2], [0, depth + 2]]);
}

// The sleeve: spigot into the middle section, tube, cam housing at the tip.
module rod_sleeve(lug = true) {
    m1 = rod_joint_mouth_z[1];
    difference() {
        union() {
            pip_spigot(m1, rod_spigot_w, rod_spigot_insert, rod_spigot_flare, rod_w);
            pip_square(m1 + rod_spigot_flare - 0.01, rod_tip_z, rod_w);
        }
        pip_square(m1 - rod_spigot_insert - 1, rod_tip_z + 1, rod_bore);
        rod_spigot_pocket(1);
        rod_at_cam() pip_lever_housing_cuts(rod_cam_spec());
        // Index notch for the ticks: a V in the +X wall's tip face.
        translate([rod_bore / 2 - 0.5, 0, rod_tip_z]) rotate([0, 90, 0])
            cylinder(r = 1.2, h = 5, $fn = 4);
    }
    rod_at_cam() pip_lever_housing_adds(rod_cam_spec());
    if (lug) rod_capture_lug();
}

// ---- Slider (slider-local: servo end z = 0, tip z = rod_slider_len) -------------------------------
// Graduation tick: a V across the +X flat, 0.4 mm deep.
module rod_tick(s, len) {
    translate([rod_slider / 2, -len / 2, s]) rotate([-90, 0, 0])
        linear_extrude(height = len) polygon([[0.1, -0.5], [-0.4, 0], [0.1, 0.5]]);
}

module rod_slider() {
    s = rod_slider;
    difference() {
        intersection() {
            translate([-s / 2, -s / 2, 0]) cube([s, s, rod_sq_len]);
            // 45-degree chamfer of the corners down to the round tip.
            cylinder(h = rod_sq_len, r1 = s / 2 + rod_sq_len, r2 = s / 2, $fn = 96);
        }
        translate([-s / 2 - 0.5, -rod_groove_w / 2, rod_groove_z0])
            cube([rod_groove_depth + 0.5, rod_groove_w, rod_groove_z1 - rod_groove_z0]);
        // Each tick is 0.02 mm longer than the one before it: no three tick ends share a
        // line, which otherwise leaves a zero-area triangle in the flat's triangulation.
        for (setting = [rods_span_min_mm:5:rods_span_min_mm + rod_coarse_travel])
            rod_tick(rod_tick_station(setting),
                     (setting % 50 == 0 ? 14 : (setting % 10 == 0 ? 9 : 5))
                     + 0.02 * (setting - rods_span_min_mm) / 5);
    }
    translate([0, 0, rod_sq_len - 0.01])
        cylinder(r = pip_get(rod_wheel_spec(), "root"), h = rod_thread_len + 0.01, $fn = 96);
    translate([0, 0, rod_ridge_z0]) pip_thread_ridge(rod_wheel_spec(), rod_ridge_a1());
}

// The slider at a setting, rod frame; `seat` pushes it across its play, away from the cam.
module rod_slider_at(length_mm, seat = false) {
    translate([0, seat ? -(rod_bore - rod_slider) / 2 : 0, rod_slider_start_z(length_mm)])
        rod_slider();
}
module rod_slider_print_pose() {
    translate([0, 0, rod_print_start_z()]) rod_slider();
}

// ---- Fork (fork frame) ------------------------------------------------------------------------
module rod_fork() {
    ws = rod_wheel_spec();
    eye_x0 = rod_fork_floor_x;
    difference() {
        union() {
            intersection() {
                rotate_extrude($fn = 160)
                    polygon(pip_wheel_body_profile(ws, rod_fork_top,
                                                   pip_get(ws, "crest") + pip_get(ws, "hgap")));
                translate([eye_x0, -50, -1]) cube([100, 100, rod_fork_top + 2]);
            }
            // Eye plate: rooted in the body above its roof, reaching past its top.
            translate([eye_x0, -rod_eye_plate_w / 2, rod_fork_top - 15])
                cube([rod_eye_plate_t, rod_eye_plate_w, rod_fork_eye_z + 6 - (rod_fork_top - 15)]);
        }
        translate([eye_x0 - 1, 0, rod_fork_eye_z]) rotate([0, 90, 0])
            cylinder(d = rod_eye_bore_d + 2 * rod_gap, h = rod_eye_plate_t + 2, $fn = 48);
    }
    pip_wheel_cage(ws, pip_get(ws, "roof_z0") + pip_get(ws, "post_r1")
                       - pip_get(ws, "wheel_r") - rod_gap + 1);
}
module rod_wheel() {
    pip_wheel(rod_wheel_spec());
}
// The wheel turned `turn` degrees from its print pose (thread and notches turn with it).
module rod_wheel_turned(turn) {
    rotate([0, 0, turn]) rod_wheel();
}
// The fork and wheel as installed at fine extension f, in the slider-tip frame (z = 0 at the
// slider's tip), wheel at its detent seat.
module rod_fork_set(f = 0) {
    translate([0, 0, -rod_wheel_tip_back + f]) {
        rod_fork();
        rod_wheel_turned(rod_wheel_seat(f));
    }
}
// The slider in the slider-tip frame.
module rod_slider_from_tip() {
    translate([0, 0, -rod_slider_len]) rod_slider();
}

// ---- Levers placed (phi: 0 as printed, 90 closed) ----------------------------------------------
module rod_cam_lever(phi = 0) {
    rod_at_cam() rotate([0, phi, 0]) pip_lever(rod_cam_spec());
}
module rod_joint_lever(j, phi = 0) {
    rod_at_joint(j) rotate([0, phi, 0]) pip_lever(rod_joint_spec());
}
module rod_joint_tooth(j, phi = 90) {
    rod_at_joint(j) rotate([0, phi, 0]) pip_lever_tooth(rod_joint_spec());
}
// The cam's housing (sleeve tip) and the parts carrying joint j's socket and spigot.
module rod_joint_socket_part(j) {
    if (j == 0) rod_servo_body(); else rod_middle_body();
}
module rod_joint_spigot_part(j) {
    if (j == 0) rod_middle_body(); else rod_sleeve();
}
// The spigot part seated by the closed lever.
module rod_joint_spigot_seated(j) {
    pip_lever_seat(rod_joint_spec(), rod_lever_side) rod_joint_spigot_part(j);
}
// The spigot's square envelope (no pocket), for the tooth probes.
module rod_spigot_envelope(j) {
    m = rod_joint_mouth_z[j];
    pip_square(m - rod_spigot_insert, m, rod_spigot_w);
}

// ---- The moving end and the fixed chain ---------------------------------------------------------
// Drawn CENTRED: the side play a closed lever takes up (0.4 mm at the slider, 0.6 mm per joint)
// is left out of the scene, so both eyes stay on their pins; a closed lever therefore overlaps
// its centred member in the drawing. The probes use the seated poses.
module rod_moving_end(length_mm) {
    rod_slider_at(length_mm);
    translate([0, 0, rod_slider_tip_z(length_mm)])
        rod_fork_set(rod_fine_extension_mm(length_mm));
}
module rod_fixed_chain() {
    rod_servo_body();
    rod_joint_lever(0, 90);
    rod_middle_body();
    rod_joint_lever(1, 90);
    rod_sleeve();
    rod_cam_lever(90);
}

// ---- Print poses -----------------------------------------------------------------------------
// Rod frame to a mouth-down print: z_print = mouth - z_rod (a rotation, not a mirror).
module rod_print_down(mouth) {
    translate([0, 0, mouth]) rotate([180, 0, 0]) children();
}
module rod_slab(z0, z1) {
    translate([-100, -100, z0]) cube([200, 200, z1 - z0]);
}

// ---- Public assembly and printable selectors -------------------------------------------------
rod_part = "assembly";
rod_len = 1066.8;   // illustrative default within the requested 940-1200 mm range

module throttle_pushrod(part = rod_part, length_mm = rod_len) {
    assert(length_mm >= rods_span_min_mm && length_mm <= rods_span_max_mm,
           "Illustrative rod setting must lie within the requested 940-1200 mm range");
    if (part == "assembly") {
        translate([rod_eye_crank_mm, 0, 0]) {
            rod_fixed_chain();
            rod_moving_end(length_mm);
        }
    } else if (part == "throttle_end") {
        // The moving end (slider, fork, wheel) in rod-local coordinates; linkage probes.
        translate([rod_eye_crank_mm, 0, 0]) rod_moving_end(length_mm);
    } else if (part == "servo_eye") {
        // The servo section with its closed joint lever, rod-local.
        translate([rod_eye_crank_mm, 0, 0]) {
            rod_servo_body();
            rod_joint_lever(0, 90);
        }
    } else if (part == "segment_lower") {
        rod_print_down(rod_joint_mouth_z[0]) {
            rod_servo_body();
            rod_joint_lever(0, 0);
        }
    } else if (part == "segment_middle") {
        rod_print_down(rod_joint_mouth_z[1]) {
            rod_middle_body();
            rod_joint_lever(1, 0);
        }
    } else if (part == "capture_assembly") {
        rod_print_down(rod_tip_z) {
            rod_sleeve();
            rod_slider_print_pose();
            rod_cam_lever(0);
        }
    } else if (part == "fork_assembly") {
        rod_fork();
        rod_wheel();
    } else if (part == "cam_coupon") {
        // Bench coupon: the real cam housing and lever on a 50 mm sleeve slice, with a plain
        // square slider stub to pull with a spring scale (closed cam, >= 185 N target).
        rod_print_down(rod_tip_z) {
            intersection() {
                rod_sleeve(lug = false);
                rod_slab(rod_tip_z - 50, rod_tip_z + 1);
            }
            pip_square(rod_tip_z - 50, rod_tip_z, rod_slider);
            rod_cam_lever(0);
        }
    } else if (part == "joint_coupon") {
        // Bench coupon: the real servo-section socket boss and its lever (50 mm slice).
        rod_print_down(rod_joint_mouth_z[0]) {
            intersection() {
                rod_servo_body();
                rod_slab(rod_joint_mouth_z[0] - 50, rod_joint_mouth_z[0] + 1);
            }
            rod_joint_lever(0, 0);
        }
    } else if (part == "joint_coupon_spigot") {
        // Mating spigot slice of the middle section (spigot, flare and 14 mm of tube).
        m = rod_joint_mouth_z[0];
        rod_print_down(m + 16)
            intersection() {
                rod_middle_body();
                rod_slab(m - rod_spigot_insert - 1, m + 16);
            }
    } else if (part == "gauge") {
        // Sliding-fit coupon: a plate with a closed square passage.
        linear_extrude(height = 14)
            difference() {
                translate([-20, -12]) square([40, 24]);
                square([rod_slider, rod_slider], center = true);
            }
    } else {
        assert(false, str("unknown pushrod part: ", part));
    }
}

throttle_pushrod(rod_part);
